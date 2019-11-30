import base64
import sys
import math

from django.contrib.auth import authenticate
from django.core import serializers
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from fcm_django.fcm import fcm_send_topic_message
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view,permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_200_OK,
)
from pathlib import Path
import os
import numpy as np
import imutils
import pickle
import cv2
import logging
import io
from PIL import Image
from rest_framework.utils import json
from rest_framework_expiring_authtoken.models import ExpiringToken
from rest_framework.authtoken.serializers import AuthTokenSerializer

from api.models import PersonRecognized
from faceRecognition.util.utilMethod import GenericResult
from family.models import Family
from record.models import Record
from users.models import CustomUser

logger = logging.getLogger('testlogger')
pathImage = os.path.join(Path().absolute(), "opencv-face-recognition/images/out_2.jpg")

def index(request):
    logger.info("index in loggg!!!!!")
    return HttpResponse("Hello, world. You're at the polls index.")

@csrf_exempt
def recieveData(request):
    if request.method == 'POST':
        try:
            body_unicode = request.body
            logger.info("llegaste0")
            photo_infile = io.BytesIO(body_unicode)
            logger.info("llegaste1")
            photo_image = Image.open(photo_infile)
            logger.info(type(photo_image))
            photo_image.save(pathImage, "JPEG", quality=80, optimize=True, progressive=True)
        except Exception as e:
            s = str(e)
            logger.info(">>ERROR: " + s)
        return JsonResponse({'ok': 'ok'})
    return JsonResponse({'error': 'error'})

@csrf_exempt
def prueba(request):
    if request.method == 'POST' and request.FILES['myfile']:
        try:
            myfile = request.FILES['myfile']
            logger.info('llegaste1!!!!')
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            logger.info('>>path:' + uploaded_file_url)
            uploaded_file_url = os.path.join(Path().absolute(), uploaded_file_url[1:])
            person = proccessRecognition(uploaded_file_url)
            logger.info('llegaste2!!!!')
            return JsonResponse({'name': person.name, 'percent': person.percent})
        except Exception as e:
            s = str(e)
            logger.info(">>ERROR: "+s)
        return JsonResponse({'name': 'paso'})
    return JsonResponse({'method': 'get'})


# myfile = request.FILES['myfile']
# now = datetime.now()
# date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
# logger.info('llegaste!!!!')
# fs = FileSystemStorage()
##filename = fs.save(myfile.name, myfile)
# uploaded_file_url = fs.url(filename)
# uploaded_file_url = os.path.join(Path().absolute(),uploaded_file_url[1:])
#return JsonResponse({'hola': 'hola'})

@csrf_exempt
@api_view(["POST"])
def faceRecognition(request):
    user = request.user
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    image = body['image']

    fakePath = os.path.join(Path().absolute(), "opencv-face-recognition/images/out_fake.jpg")
    truePath = os.path.join(Path().absolute(), "opencv-face-recognition/images/out.jpg")

    uploaded_file_url = base64ToImage(image, fakePath)

    #uploaded_file_url = bytesToImage(request, pathImage)

    if(uploaded_file_url is 'ERROR'):
        return JsonResponse((GenericResult(True, HTTP_200_OK, "ok", {'name': '', 'percent': 0})).__dict__)
        #return JsonResponse({'error':'error'})

    img_rt_90 = rotate_img(uploaded_file_url, 90)
    img_rt_90.save(truePath)
    os.remove(fakePath)

    person = proccessRecognition(truePath, user.id)
    logger.info("nombre: "+person.name)
    logger.info("percent: " + str(person.percent))
    return JsonResponse((GenericResult(True, HTTP_200_OK, "ok", {'name': person.name, 'percent': person.percent})).__dict__)

    #return JsonResponse({'name': person.name, 'percent': person.percent})

def proccessRecognition(urlImage, clientId):
    pathOpenCV = os.path.join(Path().absolute(), "opencv-face-recognition")
    person = PersonRecognized()
    protoPath = os.path.sep.join(
        [os.path.join(pathOpenCV,'face_detection_model'), "deploy.prototxt"])
    modelPath = os.path.sep.join(
        [os.path.join(pathOpenCV,'face_detection_model'),
         "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    embedder = cv2.dnn.readNetFromTorch(
        os.path.join(pathOpenCV,'openface_nn4.small2.v1.t7'))

    recognizer = pickle.loads(
        open(os.path.join(pathOpenCV, 'output/'+('' if clientId == 0 else (str(clientId)+'/'))+'recognizer.pickle'), "rb").read())
    le = pickle.loads(
        open(os.path.join(pathOpenCV, 'output/'+('' if clientId == 0 else (str(clientId)+'/'))+'le.pickle'), "rb").read())
    image = cv2.imread(urlImage)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    #cv2.imshow('image', image)
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    proba = 0.0
    name = "Desconocido"
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j] * 100
            name = le.classes_[j]
    person.name = name
    person.percent = math.trunc(proba)
    return person

def bytesToImage(request, miPath):
    try:
        body_unicode = request.body
        photo_infile = io.BytesIO(body_unicode)
        photo_image = Image.open(photo_infile)
        photo_image.save(miPath, "JPEG", quality=100, optimize=True, progressive=True)
        result = miPath
    except Exception as e:
        result = "ERROR"
        s = str(e)
        logger.info(">>ERROR: " + s)
    return result

def base64ToImage(image, miPath):
    try:
        image += "=" * ((4 - len(image) % 4) % 4)
        imgdata = base64.b64decode(image)
        with open(miPath, 'wb') as f:
            f.write(imgdata)
        result = miPath
    except Exception as e:
        result = "ERROR"
        _, _, exc_tb = sys.exc_info()
        s = str(e)
        logger.info(">>ERROR base64ToImage: " + s)
        logger.info(">>LINE ERROR: " + str(exc_tb.tb_lineno))
    return result



@csrf_exempt
@api_view(["POST"])
@permission_classes((AllowAny,))
def login(request):
    response = None
    try:
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        username = body['userName']
        password = body['password']
        if username is None or password is None:
            response = JsonResponse((GenericResult(False, HTTP_400_BAD_REQUEST, "field none", {})).__dict__)
            return response

        #serializer = AuthTokenSerializer(data={'Username': username, 'Password': password})
        #logger.info(">>valid: " + str(serializer.is_valid()))
        user = authenticate(username=username, password=password)
        if not user:
            response = JsonResponse((GenericResult(False, HTTP_404_NOT_FOUND, "no agent", {})).__dict__)
            return response

        token = ExpiringToken.objects.create(user=user)

        logger.info(">>LOGIN OK: " + user.username)

        response = JsonResponse((GenericResult(True, HTTP_200_OK, "ok", {'token': token.key, 'user': user.id})).__dict__)
        return response
    except Exception as e:
        s = str(e)
        logger.info(">>ERROR LOGIN: " + s)
        response = JsonResponse((GenericResult(False, HTTP_400_BAD_REQUEST, "error: "+s, {})).__dict__)
        return response

@csrf_exempt
@api_view(["GET"])
def logout(request):
    token, _ = ExpiringToken.objects.get_or_create(user=request.user)
    token.delete()
    logger.info(">>logout: " + request.user.username)
    response = JsonResponse((GenericResult(True, HTTP_200_OK, "ok", {})).__dict__)
    return response

@csrf_exempt
@api_view(["GET"])
def isTokenExpire(request):
    token, _ = ExpiringToken.objects.get_or_create(user=request.user)
    logger.info(">>tokenExpire: " + str(token.expired()))
    if token.expired():
        token.delete()
    response = JsonResponse((GenericResult(True, HTTP_200_OK, "ok", {'tokenExpired': token.expired()})).__dict__)
    return response

@csrf_exempt
@api_view(["GET"])
def refreshToken(request):
    token, _ = ExpiringToken.objects.get_or_create(user=request.user)
    token.delete()
    token = ExpiringToken.objects.create(user=request.user)
    response = JsonResponse((GenericResult(True, HTTP_200_OK, "ok", {'newToken': token.key})).__dict__)
    logger.info(">>newToken: " + token.key)
    return response

@csrf_exempt
@api_view(["GET"])
@permission_classes((AllowAny,))
def sendNot(request):
    fcm_send_topic_message(topic_name='my-event', message_body='un objeto se encuentra cerca de la puerta', message_title='Mensaje arduino')

    return HttpResponse("<h1>Ok</h1>")

@csrf_exempt
@api_view(["POST"])
@permission_classes((AllowAny,))
def receiveImage(request, idclient):
    logger.info(">>receive image idcliente: " + str(idclient))
    try:
        lastRecordId = Record.objects.latest('idRecord').idRecord+1
    except Record.DoesNotExist:
        lastRecordId = 1
    logger.info(">>future lastRecordId: " + str(lastRecordId))

    directoryByClient = os.path.join(Path().absolute(), "opencv-face-recognition/dataset/records/"+str(idclient))

    if not os.path.exists(directoryByClient):
        os.makedirs(directoryByClient)

    feikPath = directoryByClient+"/record_"+str(lastRecordId)+"_feik.jpg"
    truePath = directoryByClient+"/record_"+str(lastRecordId)+".jpg"

    #pathForeingImage = directoryByClient+"/record_"+str(lastRecordId)+".jpg"
    response = bytesToImage(request, feikPath)
    if response is 'ERROR':
        logger.info(">>ERROR UPLOAD IMAGEN FROM ARDUINO")
        return HttpResponse("<h1>ERROR</h1>")

    img_rt_90 = rotate_img(feikPath, -90)
    img_rt_90.save(truePath)
    os.remove(feikPath)

    personRecognition = proccessRecognition(truePath, idclient)
    logger.info("person: "+personRecognition.name)
    logger.info("percent: "+str(personRecognition.percent))
    if personRecognition.name is 'unknown':
        myFamily= Family(familyName="Desconocido", relationship="Desconocido")
    else:
        try:
            myFamily = Family.objects.get(familyName=personRecognition.name)
        except Family.DoesNotExist:
            myFamily = Family(familyName=personRecognition.name, relationship="Desconocido")


    #user = User.objects.get(id=idclient)
    user = CustomUser.objects.get(id=idclient)

    arrayPath = truePath.split("/", 2)
    photoPath = arrayPath[2]

    myRecord = Record(familyName=personRecognition.name, relationship=myFamily.relationship,
                      percent=str(personRecognition.percent), recordPhotoPath=photoPath, idClient=user)
    myRecord.save()
    logger.info(">>id record: " + str(myRecord.idRecord))
    fcm_send_topic_message(topic_name='my-event', message_body='la persona '+myRecord.familyName+' ingreso a tu hogar a las '+myRecord.dateRecord.strftime("%Y-%m-%d %H:%M:%S"),
                           message_title='Mensaje Arduino')

    return HttpResponse("<h1>OK</h1>")

def rotate_img(img_path, rt_degr):
    img = Image.open(img_path)
    return img.rotate(rt_degr, expand=1)