from django.http import HttpResponse
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
import os
import numpy as np
import imutils
import pickle
import cv2
import logging
import io
from PIL import Image
from datetime import datetime

from rest_framework.utils import json

from api.models import PersonRecognized

logger = logging.getLogger('testlogger')
pathImage = os.path.join(Path().absolute(), "opencv-face-recognition/images/out.jpg")

def index(request):
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

@csrf_exempt
def faceRecognition(request):
    # and request.FILES['myfile']:
    if request.method == 'POST':
        #myfile = request.FILES['myfile']
        #now = datetime.now()
        #date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        #logger.info('llegaste!!!!')
        #fs = FileSystemStorage()
        ##filename = fs.save(myfile.name, myfile)
        #uploaded_file_url = fs.url(filename)
        #uploaded_file_url = os.path.join(Path().absolute(),uploaded_file_url[1:])
        uploaded_file_url = bytesToImage(request)
        if(uploaded_file_url is 'ERROR'):
            return JsonResponse({'error':'error'})
        person = proccessRecognition(uploaded_file_url)
        #if os.path.exists(uploaded_file_url):
            #os.remove(uploaded_file_url)
        logger.info("nombre: "+person.name)
        logger.info("percent: " + str(person.percent))
    #return JsonResponse({'hola': 'hola'})
    return JsonResponse({'name': person.name, 'percent': person.percent})

def proccessRecognition(urlImage):
    pathOpenCV = os.path.join(Path().absolute(),"opencv-face-recognition")
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
        open(os.path.join(pathOpenCV,'output/recognizer.pickle'), "rb").read())
    le = pickle.loads(
        open(os.path.join(pathOpenCV,'output/le.pickle'), "rb").read())
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
    name = ""
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
    person.percent = proba
    return person

def bytesToImage(request):
    result = "";
    try:
        body_unicode = request.body
        photo_infile = io.BytesIO(body_unicode)
        photo_image = Image.open(photo_infile)
        photo_image.save(pathImage, "JPEG", quality=100, optimize=True, progressive=True)
        result = pathImage
    except Exception as e:
        result = "ERROR"
        s = str(e)
        logger.info(">>ERROR: " + s)
    return result