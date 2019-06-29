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
from datetime import datetime

from api.models import PersonRecognized


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

@csrf_exempt
def prueba(request):
    if request.method == 'POST':
        return JsonResponse({'method': 'post'})
    return JsonResponse({'method': 'get'})


def faceRecognition(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        #now = datetime.now()
        #date_time = now.strftime("%Y_%m_%d_%H_%M_%S")

        #fs = FileSystemStorage()
        #filename = fs.save(date_time+"_"+myfile.name, myfile)
        #uploaded_file_url = fs.url(filename)
        upurl=myfile.name
        path= Path().absolute()
        #uploaded_file_url= os.path.join(Path().absolute(),uploaded_file_url[1:])
        #person = proccessRecognition(uploaded_file_url)

    return JsonResponse({'hola': 'hola'})
    #return JsonResponse({'name': person.name, 'percent': person.percent})

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