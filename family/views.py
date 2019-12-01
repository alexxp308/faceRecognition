import base64
import os
import shutil
import threading

import imutils
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from imutils import paths
from rest_framework.decorators import api_view
from rest_framework.status import HTTP_400_BAD_REQUEST, HTTP_200_OK
from rest_framework.utils import json
import logging

from faceRecognition.util.utilMethod import GenericResult
from family.models import Family
from family.serializers import MySerialiser
import cv2
import pickle
import numpy as np
import io
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from fcm_django.fcm import fcm_send_topic_message

logger = logging.getLogger('testlogger')

@csrf_exempt
@api_view(["GET"])
def listFamily(request):
    try:
        qs = Family.objects.filter(idClient=request.user)
        serializer = MySerialiser()
        qs_json = serializer.serialize(qs)
        response = JsonResponse((GenericResult(True, HTTP_200_OK, "family list", {'familyList': qs_json})).__dict__)
    except Exception as e:
        s = str(e)
        logger.info(">>ERROR: " + s)
        response = JsonResponse((GenericResult(False, HTTP_400_BAD_REQUEST, "error: "+s, {})).__dict__)

    return response

@csrf_exempt
@api_view(["POST"])
def createFamily(request):
    try:
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        familyName = body['familyName']
        relationship = body['relationship']
        images = body['images']
        user = request.user
        myFamily = Family(familyName=familyName, relationship=relationship, idClient=user)
        myFamily.save()
        logger.info(">>id family: "+str(myFamily.idFamily))
        thread = threading.Thread(target=createFacilDetection, args=(user.id, familyName.replace(" ", "_"), images, myFamily.idFamily))
        thread.start()
        response = JsonResponse((GenericResult(True, HTTP_200_OK, "family member created", {'familyList': str(myFamily.idFamily)})).__dict__)
    except Exception as e:
        s = str(e)
        logger.info(">>ERROR: " + s)
        response = JsonResponse((GenericResult(False, HTTP_400_BAD_REQUEST, "error: "+s, {})).__dict__)

    return response

@csrf_exempt
@api_view(["POST"])
def updateFamily(request):
    try:
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        idFamily = body['idFamily']
        familyName = body['familyName']
        relationship = body['relationship']
        objFamily = Family.objects.filter(pk=idFamily)
        objFamily.update(familyName=familyName, relationship=relationship)
        serializer = MySerialiser()
        qs_json = serializer.serialize(objFamily)

        response = JsonResponse((GenericResult(True, HTTP_200_OK, "family member update", {'idFamily': qs_json})).__dict__)
    except Exception as e:
        s = str(e)
        logger.info(">>ERROR: " + s)
        response = JsonResponse((GenericResult(False, HTTP_400_BAD_REQUEST, "error: "+s, {})).__dict__)

    return response

def createFacilDetection(idClient, familyName, images, idFamily):
    listPhotos = createDataset(idClient, familyName, images)
    if len(listPhotos) > 0:
        result = createUnknowFolder(idClient)
        if result != 0:
            result = extract_embeddings(idClient, familyName, result)
            logger.info(">>extract_embeddings OK")
            logger.info(">>>>>>>>>>> value result: " + ("Si!!!" if result else "No!!!"))
            if result:
                result = train_Model(idClient)
                if result:
                    logger.info(">>train_Model OK")
                    Family.objects.filter(pk=idFamily).update(familyPhotos=','.join(listPhotos))
                    fcm_send_topic_message(topic_name='my-event',
                                           message_body='El usuario ' + familyName + " fue creado exitosamente, verifique el modulo de pruebas si desea comprobarlo",
                                           message_title='Usuario Creado')
                    return

    fcm_send_topic_message(topic_name='my-event',
                           message_body='verifique la informacion subida, o comuniquese con el proveedor',
                           message_title='Error en la creacion')
    return

def createDataset(idClient, familyName, images):
    result = []
    try:
        iterator = 0
        for image in images:
            resultConvert = bytesToImageAndSave(idClient, familyName, image, iterator)
            if resultConvert is "ERROR":
                logger.info(">>ERROR image " + iterator + " cant converter")
            else:
                result.append(resultConvert)
                iterator+=1

    except Exception as e:
        s = str(e)
        logger.info(">>ERROR createDataset: " + s)

    return result

def extract_embeddings(idClient, familyName, existsEmbedding):
    try:
        # load our serialized face detector from disk
        logger.info("[INFO] loading face detector...")

        dirRecognition = "opencv-face-recognition"
        dirRecognitionModel = dirRecognition + "/face_detection_model"

        protoPath = os.path.join(Path().absolute(), dirRecognitionModel + "/deploy.prototxt")
        modelPath = os.path.join(Path().absolute(), dirRecognitionModel + "/res10_300x300_ssd_iter_140000.caffemodel")
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        embederPath = os.path.join(Path().absolute(), dirRecognition + "/openface_nn4.small2.v1.t7")

        # load our serialized face embedding model from disk
        logger.info("[INFO] loading face recognizer...")
        embedder = cv2.dnn.readNetFromTorch(embederPath)

        datasetPath = os.path.join(Path().absolute(), dirRecognition + "/dataset/" + str(idClient) + (("/" + familyName) if existsEmbedding == 2 else ""))

        # grab the paths to the input images in our dataset
        logger.info("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(datasetPath))

        # initialize our lists of extracted facial embeddings and
        # corresponding people names

        directoryOutput = os.path.join(Path().absolute(), dirRecognition + "/output/" + str(idClient))
        filePathOutput = directoryOutput + "/embeddings.pickle"

        if not os.path.exists(directoryOutput):
            os.makedirs(directoryOutput)

        if not os.path.exists(filePathOutput):
            with open(filePathOutput, 'w'): pass

        knownEmbeddings = []
        knownNames = []

        if existsEmbedding == 2:
            data = pickle.loads(open(filePathOutput, "rb").read())
            knownEmbeddings = data["embeddings"]
            knownNames = data["names"]

        # initialize the total number of faces processed
        total = 0

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            logger.info("[INFO] processing image {}/{}".format(i + 1,
                                                         len(imagePaths)))
            name = (familyName if existsEmbedding == 2 else imagePath.split(os.path.sep)[-2])
            #name = familyName
            logger.info(">>>>>>>>NAME:" + name)


            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            detector.setInput(imageBlob)
            detections = detector.forward()

            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

        # dump the facial embeddings + names to disk
        logger.info("[INFO] serializing {} encodings...".format(total))
        data = {"embeddings": knownEmbeddings, "names": knownNames}

        f = open(filePathOutput, "wb")
        f.write(pickle.dumps(data))
        f.close()
        result = True
    except Exception as e:
        s = str(e)
        logger.info(">>ERROR extract_embeddings: " + s)
        result = False

    return result


def train_Model(idClient):
    try:
        dirOutput = os.path.join(Path().absolute(), "opencv-face-recognition/output/" + str(idClient))

        embeddingPath = dirOutput + "/embeddings.pickle"

        # load the face embeddings
        print("[INFO] loading face embeddings...")
        data = pickle.loads(open(embeddingPath, "rb").read())

        # encode the labels
        print("[INFO] encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        # train the model used to accept the 128-d embeddings of the face and
        # then produce the actual face recognition
        print("[INFO] training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)

        # write the actual face recognition model to disk
        recognizerPath = dirOutput + "/recognizer.pickle"

        if not os.path.exists(recognizerPath):
            with open(recognizerPath, 'w'): pass

        f = open(recognizerPath, "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        # write the label encoder to disk
        lePath = dirOutput + "/le.pickle"

        if not os.path.exists(lePath):
            with open(lePath, 'w'): pass

        f = open(lePath, "wb")
        f.write(pickle.dumps(le))
        f.close()
        result = True
    except Exception as e:
        s = str(e)
        logger.info(">>ERROR extract_embeddings: " + s)
        result = False

    return result


def bytesToImageAndSave(idClient,familyName,image,iterator):
    try:
        directoryClientFamily = "opencv-face-recognition/dataset/"+str(idClient) + "/" + familyName

        if not os.path.exists(directoryClientFamily):
            os.makedirs(directoryClientFamily)

        pathImage = os.path.join(Path().absolute(), directoryClientFamily + "/"+("000"+str(iterator) if iterator > 9 else "0000"+str(iterator))+".jpg")

        imgdata = base64.b64decode(image)
        with open(pathImage, 'wb') as f:
            f.write(imgdata)

        arrayPath = pathImage.split("/", 2)
        logger.info(">>array: " + arrayPath[2])
        result = arrayPath[2]
    except Exception as e:
        result = "ERROR"
        s = str(e)
        logger.info(">>ERROR bytesToImage: " + s)

    return result

def createUnknowFolder(idClient):
    try:
        datasetPath = os.path.join(Path().absolute(), "opencv-face-recognition/dataset")
        src = datasetPath+"/unknown"
        dirPhotos = datasetPath+"/"+str(idClient)
        dst = dirPhotos + "/unknown"
        number_dirs = 0

        for _, dirnames, _ in os.walk(dirPhotos):
            number_dirs += len(dirnames)

        logger.info(">>cant dirs: " + str(number_dirs))

        if number_dirs > 1:
            logger.info(">>Unknow Folder wasnt created cause its not necesary")
            result = 2
            return result

        if not os.path.exists(dst):
            os.makedirs(dst)

        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, False, None)
            else:
                shutil.copy2(s, d)

        result = 1
        logger.info(">>OK createUnknowFolder")
    except Exception as e:
        result = 0
        s = str(e)
        logger.info(">>ERROR createUnknowFolder: " + s)

    return result