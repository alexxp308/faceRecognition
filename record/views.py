import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.status import HTTP_200_OK, HTTP_400_BAD_REQUEST

from faceRecognition.util.utilMethod import GenericResult
from family.serializers import MySerialiser
from record.models import Record

logger = logging.getLogger('testlogger')

@csrf_exempt
@api_view(["GET"])
def listRecord(request):
    try:
        qs = Record.objects.filter(idClient=request.user)
        serializer = MySerialiser()
        qs_json = serializer.serialize(qs)
        logger.info(">>list record")
        response = JsonResponse((GenericResult(True, HTTP_200_OK, "record list", {'recordList': qs_json})).__dict__)
    except Exception as e:
        s = str(e)
        logger.info(">>ERROR: " + s)
        response = JsonResponse((GenericResult(False, HTTP_400_BAD_REQUEST, "error: "+s, {})).__dict__)

    return response
