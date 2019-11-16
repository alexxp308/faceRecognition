from django.db import models

from faceRecognition import settings
from django.utils.timezone import now

class Record(models.Model):
    idRecord = models.AutoField(primary_key=True)
    familyName = models.CharField(max_length=100)
    relationship = models.CharField(max_length=100)
    percent = models.CharField(max_length=100)
    recordPhotoPath = models.CharField(max_length=500)
    dateRecord = models.DateTimeField(default=now, editable=False)
    idClient = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
