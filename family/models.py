from django.db import models

from faceRecognition import settings


class Family(models.Model):
    idFamily = models.AutoField(primary_key=True)
    familyName = models.CharField(max_length=100)
    relationship = models.CharField(max_length=100)
    familyPhotos = models.CharField(max_length=5000)
    idClient = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
