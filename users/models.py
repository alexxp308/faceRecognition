from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    cell_phone_number = models.CharField(max_length=20, blank=True)
    recognizer_path = models.CharField(max_length=100, blank=True)