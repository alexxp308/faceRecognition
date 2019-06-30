from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('faceRecognition/', views.faceRecognition, name='faceRecognition'),
    path('prueba/', views.prueba, name='prueba'),
    path('recieveData/', views.recieveData, name='recieveData'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)