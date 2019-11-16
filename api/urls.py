from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('faceRecognition/', views.faceRecognition, name='faceRecognition'),
    path('prueba/', views.prueba, name='prueba'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('isTokenExpire/', views.isTokenExpire, name='isTokenExpire'),
    path('refreshToken/', views.refreshToken, name='refreshToken'),
    path('recieveData/', views.recieveData, name='recieveData'),
    #path('sendNotification/', views.sendNotification, name='sendNotification'),
    path('sendNot/', views.sendNot, name='sendNot'),
    path('receiveImage/<int:idclient>', views.receiveImage, name='receiveImage')
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)