from django.urls import path

from . import views

urlpatterns = [
    #path('', views.index, name='index'),
    path('listFamily/', views.listFamily, name='listFamily'),
    path('createFamily/', views.createFamily, name='createFamily'),
    path('updateFamily/', views.updateFamily, name='updateFamily'),
]
