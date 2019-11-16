from django.urls import path

from . import views

urlpatterns = [
    path('listRecord/', views.listRecord, name='listRecord'),
]