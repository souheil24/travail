from django.urls import path
from . import views

app_name = 'projectalger'

urlpatterns = [
    path('formulaire/', views.formulaire, name='formulaire'),

   
]