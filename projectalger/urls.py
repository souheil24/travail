from django.contrib import admin
from django.urls import path,include
from zonereghaia import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path("", include('zonereghaia.urls')),
    path('formulaire/', views.formulaire_view, name='formulaire'),
    path('', views.formulaire_view, name='home'),  # optionnel
]
