from django.db import models

# Create your models here.
from django.db import models

class Inscription(models.Model):
    nom = models.CharField(max_length=100)
    prenom = models.CharField(max_length=100)
    numero_carte = models.CharField(max_length=18, unique=True)
    taille = models.CharField(
        max_length=3,
        choices=[("S", "S"), ("M", "M"), ("L", "L"), ("XL", "XL"),("XXL","XXL")]
    )

    def __str__(self):
        return f"{self.nom} {self.prenom}"



