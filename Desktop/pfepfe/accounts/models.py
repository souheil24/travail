from django.contrib.auth.models import AbstractUser
from django.db import models
from django.conf import settings


class CustomUser(AbstractUser):
    is_employee = models.BooleanField(default=False)
    profile_image = models.ImageField(upload_to='profile_images/', null=True, blank=True)

    def __str__(self):
        return self.username

class Employee(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    firstname = models.CharField(max_length=100)
    start_date = models.DateField()
    address = models.TextField()

    def __str__(self):
        return self.firstname

        from django.db import models

class DemoRequest(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=20)
    role = models.CharField(max_length=50)
    company = models.CharField(max_length=100, blank=True)  # Champ optionnel
    created_at = models.DateTimeField(auto_now_add=True)


class Comment_Demo(models.Model):
    id_comment = models.AutoField(primary_key=True)
    text = models.TextField()
    sentiment = models.CharField(max_length=20, blank=True, null=True)
    category = models.CharField(max_length=50, blank=True, null=True)

    def __str__(self):
        return self.text
