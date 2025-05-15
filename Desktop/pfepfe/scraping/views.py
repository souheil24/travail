from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from .models import FacebookPost
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Create your views here.
