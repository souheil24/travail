from django.shortcuts import render, redirect
from .forms import InscriptionForm

def formulaire_view(request):
    if request.method == 'POST':
        form = InscriptionForm(request.POST)
        if form.is_valid():
            form.save()  # si ton form est lié à un modèle
            return render(request, 'formulaire.html', {'success': True, 'form': InscriptionForm()})
    else:
        form = InscriptionForm()
    return render(request, 'formulaire.html', {'form': form})
