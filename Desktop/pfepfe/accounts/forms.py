from django import forms
from django.contrib.auth.forms import AuthenticationForm
from .models import Employee, CustomUser
from django.db.models import Q

class EmployeeLoginForm(AuthenticationForm):
    username = forms.CharField(
        label="Email ou nom d'utilisateur",
        widget=forms.TextInput(attrs={
            'class': 'newsletter-form-text-field w-input',
            'placeholder': 'Entrez votre email ou nom d\'utilisateur'
        })
    )
    password = forms.CharField(
        label="Mot de passe",
        widget=forms.PasswordInput(attrs={
            'class': 'newsletter-form-text-field w-input',
            'placeholder': 'Entrez votre mot de passe'
        })
    )

    def clean(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')

        if username and password:
            try:
                # Chercher l'utilisateur par username ou email
                user = CustomUser.objects.get(
                    Q(username=username) | Q(email=username)
                )
                if not user.is_employee and not user.is_superuser:
                    raise forms.ValidationError("Ce compte n'a pas les permissions nécessaires.")
            except CustomUser.DoesNotExist:
                raise forms.ValidationError("Identifiants incorrects.")
        
        return super().clean()

class EmployeeCreationForm(forms.ModelForm):
    username = forms.CharField(
        max_length=150,
        label="Nom d'utilisateur",
        widget=forms.TextInput(attrs={'placeholder': 'Entrez le nom d\'utilisateur'})
    )
    email = forms.EmailField(
        label="Email",
        widget=forms.EmailInput(attrs={'placeholder': 'Entrez l\'email'})
    )
    password = forms.CharField(
        label="Mot de passe",
        widget=forms.PasswordInput(attrs={'placeholder': 'Entrez le mot de passe'})
    )
    
    class Meta:
        model = Employee
        fields = ['firstname', 'start_date', 'address']

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get('username')
        email = cleaned_data.get('email')

        if CustomUser.objects.filter(username=username).exists():
            raise forms.ValidationError('Ce nom d\'utilisateur existe déjà.')
        
        if CustomUser.objects.filter(email=email).exists():
            raise forms.ValidationError('Cet email existe déjà.')

        return cleaned_data
