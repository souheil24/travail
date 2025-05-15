import os
import sys
import django
import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime

# Ajouter le chemin du projet Django au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'PFE_APP.settings')
django.setup()

from scraping.models import poste

def publier_linkedin(message, link=None, image=None):
    """
    Publie un post sur LinkedIn en utilisant Selenium.
    
    Args:
        message (str): Le contenu du message à publier
        link (str, optional): Un lien à inclure dans le post
        image (File, optional): Une image à télécharger avec le post
    
    Returns:
        dict: Résultat de l'opération avec clés 'success' et 'error'
    """
    try:
        print("🚀 Démarrage de la publication LinkedIn...")
        
        # Configuration du navigateur
        options = uc.ChromeOptions()
        options.add_argument("--start-maximized")
        driver = uc.Chrome(version_main=136, options=options)
        
        # Accéder à LinkedIn
        driver.get("https://www.linkedin.com/")
        print("📱 Page LinkedIn ouverte. Connexion requise...")
        
        # Attendre que l'utilisateur se connecte manuellement (30 secondes max)
        wait = WebDriverWait(driver, 30)
        wait.until(lambda d: "feed" in d.current_url or "checkpoint" in d.current_url)
        
        # Vérifier si l'utilisateur est connecté
        if "feed" not in driver.current_url:
            driver.quit()
            return {"success": False, "error": "Échec de connexion à LinkedIn"}
        
        print("✅ Connecté à LinkedIn. Préparation du post...")
        
        # Cliquer sur le bouton "Créer un post"
        create_post_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Créer un post']/ancestor::button")))
        create_post_button.click()
        
        # Attendre que la fenêtre de création de post s'ouvre
        post_editor = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='textbox']")))
        
        # Saisir le message
        post_editor.send_keys(message)
        
        # Ajouter un lien si fourni
        if link:
            post_editor.send_keys("\n\n" + link)
        
        # Ajouter une image si fournie
        if image:
            # Sauvegarder temporairement l'image
            temp_path = f"/tmp/linkedin_post_image_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            with open(temp_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            
            # Cliquer sur le bouton d'ajout de média
            media_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Ajouter une photo']")))
            media_button.click()
            
            # Attendre que la boîte de dialogue de sélection de fichier apparaisse
            time.sleep(2)
            
            # Utiliser le chemin absolu de l'image
            file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
            file_input.send_keys(os.path.abspath(temp_path))
            
            # Attendre que l'image soit téléchargée
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "img[alt='Image du post']")))
            
            # Supprimer le fichier temporaire
            os.remove(temp_path)
        
        # Cliquer sur le bouton Publier
        publish_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Publier']/ancestor::button")))
        publish_button.click()
        
        # Attendre la confirmation de publication
        time.sleep(5)
        
        # Créer un enregistrement dans la base de données
        new_post = poste.objects.create(
            nom="Publication automatique",
            contenue=message,
            nombre_jaime=0,
            nombre_commentaire=0,
            nombre_republication=0
        )
        
        print(f"✅ Post LinkedIn publié avec succès! ID: {new_post.id_poste}")
        
        # Fermer le navigateur
        driver.quit()
        
        return {"success": True}
        
    except Exception as e:
        print(f"❌ Erreur lors de la publication LinkedIn: {str(e)}")
        try:
            if driver:
                driver.quit()
        except:
            pass
        return {"success": False, "error": str(e)}

# Pour tester le script en ligne de commande
if __name__ == "__main__":
    message = "Test de publication automatique depuis Python"
    result = publier_linkedin(message)
    print(result)