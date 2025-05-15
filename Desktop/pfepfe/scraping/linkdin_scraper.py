import sys
import os
import django
import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import re

# 👉 Ajouter le chemin du projet Django au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'linkedin.settings')
django.setup()

from scraping.models import comments, poste

driver = None  # ✅ Driver global pour maintenir la session ouverte

def extraire_nombre(texte):
    match = re.search(r'\d+', texte.replace(',', '').replace(' ', ''))
    return int(match.group()) if match else 0

def lancer_navigation_linkedin():
    global driver
    print("🟢 Étape 1 : Ouverture de LinkedIn dans le navigateur")
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = uc.Chrome(version_main=136, options=options)
    driver.get("https://www.linkedin.com/")
    print("🔐 Connecte-toi à LinkedIn et ouvre un post avec commentaires visibles.")

def extraire_et_enregistrer():
    global driver
    if not driver:
        print("❌ Le navigateur n’est pas lancé.")
        return

    if "login" in driver.current_url:
        print("❌ Tu n’es pas connecté à LinkedIn.")
        driver.quit()
        return

    print("🔍 Extraction du contenu du post...")

    try:
        content_elem = driver.find_element(By.CSS_SELECTOR, ".feed-shared-update-v2__description")
        author_elem = driver.find_element(By.CSS_SELECTOR, ".visually-hidden")
        nb_jaime = driver.find_element(By.CSS_SELECTOR, ".social-details-social-counts__reactions-count").text.strip()
        nb_commentaires = driver.find_element(By.CSS_SELECTOR, ".social-details-social-counts__comments").text.strip()
        nb_reposte = driver.find_element(By.CSS_SELECTOR, ".social-details-social-counts__item--truncate-text").text.strip()

        poste_obj = poste.objects.create(
            nom=author_elem.text.strip(),
            contenue=content_elem.text.strip(),
            nombre_jaime=extraire_nombre(nb_jaime),
            nombre_commentaire=extraire_nombre(nb_commentaires),
            nombre_republication=extraire_nombre(nb_reposte)
        )

        author_blocks = driver.find_elements(By.CSS_SELECTOR, ".comments-comment-meta__description-title")
        comment_blocks = driver.find_elements(By.CSS_SELECTOR, ".comments-comment-item__main-content")

        all_authors = [a.text.strip() if a else "Auteur introuvable" for a in author_blocks]
        all_comments = [c.text.strip() if c else "Commentaire introuvable" for c in comment_blocks]

        for i in range(min(len(all_authors), len(all_comments))):
            comments.objects.create(
                nom=all_authors[i],
                texte=all_comments[i],
                id_poste=poste_obj
            )

        print("✅ Données enregistrées avec succès.")
    except Exception as e:
        print(f"❌ Erreur pendant l'extraction : {e}")
    finally:
        driver.quit()
        driver = None