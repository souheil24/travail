import requests
import time
import json
from datetime import datetime
import os
import django
import sys

# 👉 Configuration personnalisable
ACCESS_TOKEN = "EAAOa69C97g4BO3vQmZBEGrYJynS3VWr7HZCn2ZAtkmXQDc5VujNmcPoPdUVnVw0kuKYtW46lGYhIqZATCWn9NWZBS7MkigS2fykag5ZBOU6CZCCS9LZAf9n72Pv9AO2ukiT7ZC6r6lLzgtYWCiqeaHylhKomC2KvqDU1ZCL7wOd7HZCeZBIVMpwDJLRjZCcdw98whZCJoZD"
PAGE_ID = "106515182375055"
GRAPH_URL = "https://graph.facebook.com/v19.0"
OUTPUT_DIR = "facebook_data"  # Dossier pour sauvegarder les données
RATE_LIMIT_DELAY = 1  # Délai entre les requêtes pour éviter les limites de taux (en secondes)

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 🔹 Fonction pour gérer les erreurs d'API
def handle_api_error(response_data):
    if "error" in response_data:
        error = response_data["error"]
        print(f"❌ Erreur API: {error.get('message', 'Erreur inconnue')} (Code: {error.get('code', 'inconnu')})")
        
        # Vérifier si le token est expiré
        if error.get('code') == 190:
            print("🔑 Le token d'accès semble être expiré ou invalide. Veuillez générer un nouveau token.")
        
        # Vérifier si on a atteint une limite de taux
        if error.get('code') == 4 or error.get('code') == 17:
            wait_time = 60  # Attendre 1 minute par défaut
            print(f"⏱️ Limite de taux atteinte. Attente de {wait_time} secondes...")
            time.sleep(wait_time)
            return True  # Indique qu'il faut réessayer
        
        return False  # Ne pas réessayer pour les autres erreurs
    return False

# 🔹 Fonction pour effectuer des requêtes API avec gestion des erreurs
def make_api_request(url, params=None, retry_count=0, max_retries=3):
    try:
        # Attendre pour respecter les limites de taux
        time.sleep(RATE_LIMIT_DELAY)
        
        # Effectuer la requête
        response = requests.get(url, params=params if '?' not in url else None)
        data = response.json()
        
        # Gérer les erreurs
        if handle_api_error(data) and retry_count < max_retries:
            return make_api_request(url, params, retry_count + 1, max_retries)
        
        return data
    except Exception as e:
        print(f"❌ Erreur lors de la requête: {str(e)}")
        if retry_count < max_retries:
            wait_time = 5 * (retry_count + 1)
            print(f"⏱️ Tentative de reconnexion dans {wait_time} secondes...")
            time.sleep(wait_time)
            return make_api_request(url, params, retry_count + 1, max_retries)
        return {"data": []}

# 🔹 Vérifier la validité du token d'accès
def verify_token():
    url = f"{GRAPH_URL}/debug_token"
    params = {
        "input_token": ACCESS_TOKEN,
        "access_token": ACCESS_TOKEN
    }
    
    response = make_api_request(url, params)
    
    if "data" in response:
        token_data = response["data"]
        if token_data.get("is_valid", False):
            expires_at = token_data.get("expires_at", 0)
            
            if expires_at == 0:
                print("✅ Token d'accès valide (ne expire pas)")
            else:
                expires_date = datetime.fromtimestamp(expires_at)
                print(f"✅ Token d'accès valide jusqu'au {expires_date}")
            
            scopes = token_data.get('scopes', [])
            print(f"🔓 Permissions: {', '.join(scopes)}")
            
            return True
        else:
            print("❌ Token d'accès invalide")
            return False
    else:
        print("❌ Impossible de vérifier le token d'accès")
        return False

# 🔹 Récupérer tous les posts de la page avec pagination complète
def get_all_posts(since_date=None):
    posts = []
    url = f"{GRAPH_URL}/{PAGE_ID}/posts"
    params = {
        "access_token": ACCESS_TOKEN,
        "fields": "id,message,full_picture,created_time,permalink_url,attachments{type,url,media,title,description},shares",
        "limit": 100  # Utiliser la limite maximale pour réduire le nombre de requêtes
    }
    
    # Ajouter un filtre de date si spécifié
    if since_date:
        params["since"] = since_date
    
    page_count = 0
    print("📃 Récupération des posts...")
    
    while url:
        page_count += 1
        print(f"  Page {page_count} de posts en cours de chargement...")
        
        data = make_api_request(url, params if '?' not in url else None)
        new_posts = data.get("data", [])
        posts.extend(new_posts)
        print(f"  ➕ {len(new_posts)} posts ajoutés")
        
        # Obtenir l'URL pour la prochaine page
        url = data.get("paging", {}).get("next")
    
    print(f"✅ Total de {len(posts)} posts récupérés")
    return posts

# 🔹 Nettoyer le texte des commentaires (mots/noms à remplacer)
def clean_comment_text(text):
    # Liste de mots communs qui peuvent être des noms (à adapter selon vos besoins)
    common_names = [
        # Ajoutez ici les noms que vous voulez explicitement remplacer
        # Par exemple votre nom, prénom, etc.
    ]
    
    # Remplacer les mentions et les tags
    text = text.replace("@", "")
    
    # Remplacer les noms spécifiques
    for name in common_names:
        if name in text:
            text = text.replace(name, "quelqu'un")
    
    return text

# 🔹 Récupérer tous les commentaires d'un post, y compris les réponses aux commentaires
def get_all_comments(post_id):
    all_comments = []
    url = f"{GRAPH_URL}/{post_id}/comments"
    params = {
        "access_token": ACCESS_TOKEN,
        # Ne demandons plus les informations sur l'auteur
        "fields": "id,message,attachment,created_time,like_count,comment_count",
        "limit": 100,
        "order": "chronological"  # Pour avoir les commentaires dans l'ordre chronologique
    }
    
    comment_count = 0
    page_count = 0
    
    print(f"  💬 Récupération des commentaires pour le post {post_id}...")
    
    while url:
        page_count += 1
        data = make_api_request(url, params if '?' not in url else None)
        comments = data.get("data", [])
        
        # Anonymiser les commentaires et nettoyer le texte
        for comment in comments:
            # Supprimer complètement le champ "from" s'il existe
            if "from" in comment:
              comment["author_id"] = comment["from"].get("id", "inconnu")
              del comment["from"]
            
            # Ajouter un champ uniformisé pour l'auteur
            comment["author"] = "commentateur"
            
            # Nettoyer le texte du commentaire
            if "message" in comment:
                comment["message"] = clean_comment_text(comment["message"])
        
        all_comments.extend(comments)
        comment_count += len(comments)
        
        # Obtenir l'URL pour la prochaine page
        url = data.get("paging", {}).get("next")
        
        if page_count % 5 == 0:
            print(f"    Page {page_count}, {comment_count} commentaires récupérés jusqu'à présent")
    
    # Récupérer les réponses aux commentaires (pour chaque commentaire principal)
    for comment in list(all_comments):
        # Ne traiter que les commentaires de premier niveau (qui n'ont pas de parent)
        if "parent" not in comment and comment.get("comment_count", 0) > 0:
            replies = get_comment_replies(comment["id"])
            all_comments.extend(replies)
    
    print(f"  ✅ Total de {len(all_comments)} commentaires et réponses récupérés")
    return all_comments

# 🔹 Récupérer les réponses à un commentaire spécifique
def get_comment_replies(comment_id):
    replies = []
    url = f"{GRAPH_URL}/{comment_id}/comments"
    params = {
        "access_token": ACCESS_TOKEN,
        # Ne demandons plus les informations sur l'auteur
        "fields": "id,message,attachment,created_time,like_count,comment_count,parent",
        "limit": 100
    }
    
    while url:
        data = make_api_request(url, params if '?' not in url else None)
        comments_data = data.get("data", [])
        
        # Anonymiser les réponses et nettoyer le texte
        for comment in comments_data:
           # Récupérer uniquement l'ID de l'auteur s'il existe
            if "from" in comment and isinstance(comment["from"], dict):
                comment["author_id"] = comment["from"].get("id", "inconnu")
                del comment["from"]  # Supprimer tout le reste (nom, etc.)
                
            # Ajouter un champ uniformisé pour l'auteur
            comment["author"] = "commentateur"
            
            # Nettoyer le texte du commentaire
            if "message" in comment:
                comment["message"] = clean_comment_text(comment["message"])
        
        replies.extend(comments_data)
        url = data.get("paging", {}).get("next")
    
    return replies

# 🔹 Récupérer les détails complets d'un post
def get_post_details(post_id):
    url = f"{GRAPH_URL}/{post_id}"
    params = {
        "access_token": ACCESS_TOKEN,
        "fields": "id,message,full_picture,created_time,permalink_url,likes.summary(true),comments.summary(true),shares,attachments{type,url,media,title,description}"
    }
    
    post_data = make_api_request(url, params)
    
    # Si le post contient un message, nettoyons-le aussi
    if "message" in post_data:
        post_data["message"] = clean_comment_text(post_data["message"])
        
    return post_data

# 🔹 Sauvegarder les données dans un fichier JSON
def save_to_json(data, filename):
    file_path = os.path.join(OUTPUT_DIR, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Données sauvegardées dans {file_path}")

# 🔹 Afficher les données d'un post et ses commentaires
def display_post_info(post_data, comments):
    print(f"\n🔹 Post ID: {post_data.get('id')}")
    print(f"📝 Message: {post_data.get('message', 'Pas de message')}")
    print(f"🖼️ Média: {post_data.get('full_picture', 'Aucun')}")
    print(f"⏰ Création: {post_data.get('created_time')}")
    print(f"🔗 URL: {post_data.get('permalink_url', 'Non disponible')}")
    
    likes = post_data.get('likes', {}).get('summary', {}).get('total_count', 0)
    comment_count = post_data.get('comments', {}).get('summary', {}).get('total_count', 0)
    shares = post_data.get('shares', {}).get('count', 0) if 'shares' in post_data else 0
    
    print(f"❤️ Likes: {likes}")
    print(f"💬 Commentaires (total selon Facebook): {comment_count}")
    print(f"🔄 Partages: {shares}")
    print(f"📊 Commentaires récupérés: {len(comments)}")
    
    if len(comments) < comment_count:
        print(f"⚠️ Attention: {comment_count - len(comments)} commentaires manquants!")
    
    # Afficher le nombre de commentaires seulement (pas d'IDs)
    print(f"\n👥 {len(comments)} commentaires récupérés au total")
    
    # Afficher les 3 premiers commentaires (si disponibles)
    if comments:
        print("\n📝 Exemples de commentaires:")
        for i, comment in enumerate(comments[:3]):
            message = comment.get('message', 'Pas de message')
            likes = comment.get('like_count', 0)
            created = comment.get('created_time', 'Date inconnue')
            
            # Afficher juste "commentateur" pour tous
            print(f"  {i+1}. commentateur ({created}): {message[:50]}{'...' if len(message) > 50 else ''} ({likes} 👍)")

# 🔹 Création d’un post Facebook
def create_facebook_post_api(message, image_url=None):
    print("envoie du post a facebook")
    url = f"{GRAPH_URL}/{PAGE_ID}/feed"
    params = {
        "access_token": ACCESS_TOKEN,
        "message": message,
    }

    if image_url:
        params["picture"] = image_url

    response = requests.post(url, data=params)
    data = response.json()

    if "id" in data:
        print(f"✅ Post créé avec succès : {data['id']}")
        return data["id"]
    else:
        print(f"❌ Erreur lors de la création du post : {data}")
        return None

def publish_photo_with_message(message, image_path):
    print("📷 Envoi d'une image réelle à Facebook...")
    print(f"📁 Fichier image à envoyer : {image_path}")
    print(f"📦 Taille : {os.path.getsize(image_path) / 1024:.2f} Ko")


    url = f"{GRAPH_URL}/{PAGE_ID}/photos"
    files = {
        'source': open(image_path, 'rb')  # fichier binaire
    }
    params = {
        "access_token": ACCESS_TOKEN,
        "message": message,
    }

    response = requests.post(url, files=files, data=params, timeout=20)
    data = response.json()

    print("📨 Réponse Facebook (photo) :", data)

    if "post_id" in data:
        return data["post_id"]
    else:
        return None

# 🔹 Suppression d’un post Facebook
def delete_facebook_post(post_id):
    url = f"{GRAPH_URL}/{post_id}"
    params = {
        "access_token": ACCESS_TOKEN
    }

    response = requests.delete(url, params=params)
    data = response.json()

    if data.get("success"):
        print("🗑️ Post supprimé avec succès")
        return True
    else:
        print("❌ Échec de la suppression :", data)
        return False

# 🔹 Traitement principal
def main():
    print("🚀 Début de la récupération des données Facebook")
    
    # Vérifier que le token est valide avant de continuer
    if not verify_token():
        print("❌ Impossible de continuer avec un token invalide.")
        return
    
    # Récupérer tous les posts (on peut spécifier une date de début en format ISO, ex: "2023-01-01")
    posts = get_all_posts()
    
    # Préparer les données complètes
    all_data = []
    
    # Pour chaque post, récupérer ses détails et commentaires
    for i, post in enumerate(posts):
        post_id = post["id"]
        print(f"\n⏳ Traitement du post {i+1}/{len(posts)} (ID: {post_id})")
        
        # Récupérer les détails complets du post
        post_data = get_post_details(post_id)
        
        # Récupérer tous les commentaires du post
        comments = get_all_comments(post_id)
        
        # Afficher les informations
        display_post_info(post_data, comments)
        
        # Stocker les données du post et ses commentaires
        post_data["fetched_comments"] = comments
        all_data.append(post_data)
        
        # Sauvegarder les données du post individuellement
        save_to_json(post_data, f"post_{post_id.split('_')[-1]}.json")
        
        # Pause pour éviter de surcharger l'API
        if i < len(posts) - 1:
            time.sleep(RATE_LIMIT_DELAY * 2)
    
    # Sauvegarder toutes les données
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_to_json(all_data, f"all_posts_{timestamp}.json")
    
    print(f"\n✅ Récupération terminée! {len(all_data)} posts traités avec leurs commentaires.")
    print(f"📊 Les fichiers JSON ont été sauvegardés dans le dossier '{OUTPUT_DIR}'.")



# Fonction pour être appelée depuis views.py
def scrape_facebook():
    # Appeler la fonction main et retourner les données formatées
    return main()

if __name__ == "__main__":
    main()

# ⚙️ Configuration Django pour accéder aux modèles
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Répertoire du projet
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PFE_APP.settings")  # Remplace par le nom de ton projet Django
django.setup()

from scraping.models import FacebookComment  # modèle à créer (voir plus bas)

# 🔹 Fonction pour sauvegarder les données dans la base de données
def save_data_to_db(post_data, comments):
    post_id = post_data.get("id")
    media_url = post_data.get("full_picture", None)
    post_likes = post_data.get("likes", {}).get("summary", {}).get("total_count", 0)

    for comment in comments:
        FacebookComment.objects.create(
             post_id=post_id,
             media_url=media_url,
             comment_id=comment.get("id"),
             commenter_id=comment.get("author_id", "anonyme"),  # author_id uniquement
             comment_text=comment.get("message", ""),
             comment_likes=comment.get("like_count", 0),
             post_likes=post_likes,
        )
    print(f"💾 {len(comments)} commentaires sauvegardés en base de données.")

# 🔹 Modifie la fonction `main()` pour qu'elle retourne les données des commentaires
def main():
    print("🚀 Début de la récupération des données Facebook")
    
    if not verify_token():
        return []  # Si le token est invalide, on retourne une liste vide.

    posts = get_all_posts()
    all_comments_data = []  # Liste pour stocker toutes les données des commentaires récupérés

    for post in posts:
        post_id = post["id"]
        post_data = get_post_details(post_id)
        comments = get_all_comments(post_id)
        display_post_info(post_data, comments)
        
        # Sauvegarder dans la base de données
        save_data_to_db(post_data, comments)
        
        # Collecter les données des commentaires pour les renvoyer
        for comment in comments:
            comment_data = {
                "post_id": post_id,
                "media_url": post_data.get("full_picture", None),
                "comment_id": comment.get("id"),
                "commenter_id": comment.get("author_id", "anonyme"),  # Utilise author_id
                "comment_text": comment.get("message", ""),
                "comment_likes": comment.get("like_count", 0),
                "post_likes": post_data.get("likes", {}).get("summary", {}).get("total_count", 0),
            }
            all_comments_data.append(comment_data)

    return all_comments_data  # Renvoie la liste des commentaires collectés