import requests
from datetime import datetime
from .models import TwitterComment
from accounts.views import safe_parse_datetime
from django.utils import timezone
import time
import json
from scraping.models import SocialAccount


# Token d'API Twitter
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAE0a1AEAAAAA8dJ57iU91lYrJ0ZCeCUmkqWfVdY%3DxlCkWm3HmGgFmSQ0ion4L036uvyHUQh50MgY1YekeoCs8aTWK4"
def get_latest_twitter_username():
    account = SocialAccount.objects.filter(platform="twitter").last()
    return account.account_name if account else None


def get_user_id(username):
    url = f"https://api.twitter.com/2/users/by/username/{username}"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["data"]["id"]
    elif response.status_code == 429:
        print(f"Limite de taux atteinte (429): {response.text}")
        return None
    else:
        print(f"Erreur récupération ID: {response.status_code} - {response.text}")
        return None

def get_user_tweets(user_id, max_results=5):
    if not user_id:
        return []
        
    url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    params = {
        "max_results": max_results,
        "tweet.fields": "author_id,created_at"
    }
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()["data"]
    elif response.status_code == 429:
        print(f"Limite de taux atteinte (429): {response.text}")
        return []
    else:
        print(f"Erreur récupération tweets: {response.status_code} - {response.text}")
        return []

def create_test_data():
    """Crée des données de test pour Twitter quand l'API n'est pas disponible."""
    print("Création de données de test pour Twitter...")
    
    # Supprimer les anciennes données de test si elles existent
    TwitterComment.objects.filter(tweet_id__startswith="test_").delete()
    
    # Créer de nouvelles données de test
   
    # Enregistrer les tweets de test dans la base de données
    created_tweets = []
    for tweet_data in test_tweets:
        tweet, created = TwitterComment.objects.update_or_create(
            tweet_id=tweet_data["tweet_id"],
            defaults={
                "text": tweet_data["text"],
                "author_id": tweet_data["author_id"],
                "created_at": tweet_data["created_at"]
            }
        )
        created_tweets.append(tweet)
    
    print(f"{len(created_tweets)} tweets de test créés avec succès")
    return created_tweets

def debug_twitter_data():
    """Fonction de débogage pour vérifier les données Twitter dans la base de données."""
    # Compter le nombre total de tweets
    total_count = TwitterComment.objects.count()
    
    # Compter le nombre de tweets de test
    test_count = TwitterComment.objects.filter(tweet_id__startswith="test_").count()
    
    # Compter le nombre de tweets réels
    real_count = total_count - test_count
    
    # Afficher les résultats
    print(f"=== DÉBOGAGE DONNÉES TWITTER ===")
    print(f"Nombre total de tweets: {total_count}")
    print(f"Tweets de test: {test_count}")
    print(f"Tweets réels: {real_count}")
    
    # Afficher les 5 derniers tweets
    print("\nDerniers tweets:")
    for tweet in TwitterComment.objects.all().order_by('-created_at')[:5]:
        print(f"- ID: {tweet.tweet_id}, Auteur: {tweet.author_id}, Texte: {tweet.text[:50]}...")
    
    return {
        'total': total_count,
        'test': test_count,
        'real': real_count
    }

def fetch_and_store_tweets():
    try:
        # Obtenir le dernier compte Twitter depuis la BDD
        username = get_latest_twitter_username()
        if not username:
            print("⚠️ Aucun compte Twitter enregistré. Utilisation de données de test.")
            return create_test_data()

        user_id = get_user_id(username)
        if user_id is None:
            print("❌ Impossible d'obtenir l'ID utilisateur. Utilisation de données de test.")
            return create_test_data()
            
        tweets = get_user_tweets(user_id)
        if not tweets:
            print("❌ Aucun tweet récupéré. Utilisation de données de test.")
            return create_test_data()
            
        # Enregistrement des tweets
        saved_tweets = []
        for tweet in tweets:
            created_at = safe_parse_datetime(tweet.get("created_at")) or timezone.now()
            tweet_obj, created = TwitterComment.objects.update_or_create(
                tweet_id=tweet["id"],
                defaults={
                    "text": tweet["text"],
                    "created_at": created_at
                }
            )
            saved_tweets.append(tweet_obj)

        print(f"✅ {len(saved_tweets)} tweets récupérés et enregistrés.")
        debug_twitter_data()
        return saved_tweets

    except Exception as e:
        print(f"💥 Erreur lors de la récupération des tweets: {str(e)}")
        return create_test_data()


    # À la fin de la fonction, ajouter:
    debug_twitter_data()