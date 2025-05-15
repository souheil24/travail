import requests
from scraping.models import YoutubeComment
from scraping.models import SocialAccount

def get_latest_youtube_channel_id():
    account = SocialAccount.objects.filter(platform='youtube').last()
    return account.account_name if account else None

API_KEY = "AIzaSyCeMZLYuSMjVQHaNLZPgYNvu-8Ob9F4Hec"
RESULTS = []


# 🔹 Récupère les ID des vidéos d'une chaîne
def get_video_ids_from_channel(channel_id):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "key": API_KEY,
        "channelId": channel_id,
        "part": "snippet",
        "order": "date",
        "maxResults": 1 # max c'est 50 autorisé par API
    }
    response = requests.get(url, params=params)
    data = response.json()
    return [
        item["id"]["videoId"]
        for item in data.get("items", [])
        if item["id"]["kind"] == "youtube#video"
    ]

# 🔹 Récupère les infos vidéo : titre, vues, et likes
def get_video_details(video_id):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    item = response.json()["items"][0]
    return {
        "video_id": video_id,
        "title": item["snippet"]["title"],
        "views": item["statistics"].get("viewCount", "0"),
        "likes_video": item["statistics"].get("likeCount", "0")
    }

# 🔹 Récupère les commentaires
def get_comments(video_id):
    comments = []
    page_token = None
    while True:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video_id,
            "key": API_KEY,
            "maxResults": 100,
            "pageToken": page_token
        }
        response = requests.get(url, params=params)
        data = response.json()

        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": snippet["textDisplay"]
            })

        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return comments

# 🔹 Fonction principale pour récupérer les données et les enregistrer
def scrape_youtube_data():
    try:
        print("🚀 Démarrage du scraping YouTube...")

        # 🧠 Récupérer dynamiquement le Channel ID
        channel_id = get_latest_youtube_channel_id()
        if not channel_id:
            print("❌ Aucun Channel ID YouTube enregistré.")
            return []

        video_ids = get_video_ids_from_channel(channel_id)
        print(f"Vidéos trouvées: {len(video_ids)}")

        results = []

        if not video_ids:
            print("Aucune vidéo trouvée, ajout de données de test")
            # ... (bloc test inchangé)
            return results

        for video_id in video_ids[:5]:
            try:
                video_info = get_video_details(video_id)
                comments = get_comments(video_id)

                if not comments:
                    data = {
                        "video_id": video_info["video_id"],
                        "title": video_info["title"],
                        "views": video_info["views"],
                        "likes_video": video_info["likes_video"],
                        "comment_text": "Aucun commentaire"
                    }
                    YoutubeComment.objects.create(**data)
                    results.append(data)
                else:
                    for c in comments[:10]:
                        data = {
                            "title": video_info["title"],
                            "views": video_info["views"],
                            "likes_video": video_info["likes_video"],
                            "comment_text": c["text"]
                        }
                        YoutubeComment.objects.create(**data)
                        results.append(data)

            except Exception as e:
                print(f"Erreur pour la vidéo {video_id}: {str(e)}")
                continue

        print(f"✅ Scraping terminé: {len(results)} commentaires récupérés")
        return results

    except Exception as e:
        print(f"💥 Erreur globale dans scrape_youtube_data: {str(e)}")
        import traceback
        traceback.print_exc()
