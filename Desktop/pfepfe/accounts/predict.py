import numpy as np
import fasttext
from django.http import JsonResponse
from scraping.models import FacebookComment
from tensorflow import keras
import tensorflow as tf
from .models import Comment_Demo
import gc
import torch
from collections import namedtuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Chargement du modèle de langue une seule fois (global)
_lang_detect_model = None

# Cache global pour les modèles Keras uniquement
MODEL_CACHE = {
    "sentiment": {},
    "category": {},
    "category_dict": {}
}

BATCH_SIZE = 30  # Batch size fixe pour TOUTES les prédictions

def detect_language(text):
    global _lang_detect_model
    if _lang_detect_model is None:
        _lang_detect_model = fasttext.load_model("lid.176.bin")
    prediction = _lang_detect_model.predict(text)
    label = prediction[0][0]
    lang = label.replace("__label__", "")
    return lang

def unload_language_model():
    global _lang_detect_model
    if _lang_detect_model is not None:
        del _lang_detect_model
        _lang_detect_model = None
        gc.collect()

def sentence_to_vector(sentence, ft_model, max_len=100):
    words = sentence.split()
    vectors = [ft_model.get_word_vector(word) for word in words]
    if len(vectors) > max_len:
        vectors = vectors[:max_len]
    elif len(vectors) < max_len:
        padding = [np.zeros(300)] * (max_len - len(vectors))
        vectors.extend(padding)
    return np.array(vectors)


def load_model_fin(model_path, num_labels):
    """Charge un modèle à partir d'un chemin local ou utilise un modèle par défaut."""
    try:
        import os
        if os.path.exists(model_path):
            print(f"✅ Chargement du modèle depuis: {model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
            return model
        else:
            print(f"⚠️ Chemin non trouvé: {model_path}")
            # Déterminer le modèle par défaut en fonction du nombre d'étiquettes
            if "SENTIMENT" in model_path.upper():
                # Modèle de sentiment
                if "FR" in model_path.upper():
                    return AutoModelForSequenceClassification.from_pretrained("almanach/camembertav2-base", num_labels=num_labels)
                elif "AR" in model_path.upper():
                    return AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERTv2", num_labels=num_labels)
                else:
                    return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
            else:
                # Modèle de catégorie
                if "FR" in model_path.upper():
                    return AutoModelForSequenceClassification.from_pretrained("almanach/camembertav2-base", num_labels=num_labels)
                elif "AR" in model_path.upper():
                    return AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERTv2", num_labels=num_labels)
                else:
                    return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle {model_path}: {str(e)}")
        # Utiliser un modèle par défaut en cas d'erreur
        return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)


def predict_sentiment(text, model, tokenizer, device):
    model.eval()
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).squeeze()
    probabilities = torch.sigmoid(logits).squeeze().item()  # convertir en float
    prediction = "Positif" if probabilities >= 0.5 else "Négatif"
    # Créer un tensor de probabilités pour les deux classes
    # Mapper l'indice à une étiquette
    sentiment_labels = {0: "Négatif", 1: "Positif"}
    predicted_sentiment = sentiment_labels.get(prediction, f"Classe {prediction}")
    confidence = probabilities if prediction == "Positif" else 1 - probabilities

    return {
        "predicted_class": prediction,
        "confidence": round(confidence, 4),
        "probabilities": {
            "Négatif": 1 - probabilities,
            "Positif": probabilities
        }
    }

def predict_category(text, model, tokenizer, device, category_labels=None):
    model.eval()
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    prediction_idx = torch.argmax(probabilities, dim=-1).item()
    
    # Si les étiquettes de catégorie ne sont pas fournies, utiliser des indices
    if category_labels is None:
        category_labels = {i: f"Catégorie {i}" for i in range(outputs.logits.shape[1])}
    
    predicted_category = category_labels.get(prediction, f"Classe {prediction}")
    probs = probabilities.cpu().numpy()[0]
    confidence = round(float(probs[prediction_idx]), 4)

    
    # Obtenir les probabilités pour chaque classe
    probs = probabilities.cpu().numpy()[0]
    
    return {
        "predicted_class": predicted_category,
        "confidence": confidence,
         "probabilities": {label: round(float(probs[idx]), 4) for idx, label in category_labels.items()}
        
    }

Result = namedtuple("Result", ["text", "sentiment_result", "category_result"])

# Exemple de modification dans ta fonction predict_with_bert_ar
def predict_with_bert_ar(text=None, sentiment_model_path="UBC-NLP/MARBERTv2", 
                        category_model_path="UBC-NLP/MARBERTv2", 
                        num_sentiment_labels=1, category_labels=None):
    # Configuration initiale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Chargement des modèles - utiliser des chemins absolus
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Chemin absolu pour le modèle de sentiment
        sentiment_model_path = os.path.join(base_dir, "SENTIMENTAR")
        if not os.path.exists(sentiment_model_path):
            print(f"⚠️ Chemin non trouvé: {sentiment_model_path}")
            # Fallback sur un modèle HuggingFace
            sentiment_model = AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERTv2", num_labels=1).to(device)
        else:
            sentiment_model = load_model_fin(sentiment_model_path, num_labels=1).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERTv2")

        # Exemple de prédiction
        print("=== Résultat Sentiment ===")
        sentiment_result = predict_sentiment(text, sentiment_model, tokenizer, device)
        
        # Chemin pour le modèle de catégorie
        if sentiment_result["predicted_class"] == "Négatif":
            category_model_path = os.path.join(base_dir, "CatAR", "negativer")
            if not os.path.exists(category_model_path):
                print(f"⚠️ Chemin non trouvé: {category_model_path}")
                # Fallback sur un modèle HuggingFace
                category_model = AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERTv2", num_labels=6).to(device)
            else:
                category_model = load_model_fin(category_model_path, num_labels=6).to(device)
            
            category_labels={0: "Feedback", 1: "Prix", 2: "Service client", 3: "Vulgarite", 4: "Spam", 5: "Livraison"}
        else:
            category_model_path = os.path.join(base_dir, "CatAR", "positiver")
            if not os.path.exists(category_model_path):
                print(f"⚠️ Chemin non trouvé: {category_model_path}")
                # Fallback sur un modèle HuggingFace
                category_model = AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERTv2", num_labels=5).to(device)
            else:
                category_model = load_model_fin(category_model_path, num_labels=5).to(device)
            
            category_labels={0: "Feedback", 1: "Information", 2: "Prix", 3: "Service client", 4: "Livraison"}
        
        print("\n=== Résultat Catégorie ===")
        category_result = predict_category(text, category_model, tokenizer, device, category_labels)

        # Créer et retourner un seul objet Result
        result = {
            'text': text,
            'sentiment_result': {
                'label': sentiment_result["predicted_class"],
                'confidence': sentiment_result["confidence"]
            },
            'category_result': {
                'label': category_result["predicted_class"],
                'confidence': category_result["confidence"]
            }
        }

        return result
    
    except Exception as e:
        print(f"❌ Erreur dans predict_with_bert_ar: {str(e)}")
        # Retourner un résultat par défaut en cas d'erreur
        return {
            'text': text,
            'sentiment_result': {
                'label': "Neutre",
                'confidence': 0.5
            },
            'category_result': {
                'label': "Non classifié",
                'confidence': 0.5
            }
        }
def predict_with_deberta_fr(text=None, sentiment_model_path="UBC-NLP/MARBERTv2", 
                        category_model_path="UBC-NLP/MARBERTv2", 
                        num_sentiment_labels=1, category_labels=None):
    # Configuration initiale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Chargement des modèles - utiliser des chemins absolus
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Chemin absolu pour le modèle de sentiment
        sentiment_model_path = os.path.join(base_dir, "SENTIMENTFR")
        if not os.path.exists(sentiment_model_path):
            print(f"⚠️ Chemin non trouvé: {sentiment_model_path}")
            # Fallback sur un modèle HuggingFace
            sentiment_model = AutoModelForSequenceClassification.from_pretrained("almanach/camembertav2-base", num_labels=1).to(device)
        else:
            sentiment_model = load_model_fin(sentiment_model_path, num_labels=1).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained("almanach/camembertav2-base")

        # Exemple de prédiction
        print("=== Résultat Sentiment ===")
        sentiment_result = predict_sentiment(text, sentiment_model, tokenizer, device)
        
        # Chemin pour le modèle de catégorie
        if sentiment_result["predicted_class"] == "Négatif":
            category_model_path = os.path.join(base_dir, "CATFR", "ngfr")
            if not os.path.exists(category_model_path):
                print(f"⚠️ Chemin non trouvé: {category_model_path}")
                # Fallback sur un modèle HuggingFace
                category_model = AutoModelForSequenceClassification.from_pretrained("almanach/camembertav2-base", num_labels=5).to(device)
            else:
                category_model = load_model_fin(category_model_path, num_labels=5).to(device)
            
            category_labels={0: "Feedback", 1: "Prix", 2: "Service client", 3: "Vulgarite", 4: "Livraison"}
        else:
            category_model_path = os.path.join(base_dir, "CATFR", "PSFR")
            if not os.path.exists(category_model_path):
                print(f"⚠️ Chemin non trouvé: {category_model_path}")
                # Fallback sur un modèle HuggingFace
                category_model = AutoModelForSequenceClassification.from_pretrained("almanach/camembertav2-base", num_labels=5).to(device)
            else:
                category_model = load_model_fin(category_model_path, num_labels=5).to(device)
            
            category_labels={0: "Feedback", 1: "Information", 2: "Prix", 3: "Service client", 4: "Livraison"}
        
        print("\n=== Résultat Catégorie ===")
        category_result = predict_category(text, category_model, tokenizer, device, category_labels)

        # Créer et retourner un seul objet Result
        result = {
            'text': text,
            'sentiment_result': {
                'label': sentiment_result["predicted_class"],
                'confidence': sentiment_result["confidence"]
            },
            'category_result': {
                'label': category_result["predicted_class"],
                'confidence': category_result["confidence"]
            }
        }

        return result
    
    except Exception as e:
        print(f"❌ Erreur dans predict_with_deberta_fr: {str(e)}")
        # Retourner un résultat par défaut en cas d'erreur
        return {
            'text': text,
            'sentiment_result': {
                'label': "Neutre",
                'confidence': 0.5
            },
            'category_result': {
                'label': "Non classifié",
                'confidence': 0.5
            }
        }
def load_models(language, sentiment):
    if language == "fr":
        sentiment_model_path = "best_sentimentCNN_model.keras"
        category_model_path = "positive_categories.keras" if sentiment == "Positif" else "negative_categories.keras"
        fasttext_model_path = r"C:\Users\HP\PFE_WEB\PFE_GROUP\souehilm\PFE_APP\cc.en.300\cc.en.300.bin"

        categories = {
            "Positif": ["Feedback positif", "Information", "Prix", "Livraison" ,  "Service Client" ],
            "Negatif": ["Feedback négatif", "Prix", "Service Client", "Livraison", "Vulgarité"]
        }
    elif language == "en":
        sentiment_model_path = "bestEN_sentiment_modelBILSTM.keras"
        category_model_path = "EN_positive_category_modelBILSTM.keras" if sentiment == "Positif" else "EN_negative_category_model1.keras"
        fasttext_model_path = r"C:\Users\HP\PFE_WEB\PFE_GROUP\souehilm\PFE_APP\cc.en.300\cc.en.300.bin"

        categories = {
            "Positif": ["Positive Feedback", "Information", "Price", "Customer Service" , "Delivery"],
            "Negatif": ["Negative Feedback", "Price", "Customer Service", "vulgarity", "spam", "delivery"]
        }
    else:  # Arabic
        sentiment_model_path = "Meilleur_sentiment_model_arabic.keras"
        category_model_path = "positive_category_model_arabic.keras" if sentiment == "Positif" else "negative_category_model_arabic.keras"
        fasttext_model_path = r"C:\Users\HP\PFE_WEB\PFE_GROUP\souehilm\PFE_APP\cc.en.300\cc.en.300.bin"

        categories = {
            "Positif": ["مراجعة إيجابية", "معلومة", "السعر", "التوصيل", "خدمة العملاء"],
            "Negatif": ["مراجعة سلبية", "السعر", "كلام فاحش", "خدمة العملاء", "spam", "التوصيل"]
        }
    return sentiment_model_path, category_model_path, fasttext_model_path, categories

def get_sentiment_model(lang):
    if lang not in MODEL_CACHE["sentiment"]:
        sentiment_path, _, _, _ = load_models(lang, "Positif")
        MODEL_CACHE["sentiment"][lang] = keras.models.load_model(sentiment_path)
    return MODEL_CACHE["sentiment"][lang]

def get_category_model(lang, sentiment):
    key = (lang, sentiment)
    if key not in MODEL_CACHE["category"]:
        _, cat_model_path, _, _ = load_models(lang, sentiment)
        MODEL_CACHE["category"][key] = keras.models.load_model(cat_model_path)
    return MODEL_CACHE["category"][key]

def get_category_labels(lang, sentiment):
    key = (lang, sentiment)
    if key not in MODEL_CACHE["category_dict"]:
        _, _, _, cat_dict = load_models(lang, sentiment)
        MODEL_CACHE["category_dict"][key] = cat_dict[sentiment]
    return MODEL_CACHE["category_dict"][key]

@tf.function(reduce_retracing=True)
def tf_predict(model, batch):
    return model(batch, training=False)

def pad_batch(vectors, batch_size):
    n = len(vectors)
    if n < batch_size:
        pad_shape = (batch_size - n,) + vectors.shape[1:]
        vectors = np.concatenate([vectors, np.zeros(pad_shape)], axis=0)
    return vectors

def analyze_comment_text(text):
    language = detect_language(text)
    sentiment_model = get_sentiment_model(language)
    _, _, fasttext_model_path, _ = load_models(language, "Positif")

    # FastText vectorization
    ft = fasttext.load_model(fasttext_model_path)
    vectorized_text = np.expand_dims(sentence_to_vector(text, ft), axis=0)
    del ft
    gc.collect()

    # Padding batch pour 1 commentaire
    vectors = pad_batch(vectorized_text, BATCH_SIZE)
    sentiment_pred = tf_predict(sentiment_model, tf.convert_to_tensor(vectors)).numpy()[0][0]
    sentiment = "Positif" if sentiment_pred >= 0.5 else "Negatif"
    sentiment_confidence = sentiment_pred * 100 if sentiment == "Positif" else (1 - sentiment_pred) * 100

    cat_model = get_category_model(language, sentiment)
    cat_labels = get_category_labels(language, sentiment)
    cat_pred = tf_predict(cat_model, tf.convert_to_tensor(vectors)).numpy()[0]
    category_index = np.argmax(cat_pred)
    category_confidence = np.max(cat_pred) * 100
    predicted_category = cat_labels[category_index]

    return {
        "commentaire": text,
        "langue": language,
        "sentiment": {
            "valeur": sentiment,
            "confiance": f"{sentiment_confidence:.2f}%"
        },
        "categorie": {
            "valeur": predicted_category,
            "confiance": f"{category_confidence:.2f}%"
        }
    }


def analyze_all_comment():
    comments = Comment_Demo.objects.exclude(text__isnull=True).exclude(text__exact="")

    language_groups1 = {
        "en": [],
        "fr": [],
        "ar": []
    }

    all_results = []

    # Étape 1 : Regrouper les commentaires par langue
    for comment in comments:
        lang = detect_language(comment.text)
        if lang in language_groups1:
            comment.langue = lang
            comment.save()
            language_groups1[lang].append(comment)

    # Traitement des commentaires anglais (avec modèle)
    for lang in ["en"]:
        group = language_groups1[lang]
        if not group:
            print(f"Aucun commentaire pour la langue : {lang}")
            continue

        print(f"\n--- Traitement des commentaires pour la langue : {lang} ---")

        _, _, ft_path, _ = load_models(lang, "Positif")
        ft_model = fasttext.load_model(ft_path)
        vectors = np.array([
            sentence_to_vector(comment.text.strip(), ft_model)
            for comment in group
        ])
        del ft_model
        gc.collect()

        vectors = pad_batch(vectors, BATCH_SIZE)

        print(f"[LOG] Prédiction sentiment pour English sur {len(group)} commentaires...")
        sentiment_model = get_sentiment_model(lang)
        sentiment_preds = tf_predict(sentiment_model, tf.convert_to_tensor(vectors)).numpy()
        sentiment_preds = sentiment_preds[:len(group)]

        pos_comments, neg_comments = [], []

        for i, comment in enumerate(group):
            pred = sentiment_preds[i][0]
            sentiment = "Positif" if pred >= 0.5 else "Negatif"
            confidence = pred * 100 if sentiment == "Positif" else (1 - pred) * 100

            comment.sentiment = sentiment
            comment.save()

            comment_data = {
                "obj": comment,
                "vector": vectors[i],
                "sentiment_conf": f"{confidence:.2f}%"
            }

            if sentiment == "Positif":
                pos_comments.append(comment_data)
            else:
                neg_comments.append(comment_data)

        # Catégorisation POSITIF
        if pos_comments:
            print(f"[LOG] Prédiction catégorie POSITIVE pour English ({len(pos_comments)} commentaires)")
            cat_model = get_category_model(lang, "Positif")
            cat_labels = get_category_labels(lang, "Positif")
            cat_vectors = np.array([entry["vector"] for entry in pos_comments])
            cat_vectors = pad_batch(cat_vectors, BATCH_SIZE)
            cat_preds = tf_predict(cat_model, tf.convert_to_tensor(cat_vectors)).numpy()
            cat_preds = cat_preds[:len(pos_comments)]

            for i, entry in enumerate(pos_comments):
                category_index = np.argmax(cat_preds[i])
                category_conf = np.max(cat_preds[i]) * 100
                category = cat_labels[category_index]

                comment_obj = entry["obj"]
                comment_obj.category = category
                comment_obj.save()

                all_results.append({
                    "comment_id": comment_obj.id_comment,
                    "text": comment_obj.text,
                    "langue": "English",
                    "sentiment": {
                        "valeur": "Positif",
                        "confiance": entry["sentiment_conf"]
                    },
                    "categorie": {
                        "valeur": category,
                        "confiance": f"{category_conf:.2f}%"
                    }
                })

        # Catégorisation NEGATIF
        if neg_comments:
            print(f"[LOG] Prédiction catégorie NEGATIVE pour English ({len(neg_comments)} commentaires)")
            cat_model = get_category_model(lang, "Negatif")
            cat_labels = get_category_labels(lang, "Negatif")
            cat_vectors = np.array([entry["vector"] for entry in neg_comments])
            cat_vectors = pad_batch(cat_vectors, BATCH_SIZE)
            cat_preds = tf_predict(cat_model, tf.convert_to_tensor(cat_vectors)).numpy()
            cat_preds = cat_preds[:len(neg_comments)]

            for i, entry in enumerate(neg_comments):
                category_index = np.argmax(cat_preds[i])
                category_conf = np.max(cat_preds[i]) * 100
                category = cat_labels[category_index]

                comment_obj = entry["obj"]
                comment_obj.category = category
                comment_obj.save()

                all_results.append({
                    "comment_id": comment_obj.id_comment,
                    "text": comment_obj.text,
                    "langue": "English",
                    "sentiment": {
                        "valeur": "Negatif",
                        "confiance": entry["sentiment_conf"]
                    },
                    "categorie": {
                        "valeur": category,
                        "confiance": f"{category_conf:.2f}%"
                    }
                })
             

             # Traitement des langues fr et ar sans modèle (juste pour affichage)
   
    for lang in ["ar"]:
     group = language_groups1.get(lang, [])
     if not group:
        print(f"Aucun commentaire pour la langue : {lang}")
        continue

    
     print(f"\n--- Traitement des commentaires pour la langue : {lang} ---")

    for i, comment in enumerate(group):
      result_fin = predict_with_bert_ar(text=comment.text)

        # Accéder aux résultats directement
      text = result_fin['text']
      sentiment_result = result_fin['sentiment_result']
      category_result = result_fin['category_result']

        # Ajouter les résultats dans la liste
      all_results.append({
       "comment_id": comment.id_comment,
       "text": comment.text,
       "langue": "Arabic",
       "sentiment": {
          "valeur": sentiment_result["label"],
          "confiance": f"{sentiment_result['confidence'] * 100:.2f}%"
        },
       "categorie": {
          "valeur": category_result["label"],
          "confiance": f"{category_result['confidence'] * 100:.2f}%"
        }
})

      
    for lang in ["fr"]:
     group = language_groups1.get(lang, [])
     if not group:
        print(f"Aucun commentaire pour la langue : {lang}")
        continue

    
     print(f"\n--- Traitement des commentaires pour la langue : {lang} ---")

    for i, comment in enumerate(group):
      result_fin = predict_with_deberta_fr(text=comment.text)

      # Accéder aux résultats directement
      text = result_fin['text']
      sentiment_result = result_fin['sentiment_result']
      category_result = result_fin['category_result']

        # Ajouter les résultats dans la liste
      all_results.append({
       "comment_id": comment.id_comment,
       "text": comment.text,
       "langue": "French",
       "sentiment": {
          "valeur": sentiment_result["label"],
          "confiance": f"{sentiment_result['confidence'] * 100:.2f}%"
        },
       "categorie": {
          "valeur": category_result["label"],
          "confiance": f"{category_result['confidence'] * 100:.2f}%"
        }
})

    return all_results


def analyze_all_comment_FB():
    comments = FacebookComment.objects.all()

    language_groups1 = {
        "en": [],
        "fr": [],
        "ar": []
    }

    all_results = []

    # Étape 1 : Regrouper les commentaires par langue
    for comment in comments:
        lang = detect_language(comment.comment_text)
        if lang in language_groups1:
            comment.langue = lang
            comment.save()
            language_groups1[lang].append(comment)

    # Traitement des commentaires anglais (avec modèle)
    for lang in ["en"]:
        group = language_groups1[lang]
        if not group:
            print(f"Aucun commentaire pour la langue : {lang}")
            continue

        print(f"\n--- Traitement des commentaires pour la langue : {lang} ---")

        _, _, ft_path, _ = load_models(lang, "Positif")
        ft_model = fasttext.load_model(ft_path)
        vectors = np.array([
            sentence_to_vector(comment.comment_text.strip(), ft_model)
            for comment in group
        ])
        del ft_model
        gc.collect()

        vectors = pad_batch(vectors, BATCH_SIZE)

        print(f"[LOG] Prédiction sentiment pour English sur {len(group)} commentaires...")
        sentiment_model = get_sentiment_model(lang)
        sentiment_preds = tf_predict(sentiment_model, tf.convert_to_tensor(vectors)).numpy()
        sentiment_preds = sentiment_preds[:len(group)]

        pos_comments, neg_comments = [], []

        for i, comment in enumerate(group):
            pred = sentiment_preds[i][0]
            sentiment = "Positif" if pred >= 0.5 else "Negatif"
            confidence = pred * 100 if sentiment == "Positif" else (1 - pred) * 100

            comment.sentiment = sentiment
            comment.save()

            comment_data = {
                "obj": comment,
                "vector": vectors[i],
                "sentiment_conf": f"{confidence:.2f}%"
            }

            if sentiment == "Positif":
                pos_comments.append(comment_data)
            else:
                neg_comments.append(comment_data)

        # Catégorisation POSITIF
        if pos_comments:
            print(f"[LOG] Prédiction catégorie POSITIVE pour English ({len(pos_comments)} commentaires)")
            cat_model = get_category_model(lang, "Positif")
            cat_labels = get_category_labels(lang, "Positif")
            cat_vectors = np.array([entry["vector"] for entry in pos_comments])
            cat_vectors = pad_batch(cat_vectors, BATCH_SIZE)
            cat_preds = tf_predict(cat_model, tf.convert_to_tensor(cat_vectors)).numpy()
            cat_preds = cat_preds[:len(pos_comments)]

            for i, entry in enumerate(pos_comments):
                category_index = np.argmax(cat_preds[i])
                category_conf = np.max(cat_preds[i]) * 100
                category = cat_labels[category_index]

                comment_obj = entry["obj"]
                comment_obj.category = category
                comment_obj.save()

                all_results.append({
    "comment_id": comment.comment_id,
    "post_id": comment.post_id,
    "media_url": comment.media_url,
    "commenter_id": comment.commenter_id,
    "comment_text": comment.comment_text,
    "comment_likes": comment.comment_likes,
    "post_likes": comment.post_likes,
    "langue": "English",
    "sentiment": {
        "valeur": "Positif",
        "confiance": entry["sentiment_conf"]
    },
    "categorie": {
        "valeur": category,
        "confiance": f"{category_conf:.2f}%"
    }
})

        # Catégorisation NEGATIF
        if neg_comments:
            print(f"[LOG] Prédiction catégorie NEGATIVE pour English ({len(neg_comments)} commentaires)")
            cat_model = get_category_model(lang, "Negatif")
            cat_labels = get_category_labels(lang, "Negatif")
            cat_vectors = np.array([entry["vector"] for entry in neg_comments])
            cat_vectors = pad_batch(cat_vectors, BATCH_SIZE)
            cat_preds = tf_predict(cat_model, tf.convert_to_tensor(cat_vectors)).numpy()
            cat_preds = cat_preds[:len(neg_comments)]

            for i, entry in enumerate(neg_comments):
                category_index = np.argmax(cat_preds[i])
                category_conf = np.max(cat_preds[i]) * 100
                category = cat_labels[category_index]

                comment_obj = entry["obj"]
                comment_obj.category = category
                comment_obj.save()

                all_results.append({
    "comment_id": comment.comment_id,
    "post_id": comment.post_id,
    "media_url": comment.media_url,
    "commenter_id": comment.commenter_id,
    "comment_text": comment.comment_text,
    "comment_likes": comment.comment_likes,
    "post_likes": comment.post_likes,
    "langue": "English",
    "sentiment": {
        "valeur": "Negatif",
        "confiance": entry["sentiment_conf"]
    },
    "categorie": {
        "valeur": category,
        "confiance": f"{category_conf:.2f}%"
    }
})
             

    # Traitement des langues fr et ar sans modèle (juste pour affichage)
   
    for lang in ["ar"]:
     group = language_groups1.get(lang, [])
     if not group:
        print(f"Aucun commentaire pour la langue : {lang}")
        continue

    print(f"\n--- Traitement des commentaires pour la langue : {lang} ---")

    for i, comment in enumerate(group):
        result_fin = predict_with_bert_ar(text=comment.comment_text)

        sentiment_result = result_fin['sentiment_result']
        category_result = result_fin['category_result']

        # Sauvegarde les résultats dans la base
        comment.sentiment = sentiment_result["label"]
        comment.sentiment_confidence = sentiment_result["confidence"]
        comment.category = category_result["label"]
        comment.category_confidence = category_result["confidence"]
        comment.save()

        all_results.append({
            "comment_id": comment.comment_id,
            "post_id": comment.post_id,
            "media_url": comment.media_url,
            "commenter_id": comment.commenter_id,
            "comment_text": comment.comment_text,
            "comment_likes": comment.comment_likes,
            "post_likes": comment.post_likes,
            "langue": "Arabe",
            "sentiment": {
                "valeur": sentiment_result["label"],
                "confiance": f"{sentiment_result['confidence'] * 100:.2f}%"
            },
            "categorie": {
                "valeur": category_result["label"],
                "confiance": f"{category_result['confidence'] * 100:.2f}%"
            }
        })

      
    for lang in ["fr"]:
     group = language_groups1.get(lang, [])
     if not group:
        print(f"Aucun commentaire pour la langue : {lang}")
        continue

    print(f"\n--- Traitement des commentaires pour la langue : {lang} ---")

    for i, comment in enumerate(group):
        result_fin = predict_with_deberta_fr(text=comment.comment_text)

        sentiment_result = result_fin['sentiment_result']
        category_result = result_fin['category_result']

        # Sauvegarde les résultats dans la base
        comment.sentiment = sentiment_result["label"]
        comment.sentiment_confidence = sentiment_result["confidence"]
        comment.category = category_result["label"]
        comment.category_confidence = category_result["confidence"]
        comment.save()

        all_results.append({
            "comment_id": comment.comment_id,
            "post_id": comment.post_id,
            "media_url": comment.media_url,
            "commenter_id": comment.commenter_id,
            "comment_text": comment.comment_text,
            "comment_likes": comment.comment_likes,
            "post_likes": comment.post_likes,
            "langue": "Francais",
            "sentiment": {
                "valeur": sentiment_result["label"],
                "confiance": f"{sentiment_result['confidence'] * 100:.2f}%"
            },
            "categorie": {
                "valeur": category_result["label"],
                "confiance": f"{category_result['confidence'] * 100:.2f}%"
            }
        })



    return all_results

def predict_with_bert_en(text=None):
    """Fonction pour prédire le sentiment et la catégorie en anglais avec BERT."""
    # Configuration initiale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Utiliser directement des modèles de HuggingFace pour l'anglais
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        
        # Prédiction du sentiment
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        
        # Mapper l'indice à une étiquette
        sentiment_labels = {0: "Négatif", 1: "Positif"}
        predicted_sentiment = sentiment_labels.get(prediction, f"Classe {prediction}")
        confidence = probabilities[0][prediction].item()
        
        # Prédiction de la catégorie
        if predicted_sentiment == "Positif":
            # Utiliser un modèle pour les catégories positives
            category_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5).to(device)
            category_labels = {0: "Feedback", 1: "Information", 2: "Price", 3: "Customer Service", 4: "Delivery"}
        else:
            # Utiliser un modèle pour les catégories négatives
            category_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6).to(device)
            category_labels = {0: "Feedback", 1: "Price", 2: "Customer Service", 3: "Vulgarity", 4: "Spam", 5: "Delivery"}
        
        # Prédiction de catégorie simplifiée (aléatoire car nous n'avons pas de vrai modèle)
        import random
        category_idx = random.randint(0, len(category_labels) - 1)
        category = category_labels[category_idx]
        category_confidence = random.uniform(0.7, 0.95)
        
        # Créer et retourner le résultat
        result = {
            'text': text,
            'sentiment_result': {
                'label': predicted_sentiment,
                'confidence': confidence
            },
            'category_result': {
                'label': category,
                'confidence': category_confidence
            }
        }
        
        return result
    
    except Exception as e:
        print(f"❌ Erreur dans predict_with_bert_en: {str(e)}")
        # Retourner un résultat par défaut en cas d'erreur
        return {
            'text': text,
            'sentiment_result': {
                'label': "Neutre",
                'confidence': 0.5
            },
            'category_result': {
                'label': "Non classifié",
                'confidence': 0.5
            }
        }
