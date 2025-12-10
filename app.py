import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from textblob import TextBlob

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Hotel Recommender AI", page_icon="🏨", layout="wide")

# --- 2. CHARGEMENT DES DONNÉES (Mis en cache pour la rapidité) ---
@st.cache_resource
def load_resources():
    # Chargement Modèles
    model = joblib.load('modele_hotel_xgboost.pkl')
    tfidf = joblib.load('vectorizer_tfidf.pkl')
    return model, tfidf

@st.cache_data
def load_data():
    # Chargement Dataset (On ne garde que les colonnes utiles pour aller vite)
    # Assure-toi que le fichier csv est dans le même dossier !
    df = pd.read_csv("archive.zip")
    
    # On nettoie un peu pour avoir une liste unique d'hôtels par pays
    # On groupe par Hôtel pour avoir ses stats moyennes
    # Astuce : On garde les Tags concaténés pour s'en servir comme "Description" pour l'IA
    hotel_stats = df.groupby(['Hotel_Name', 'Hotel_Address']).agg({
        'Average_Score': 'mean',
        'Total_Number_of_Reviews': 'first',
        'Tags': lambda x: ' '.join(x) # On combine tous les tags pour l'analyse de texte
    }).reset_index()
    
    # Extraction du pays
    hotel_stats['Country'] = hotel_stats['Hotel_Address'].apply(lambda x: x.split()[-1])
    
    # On nettoie les pays (parfois "Kingdom" pour UK)
    hotel_stats['Country'] = hotel_stats['Country'].replace('Kingdom', 'United Kingdom')
    
    return hotel_stats

try:
    with st.spinner("Chargement de l'IA et de la base de données..."):
        model, tfidf = load_resources()
        df_hotels = load_data()
    st.sidebar.success("✅ Système prêt !")
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# --- 3. SIDEBAR : LE PROFIL UTILISATEUR ---
st.sidebar.header("🔎 Vos Critères")

# Choix du pays (Filtrage de base)
pays_dispo = sorted(df_hotels['Country'].unique())
choix_pays = st.sidebar.selectbox("Destination", pays_dispo, index=pays_dispo.index("France") if "France" in pays_dispo else 0)

# Paramètres Personnels (Ceux qui influencent l'IA)
st.sidebar.subheader("Votre Profil")
type_voyage = st.sidebar.radio("Type", ["Loisir (Leisure)", "Affaires (Business)"])
groupe = st.sidebar.selectbox("Groupe", ["Couple", "Famille", "Solo", "Groupe/Amis"])
duree = st.sidebar.slider("Durée (Nuits)", 1, 14, 3)

# Bouton de recherche
lancer_recherche = st.sidebar.button("Trouver mon Hôtel Idéal 🌟", type="primary")

# --- 4. LA LOGIQUE DE RECOMMANDATION (IA) ---
if lancer_recherche:
    st.title(f"🏨 Top Hôtels recommandés en : {choix_pays}")
    
    # ÉTAPE A : Filtrer les hôtels du pays choisi
    candidats = df_hotels[df_hotels['Country'] == choix_pays].copy()
    
    if candidats.empty:
        st.warning("Aucun hôtel trouvé pour ce pays.")
    else:
        # ÉTAPE B : Préparer les données pour l'IA
        # On va simuler que CE client visite CHAQUE hôtel
        
        # 1. Variables Binaires (Tags Utilisateur)
        is_leisure = 1 if "Loisir" in type_voyage else 0
        is_business = 1 if "Affaires" in type_voyage else 0
        is_couple = 1 if groupe == "Couple" else 0
        is_family = 1 if groupe == "Famille" else 0
        is_solo = 1 if groupe == "Solo" else 0
        
        # 2. One-Hot Encoding du Pays
        countries_order = ['Austria', 'France', 'Italy', 'Netherlands', 'Spain', 'United Kingdom']
        country_vector = [1 if c == choix_pays else 0 for c in countries_order]
        
        # 3. Sentiment & Texte
        # Pour recommander, on utilise les TAGS de l'hôtel comme "Texte" à analyser par le TF-IDF
        # Cela permet à l'IA de voir si les mots-clés de l'hôtel matchent le succès
        # On simule un sentiment neutre/positif (0.5) car on espère que ça se passera bien
        sentiment_simule = 0.5 
        neg_words = 0
        pos_words = 10 # Valeur arbitraire positive
        
        # ÉTAPE C : Boucle de Prédiction (Vectorisation en masse)
        # Pour aller vite, on construit une grosse matrice
        
        # Convertir les tags des hôtels en vecteurs TF-IDF
        X_text = tfidf.transform(candidats['Tags'].astype(str))
        
        # Construire la matrice numérique pour tous les hôtels d'un coup
        # On répète les infos utilisateur pour chaque ligne d'hôtel
        nb_hotels = len(candidats)
        
        # Colonnes: Average_Score, NegWords, PosWords, TotalReviews, History(5), Tags..., Duration, Sentiment, Countries...
        # Attention à respecter l'ordre EXACT de l'entraînement
        # Colonnes: Average_Score, NegWords, PosWords, TotalReviews, Tags..., Duration, Sentiment, Countries...
        # Attention à respecter l'ordre EXACT de l'entraînement
        static_user_data = [
            0, # Placeholder pour Average_Score (sera remplacé plus bas)
            neg_words, 
            pos_words,
            0, # Placeholder pour Total_Reviews (sera remplacé plus bas)
            # --- LIGNE SUPPRIMÉE ICI (le "5") ---
            is_leisure, is_couple, is_solo, is_family, is_business,
            duree,
            sentiment_simule
        ] + country_vector
        
        # Création de la matrice numpy
        X_numeric = np.tile(static_user_data, (nb_hotels, 1)).astype(float)
        
        # On remplit les vraies valeurs de l'hôtel (Score et Total Reviews)
        X_numeric[:, 0] = candidats['Average_Score'].values
        X_numeric[:, 3] = candidats['Total_Number_of_Reviews'].values
        
        # Fusion
        X_final = hstack([X_text, X_numeric])
        
        # ÉTAPE D : Prédiction des Probabilités
        # On demande la probabilité d'être classe 1 (Bon hôtel)
        probs = model.predict_proba(X_final)[:, 1]
        
        # On ajoute le score IA au dataframe
        candidats['AI_Score'] = probs
        
        # On trie par le Score IA (et non par la note moyenne classique !)
        top_hotels = candidats.sort_values(by='AI_Score', ascending=False).head(10)
        
        # --- 5. AFFICHAGE DES RÉSULTATS ---
        for index, row in top_hotels.iterrows():
            with st.container():
                c1, c2, c3 = st.columns([3, 1, 1])
                
                with c1:
                    st.subheader(f"🏨 {row['Hotel_Name']}")
                    st.caption(f"📍 {row['Hotel_Address']}")
                
                with c2:
                    st.metric("Note Booking", f"{row['Average_Score']}/10")
                
                with c3:
                    # Affichage du Score IA en vert
                    score_ia = row['AI_Score'] * 100
                    st.metric("Score IA (Match)", f"{score_ia:.1f}%", delta="Recommandé")
                
                st.progress(int(score_ia))
                st.divider()

else:
    st.info("👈 Configurez votre voyage dans le menu de gauche pour lancer l'IA.")
    st.markdown("""
    ### 🧠 Comment ça marche ?
    Ce n'est pas un simple classement par étoiles.
    
    Notre modèle **XGBoost Hybride** analyse :
    1. **Votre profil** (Solo, Famille, Business...).
    2. **Les caractéristiques de l'hôtel** (Basées sur l'analyse de milliers d'avis précédents).
    3. **La sémantique des tags** de l'hôtel via TF-IDF.
    
    Il calcule ensuite la probabilité que **VOUS** mettiez une note > 8/10 à cet hôtel.
    """)