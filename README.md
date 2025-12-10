# 🏨 AI Hotel Recommender
Une application intelligente de recommandation d'hôtels alimentée par le Machine Learning. Ce projet utilise une approche hybride combinant l'analyse sémantique (NLP) des avis clients et un modèle prédictif XGBoost pour suggérer les établissements les plus adaptés au profil de chaque voyageur.


🚀 Fonctionnalités Clés
Moteur de Recommandation Personnalisé : Suggère les meilleurs hôtels en fonction du type de voyage (Solo, Couple, Famille, Affaires), de la durée du séjour et de la destination.

Analyse Hybride Avancée :

Métadonnées : Analyse les notes, la popularité et les caractéristiques de l'hôtel.

Contenu (NLP) : Utilise TF-IDF pour analyser les "Tags" et descriptions des hôtels afin de trouver les correspondances sémantiques avec le succès.

Prédiction de Satisfaction : Calcule une probabilité de satisfaction (Score IA) pour chaque hôtel candidat.

Interface Interactive : Dashboard simple et intuitif construit avec Streamlit.

🛠️ Stack Technique
Langage : Python 3.x

Interface Web : Streamlit

Machine Learning : XGBoost, Scikit-Learn

NLP (Traitement du Langage) : TextBlob, TF-IDF

Manipulation de Données : Pandas, NumPy

Sérialisation : Joblib

📦 Installation et Lancement
Pour exécuter ce projet localement sur votre machine :

Cloner le dépôt :

Bash

git clone https://github.com/sefraniabdou1937/proje_ML_HOTEL.git
cd proje_ML_HOTEL
Installer les dépendances :

Bash

pip install -r requirements.txt
Lancer l'application :

Bash

streamlit run app.py
L'application s'ouvrira automatiquement dans votre navigateur à l'adresse http://localhost:8501.

📂 Structure du Projet
app.py : Le cœur de l'application. Contient l'interface Streamlit et la logique de recommandation en temps réel.

model.ipynb : Le notebook Jupyter utilisé pour l'analyse exploratoire des données (EDA), le nettoyage et l'entraînement du modèle.

modele_hotel_xgboost.pkl : Le modèle XGBoost entraîné et sauvegardé.

vectorizer_tfidf.pkl : Le vectoriseur TF-IDF pour le traitement du texte.

requirements.txt : Liste de toutes les bibliothèques Python néces saires.

Hotel_Reviews.zip (Non inclus/À télécharger) : Le dataset source .

🧠 À propos du Modèle
Le modèle a été entraîné sur le dataset "515K Hotel Reviews Data in Europe" (Source : Kaggle).

Entrées (Features) : 1017 caractéristiques, incluant des vecteurs de texte (TF-IDF sur les avis/tags) et des variables catégorielles encodées (Pays, Type de voyageur).

Algorithme : XGBoost Classifier optimisé.

Performance : Précision (Accuracy) supérieure à 80% sur le jeu de test.

📝 Données
Le projet nécessite le fichier de données Hotel_Reviews.csv. Pour des raisons de taille, il peut être nécessaire de le télécharger manuellement depuis Kaggle si le fichier zip n'est pas présent https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe

