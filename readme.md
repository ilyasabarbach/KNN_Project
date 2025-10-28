# Projet de Classification KNN (avec API Flask)

Ce projet implémente un modèle de Machine Learning (K-Nearest Neighbors) pour prédire si un utilisateur achètera un produit en fonction de son genre, de son âge et de son salaire estimé.

Le projet comprend l'analyse exploratoire complète (Notebook) ainsi qu'une **API Flask déployable** pour les prédictions en temps réel.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-black?style=for-the-badge&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)

## 📂 Structure du Projet

Ce projet adopte une structure MLOps propre pour séparer les données, l'analyse, les modèles et le code de l'application.
KNN_Project/ ├── data/ │ └── Social_Network_Ads.csv ├── models/ │ ├── knn_model.pkl │ └── scaler.pkl ├── notebooks/ │ └── KNN_Classification_TP.ipynb ├── src/ │ └── app.py ├── README.md └── requirements.txt

## 🚀 Démarrage (Lancer l'API)

Pour exécuter ce projet et lancer le service de prédiction localement, suivez ces étapes.

### 1. Prérequis

Assurez-vous d'avoir Python 3.8+ et `pip` installés.

### 2. Installation

1.  **Clonez ce dépôt :**
    ```bash
    git clone [https://github.com/ilyasabarbach/KNN_Project.git](https://github.com/ilyasabarbach/KNN_Project.git)
    cd KNN_Project
    ```

2.  **Installez les dépendances :**
    Toutes les bibliothèques nécessaires sont listées dans `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### 3. Lancement du Serveur

Lancez le serveur Flask depuis la racine du projet :

```bash
python src/app.py
```

Le serveur démarrera sur http://127.0.0.1:5000/.

### 4. Comment tester l'API

Une fois le serveur lancé, vous pouvez tester le point de terminaison /predict à l'aide d'un outil comme Postman ou curl.

URL : http://127.0.0.1:5000/predict

Méthode : POST

Body (raw, JSON) :
{
    "Age": 30,
    "EstimatedSalary": 50000,
    "Gender": 1
}

Si la requête est réussie, vous recevrez une réponse au format JSON :

{
    "message": "The model predicts the user will NOT purchase.",
    "prediction": 0
}

### 5. Analyse et Modèle

1.  **Analyse Exploratoire :**

Pour une analyse détaillée, la préparation des données et les étapes d'entraînement du modèle, veuillez consulter le Jupyter Notebook situé dans le dossier /notebooks/.

2.  **Résultats du Modèle :**


Le modèle final est un K-Nearest Neighbors (KNN) entraîné sur 3 caractéristiques (Genre, Âge, Salaire Estimé).

Précision (Accuracy) sur l'ensemble de test : 93 %