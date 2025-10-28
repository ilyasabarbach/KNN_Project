import joblib
from flask import Flask, request, jsonify
import numpy as np
import pathlib 

# --- 1. CONSTRUCTION D'UN CHEMIN ABSOLU (ROBUSTE) ---
APP_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

# --- 2. CHARGEMENT DES MODELES ---
try:
    model_path = MODELS_DIR / 'knn_model.pkl'
    scaler_path = MODELS_DIR / 'scaler.pkl'
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Modèles chargés depuis : {MODELS_DIR}")

except FileNotFoundError:
    print(f"ERREUR: Fichiers .pkl introuvables. Vérifiez qu'ils existent bien dans : {MODELS_DIR}")

# --- 3. INITIALISATION DE L'APPLICATION FLASK ---
app = Flask(__name__)

# --- 4. DÉFINITION DE LA ROUTE D'API (MISE À JOUR) ---
@app.route('/predict', methods=['POST'])
def predict():
    # Réception des données au format JSON (clé-valeur)
    data = request.get_json(force=True)
    
    try:
        # A. Extraire les données par clé (Key-Value)
        # Note : L'ordre dans le JSON n'a pas d'importance
        age = data['Age']
        salary = data['EstimatedSalary']
        gender = data['Gender']
        
        # B. Assembler les données dans l'ORDRE EXACT de l'entraînement
        # (Basé sur notre débogage, l'ordre était [Gender, Age, Salary])
        new_data = np.array([[gender, age, salary]])

    except KeyError:
        return jsonify({'error': 'Clé(s) manquante(s). Le JSON doit contenir "Age", "EstimatedSalary", et "Gender".'}), 400
    except TypeError:
         return jsonify({'error': 'Format de données invalide. Vérifiez les valeurs.'}), 400

    # Standardisation des nouvelles données
    scaled_data = scaler.transform(new_data)
    
    # Prédiction par le modèle
    prediction = model.predict(scaled_data)
    prediction_value = int(prediction[0]) # Obtenir 0 ou 1
    
    # C. Formater la nouvelle réponse personnalisée
    if prediction_value == 1:
        message = "The model predicts the user WILL purchase."
    else:
        message = "The model predicts the user will NOT purchase."
    
    # Retour du résultat au format JSON souhaité
    return jsonify({
        'message': message,
        'prediction': prediction_value
    })

# --- 5. LANCEMENT DU SERVEUR ---
if __name__ == '__main__':
    # Lance l'application en mode debug
    app.run(debug=True)