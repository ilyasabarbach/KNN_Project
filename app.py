import joblib
from flask import Flask, request, jsonify
import numpy as np

# --- 1. CHARGEMENT DES MODELES ---
try:
    model = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    print("Erreur: Les fichiers knn_model.pkl ou scaler.pkl sont introuvables.")

# --- 2. INITIALISATION DE L'APPLICATION FLASK ---
app = Flask(__name__)

# --- 3. DÉFINITION DE LA ROUTE D'API ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    try:
        new_data = np.array(data['features'])
    except KeyError:
        return jsonify({'error': 'Le format JSON doit contenir la clé "features".'}), 400
    scaled_data = scaler.transform(new_data)
    
    prediction = model.predict(scaled_data)
    
    return jsonify({'prediction': int(prediction[0])})

# --- 4. LANCEMENT DU SERVEUR ---
if __name__ == '__main__':
    app.run(debug=True)