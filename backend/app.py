from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # prevent GUI errors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    result = model.predict(features)[0]
    return jsonify({'result': 'Parkinsons Detected' if result == 1 else 'Healthy'})

@app.route('/generate-plots', methods=['POST'])
def generate_plots():
    # Simulated test results
    y_test = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion.png")
    plt.close()

    # Feature Importance
    importances = model.feature_importances_
    features = model.feature_names_in_
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[sorted_idx], y=features[sorted_idx])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("importance.png")
    plt.close()

    return jsonify({"status": "success"})

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)
