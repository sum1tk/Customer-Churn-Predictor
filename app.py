"""
app.py
Minimal inference API for the churn model. Deployed to EC2 by CodeDeploy.
"""
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].tolist()
    return jsonify({
        "prediction": int(prediction),
        "churn_probability": probability[1]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
