#app.py

import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify

MODEL_PATH = os.getenv("MODEL_PATH", "model/Churn_model.pkl")

app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Couldn't load the model from {MODEL_PATH}: {e}")

@app.route('/')
def home():
    return jsonify({
        "message": "Churn Prediction API is live!",
        "routes": {
            "/health": "Health check endpoint",
            "/predict": "POST endpoint for churn prediction"
        },
        "usage": {
            "example_input": {"input": [[1, 0, 23, 45, 12000, 2]]}
        }
    })

@app.get("/health")
def health():
    return {"status": "ok"}, 200

@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        x = payload.get("input")
        if x is None:
            return jsonify(error="Missing input"), 400

        if isinstance(x, list) and len(x) > 0 and not isinstance(x[0], list):
            x = [x]

        X = np.array(x, dtype=float)
        preds = model.predict(X)
        return jsonify(prediction=preds.tolist()), 200

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
