"""Flask application for the Mental Health Risk Detector API."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS

from config import (
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    RISK_LEVELS,
    MIN_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
    EVALUATION_REPORT,
)
from pipeline.model_trainer import ModelTrainer
from pipeline.feature_extractor import FeatureExtractor
from pipeline.explainer import Explainer
from pipeline.topic_modeler import TopicModeler
from utils.helpers import validate_text, format_risk_response

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global objects
trainer = None
feature_extractor = None
explainer = None
topic_modeler = None
models_loaded = False


def load_models():
    """Load all ML models at server startup."""
    global trainer, feature_extractor, explainer, topic_modeler, models_loaded

    try:
        print("🔄 Loading ML models...")

        # Load classifier
        trainer = ModelTrainer()
        trainer.load_model()

        # Feature extractor
        feature_extractor = FeatureExtractor()

        # Explainer
        explainer = Explainer(trainer.best_model, trainer.feature_extractor.tfidf)

        # Topic modeler
        topic_modeler = TopicModeler()
        topic_modeler.load()

        models_loaded = True
        print("✅ All models loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("   Please run 'python train.py' first to train the models.")
        models_loaded = False


# ─── API ENDPOINTS ─────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    """System health check endpoint."""
    return jsonify({
        "status": "healthy" if models_loaded else "models_not_loaded",
        "models_loaded": models_loaded,
        "model_name": trainer.best_model_name if trainer and trainer.best_model_name else None,
        "disclaimer": "This tool is NOT a clinical diagnosis system. For educational and research purposes only.",
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Predict mental health risk level with explanations."""
    if not models_loaded:
        return jsonify({"error": "Models not loaded. Please train models first."}), 503

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    text = data["text"]

    # Validate input
    is_valid, error_msg = validate_text(text, MIN_TEXT_LENGTH, MAX_TEXT_LENGTH)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    try:
        # Get prediction
        prediction = trainer.predict(text)

        # Format risk response
        risk_response = format_risk_response(prediction, RISK_LEVELS)

        # Get LIME explanation
        explanation = explainer.explain_lime(text, num_features=10)

        # Get sentiment and emotion features
        features = feature_extractor.get_combined_features(text)

        # Get topic analysis
        topic = topic_modeler.predict_topic(text) if topic_modeler.model else {}

        response = {
            "prediction": risk_response,
            "explanation": explanation,
            "features": features,
            "topic": topic,
            "input_text": text,
            "disclaimer": "This is NOT a clinical diagnosis. Please consult a mental health professional.",
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze text for sentiment, emotion, and topics without prediction."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    text = data["text"]

    is_valid, error_msg = validate_text(text, MIN_TEXT_LENGTH, MAX_TEXT_LENGTH)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    try:
        fe = feature_extractor if feature_extractor else FeatureExtractor()

        # Get features
        features = fe.get_combined_features(text)

        # Get topic if available
        topic = {}
        if topic_modeler and topic_modeler.model:
            topic = topic_modeler.predict_topic(text)

        response = {
            "sentiment": {
                "vader": features["vader_sentiment"],
                "textblob": features["textblob_sentiment"],
            },
            "emotional_tone": features["emotional_tone"],
            "top_unigrams": features["top_unigrams"],
            "top_bigrams": features["top_bigrams"],
            "topic": topic,
            "input_text": text,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/evaluation", methods=["GET"])
def evaluation():
    """Get model evaluation report."""
    if os.path.exists(EVALUATION_REPORT):
        with open(EVALUATION_REPORT, "r") as f:
            report = json.load(f)
        return jsonify(report)
    else:
        return jsonify({"error": "Evaluation report not found. Train models first."}), 404


# ─── MAIN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    load_models()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
