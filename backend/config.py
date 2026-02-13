"""Configuration settings for the Mental Health Risk Detector backend."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Model filenames
LOGISTIC_MODEL = os.path.join(MODELS_DIR, "logistic_regression.joblib")
RANDOM_FOREST_MODEL = os.path.join(MODELS_DIR, "random_forest.joblib")
SVM_MODEL = os.path.join(MODELS_DIR, "svm_model.joblib")
TFIDF_VECTORIZER = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LABEL_ENCODER = os.path.join(MODELS_DIR, "label_encoder.joblib")
TOPIC_MODEL = os.path.join(MODELS_DIR, "topic_model.joblib")
TOPIC_VECTORIZER = os.path.join(MODELS_DIR, "topic_vectorizer.joblib")
EVALUATION_REPORT = os.path.join(MODELS_DIR, "evaluation_report.json")

# Dataset
DATASET_PATH = os.path.join(DATA_DIR, "mental_health_dataset.csv")

# Risk levels
RISK_LEVELS = {
    0: {"label": "Low", "color": "#22c55e", "description": "Neutral or mildly emotional language"},
    1: {"label": "Medium", "color": "#eab308", "description": "Persistent negative emotions, distress signals"},
    2: {"label": "High", "color": "#ef4444", "description": "Strong negative sentiment, crisis indicators"},
}

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"

# NLP settings
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 5000
TOP_N_FEATURES = 10
N_TOPICS = 5
