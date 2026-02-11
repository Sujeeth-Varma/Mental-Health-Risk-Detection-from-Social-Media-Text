"""Model training and evaluation module for the Mental Health Risk Detector."""

import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from scipy.sparse import hstack, issparse
import pandas as pd

from config import (
    EVALUATION_REPORT,
    LABEL_ENCODER,
    LOGISTIC_MODEL,
    MODELS_DIR,
    RANDOM_FOREST_MODEL,
    SVM_MODEL,
    TFIDF_VECTORIZER,
)
from pipeline.feature_extractor import FeatureExtractor
from pipeline.preprocessor import TextPreprocessor


class ModelTrainer:
    """Train, evaluate, and save ML models for mental health risk classification."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.best_model_name = None
        self.best_model = None

    def prepare_features(self, texts: list, fit: bool = True) -> np.ndarray:
        """Prepare feature matrix from raw texts."""
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_batch(texts)

        # TF-IDF features
        if fit:
            self.feature_extractor.fit_tfidf(processed_texts)
        tfidf_matrix = self.feature_extractor.transform_tfidf(processed_texts)

        # Linguistic features
        linguistic_features = []
        for text in texts:
            features = self.feature_extractor.build_feature_vector(text)
            linguistic_features.append(list(features.values()))

        linguistic_matrix = np.array(linguistic_features)

        # Combine TF-IDF and linguistic features
        if issparse(tfidf_matrix):
            from scipy.sparse import csr_matrix
            linguistic_sparse = csr_matrix(linguistic_matrix)
            combined = hstack([tfidf_matrix, linguistic_sparse])
        else:
            combined = np.hstack([tfidf_matrix, linguistic_matrix])

        return combined

    def train(self, texts: list, labels: list) -> dict:
        """Train multiple models and select the best one."""
        print("\n📊 Preparing features...")
        X = self.prepare_features(texts, fit=True)
        y = np.array(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        print(f"   Feature dimensions: {X_train.shape[1]}")

        # Define models
        model_configs = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
                C=1.0,
                multi_class="multinomial",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "svm": SVC(
                kernel="linear",
                probability=True,
                random_state=42,
                class_weight="balanced",
                C=1.0,
            ),
        }

        results = {}
        best_f1 = 0

        for name, model in model_configs.items():
            print(f"\n🔧 Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()
            report = classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"], output_dict=True, zero_division=0)

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1_weighted")

            results[name] = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "cv_f1_mean": round(cv_scores.mean(), 4),
                "cv_f1_std": round(cv_scores.std(), 4),
                "confusion_matrix": cm,
                "classification_report": report,
            }

            self.models[name] = model

            print(f"   Accuracy: {accuracy:.4f} | F1: {f1:.4f} | CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name
                self.best_model = model

        print(f"\n🏆 Best model: {self.best_model_name} (F1: {best_f1:.4f})")
        return results

    def save_models(self):
        """Save all trained models and artifacts."""
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Save models
        if "logistic_regression" in self.models:
            joblib.dump(self.models["logistic_regression"], LOGISTIC_MODEL)
        if "random_forest" in self.models:
            joblib.dump(self.models["random_forest"], RANDOM_FOREST_MODEL)
        if "svm" in self.models:
            joblib.dump(self.models["svm"], SVM_MODEL)

        # Save TF-IDF vectorizer
        if self.feature_extractor.tfidf is not None:
            joblib.dump(self.feature_extractor.tfidf, TFIDF_VECTORIZER)

        # Save best model name
        meta = {"best_model": self.best_model_name}
        joblib.dump(meta, LABEL_ENCODER)

        print(f"\n💾 Models saved to {MODELS_DIR}/")

    def save_evaluation(self, results: dict):
        """Save evaluation results to JSON."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(EVALUATION_REPORT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"📝 Evaluation report saved to {EVALUATION_REPORT}")

    def load_model(self, model_name: str = None):
        """Load a trained model."""
        # Load meta to find best model
        if model_name is None:
            if os.path.exists(LABEL_ENCODER):
                meta = joblib.load(LABEL_ENCODER)
                model_name = meta.get("best_model", "logistic_regression")
            else:
                model_name = "logistic_regression"

        model_paths = {
            "logistic_regression": LOGISTIC_MODEL,
            "random_forest": RANDOM_FOREST_MODEL,
            "svm": SVM_MODEL,
        }

        path = model_paths.get(model_name)
        if path and os.path.exists(path):
            self.best_model = joblib.load(path)
            self.best_model_name = model_name
            print(f"✅ Loaded model: {model_name}")
        else:
            raise FileNotFoundError(f"Model file not found for {model_name}")

        # Load TF-IDF
        if os.path.exists(TFIDF_VECTORIZER):
            self.feature_extractor.tfidf = joblib.load(TFIDF_VECTORIZER)
            print("✅ Loaded TF-IDF vectorizer")
        else:
            raise FileNotFoundError("TF-IDF vectorizer not found")

    def predict(self, text: str) -> dict:
        """Predict risk level for a single text."""
        if self.best_model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        processed = self.preprocessor.preprocess(text)
        tfidf_features = self.feature_extractor.transform_tfidf([processed])
        linguistic = self.feature_extractor.build_feature_vector(text)
        linguistic_array = np.array([list(linguistic.values())])

        if issparse(tfidf_features):
            from scipy.sparse import csr_matrix
            combined = hstack([tfidf_features, csr_matrix(linguistic_array)])
        else:
            combined = np.hstack([tfidf_features, linguistic_array])

        prediction = int(self.best_model.predict(combined)[0])

        # Get probabilities if available
        probabilities = {}
        if hasattr(self.best_model, "predict_proba"):
            proba = self.best_model.predict_proba(combined)[0]
            probabilities = {
                "low": round(float(proba[0]), 4),
                "medium": round(float(proba[1]), 4),
                "high": round(float(proba[2]), 4),
            }

        risk_map = {0: "Low", 1: "Medium", 2: "High"}

        return {
            "risk_level": risk_map.get(prediction, "Unknown"),
            "risk_code": prediction,
            "probabilities": probabilities,
            "model_used": self.best_model_name,
        }
