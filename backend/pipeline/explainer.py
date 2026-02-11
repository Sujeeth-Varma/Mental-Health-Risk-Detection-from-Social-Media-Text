"""Explainability module for the Mental Health Risk Detector (LIME-based)."""

import numpy as np
from lime.lime_text import LimeTextExplainer
from scipy.sparse import hstack, csr_matrix, issparse

from pipeline.preprocessor import TextPreprocessor
from pipeline.feature_extractor import FeatureExtractor


class Explainer:
    """Provide explanations for model predictions using LIME."""

    def __init__(self, model, tfidf_vectorizer):
        self.model = model
        self.tfidf = tfidf_vectorizer
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.tfidf = tfidf_vectorizer
        self.lime_explainer = LimeTextExplainer(
            class_names=["Low", "Medium", "High"],
            split_expression=r"\W+",
            random_state=42,
        )

    def _predict_proba(self, texts: list) -> np.ndarray:
        """Predict probabilities for a list of texts (used by LIME)."""
        results = []
        for text in texts:
            processed = self.preprocessor.preprocess(text)
            tfidf_features = self.tfidf.transform([processed])
            linguistic = self.feature_extractor.build_feature_vector(text)
            linguistic_array = np.array([list(linguistic.values())])

            if issparse(tfidf_features):
                combined = hstack([tfidf_features, csr_matrix(linguistic_array)])
            else:
                combined = np.hstack([tfidf_features, linguistic_array])

            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(combined)[0]
            else:
                pred = self.model.predict(combined)[0]
                proba = np.zeros(3)
                proba[int(pred)] = 1.0

            results.append(proba)

        return np.array(results)

    def explain_lime(self, text: str, num_features: int = 10) -> dict:
        """Generate LIME explanation for a prediction."""
        try:
            explanation = self.lime_explainer.explain_instance(
                text,
                self._predict_proba,
                num_features=num_features,
                num_samples=200,
            )

            # Get feature contributions
            feature_weights = explanation.as_list()
            word_contributions = [
                {
                    "word": word,
                    "weight": round(float(weight), 4),
                    "impact": "positive" if weight > 0 else "negative",
                }
                for word, weight in feature_weights
            ]

            # Sort by absolute weight
            word_contributions.sort(key=lambda x: abs(x["weight"]), reverse=True)

            # Get prediction probabilities from the explanation
            prediction_probabilities = {}
            try:
                proba = explanation.predict_proba
                prediction_probabilities = {
                    "low": round(float(proba[0]), 4),
                    "medium": round(float(proba[1]), 4),
                    "high": round(float(proba[2]), 4),
                }
            except Exception:
                pass

            return {
                "word_contributions": word_contributions,
                "prediction_probabilities": prediction_probabilities,
                "num_features_shown": len(word_contributions),
            }

        except Exception as e:
            return {
                "word_contributions": [],
                "prediction_probabilities": {},
                "error": str(e),
            }

    def get_feature_importance(self, feature_names: list = None) -> list:
        """Get global feature importance from the model."""
        importances = []

        if hasattr(self.model, "feature_importances_"):
            # Random Forest
            raw_importances = self.model.feature_importances_
            if feature_names and len(feature_names) == len(raw_importances):
                for name, imp in zip(feature_names, raw_importances):
                    importances.append({"feature": name, "importance": round(float(imp), 6)})
            else:
                for i, imp in enumerate(raw_importances):
                    importances.append({"feature": f"feature_{i}", "importance": round(float(imp), 6)})

        elif hasattr(self.model, "coef_"):
            # Logistic Regression / SVM
            coefs = np.abs(self.model.coef_).mean(axis=0)
            if feature_names and len(feature_names) == len(coefs):
                for name, coef in zip(feature_names, coefs):
                    importances.append({"feature": name, "importance": round(float(coef), 6)})
            else:
                for i, coef in enumerate(coefs):
                    importances.append({"feature": f"feature_{i}", "importance": round(float(coef), 6)})

        # Sort and return top features
        importances.sort(key=lambda x: x["importance"], reverse=True)
        return importances[:20]
