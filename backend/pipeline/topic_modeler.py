"""Topic modeling module for the Mental Health Risk Detector."""

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import os

from config import TOPIC_MODEL, TOPIC_VECTORIZER, N_TOPICS


class TopicModeler:
    """Topic modeling using LDA and NMF."""

    def __init__(self, n_topics: int = N_TOPICS, method: str = "lda"):
        self.n_topics = n_topics
        self.method = method
        self.model = None
        self.vectorizer = None

    def fit(self, texts: list):
        """Fit topic model on a corpus of texts."""
        if self.method == "lda":
            self.vectorizer = CountVectorizer(
                max_features=2000,
                max_df=0.95,
                min_df=2,
                stop_words="english",
            )
            doc_term_matrix = self.vectorizer.fit_transform(texts)
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=20,
                learning_method="online",
            )
        else:  # NMF
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                max_df=0.95,
                min_df=2,
                stop_words="english",
            )
            doc_term_matrix = self.vectorizer.fit_transform(texts)
            self.model = NMF(
                n_components=self.n_topics,
                random_state=42,
                max_iter=200,
            )

        self.model.fit(doc_term_matrix)
        return self

    def get_topics(self, n_words: int = 8) -> list:
        """Get top words for each topic."""
        if self.model is None or self.vectorizer is None:
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for idx, topic in enumerate(self.model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            topics.append({
                "topic_id": idx,
                "keywords": top_words,
                "label": f"Topic {idx + 1}",
            })
        return topics

    def predict_topic(self, text: str) -> dict:
        """Predict the dominant topic for a given text."""
        if self.model is None or self.vectorizer is None:
            return {"topic_id": -1, "keywords": [], "confidence": 0.0}

        try:
            doc_term = self.vectorizer.transform([text])
            topic_distribution = self.model.transform(doc_term)[0]
            dominant_topic = topic_distribution.argmax()
            confidence = float(topic_distribution[dominant_topic])

            feature_names = self.vectorizer.get_feature_names_out()
            topic_words = self.model.components_[dominant_topic]
            top_words = [feature_names[i] for i in topic_words.argsort()[:-6:-1]]

            return {
                "topic_id": int(dominant_topic),
                "keywords": top_words,
                "confidence": round(confidence, 4),
                "distribution": [round(float(d), 4) for d in topic_distribution],
            }
        except Exception:
            return {"topic_id": -1, "keywords": [], "confidence": 0.0}

    def save(self):
        """Save topic model and vectorizer."""
        os.makedirs(os.path.dirname(TOPIC_MODEL), exist_ok=True)
        joblib.dump(self.model, TOPIC_MODEL)
        joblib.dump(self.vectorizer, TOPIC_VECTORIZER)

    def load(self):
        """Load topic model and vectorizer."""
        if os.path.exists(TOPIC_MODEL) and os.path.exists(TOPIC_VECTORIZER):
            self.model = joblib.load(TOPIC_MODEL)
            self.vectorizer = joblib.load(TOPIC_VECTORIZER)
            return True
        return False
