"""Feature extraction module for the Mental Health Risk Detector."""

from collections import Counter

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.preprocessor import TextPreprocessor

# NRC-style emotion lexicon (keyword-based)
_EMOTION_LEXICON = {
    "joy": {"happy", "joy", "love", "wonderful", "amazing", "great", "excited",
            "fantastic", "fun", "laugh", "smile", "delight", "cheerful", "pleased",
            "grateful", "thankful", "blessed", "beautiful", "enjoy", "celebrate",
            "proud", "bright", "awesome", "glad", "thrilled", "content", "peaceful"},
    "sadness": {"sad", "depressed", "unhappy", "miserable", "hopeless", "lonely",
                "cry", "crying", "tears", "grief", "sorrow", "empty", "broken",
                "heartbroken", "gloomy", "despair", "lost", "abandoned", "pain",
                "suffer", "worthless", "numb", "dark", "darkness", "meaningless"},
    "anger": {"angry", "furious", "rage", "hate", "mad", "annoyed", "irritated",
              "frustrated", "hostile", "bitter", "resentful", "outraged", "livid",
              "enraged", "aggressive", "violent", "destroy", "kill", "disgusted"},
    "fear": {"afraid", "scared", "terrified", "anxious", "panic", "nervous",
             "worried", "dread", "horror", "frightened", "phobia", "paranoid",
             "uneasy", "tense", "overwhelmed", "helpless", "trapped", "threat"},
    "surprise": {"surprised", "shocked", "amazed", "astonished", "unexpected",
                 "sudden", "startled", "speechless", "unbelievable", "wow",
                 "incredible", "stunning", "remarkable"},
    "disgust": {"disgusting", "gross", "revolting", "sick", "nasty", "awful",
                "terrible", "horrible", "repulsive", "vile", "pathetic", "loathe"},
    "trust": {"trust", "believe", "faith", "honest", "reliable", "loyal",
              "support", "friend", "safe", "comfort", "confident", "caring"},
    "anticipation": {"hope", "expect", "excited", "looking forward", "plan",
                     "eager", "curious", "ready", "wish", "dream", "future",
                     "potential", "opportunity", "goal"},
}


class FeatureExtractor:
    """Extracts linguistic features: sentiment, emotion, n-grams, TF-IDF."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vader = SentimentIntensityAnalyzer()
        self.tfidf = None

    def get_vader_sentiment(self, text: str) -> dict:
        """Get VADER sentiment scores."""
        scores = self.vader.polarity_scores(text)
        return {
            "compound": round(scores["compound"], 4),
            "positive": round(scores["pos"], 4),
            "negative": round(scores["neg"], 4),
            "neutral": round(scores["neu"], 4),
        }

    def get_textblob_sentiment(self, text: str) -> dict:
        """Get TextBlob sentiment analysis."""
        blob = TextBlob(text)
        return {
            "polarity": round(blob.sentiment.polarity, 4),
            "subjectivity": round(blob.sentiment.subjectivity, 4),
        }

    def get_emotional_tone(self, text: str) -> dict:
        """Get emotional tone using keyword-based NRC-style lexicon."""
        words = set(text.lower().split())
        raw_scores = {}
        for emotion, lexicon in _EMOTION_LEXICON.items():
            score = len(words & lexicon)
            raw_scores[emotion] = score

        total = sum(raw_scores.values()) or 1
        normalized = {k: round(v / total, 4) for k, v in raw_scores.items()}
        return normalized

    def get_ngrams(self, text: str, n: int = 2, top_k: int = 10) -> list:
        """Extract top n-grams from text."""
        tokens = self.preprocessor.get_tokens(text)
        if len(tokens) < n:
            return [(" ".join(tokens), 1)] if tokens else []

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i + n])
            ngrams.append(ngram)

        counter = Counter(ngrams)
        return counter.most_common(top_k)

    def get_combined_features(self, text: str) -> dict:
        """Get all features combined for a single text."""
        vader = self.get_vader_sentiment(text)
        textblob = self.get_textblob_sentiment(text)
        emotions = self.get_emotional_tone(text)
        unigrams = self.get_ngrams(text, n=1, top_k=10)
        bigrams = self.get_ngrams(text, n=2, top_k=10)

        return {
            "vader_sentiment": vader,
            "textblob_sentiment": textblob,
            "emotional_tone": emotions,
            "top_unigrams": [{"term": term, "count": count} for term, count in unigrams],
            "top_bigrams": [{"term": term, "count": count} for term, count in bigrams],
        }

    def build_feature_vector(self, text: str) -> dict:
        """Build a numeric feature vector for ML input."""
        vader = self.get_vader_sentiment(text)
        textblob = self.get_textblob_sentiment(text)
        emotions = self.get_emotional_tone(text)

        features = {
            "vader_compound": vader["compound"],
            "vader_positive": vader["positive"],
            "vader_negative": vader["negative"],
            "vader_neutral": vader["neutral"],
            "textblob_polarity": textblob["polarity"],
            "textblob_subjectivity": textblob["subjectivity"],
        }

        # Add emotion features
        for emotion, score in emotions.items():
            features[f"emotion_{emotion}"] = score

        return features

    def fit_tfidf(self, texts: list, max_features: int = 5000) -> TfidfVectorizer:
        """Fit TF-IDF vectorizer on a corpus."""
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.tfidf.fit(texts)
        return self.tfidf

    def transform_tfidf(self, texts: list):
        """Transform texts using fitted TF-IDF vectorizer."""
        if self.tfidf is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        return self.tfidf.transform(texts)
