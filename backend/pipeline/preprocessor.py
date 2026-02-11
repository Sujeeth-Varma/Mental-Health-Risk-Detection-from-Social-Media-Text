"""Text preprocessing module for the Mental Health Risk Detector."""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Handles all text cleaning and preprocessing operations."""

    def __init__(self):
        # Ensure NLTK data is available
        for resource in ["punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

        self.stop_words = set(stopwords.words("english"))
        # Keep negation words as they are important for sentiment
        self.keep_words = {"not", "no", "never", "neither", "nobody", "nothing",
                          "nowhere", "nor", "cannot", "without", "hardly", "barely"}
        self.stop_words -= self.keep_words
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Basic text cleaning: lowercasing, removing URLs, mentions, hashtags, etc."""
        if not text or not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove user mentions
        text = re.sub(r"@\w+", "", text)

        # Remove hashtag symbols but keep the text
        text = re.sub(r"#", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove special characters and numbers (keep letters and spaces)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> list:
        """Tokenize text into words."""
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def remove_stopwords(self, tokens: list) -> list:
        """Remove stopwords while keeping negation words."""
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]

    def lemmatize(self, tokens: list) -> list:
        """Lemmatize tokens to their base form."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline: clean → tokenize → stopwords → lemmatize → join."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)

    def preprocess_batch(self, texts: list) -> list:
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in texts]

    def get_tokens(self, text: str) -> list:
        """Get preprocessed tokens from text."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens
