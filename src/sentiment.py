# src/sentiment.py
"""
Sentiment detection module.
Uses NLTK VADER as default (quick, lightweight).
Also includes optional transformer-based pipeline (if available).
"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure vader lexicon is available
nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

def get_sentiment_vader(text: str):
    """Return ('positive'|'neutral'|'negative', score) using VADER."""
    scores = sia.polarity_scores(text)
    comp = scores['compound']
    if comp >= 0.05:
        return "positive", comp
    elif comp <= -0.05:
        return "negative", comp
    else:
        return "neutral", comp

# Optional transformer-based sentiment (may take longer to download)
transformer_sentiment = None
try:
    from transformers import pipeline
    transformer_sentiment = pipeline("sentiment-analysis")
except Exception:
    transformer_sentiment = None

def detect_sentiment(text: str, method: str = "auto"):
    """
    Detect sentiment for given text.
    method: "auto" (use VADER), or "transformer" (if transformer available).
    Returns (label, score).
    """
    if method == "transformer" and transformer_sentiment:
        res = transformer_sentiment(text)[0]
        # res typically has 'label' and 'score'
        label = res.get("label", "").lower()
        score = float(res.get("score", 0.0))
        # normalize common label names
        if label in ("positive", "negative", "neutral"):
            return label, score
        # some models return LABEL_0 etc. try to map
        if label.startswith("label_"):
            try:
                idx = int(label.split("_")[1])
                mapping = {0: "negative", 1: "neutral", 2: "positive"}
                return mapping.get(idx, "neutral"), score
            except Exception:
                return "neutral", score
        return label, score

    # fallback/default: VADER
    return get_sentiment_vader(text)
