# src/app_core.py
"""
Core integration: uses sentiment detector and generator to produce outputs.
"""

from .sentiment import detect_sentiment
from .generator import generate_text

def ai_text_generator(prompt: str,
                      manual_sentiment: str = None,
                      length: int = 150,
                      temperature: float = 0.7,
                      top_p: float = 0.9):
    """
    Returns (sentiment_label, generated_text).
    If manual_sentiment (positive/neutral/negative) is provided, it overrides auto-detection.
    """
    if manual_sentiment:
        sentiment = manual_sentiment
    else:
        sentiment, _ = detect_sentiment(prompt)
    text = generate_text(prompt, sentiment=sentiment,
                         max_length=length, temperature=temperature, top_p=top_p)
    return sentiment, text
