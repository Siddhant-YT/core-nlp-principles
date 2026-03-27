"""
utils/sentiment.py

Sentiment Analysis using VADER (from NLTK).

What is Sentiment Analysis?
    Sentiment analysis determines the emotional tone of text.
    It classifies text as Positive, Negative, or Neutral.

Two main approaches:
    1. Lexicon-based (VADER) — uses a dictionary of words tagged with sentiment scores
       Fast, no training needed, great for social media / short texts
    2. ML-based (BERT, etc.) — trained neural network
       More accurate for complex sentences but heavier

We use VADER here because:
    - It's lightweight and fast
    - Works well for general text
    - Handles punctuation emphasis (!! = more intense) and capitalization (GREAT vs great)
    - Returns scores for Positive, Negative, Neutral, and a Compound score

VADER Score Guide:
    compound >= 0.05  → Positive
    compound <= -0.05 → Negative
    in between        → Neutral
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER's lexicon if not already available
nltk.download("vader_lexicon", quiet=True)

# Initialize the VADER sentiment analyzer once
# (avoids recreating it on every function call)
_vader = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of the input text using VADER.

    Args:
        text: Any input text string

    Returns:
        dict with:
            - 'label':    'Positive', 'Negative', or 'Neutral'
            - 'compound': float from -1.0 (very negative) to +1.0 (very positive)
            - 'positive': fraction of text that is positive (0.0 to 1.0)
            - 'negative': fraction of text that is negative (0.0 to 1.0)
            - 'neutral':  fraction of text that is neutral (0.0 to 1.0)

    Example:
        analyze_sentiment("I absolutely love this product!")
        → {'label': 'Positive', 'compound': 0.6588, 'positive': 0.5, ...}
    """
    # polarity_scores returns a dict: {'neg': x, 'neu': x, 'pos': x, 'compound': x}
    scores = _vader.polarity_scores(text)

    # Determine the overall label based on compound score thresholds
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label":    label,
        "compound": round(compound, 4),
        "positive": round(scores["pos"], 4),
        "negative": round(scores["neg"], 4),
        "neutral":  round(scores["neu"], 4),
    }
