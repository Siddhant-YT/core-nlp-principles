"""
utils/preprocessor.py

Handles all text cleaning steps before analysis:
  - Lowercasing
  - Punctuation removal
  - Stopword removal
  - Lemmatization

These are the foundational steps that every NLP pipeline begins with.
"""

import re
import nltk
import spacy
from nltk.corpus import stopwords

# Download NLTK resources if not already present
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# Load spaCy English model (used for lemmatization)
nlp = spacy.load("en_core_web_sm")

# Load English stopwords once so we don't reload on every call
STOP_WORDS = set(stopwords.words("english"))


def lowercase(text: str) -> str:
    """
    Convert all characters to lowercase.
    Reason: 'Apple' and 'apple' should be treated as the same word by the model.
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    Remove all characters that are not letters, digits, or spaces.
    Reason: Punctuation like '!!', '...', '?' adds noise to analysis.
    """
    # [^a-zA-Z0-9\s] means: match anything that is NOT a letter, digit, or whitespace
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_stopwords(tokens: list) -> list:
    """
    Remove common English words that carry little meaning.
    Examples of stopwords: 'the', 'is', 'in', 'a', 'an', 'and', 'of'

    Args:
        tokens: A list of word strings

    Returns:
        A filtered list with stopwords removed
    """
    return [word for word in tokens if word not in STOP_WORDS]


def lemmatize(text: str) -> list:
    """
    Convert each word to its base/dictionary form using spaCy.
    Examples:
        'running' -> 'run'
        'studies' -> 'study'
        'better'  -> 'good'

    Lemmatization is smarter than stemming because it uses vocabulary context.

    Args:
        text: A cleaned string of text

    Returns:
        A list of lemmatized tokens (stopwords and short tokens removed)
    """
    doc = nlp(text)
    tokens = [
        token.lemma_          # Get the base/root form of the word
        for token in doc
        if not token.is_stop  # Skip stopwords
        and not token.is_punct  # Skip punctuation tokens
        and not token.is_space  # Skip whitespace tokens
        and len(token.text) > 2  # Skip very short words (e.g., 'a', 'is')
        and token.is_alpha   # Keep only alphabetic tokens (no numbers)
    ]
    return tokens


def full_preprocess(text: str) -> dict:
    """
    Run the complete preprocessing pipeline and return each step's result.
    Steps:
        1. Lowercase
        2. Remove punctuation
        3. Lemmatize + remove stopwords (both done by spaCy in one pass)

    Args:
        text: Raw input text from user

    Returns:
        dict with keys:
            - 'original': the original text
            - 'lowercased': after lowercasing
            - 'no_punctuation': after removing punctuation
            - 'tokens': final cleaned list of lemmatized tokens
            - 'cleaned_string': tokens joined as a single string (used by TF-IDF)
    """
    original = text.strip()

    # Step 1: Lowercase
    lowercased = lowercase(original)

    # Step 2: Remove punctuation
    no_punct = remove_punctuation(lowercased)

    # Step 3: Lemmatize and remove stopwords in one spaCy pass
    tokens = lemmatize(no_punct)

    # Join tokens into a string (TF-IDF needs a string, not a list)
    cleaned_string = " ".join(tokens)

    return {
        "original": original,
        "lowercased": lowercased,
        "no_punctuation": no_punct,
        "tokens": tokens,
        "cleaned_string": cleaned_string,
    }
