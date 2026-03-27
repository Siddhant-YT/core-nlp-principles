"""
utils/keywords.py

Extracts important keywords from text using TF-IDF
(Term Frequency - Inverse Document Frequency).

TF-IDF Intuition:
    - TF  (Term Frequency)        = How often a word appears in THIS document
    - IDF (Inverse Document Freq) = How rare the word is across ALL documents
    - TF-IDF = TF x IDF
    - High score = word is frequent here but rare elsewhere = important keyword
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(cleaned_text: str, top_n: int = 10) -> list[tuple[str, float]]:
    """
    Extract top N keywords from preprocessed text using TF-IDF scoring.

    Why TF-IDF over simple word counts?
        Simple counts give high scores to 'the', 'is', 'a' (useless common words).
        TF-IDF penalizes words that are common everywhere and rewards unique, relevant words.

    Args:
        cleaned_text: Preprocessed string (already lemmatized, stopwords removed)
        top_n: How many top keywords to return

    Returns:
        List of (word, score) tuples, sorted by score descending
        Example: [('machine', 0.45), ('learning', 0.38), ('model', 0.30), ...]
    """
    # Need at least a few words to compute TF-IDF meaningfully
    if not cleaned_text or len(cleaned_text.split()) < 2:
        return []

    # TfidfVectorizer converts text into a matrix of TF-IDF scores
    # stop_words='english' adds another layer of stopword filtering
    vectorizer = TfidfVectorizer(stop_words="english")

    # fit_transform: 
    #   - 'fit'      = learn the vocabulary from this text
    #   - 'transform' = convert to TF-IDF score matrix
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])

    # Get the list of all unique words (features) the vectorizer learned
    feature_names = vectorizer.get_feature_names_out()

    # Get the scores as a flat array (we only have 1 document, so index [0])
    scores = tfidf_matrix.toarray()[0]

    # Pair each word with its score, then sort highest first
    word_scores = sorted(
        zip(feature_names, scores),
        key=lambda pair: pair[1],
        reverse=True
    )

    # Return only the top N keywords
    return word_scores[:top_n]
