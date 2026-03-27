"""
utils/similarity.py

Text Similarity using Sentence Embeddings.

What are Embeddings?
    Embeddings convert text into a list of numbers (a vector) in such a way
    that texts with similar meanings get similar vectors.

    Example:
        "I love programming"  → [0.23, -0.45, 0.88, ...]
        "Coding is my passion" → [0.21, -0.43, 0.85, ...]  <- close to above!
        "The sky is blue"     → [-0.55, 0.12, -0.30, ...]  <- far away

    This is fundamentally different from TF-IDF, which treats words as isolated tokens.
    Embeddings capture MEANING, not just word overlap.

Cosine Similarity:
    We measure how "close" two vectors are using cosine similarity.
    Value ranges from 0.0 (totally different) to 1.0 (identical meaning).

    Think of it as: how much do two arrows point in the same direction?

Why this matters :
    - RAG systems use this exact approach to find relevant documents
    - Semantic search is built on embeddings + cosine similarity
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st


# Cache the model in Streamlit's session — it's ~90MB, we only want to load once
@st.cache_resource
def load_embedding_model():
    """
    Load the sentence transformer model.
    'all-MiniLM-L6-v2' is a small, fast model with good quality.
    It produces 384-dimensional embeddings for any input text.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Convert a single text string into a numerical vector (embedding).

    Args:
        text:  Any input string
        model: Loaded SentenceTransformer model

    Returns:
        numpy array of shape (384,) — a list of 384 numbers representing the text
    """
    # model.encode returns an array; [0] gets the first (only) result
    return model.encode([text])[0]


def compute_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """
    Compute the semantic similarity between two texts.

    Args:
        text1, text2: Any two strings to compare
        model: Loaded SentenceTransformer model

    Returns:
        float between 0.0 and 1.0
        - 1.0 = identical meaning
        - 0.5 = moderately related
        - 0.0 = completely unrelated
    """
    # Encode both texts into vectors
    embeddings = model.encode([text1, text2])

    # cosine_similarity expects 2D arrays — reshape to (1, 384)
    sim = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )

    # sim is a 2D array [[value]], so we extract the scalar
    return round(float(sim[0][0]), 4)


def rank_by_similarity(query: str, candidates: list[str], model: SentenceTransformer) -> list[dict]:
    """
    Rank a list of candidate texts by their similarity to a query.
    This is the core logic of semantic search (and RAG retrieval).

    Args:
        query:      The input text to compare against
        candidates: List of texts to rank
        model:      Loaded SentenceTransformer model

    Returns:
        List of dicts sorted by similarity score (highest first):
        [
            {"text": "...", "score": 0.87},
            {"text": "...", "score": 0.52},
            ...
        ]
    """
    if not candidates:
        return []

    # Encode query + all candidates together (more efficient than one-by-one)
    all_texts = [query] + candidates
    all_embeddings = model.encode(all_texts)

    # First embedding is the query; rest are candidates
    query_emb = all_embeddings[0].reshape(1, -1)
    candidate_embs = all_embeddings[1:]

    # Compute similarity of query against every candidate at once
    similarities = cosine_similarity(query_emb, candidate_embs)[0]

    # Build result list with score attached to each candidate
    results = [
        {"text": text, "score": round(float(score), 4)}
        for text, score in zip(candidates, similarities)
    ]

    # Sort by score, highest first
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
