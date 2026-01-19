from typing import List, Tuple
import os
import numpy as np

from embeddings.embedder import Embedder


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def rerank_by_cosine(query: str, candidate_indices: List[int], vector_store, model_name: str = "all-MiniLM-L6-v2") -> List[Tuple[int, float]]:
    """Rerank candidate document indices by cosine similarity with the query.

    Returns list of (index, score) sorted descending.
    Attempts to load a saved TF-IDF vectorizer at `data/index/vectorizer.pkl` so
    query embeddings use the same dimensionality as the index.
    """
    if not candidate_indices:
        return []

    # Prefer TF-IDF vectorizer saved during ingestion if present
    vectorizer_path = os.path.join("data", "index", "vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        embedder = Embedder(vectorizer_path=vectorizer_path)
    else:
        embedder = Embedder(model_name)
    q_emb = embedder.encode([query])[0]

    scores = []
    for idx in candidate_indices:
        doc_emb = vector_store.embeddings[int(idx)]
        score = cosine_sim(q_emb, doc_emb)
        scores.append((int(idx), score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
def rerank(chunks, scores):
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1])
    return [r[0] for r in ranked]
