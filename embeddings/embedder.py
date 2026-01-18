from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wrapper around SentenceTransformer to produce numpy embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts and return a 2D numpy array (n_texts, dim)."""
        arr = self.model.encode(texts, show_progress_bar=False)
        embs = np.asarray(arr, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return embs
