import faiss
import numpy as np
from typing import Tuple


class VectorStore:
    """Simple FAISS-backed vector store using L2 distance.

    Embeddings must be a 2D numpy array of dtype float32.
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings must be a non-empty 2D numpy array")

        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D")

        self.dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) for the top-k nearest neighbors.

        `query_embedding` should be shaped (1, dim).
        """
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        if q.shape[1] != self.dim:
            raise ValueError(f"Query embedding dimension {q.shape[1]} != index dim {self.dim}")

        distances, indices = self.index.search(q, k)
        return distances, indices
