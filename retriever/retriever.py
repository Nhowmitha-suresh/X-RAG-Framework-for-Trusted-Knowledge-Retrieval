from typing import List, Tuple
import numpy as np


class Retriever:
    def __init__(self, vector_store, documents: List[str]):
        self.vector_store = vector_store
        self.documents = documents

    def retrieve(self, query_embedding: np.ndarray, k: int = 3) -> Tuple[List[str], np.ndarray]:
        """Return top-k documents and their distances for the provided query embedding.

        query_embedding should be shaped (1, dim).
        """
        distances, indices = self.vector_store.search(query_embedding, k)

        # faiss returns (n_queries, k) arrays
        if indices is None or indices.size == 0:
            return [], np.array([])

        idx_row = indices[0]
        dist_row = distances[0]

        docs = []
        for i in idx_row:
            if i < 0 or i >= len(self.documents):
                docs.append("")
            else:
                docs.append(self.documents[int(i)])

        return docs, dist_row
