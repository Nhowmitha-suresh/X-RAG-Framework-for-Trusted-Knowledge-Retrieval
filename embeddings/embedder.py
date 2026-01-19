from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    """Embedder wrapper.

    - If `sentence_transformers` is available and no `vectorizer_path` is provided, uses a SentenceTransformer model.
    - Otherwise uses a TF-IDF vectorizer. When used with TF-IDF, the vectorizer can be saved/loaded
      to ensure consistent dimensions between indexing and querying.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vectorizer_path: str = None) -> None:
        self.model_name = model_name
        self.vectorizer_path = vectorizer_path

        if vectorizer_path is not None:
            # force TF-IDF mode and load vectorizer
            import pickle

            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self.use_tfidf = True
            self._fitted = True
        else:
            if _HAS_ST:
                self.model = SentenceTransformer(model_name)
                self.use_tfidf = False
            else:
                self.vectorizer = TfidfVectorizer(max_features=768)
                self.use_tfidf = True
                self._fitted = False

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into numpy array. If TF-IDF fallback is used, the vectorizer
        will be fit on the input texts (suitable for single-batch ingestion) or used
        to transform when loaded from disk.
        """
        if self.use_tfidf:
            if not self._fitted:
                X = self.vectorizer.fit_transform(texts)
                self._fitted = True
            else:
                X = self.vectorizer.transform(texts)
            arr = X.toarray().astype(np.float32)
            return arr

        arr = self.model.encode(texts, show_progress_bar=False)
        embs = np.asarray(arr, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return embs

    def save_vectorizer(self, path: str):
        """Save the TF-IDF vectorizer to `path`. Only valid if using TF-IDF."""
        if not self.use_tfidf:
            raise RuntimeError("Vectorizer save is only available when using TF-IDF fallback")
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
