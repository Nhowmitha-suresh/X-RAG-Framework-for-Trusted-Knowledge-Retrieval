from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts):
        return np.array(self.model.encode(texts))
