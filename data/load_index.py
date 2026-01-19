import json
import os
from typing import List, Tuple

import numpy as np

from embeddings.vector_store import VectorStore
from embeddings.embedder import Embedder


def load_metadata(meta_path: str) -> List[dict]:
    docs = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def load_index(index_dir: str = "data/index") -> Tuple[VectorStore, List[dict]]:
    emb_path = os.path.join(index_dir, "embeddings.npy")
    meta_path = os.path.join(index_dir, "metadata.jsonl")
    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Index files not found in {index_dir}")

    embeddings = np.load(emb_path)
    metadata = load_metadata(meta_path)
    vs = VectorStore(embeddings)
    return vs, metadata


def query_index(query: str, vs: VectorStore, metadata: List[dict], model_name: str = "all-MiniLM-L6-v2", k: int = 3):
    embedder = Embedder(model_name)
    q_emb = embedder.encode([query])
    distances, indices = vs.search(q_emb, k=k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx].copy()
        m["distance"] = float(dist)
        results.append(m)
    return results
