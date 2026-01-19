import json
from typing import List

from data.load_index import load_index
from retriever.retriever import Retriever
from retriever.reranker import rerank_by_cosine


def load_test_queries(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(test_queries_path: str = "evaluation/test_queries.json", index_dir: str = "data/index"):
    vs, metadata = load_index(index_dir)
    docs = [m["text"] for m in metadata]
    retriever = Retriever(vs, docs)

    tests = load_test_queries(test_queries_path)
    results = []
    for t in tests:
        q = t.get("query")
        expected = t.get("expected", "")

        from embeddings.embedder import Embedder
        import os
        vectorizer_path = os.path.join("data", "index", "vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            embedder = Embedder(vectorizer_path=vectorizer_path)
        else:
            embedder = Embedder()
        q_emb = embedder.encode([q])
        docs_ret, dists, indices = retriever.retrieve(q_emb, k=5)
        ranked = rerank_by_cosine(q, list(indices), vs)
        ranked_indices = [r[0] for r in ranked]
        top_texts = [metadata[i]["text"] for i in ranked_indices]

        hit = any(expected.lower() in t.lower() for t in top_texts)
        results.append({"query": q, "expected": expected, "hit": hit, "top": top_texts})

    # simple summary
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    print(f"{hits}/{total} hits")
    return results


if __name__ == "__main__":
    evaluate()
import time

def measure_latency(func, *args):
    start = time.time()
    result = func(*args)
    return result, round(time.time() - start, 3)
