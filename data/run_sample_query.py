from data.load_index import load_index
from retriever.retriever import Retriever
from retriever.reranker import rerank_by_cosine
from generator.llm import LLM
from generator.answer_generator import generate_answer
from embeddings.embedder import Embedder


def main():
    vs, metadata = load_index("data/index")
    docs = [m["text"] for m in metadata]
    retriever = Retriever(vs, docs)

    query = "What services does Poorna Multi Speciality Hospital provide as a multi speciality hospital?"
    # If TF-IDF vectorizer was saved during ingestion, load it to ensure matching dims
    import os
    vectorizer_path = os.path.join("data", "index", "vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        embedder = Embedder(vectorizer_path=vectorizer_path)
    else:
        embedder = Embedder()

    print("vectorizer_path=", vectorizer_path, "exists=", os.path.exists(vectorizer_path))
    print("embedder.__dict__=", getattr(embedder, '__dict__', None))
    print(f"Embedder using TF-IDF: {getattr(embedder, 'use_tfidf', False)}")
    q_emb = embedder.encode([query])
    import numpy as np
    print("q_emb.shape=", np.asarray(q_emb).shape)
    docs_ret, dists, indices = retriever.retrieve(q_emb, k=5)

    ranked = rerank_by_cosine(query, list(indices), vs)
    ranked_indices = [r[0] for r in ranked]
    contexts = [metadata[i] for i in ranked_indices]

    llm = LLM()
    out = generate_answer(query, contexts, llm=llm)

    print("\n=== QUESTION ===\n", query)
    print("\n=== ANSWER ===\n", out.get("answer"))
    print("\n=== SOURCES ===")
    for c in contexts:
        print(f"[{c.get('source_id')}] {c.get('department')}: {c.get('text')[:200]}...")


if __name__ == '__main__':
    main()
