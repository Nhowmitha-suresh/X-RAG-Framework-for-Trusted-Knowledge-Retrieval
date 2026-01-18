import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import clean_text, chunk_text
from embeddings import Embedder
from retriever import Retriever
from generator import generate_answer
from explainability import explain


def build_trustrag():
    with open("data/knowledge.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_text(text)
    chunks = chunk_text(text)

    embedder = Embedder()
    embeddings = embedder.encode(chunks)

    retriever = Retriever(embeddings, chunks)
    return embedder, retriever


EMBEDDER, RETRIEVER = build_trustrag()


def query_trustrag(question):
    q_emb = EMBEDDER.encode([question])
    docs, distances = RETRIEVER.retrieve(q_emb)

    answer = generate_answer(" ".join(docs), question)
    explanation = explain(docs, distances)

    return answer, explanation


if __name__ == "__main__":
    while True:
        q = input("\nAsk (exit to quit): ")
        if q.lower() == "exit":
            break

        ans, exp = query_trustrag(q)
        print("\nANSWER:\n", ans)
        print("\nEXPLANATION:\n", exp)
