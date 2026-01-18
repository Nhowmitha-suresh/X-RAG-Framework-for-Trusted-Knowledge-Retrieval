import os
from typing import List, Tuple, Dict, Any
import numpy as np

from utils.text_cleaner import clean_text
from utils.chunker import chunk_text
from utils.logger import get_logger
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore
from retriever.retriever import Retriever
from generator.answer_generator import generate_answer
from explainability.explanation import explain

logger = get_logger(__name__)


def build_rag(data_path: str = "data/knowledge.txt") -> Tuple[Embedder, Retriever]:
    """Build the embedding model and retriever from a local text file.

    Looks for `data/knowledge.txt` or `data/documents.txt`.
    """
    if not os.path.exists(data_path):
        alt = "data/documents.txt"
        if os.path.exists(alt):
            data_path = alt
        else:
            raise FileNotFoundError("No data file found at data/knowledge.txt or data/documents.txt")

    with open(data_path, "r", encoding="utf-8") as f:
        raw = f.read()

    text = clean_text(raw)
    chunks = chunk_text(text)

    embedder = Embedder()
    embeddings = embedder.encode(chunks)

    # Ensure embeddings are 2D numpy array
    embeddings = np.asarray(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    vector_store = VectorStore(embeddings)
    retriever = Retriever(vector_store, chunks)
    logger.info("Built RAG components: %d chunks, embedding dim=%d", len(chunks), embeddings.shape[1])
    return embedder, retriever


EMBEDDER = None
RETRIEVER = None


def query_rag(question: str, k: int = 3) -> Dict[str, Any]:
    """Query the RAG system and return a structured response.

    Returns a dict with keys: `answer`, `sources`, `similarities`, `confidence`.
    """
    if EMBEDDER is None or RETRIEVER is None:
        raise RuntimeError("RAG system is not initialized")

    if not question or not question.strip():
        return {"answer": "", "sources": [], "similarities": [], "confidence": 0.0}

    global EMBEDDER, RETRIEVER

    # Lazy initialize heavy components on first query to allow the API to start without models
    if EMBEDDER is None or RETRIEVER is None:
        EMBEDDER, RETRIEVER = build_rag()

    q_emb = EMBEDDER.encode([question])
    q_emb = np.asarray(q_emb)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)

    docs, distances = RETRIEVER.retrieve(q_emb, k=k)

    # Generate answer using only retrieved documents
    answer = generate_answer(docs, question)
    explanation = explain(docs, distances)

    return {
        "answer": answer,
        "sources": explanation.get("sources", []),
        "similarities": explanation.get("similarities", []),
        "confidence": explanation.get("confidence", 0.0),
    }


if __name__ == "__main__":
    # simple CLI for quick local testing
    while True:
        q = input("\nAsk (exit to quit): ")
        if q.lower() == "exit":
            break

        try:
            resp = query_rag(q, k=3)
            print("\nANSWER:\n", resp["answer"])
            print("\nSOURCES:\n")
            for s in resp["sources"]:
                print(f"- {s.get('source_text')[:200]}")
            print("\nCONFIDENCE:\n", resp["confidence"])
        except Exception as e:
            logger.exception("Error answering question: %s", e)
