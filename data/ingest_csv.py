import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd

from embeddings.embedder import Embedder
from utils.chunker import chunk_text
from utils.text_cleaner import clean_text


def load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_documents(df: pd.DataFrame, chunk_size: int = 80) -> List[dict]:
    docs = []
    for _, row in df.iterrows():
        src_id = int(row.get("id", -1))
        hospital = str(row.get("hospital", ""))
        department = str(row.get("department", ""))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))

        text = f"{question}\n\n{answer}"
        text = clean_text(text)
        chunks = chunk_text(text, chunk_size=chunk_size)

        for ci, chunk in enumerate(chunks):
            docs.append({
                "source_id": src_id,
                "hospital": hospital,
                "department": department,
                "chunk_index": ci,
                "text": chunk,
            })

    return docs


def embed_documents(docs: List[dict], embedder: Embedder, batch_size: int = 64):
    texts = [d["text"] for d in docs]
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        e = embedder.encode(batch)
        embs.append(e)
    if embs:
        embs_arr = np.vstack(embs)
    else:
        embs_arr = np.zeros((0, 0), dtype=np.float32)
    return embs_arr


def save_index(embeddings: np.ndarray, docs: List[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    emb_path = os.path.join(out_dir, "embeddings.npy")
    meta_path = os.path.join(out_dir, "metadata.jsonl")
    np.save(emb_path, embeddings)

    with open(meta_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Saved embeddings ({embeddings.shape}) to {emb_path}")
    print(f"Saved metadata ({len(docs)}) to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest CSV and build embeddings + metadata")
    parser.add_argument("csv_path", help="Path to the CSV dataset")
    parser.add_argument("--out_dir", default="data/index", help="Output directory for embeddings and metadata")
    parser.add_argument("--chunk_size", type=int, default=80, help="Chunk size in words")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    args = parser.parse_args()

    df = load_csv(args.csv_path)
    docs = build_documents(df, chunk_size=args.chunk_size)
    if not docs:
        print("No documents found in CSV")
        return

    embedder = Embedder(args.model)
    embeddings = embed_documents(docs, embedder)
    save_index(embeddings, docs, args.out_dir)

    # If TF-IDF fallback was used, save the vectorizer so queries use the same mapping
    try:
        if getattr(embedder, "use_tfidf", False):
            vec_path = os.path.join(args.out_dir, "vectorizer.pkl")
            embedder.save_vectorizer(vec_path)
            print(f"Saved TF-IDF vectorizer to {vec_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
