import streamlit as st

from data.load_index import load_index
from retriever.retriever import Retriever
from retriever.reranker import rerank_by_cosine
from generator.llm import LLM
from generator.answer_generator import generate_answer
import streamlit as st
import subprocess
import os
from typing import Optional

from data.load_index import load_index
from retriever.retriever import Retriever
from retriever.reranker import rerank_by_cosine
from generator.llm import LLM
from generator.answer_generator import generate_answer


@st.cache(allow_output_mutation=True)
def load_resources(index_dir: str = "data/index"):
    vs, metadata = load_index(index_dir)
    docs = [m["text"] for m in metadata]
    return vs, metadata, docs


def run_ingest(csv_path: str, out_dir: str = "data/index") -> str:
    """Run the ingestion script in a subprocess and return its output."""
    cmd = ["python", "data/ingest_csv.py", csv_path, "--out_dir", out_dir]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return proc.stdout + proc.stderr
    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr


def main():
    st.set_page_config(page_title="Hospital RAG", layout="wide")
    st.title("Hospital RAG — Demo UI")

    # Sidebar: ingestion and settings
    st.sidebar.header("Ingest / Settings")
    uploaded = st.sidebar.file_uploader("Upload CSV to ingest (rows -> chunks)", type=["csv"])
    reindex_path = st.sidebar.text_input("Or provide CSV path to ingest", "")
    model_provider = st.sidebar.selectbox("LLM provider", ["google", "openai", "none"])
    top_k = st.sidebar.slider("Default top-k", 1, 10, 3)

    if st.sidebar.button("Ingest and (re)build index"):
        csv_to_use = None
        if uploaded is not None:
            tmp = os.path.join("data", "uploaded.csv")
            with open(tmp, "wb") as f:
                f.write(uploaded.getbuffer())
            csv_to_use = tmp
        elif reindex_path:
            csv_to_use = reindex_path

        if not csv_to_use:
            st.sidebar.error("Provide a CSV by upload or path to ingest")
        else:
            with st.sidebar.spinner("Running ingestion..."):
                out = run_ingest(csv_to_use, out_dir="data/index")
            st.sidebar.success("Ingestion finished")
            st.sidebar.text_area("Ingest log", out, height=200)

    # Main UI: load index
    try:
        vs, metadata, docs = load_resources()
    except Exception as e:
        st.error(f"Could not load index: {e}")
        return

    col1, col2 = st.columns([2, 3])

    with col1:
        q = st.text_area("Ask a question about the hospital:")
        k = st.number_input("Top-k retrieval", min_value=1, max_value=20, value=top_k)
        use_llm = st.checkbox("Use LLM for final answer (may cost)", value=(model_provider != "none"))
        if st.button("Ask") and q:
            retriever = Retriever(vs, docs)
            vectorizer_path = os.path.join("data", "index", "vectorizer.pkl")
            if os.path.exists(vectorizer_path):
                from embeddings.embedder import Embedder

                embedder = Embedder(vectorizer_path=vectorizer_path)
            else:
                from embeddings.embedder import Embedder

                embedder = Embedder()

            q_emb = embedder.encode([q])
            docs_ret, dists, indices = retriever.retrieve(q_emb, k=k)

            ranked = rerank_by_cosine(q, list(indices), vs)
            ranked_indices = [r[0] for r in ranked]
            contexts = [metadata[i] for i in ranked_indices]

            llm = None
            if use_llm:
                # Initialize LLM with env keys; the LLM wrapper auto-detects provider keys
                llm = LLM()

            out = generate_answer(q, contexts, llm=llm)

            st.subheader("Answer")
            st.write(out.get("answer") if isinstance(out, dict) else out)
            st.caption(f"Used LLM: {out.get('used_llm')}")

            st.subheader("Retrieved Contexts")
            for i, c in enumerate(contexts, 1):
                with st.expander(f"[{c.get('source_id')}] {c.get('department')} (chunk {c.get('chunk_index')})"):
                    st.write(c.get("text"))
                    st.json({"source_id": c.get("source_id"), "department": c.get("department"), "distance": None})

    with col2:
        st.subheader("Debug / Raw Outputs")
        if st.button("Show raw metadata (first 50)"):
            st.write(metadata[:50])


if __name__ == "__main__":
    main()
