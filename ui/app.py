import streamlit as st
import requests

st.set_page_config(page_title="TrustRAG", layout="centered")
st.title("TrustRAG — Explainable Retrieval-Augmented Generation")

question = st.text_input("Ask a question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        try:
            res = requests.post("http://localhost:8000/ask", json={"question": question}, timeout=15)
            res.raise_for_status()
            data = res.json()

            st.subheader("Answer")
            st.write(data.get("answer", ""))

            st.subheader("Sources")
            sources = data.get("sources", [])
            for s in sources:
                sim = s.get("similarity")
                st.markdown(f"**Source {s.get('index')}:** {s.get('source_text')[:400]}")
                if sim is not None:
                    st.caption(f"Similarity: {sim:.4f} — Distance: {s.get('distance')}")

            st.subheader("Confidence")
            conf = data.get("confidence", 0.0)
            st.progress(min(max(conf, 0.0), 1.0))

        except Exception as e:
            st.error(f"Failed to get answer: {e}")
