from typing import List
from generator.llm import call_llm


def generate_answer(chunks: List[str], query: str) -> str:
    """Generate an answer strictly from the provided `chunks`.

    If the information required to answer `query` is not present in `chunks`,
    the model must reply with the exact string: "Not found in documents".
    """
    # If no chunks, immediately return Not found
    if not chunks:
        return "Not found in documents"

    # Build numbered context so model can refer to sources
    numbered = []
    for i, c in enumerate(chunks, 1):
        numbered.append(f"[SOURCE {i}] {c}")

    system = (
        "You are an assistant that must answer using ONLY the provided sources. "
        "Do NOT hallucinate or use external knowledge. If the answer cannot be found in the sources, "
        "respond exactly with: Not found in documents"
    )

    user = (
        "Context:\n" + "\n---\n".join(numbered) + f"\n\nQuestion: {query}\n\n"
        "Provide a concise answer and cite the source number(s) you used in square brackets."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    return call_llm(messages)
