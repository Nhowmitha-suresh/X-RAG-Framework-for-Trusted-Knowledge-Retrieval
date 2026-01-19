from typing import List, Optional

from generator.llm import LLM


PROMPT_TEMPLATE = (
    "You are a helpful, concise medical support assistant. Use the provided context and answer the question. "
    "Cite sources by their `source_id` and `department` when relevant.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:" 
)


def generate_answer(question: str, contexts: List[dict], llm: Optional[LLM] = None) -> dict:
    """Generate an answer given retrieved contexts. If `llm` is None or no API key, fallback to concatenation."""
    context_text = "\n\n".join([f"[{c.get('source_id')}] ({c.get('department')}) {c.get('text')}" for c in contexts])

    if llm is not None:
        system = "You are a concise assistant answering hospital policy and services queries. Provide sources and be factual."
        prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
        out = llm.chat_completion(system, prompt)
        if out is not None:
            return {"answer": out, "used_llm": True, "sources": [c.get("source_id") for c in contexts]}

    # Fallback: combine top context snippets into a short answer
    snippet = " \n\n ".join([c.get("text") for c in contexts[:3]])
    ans = f"Based on retrieved documents: {snippet}"
    return {"answer": ans, "used_llm": False, "sources": [c.get("source_id") for c in contexts]}

