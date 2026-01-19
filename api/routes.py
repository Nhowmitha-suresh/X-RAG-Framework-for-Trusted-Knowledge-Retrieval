from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from data.load_index import load_index
from retriever.retriever import Retriever
from retriever.reranker import rerank_by_cosine
from generator.llm import LLM
from generator.answer_generator import generate_answer


router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 3


@router.post("/query")
def query(req: QueryRequest):
    try:
        vs, metadata = load_index("data/index")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    retriever = Retriever(vs, [m["text"] for m in metadata])
    from embeddings.embedder import Embedder

    import os
    vectorizer_path = os.path.join("data", "index", "vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        embedder = Embedder(vectorizer_path=vectorizer_path)
    else:
        embedder = Embedder()
    q_emb = embedder.encode([req.query])
    docs, distances, indices = retriever.retrieve(q_emb, k=req.k)

    # Rerank
    ranked = rerank_by_cosine(req.query, list(indices), vs)
    ranked_indices = [r[0] for r in ranked]

    contexts = [metadata[i] for i in ranked_indices]

    llm = LLM()
    result = generate_answer(req.query, contexts, llm=llm)
    result.update({"retrieved": contexts})
    return result
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any

from main import query_rag

router = APIRouter()


class Query(BaseModel):
    question: str


@router.post("/ask")
def ask_rag(query: Query) -> Any:
    try:
        return query_rag(query.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
