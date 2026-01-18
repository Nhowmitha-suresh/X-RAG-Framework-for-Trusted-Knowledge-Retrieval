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
