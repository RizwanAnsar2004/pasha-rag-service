"""FastAPI application exposing the secure RAG service."""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, status

from . import vectorstore
from .config import get_settings
from .rag import answer_question
from .schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(
    title="Secure RAG Service",
    version="0.1.0",
    description=(
        "Ingest JSON documents into ChromaDB and answer questions strictly from "
        "that context, with guardrails against prompt injection and "
        "out-of-context answering."
    ),
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Simple API-key gate. Disabled when SERVICE_API_KEY is unset."""
    settings = get_settings()
    if not settings.auth_enabled:
        return
    if x_api_key != settings.service_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "documents": vectorstore.count()}


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest(req: IngestRequest) -> IngestResponse:
    settings = get_settings()
    ingested, total = vectorstore.ingest(req.documents)
    return IngestResponse(
        ingested=ingested,
        collection=settings.collection_name,
        total_in_collection=total,
    )


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest) -> QueryResponse:
    return answer_question(req.question, req.top_k)
