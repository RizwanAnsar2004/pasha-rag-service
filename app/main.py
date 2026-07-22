"""FastAPI application exposing the secure RAG service."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import databank, vectorstore
from .config import get_settings
from .rag import answer_question
from .schemas import (
    DatabankEvent,
    DatabankSyncResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)


def _client_key(request: Request) -> str:
    """Identify the caller for rate limiting.

    nginx sits in front of this service (deploy/nginx.conf), so `request.client`
    is always 127.0.0.1 — using it directly would put every caller in the world
    into one shared bucket. X-Forwarded-For's first entry is the original client.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_key)

app = FastAPI(
    title="Secure RAG Service",
    version="0.1.0",
    description=(
        "Ingest JSON documents into ChromaDB and answer questions strictly from "
        "that context, with guardrails against prompt injection and "
        "out-of-context answering."
    ),
)


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
@limiter.limit("5/hour")
def query(request: Request, req: QueryRequest) -> QueryResponse:
    return answer_question(req.question, req.top_k)


@app.post("/databank/event", dependencies=[Depends(require_api_key)])
def databank_event(event: DatabankEvent) -> dict[str, object]:
    """Databank change sink — re-syncs the affected row from the source of truth
    (the payload is never trusted for content).

    Returns 200 on success and on an unactionable payload (no id — retrying
    won't help). On a *genuine* sync failure (e.g. Supabase/OpenAI hiccup) it
    raises 500 so the caller's bounded retry kicks in and the row still lands —
    keeping ingestion reliable on every approve/edit."""
    row_id = event.row_id()
    if not row_id:
        return {"ok": False, "reason": "no row id in payload"}
    try:
        action = databank.sync_row(row_id)
        return {"ok": True, "id": row_id, "action": action}
    except Exception as exc:  # noqa: BLE001
        logger.exception("databank event sync failed for %s", row_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"sync failed for {row_id}: {exc}",
        ) from exc


@app.post(
    "/databank/sync",
    response_model=DatabankSyncResponse,
    dependencies=[Depends(require_api_key)],
)
def databank_sync() -> DatabankSyncResponse:
    """Full backfill + reconcile of the databank table into the vector store."""
    result = databank.sync_all()
    return DatabankSyncResponse(**result)
