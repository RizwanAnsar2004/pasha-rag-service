"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single record to be embedded and stored."""

    id: str | None = Field(
        default=None,
        description="Optional stable id. Generated from content if omitted.",
    )
    text: str = Field(..., min_length=1, description="The text to embed.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata stored alongside."
    )


class IngestRequest(BaseModel):
    """Payload for ingesting JSON data into the vector store."""

    documents: list[Document] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    ingested: int
    collection: str
    total_in_collection: int


class DatabankEvent(BaseModel):
    """Supabase Database Webhook payload for the `databank` table.

    Supabase posts `{ type, table, schema, record, old_record }`. We only need
    the row id (and the event type to distinguish deletes); the row is re-fetched
    from Supabase as the source of truth rather than trusting `record`."""

    type: str = Field(..., description="INSERT | UPDATE | DELETE")
    table: str | None = None
    record: dict[str, Any] | None = None
    old_record: dict[str, Any] | None = None

    def row_id(self) -> str | None:
        """The affected row id, from the new record or (on delete) the old one."""
        for src in (self.record, self.old_record):
            if src and src.get("id"):
                return str(src["id"])
        return None


class DatabankSyncResponse(BaseModel):
    upserted: int
    deleted: int
    skipped: int = Field(
        default=0, description="Rows whose embedded text was unchanged."
    )
    total_in_collection: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    # Deliberately unconstrained: both are opaque client metadata, and a
    # malformed one must not cost the caller their answer. Length is enforced
    # where each is actually used (see MAX_SESSION_ID_LEN in app.main), which
    # degrades to the IP bucket instead of returning 422.
    session_id: str | None = Field(
        default=None,
        description=(
            "Caller's conversation id, used as the rate-limit bucket. Sent in "
            "the body rather than a header so proxies can't strip it. Falls "
            "back to the client IP when absent or unusable."
        ),
    )
    request_id: str | None = Field(
        default=None,
        description="Per-query id for correlating logs across services.",
    )


class SourceChunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class QueryResponse(BaseModel):
    answer: str
    grounded: bool = Field(
        description="True when the answer was drawn from retrieved context."
    )
    refused: bool = Field(
        default=False,
        description="True when the request was blocked or could not be grounded.",
    )
    reason: str | None = Field(
        default=None, description="Why the request was refused, if applicable."
    )
    sources: list[SourceChunk] = Field(default_factory=list)


class VoiceQueryResponse(QueryResponse):
    """A /query/voice answer, plus what the service heard — echoed back so the
    UI can show the user their transcribed question."""

    transcription: str = Field(
        default="", description="The transcript the answer was generated from."
    )
