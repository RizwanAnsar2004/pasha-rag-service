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


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


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
