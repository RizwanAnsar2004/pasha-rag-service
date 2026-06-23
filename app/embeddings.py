"""Embeddings via OpenAI's embedding model.

Vectors are L2-normalized so cosine distance in Chroma is consistent.
"""

from __future__ import annotations

import math
from functools import lru_cache

from openai import OpenAI

from .config import get_settings


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key or None)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _embed(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    response = _client().embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [_normalize(list(item.embedding)) for item in response.data]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents for storage."""
    if not texts:
        return []
    return _embed(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single search query."""
    return _embed([text])[0]
