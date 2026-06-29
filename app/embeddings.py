"""Embeddings via OpenAI's embedding model.

Vectors are L2-normalized so cosine distance in Chroma is consistent. Inputs are
chunked to stay under OpenAI's per-request limits (300k tokens / 2048 inputs,
~8191 tokens per individual input), with a recursive split as a safety net if a
token estimate is off.
"""

from __future__ import annotations

import math
from functools import lru_cache

from openai import OpenAI, BadRequestError

from .config import get_settings


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key or None)


# OpenAI embedding request limits for text-embedding-3-*: 300k tokens and 2048
# inputs per request, ~8191 tokens per single input. We batch well under these.
# Tokens are estimated from characters with a deliberately LOW chars/token ratio
# so we over-count and under-fill rather than overflow.
_TOKEN_BUDGET_PER_REQUEST = 200_000
_MAX_INPUTS_PER_REQUEST = 1000
_CHARS_PER_TOKEN = 3                 # conservative (English is ~4) → safe margin
_MAX_CHARS_PER_INPUT = 20_000        # ≈ under the per-input 8191-token cap


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _truncate(text: str) -> str:
    """Cap a single input so it can't exceed the model's per-item token limit."""
    return text[:_MAX_CHARS_PER_INPUT]


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN + 1


def _batches(texts: list[str]):
    """Yield batches that stay under both the token budget and the input cap."""
    batch: list[str] = []
    tokens = 0
    for t in texts:
        tt = _estimate_tokens(t)
        if batch and (
            len(batch) >= _MAX_INPUTS_PER_REQUEST
            or tokens + tt > _TOKEN_BUDGET_PER_REQUEST
        ):
            yield batch
            batch, tokens = [], 0
        batch.append(t)
        tokens += tt
    if batch:
        yield batch


def _embed_request(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    response = _client().embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [_normalize(list(item.embedding)) for item in response.data]


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed one pre-sized batch, splitting in half and retrying if the request
    still overflows the token limit (safety net for an off token estimate)."""
    try:
        return _embed_request(texts)
    except BadRequestError as e:
        overflow = (
            getattr(e, "code", "") == "max_tokens_per_request"
            or "max_tokens_per_request" in str(e)
        )
        if not overflow or len(texts) <= 1:
            raise
        mid = len(texts) // 2
        return _embed_batch(texts[:mid]) + _embed_batch(texts[mid:])


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents for storage, chunked under OpenAI's limits."""
    if not texts:
        return []
    out: list[list[float]] = []
    for batch in _batches([_truncate(t) for t in texts]):
        out.extend(_embed_batch(batch))
    return out


def embed_query(text: str) -> list[float]:
    """Embed a single search query."""
    return _embed_request([_truncate(text)])[0]
