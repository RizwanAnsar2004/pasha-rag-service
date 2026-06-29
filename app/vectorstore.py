"""ChromaDB persistent vector store wrapper."""

from __future__ import annotations

import hashlib
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import get_settings
from .embeddings import embed_documents
from .schemas import Document, SourceChunk


@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(
        path=settings.chroma_path,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
    )


def _collection() -> chromadb.Collection:
    settings = get_settings()
    # Cosine space so distances align with the `max_distance` guardrail.
    return _client().get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _doc_id(doc: Document) -> str:
    if doc.id:
        return doc.id
    return hashlib.sha256(doc.text.encode("utf-8")).hexdigest()[:32]


def ingest(documents: list[Document]) -> tuple[int, int]:
    """Embed and upsert documents. Returns (ingested_count, total_in_collection)."""
    collection = _collection()
    ids = [_doc_id(d) for d in documents]
    texts = [d.text for d in documents]
    # Chroma rejects empty-dict metadata in some versions; ensure a non-empty map.
    metadatas = [d.metadata or {"_source": "ingest"} for d in documents]

    embeddings = embed_documents(texts)
    collection.upsert(
        ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
    )
    return len(ids), collection.count()


def query(question_embedding: list[float], top_k: int) -> list[SourceChunk]:
    """Retrieve the nearest chunks for a query embedding."""
    collection = _collection()
    if collection.count() == 0:
        return []

    n = min(top_k, collection.count())
    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[SourceChunk] = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    for cid, text, meta, dist in zip(ids, docs, metas, dists):
        chunks.append(
            SourceChunk(
                id=cid, text=text, metadata=meta or {}, distance=float(dist)
            )
        )
    return chunks


def count() -> int:
    return _collection().count()


def delete_ids(ids: list[str]) -> int:
    """Remove documents by id. Returns the number requested for deletion."""
    if not ids:
        return 0
    _collection().delete(ids=ids)
    return len(ids)


def get_meta(doc_id: str) -> dict | None:
    """Return the stored metadata for a single id, or None if absent."""
    result = _collection().get(ids=[doc_id], include=["metadatas"])
    metas = result.get("metadatas") or []
    if not metas:
        return None
    return metas[0] or {}


def list_ids(where: dict | None = None) -> list[str]:
    """List stored ids, optionally filtered by a metadata `where` clause."""
    result = _collection().get(where=where, include=[])
    return list(result.get("ids") or [])


def update_meta(doc_id: str, metadata: dict) -> None:
    """Update only the metadata of an existing document (no re-embedding).

    Used when a row's embedded text is unchanged but its metadata moved (e.g.
    a verification toggle) — avoids paying for a fresh embedding."""
    _collection().update(ids=[doc_id], metadatas=[metadata or {"_source": "ingest"}])
