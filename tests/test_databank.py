"""Tests for the databank → vector store sync.

`build_document` / `build_text` are pure and tested directly. The sync paths
exercise the real ChromaDB wiring in a temp dir, stubbing only the Supabase
fetch and the OpenAI embedding call (same pattern as test_pipeline.py).
"""

from __future__ import annotations

import math

import pytest


def _fake_vec(text: str) -> list[float]:
    dim = 64
    vec = [0.0] * dim
    for tok in text.lower().split():
        vec[hash(tok) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


SAMPLE_ROW = {
    "id": "11111111-1111-1111-1111-111111111111",
    "startup_name": "Acme Health",
    "company_name": "Acme Health Pvt Ltd",
    "tagline": "<p>Telemedicine for <strong>everyone</strong></p>",
    "startup_idea": "Connecting rural patients to doctors.",
    "business_model": "B2C subscription.",
    "social_impact": "Healthcare access.",
    "awards": None,
    "certifications": "",
    "sdgs": "SDG 3",
    "primary_industry": "HealthTech",
    "city": "Karachi",
    "hq_country": "Pakistan",
    "pasha_verified": True,
    "women_led": False,
    "website": "https://acme.health",
    "source": "submission",
    "updated_at": "2026-06-29T00:00:00Z",
    "key_persons": [
        {"name": "Sara Khan", "role": "CEO"},
        {"name": "Ali Raza", "role": "CTO"},
    ],
    "answers": {
        "problem_statement": "Rural patients lack specialists.",
        "usp": "24/7 video consults.",
        "legal_company_name": "Acme Health Pvt Ltd",
        "currently_hiring": True,
        "operating_markets": ["Karachi", "Lahore"],
        "secret_internal_note": "keep me out",
    },
}


# --------------------------- pure builders ---------------------------------- #
def test_build_text_strips_html_and_includes_public_fields():
    from app import databank

    text = databank.build_text(SAMPLE_ROW)
    assert "Telemedicine for everyone" in text  # HTML stripped
    assert "<strong>" not in text and "<p>" not in text
    assert "Connecting rural patients" in text
    assert "Sara Khan — CEO" in text
    assert "Rural patients lack specialists" in text  # answer field


def test_build_text_embeds_all_answer_fields():
    from app import databank

    text = databank.build_text(SAMPLE_ROW)
    # Every answer-bag field is embedded so any admin edit is searchable.
    assert "Acme Health Pvt Ltd" in text          # legal_company_name (scalar)
    assert "24/7 video consults" in text           # usp
    assert "Yes" in text                            # currently_hiring (bool → Yes)
    assert "Karachi, Lahore" in text               # operating_markets (list join)


def test_excluded_answer_keys_are_dropped(monkeypatch):
    from app import databank

    monkeypatch.setattr(databank, "EXCLUDED_ANSWER_KEYS", {"secret_internal_note"})
    text = databank.build_text(SAMPLE_ROW)
    assert "keep me out" not in text               # excluded key omitted
    assert "Acme Health Pvt Ltd" in text           # others still present


def test_build_metadata_is_non_sensitive_and_scalar():
    from app import databank

    doc = databank.build_document(SAMPLE_ROW)
    meta = doc.metadata
    assert doc.id == "startup:11111111-1111-1111-1111-111111111111"
    assert meta["type"] == "startup"
    assert meta["row_id"] == SAMPLE_ROW["id"]
    assert meta["primary_industry"] == "HealthTech"
    assert meta["pasha_verified"] is True
    assert "content_hash" in meta
    # None / empty columns are dropped (Chroma rejects None metadata values).
    assert "awards" not in meta
    assert "certifications" not in meta
    # All metadata values are Chroma-safe scalars.
    for v in meta.values():
        assert isinstance(v, (str, int, float, bool))


def test_build_document_handles_sparse_row():
    from app import databank

    doc = databank.build_document({"id": "abc", "startup_name": "Lonely Co"})
    assert doc.text  # never empty
    assert "Lonely Co" in doc.text


# --------------------------- sync wiring ------------------------------------ #
@pytest.fixture
def synced(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("COLLECTION_NAME", "test_databank")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-role-key")

    from app.config import get_settings

    get_settings.cache_clear()

    from app import databank, vectorstore

    vectorstore._client.cache_clear()
    monkeypatch.setattr(
        vectorstore, "embed_documents", lambda texts: [_fake_vec(t) for t in texts]
    )

    yield databank, vectorstore, monkeypatch

    get_settings.cache_clear()
    vectorstore._client.cache_clear()


def _startup_ids(vectorstore):
    return vectorstore.list_ids(where={"type": "startup"})


def test_sync_row_upsert_then_skip(synced):
    databank, vectorstore, monkeypatch = synced
    monkeypatch.setattr(databank, "_fetch_rows", lambda ids=None: [SAMPLE_ROW])

    assert databank.sync_row(SAMPLE_ROW["id"]) == "upsert"
    assert len(_startup_ids(vectorstore)) == 1
    # Same content again → embedded text unchanged → skip (no re-embed).
    assert databank.sync_row(SAMPLE_ROW["id"]) == "skip"
    assert len(_startup_ids(vectorstore)) == 1


def test_sync_row_metadata_only_change_skips_embedding(synced):
    databank, vectorstore, monkeypatch = synced
    monkeypatch.setattr(databank, "_fetch_rows", lambda ids=None: [SAMPLE_ROW])
    databank.sync_row(SAMPLE_ROW["id"])

    # Flip a metadata-only column (no change to embedded text) → skip + updated.
    changed = {**SAMPLE_ROW, "pasha_verified": False}
    monkeypatch.setattr(databank, "_fetch_rows", lambda ids=None: [changed])

    # Embedding must NOT be called on a skip.
    def _boom(texts):
        raise AssertionError("embedding should be skipped")

    monkeypatch.setattr(vectorstore, "embed_documents", _boom)
    assert databank.sync_row(SAMPLE_ROW["id"]) == "skip"
    meta = vectorstore.get_meta(databank.chroma_id(SAMPLE_ROW["id"]))
    assert meta["pasha_verified"] is False


def test_sync_row_missing_deletes_vector(synced):
    databank, vectorstore, monkeypatch = synced
    monkeypatch.setattr(databank, "_fetch_rows", lambda ids=None: [SAMPLE_ROW])
    databank.sync_row(SAMPLE_ROW["id"])
    assert len(_startup_ids(vectorstore)) == 1

    # Row now gone from Supabase → event re-sync removes the vector, and the
    # now-empty databank leaves no summary docs behind either.
    monkeypatch.setattr(databank, "_fetch_rows", lambda ids=None: [])
    assert databank.sync_row(SAMPLE_ROW["id"]) == "delete"
    assert vectorstore.count() == 0


def test_sync_all_reconciles_orphans(synced):
    databank, vectorstore, monkeypatch = synced

    other = {**SAMPLE_ROW, "id": "22222222-2222-2222-2222-222222222222",
             "startup_name": "Beta Co"}
    monkeypatch.setattr(databank, "_fetch_rows", lambda ids=None: [SAMPLE_ROW, other])
    res = databank.sync_all()
    assert res["upserted"] == 2
    assert len(_startup_ids(vectorstore)) == 2

    # Second startup disappears → next full sync prunes its orphan vector.
    monkeypatch.setattr(databank, "_fetch_rows", lambda ids=None: [SAMPLE_ROW])
    res = databank.sync_all()
    assert res["deleted"] == 1
    assert len(_startup_ids(vectorstore)) == 1


# --------------------------- summary documents ------------------------------ #
def test_build_summary_documents_counts_and_hygiene():
    from app import databank

    metas = [
        {"primary_industry": "HealthTech", "city": "Karachi"},
        {"primary_industry": "HealthTech", "city": "Lahore"},
        {"primary_industry": "Fintech"},
        # Placeholder / junk values must not become categories.
        {"primary_industry": "Other"},
        {"primary_industry": "41509457-8b90-4af5-949b-b620866ea5d4"},
    ]
    docs = {d.id: d for d in databank.build_summary_documents(metas)}

    cat = docs["startup-summary:categories"]
    assert cat.metadata["type"] == "startup_summary"
    assert "2 distinct startup categories" in cat.text
    assert "HealthTech: 2" in cat.text and "Fintech: 1" in cat.text
    assert "Other" not in cat.text and "41509457" not in cat.text

    assert "2 distinct startup cities" in docs["startup-summary:cities"].text
    # No row carries a product stage or NIC → those facet docs are omitted.
    assert "startup-summary:product-stages" not in docs
    assert "startup-summary:incubation-centers" not in docs
    assert "contains 5 startups in total" in docs["startup-summary:overview"].text

    assert databank.build_summary_documents([]) == []


def test_sync_all_builds_and_updates_summaries(synced):
    databank, vectorstore, monkeypatch = synced

    other = {**SAMPLE_ROW, "id": "22222222-2222-2222-2222-222222222222",
             "startup_name": "Beta Co", "primary_industry": "Fintech"}
    monkeypatch.setattr(databank, "_fetch_rows", lambda ids=None: [SAMPLE_ROW, other])
    databank.sync_all()

    def summary_text(slug):
        result = vectorstore._collection().get(
            ids=[f"startup-summary:{slug}"], include=["documents"]
        )
        docs = result.get("documents") or []
        return docs[0] if docs else None

    text = summary_text("categories")
    assert "2 distinct startup categories" in text
    assert "HealthTech: 1" in text and "Fintech: 1" in text

    # A category change flows into the summary on the next sync.
    changed = {**other, "primary_industry": "HealthTech"}
    monkeypatch.setattr(databank, "_fetch_rows", lambda ids=None: [SAMPLE_ROW, changed])
    databank.sync_all()
    text = summary_text("categories")
    assert "1 distinct startup categories" in text
    assert "HealthTech: 2" in text and "Fintech" not in text


def test_sync_summaries_skips_unchanged(synced):
    databank, vectorstore, monkeypatch = synced
    monkeypatch.setattr(databank, "_fetch_rows", lambda ids=None: [SAMPLE_ROW])
    databank.sync_all()

    # Corpus unchanged → every summary hash matches → no re-embedding.
    def _boom(texts):
        raise AssertionError("embedding should be skipped")

    monkeypatch.setattr(vectorstore, "embed_documents", _boom)
    res = databank.sync_summaries()
    assert res["upserted"] == 0 and res["deleted"] == 0
    assert res["skipped"] > 0


def test_databank_event_routes_delete(synced):
    databank, vectorstore, monkeypatch = synced
    from app.schemas import DatabankEvent

    evt = DatabankEvent(type="DELETE", old_record={"id": SAMPLE_ROW["id"]})
    assert evt.row_id() == SAMPLE_ROW["id"]
    evt_insert = DatabankEvent(type="INSERT", record={"id": "x"})
    assert evt_insert.row_id() == "x"
