"""Sync the Pasha `databank` table (Supabase) into the vector store.

Supabase is the source of truth. On a webhook event we re-fetch the affected
row by id and upsert/delete its vector; a full `sync_all()` backfills every row
and prunes orphans. Only public-profile fields are embedded — mirroring what the
Pasha `/directory/[slug]` page exposes — so no private contact / outreach /
financial data ever lands in the corpus.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re

import httpx

from .config import get_settings
from .schemas import Document
from . import vectorstore

logger = logging.getLogger(__name__)

# Stable id prefix so startup vectors are addressable + distinguishable from the
# manually-ingested pasha.org.pk site docs sharing the `documents` collection.
ID_PREFIX = "startup:"

# We embed EVERY form field stored in the answers bag (admins want all fields
# searchable, so any edit is reflected). Empty values are skipped; booleans
# render as Yes/No and lists are comma-joined. Add a key here only if a specific
# answer field should be kept OUT of the embedding (e.g. genuinely sensitive).
EXCLUDED_ANSWER_KEYS: set[str] = set()

# Rich-text / long-text columns embedded as the semantic body of a startup.
_CONTENT_COLUMNS = [
    ("startup_name", "Startup"),
    ("company_name", "Company"),
    ("tagline", "Tagline"),
    ("startup_idea", "Idea"),
    ("business_model", "Business model"),
    ("social_impact", "Social impact"),
    ("awards", "Awards"),
    ("certifications", "Certifications"),
    ("sdgs", "SDGs"),
]

# Non-sensitive columns kept as Chroma metadata for filtering / display.
_METADATA_COLUMNS = [
    "primary_industry",
    "secondary_industries",
    "product_stage",
    "city",
    "hq_country",
    "nic_name",
    "incubation_stage",
    "cohort",
    "website",
    "pasha_verified",
    "women_led",
    "hiring",
    "fundraising",
    "source",
    "updated_at",
]

# Columns pulled from PostgREST (content + metadata + id + JSONB bags). Kept as a
# list so we can prune any column the live table doesn't have yet (e.g. a
# not-yet-applied migration) instead of failing the whole sync with a 400.
_BASE_COLUMNS: list[str] = (
    ["id"]
    + [c for c, _ in _CONTENT_COLUMNS]
    + _METADATA_COLUMNS
    + ["key_persons", "answers"]
)
# Discovered-valid subset, cached after the first prune so later calls are fast.
_active_columns: list[str] | None = None

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(value: object) -> str:
    """Flatten rich-text HTML to readable plain text."""
    if not value:
        return ""
    text = _TAG_RE.sub(" ", str(value))
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chroma_id(row_id: str) -> str:
    return f"{ID_PREFIX}{row_id}"


# --------------------------------------------------------------------------- #
# Supabase REST access (service role — bypasses RLS).
# --------------------------------------------------------------------------- #
def _rest_headers() -> dict[str, str]:
    settings = get_settings()
    key = settings.supabase_service_role_key
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def _select_columns() -> str:
    return ",".join(_active_columns or _BASE_COLUMNS)


def _prune_missing_column(message: str) -> bool:
    """If a PostgREST error names a databank column we requested but the table
    lacks (`column ... does not exist`), drop it from the active set. Returns
    True when a column was pruned, so the caller retries without it."""
    global _active_columns
    if "does not exist" not in (message or ""):
        return False
    current = list(_active_columns or _BASE_COLUMNS)
    for col in current:
        if col == "id":
            continue
        if re.search(rf"\b{re.escape(col)}\b", message):
            current.remove(col)
            _active_columns = current
            logger.warning(
                "databank.%s not present in table — skipping it in sync", col
            )
            return True
    return False


def _get(
    client: httpx.Client, base: str, params: dict, headers: dict
) -> httpx.Response:
    """GET with strip-and-retry on PostgREST 'column does not exist' 400s. The
    `select` param is refreshed from the active column set on each attempt, and
    any other error surfaces the response body (far clearer than a bare status)."""
    while True:
        attempt = {**params, "select": _select_columns()}
        resp = client.get(base, params=attempt, headers=headers)
        if resp.status_code == 400 and _prune_missing_column(resp.text):
            continue
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Supabase databank fetch failed ({resp.status_code}): {resp.text}"
            )
        return resp


def _fetch_rows(ids: list[str] | None = None) -> list[dict]:
    """Fetch databank rows from Supabase. `ids=None` → all rows (paginated)."""
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not configured."
        )

    base = settings.supabase_url.rstrip("/") + "/rest/v1/databank"

    # Targeted fetch (event path): a small id set via the `in` filter.
    if ids is not None:
        if not ids:
            return []
        params = {"order": "id", "id": "in.(" + ",".join(ids) + ")"}
        with httpx.Client(timeout=30) as client:
            return _get(client, base, params, _rest_headers()).json()

    # Full fetch (backfill): page past PostgREST's 1000-row cap.
    rows: list[dict] = []
    page_size = 1000
    offset = 0
    with httpx.Client(timeout=60) as client:
        while True:
            headers = {
                **_rest_headers(),
                "Range-Unit": "items",
                "Range": f"{offset}-{offset + page_size - 1}",
            }
            batch = _get(client, base, {"order": "id"}, headers).json()
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
    return rows


# --------------------------------------------------------------------------- #
# Document assembly.
# --------------------------------------------------------------------------- #
def _key_person_lines(key_persons: object) -> list[str]:
    if not isinstance(key_persons, list):
        return []
    out: list[str] = []
    for p in key_persons:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()
        role = str(p.get("role") or "").strip()
        if name and role:
            out.append(f"{name} — {role}")
        elif name:
            out.append(name)
    return out


def _format_answer(value: object) -> str:
    """Render an answer value as readable text for embedding.

    Booleans → Yes/No, lists → comma-joined non-empty items, everything else →
    HTML-stripped string. Empty/blank values return "" so the caller skips them."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        parts = [_strip_html(v) for v in value]
        return ", ".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = [_strip_html(v) for v in value.values()]
        return ", ".join(p for p in parts if p)
    return _strip_html(value)


def _answer_lines(answers: object) -> list[str]:
    """Embed EVERY answer-bag field (minus EXCLUDED_ANSWER_KEYS) so any form
    field an admin edits is searchable. Empty values are skipped."""
    if not isinstance(answers, dict):
        return []
    out: list[str] = []
    for key, raw in answers.items():
        if key in EXCLUDED_ANSWER_KEYS:
            continue
        val = _format_answer(raw)
        if val:
            label = key.replace("_", " ").capitalize()
            out.append(f"{label}: {val}")
    return out


def build_text(row: dict) -> str:
    """Assemble the embedded, public-facing text block for a databank row."""
    lines: list[str] = []
    for col, label in _CONTENT_COLUMNS:
        val = _strip_html(row.get(col))
        if val:
            lines.append(f"{label}: {val}")

    people = _key_person_lines(row.get("key_persons"))
    if people:
        lines.append("Team: " + "; ".join(people))

    lines.extend(_answer_lines(row.get("answers")))
    return "\n".join(lines)


def build_metadata(row: dict, content_hash: str) -> dict:
    """Non-sensitive filter/display metadata. Chroma rejects None values, so we
    only include keys that carry a usable scalar."""
    meta: dict[str, object] = {
        "type": "startup",
        "row_id": str(row.get("id")),
        "startup_name": _strip_html(row.get("startup_name")) or "Unnamed",
        "content_hash": content_hash,
    }
    for col in _METADATA_COLUMNS:
        val = row.get(col)
        if val is None or val == "":
            continue
        meta[col] = val if isinstance(val, (bool, int, float)) else str(val)
    return meta


def build_document(row: dict) -> Document:
    """Build a vector-store Document (id, text, metadata) for a databank row."""
    text = build_text(row)
    # Guarantee a non-empty body even for a sparse row (Document requires it).
    if not text:
        text = f"Startup: {_strip_html(row.get('startup_name')) or 'Unnamed'}"
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Document(
        id=chroma_id(str(row.get("id"))),
        text=text,
        metadata=build_metadata(row, content_hash),
    )


# --------------------------------------------------------------------------- #
# Sync operations.
# --------------------------------------------------------------------------- #
def sync_row(row_id: str) -> str:
    """Sync a single databank row by id. Returns 'upsert' | 'skip' | 'delete'."""
    rows = _fetch_rows([row_id])
    if not rows:
        # Row is gone (deleted / never existed) → drop its vector.
        vectorstore.delete_ids([chroma_id(row_id)])
        return "delete"

    doc = build_document(rows[0])
    new_hash = doc.metadata.get("content_hash")
    existing = vectorstore.get_meta(doc.id)

    if existing and existing.get("content_hash") == new_hash:
        # Embedded text unchanged — refresh metadata only, skip the embedding.
        vectorstore.update_meta(doc.id, doc.metadata)
        return "skip"

    vectorstore.ingest([doc])
    return "upsert"


def sync_all() -> dict:
    """Backfill every databank row, then prune orphaned startup vectors.

    Returns counts: {upserted, skipped, deleted, total_in_collection}."""
    rows = _fetch_rows(None)
    upserted = 0
    skipped = 0

    fresh: list[Document] = []
    seen_ids: set[str] = set()
    for row in rows:
        doc = build_document(row)
        seen_ids.add(doc.id)
        existing = vectorstore.get_meta(doc.id)
        if existing and existing.get("content_hash") == doc.metadata.get("content_hash"):
            vectorstore.update_meta(doc.id, doc.metadata)
            skipped += 1
        else:
            fresh.append(doc)

    if fresh:
        vectorstore.ingest(fresh)
        upserted = len(fresh)

    # Reconcile: delete any startup vector whose row no longer exists.
    stored = set(vectorstore.list_ids(where={"type": "startup"}))
    orphans = [sid for sid in stored if sid not in seen_ids]
    deleted = vectorstore.delete_ids(orphans)

    logger.info(
        "databank sync_all: upserted=%d skipped=%d deleted=%d rows=%d",
        upserted, skipped, deleted, len(rows),
    )
    return {
        "upserted": upserted,
        "skipped": skipped,
        "deleted": deleted,
        "total_in_collection": vectorstore.count(),
    }
