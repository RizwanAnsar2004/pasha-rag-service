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
from collections import Counter

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


# Profile columns folded INTO the embedded text (not just metadata) so even a
# sparse record — one with no idea/business-model write-up — still carries a few
# real facts the assistant can answer from (sector, location, stage, etc.).
_PROFILE_COLUMNS = [
    ("primary_industry", "Sector"),
    ("secondary_industries", "Also operates in"),
    ("product_stage", "Product stage"),
    ("city", "City"),
    ("hq_country", "Country"),
    ("nic_name", "Incubation center"),
    ("incubation_stage", "Incubation stage"),
    ("cohort", "Cohort"),
    ("website", "Website"),
]


def build_text(row: dict) -> str:
    """Assemble the embedded, public-facing text block for a databank row."""
    lines: list[str] = []
    for col, label in _CONTENT_COLUMNS:
        val = _strip_html(row.get(col))
        if val:
            lines.append(f"{label}: {val}")

    for col, label in _PROFILE_COLUMNS:
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
# Aggregate summary documents.
#
# Top-k retrieval only ever shows the model a handful of individual startup
# profiles, so aggregate questions ("how many categories are there?", "how many
# startups in Lahore?") can never be answered from per-row vectors. These
# pre-computed roll-up documents put the totals themselves into the corpus.
# --------------------------------------------------------------------------- #
SUMMARY_ID_PREFIX = "startup-summary:"

# One roll-up document per facet: (metadata column, id slug, plural phrasing,
# singular phrasing). The phrasing is written into the document text, so it
# should use the words people actually ask with.
_SUMMARY_FACETS = [
    ("primary_industry", "categories", "categories (industries / sectors)", "category"),
    ("city", "cities", "cities", "city"),
    ("nic_name", "incubation-centers", "incubation centers (NICs)", "incubation center"),
    ("product_stage", "product-stages", "product stages", "product stage"),
]

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def _facet_value(meta: dict, col: str) -> str | None:
    """A countable facet value, or None. Skips empties, the placeholder values
    the corpus treats as missing (NULL / Other / None), and raw UUIDs that
    leaked into text columns in a few source rows."""
    val = str(meta.get(col) or "").strip()
    if not val or val.lower() in {"null", "none", "other"} or _UUID_RE.match(val):
        return None
    return val


def _summary_doc(slug: str, text: str) -> Document:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Document(
        id=f"{SUMMARY_ID_PREFIX}{slug}",
        text=text,
        metadata={
            "type": "startup_summary",
            "facet": slug,
            "content_hash": content_hash,
        },
    )


def build_summary_documents(metas: list[dict]) -> list[Document]:
    """Build the aggregate documents from startup vector metadata. Facets with
    no usable values are omitted; an empty databank yields no documents."""
    total = len(metas)
    if total == 0:
        return []

    counts: dict[str, Counter] = {}
    for col, slug, _, _ in _SUMMARY_FACETS:
        counter: Counter = Counter()
        for meta in metas:
            val = _facet_value(meta, col)
            if val:
                counter[val] += 1
        counts[slug] = counter

    docs: list[Document] = []
    for _, slug, plural, singular in _SUMMARY_FACETS:
        counter = counts[slug]
        if not counter:
            continue
        breakdown = "; ".join(f"{name}: {n}" for name, n in counter.most_common())
        docs.append(
            _summary_doc(
                slug,
                f"P@SHA Startup Databank summary — startup {plural}.\n"
                f"The databank lists {total} startups. There are "
                f"{len(counter)} distinct startup {plural}.\n"
                f"Number of startups per {singular}: {breakdown}.\n"
                f"This answers questions like: how many {plural} are there; "
                f"how many startups in each {singular}; which {singular} has "
                f"the most startups.",
            )
        )

    docs.append(
        _summary_doc(
            "overview",
            "P@SHA Startup Databank summary — overview.\n"
            f"The startup databank contains {total} startups in total, spanning "
            f"{len(counts['categories'])} categories (industries / sectors), "
            f"{len(counts['cities'])} cities, and "
            f"{len(counts['incubation-centers'])} incubation centers.\n"
            "This answers questions like: how many startups are there in "
            "total; how big is the startup databank.",
        )
    )
    return docs


def sync_summaries() -> dict:
    """Rebuild the aggregate summary documents from the startup vectors
    currently in the store. Called after every row/full sync so the totals
    always match the corpus; unchanged summaries are skipped via content_hash
    (no re-embed), and stale ones are pruned.

    Returns counts: {upserted, skipped, deleted}."""
    metas = vectorstore.list_meta(where={"type": "startup"})
    docs = build_summary_documents(metas)

    fresh: list[Document] = []
    for doc in docs:
        existing = vectorstore.get_meta(doc.id)
        if existing and existing.get("content_hash") == doc.metadata["content_hash"]:
            continue
        fresh.append(doc)
    if fresh:
        vectorstore.ingest(fresh)

    wanted = {doc.id for doc in docs}
    stored = vectorstore.list_ids(where={"type": "startup_summary"})
    stale = [sid for sid in stored if sid not in wanted]
    deleted = vectorstore.delete_ids(stale)
    return {
        "upserted": len(fresh),
        "skipped": len(docs) - len(fresh),
        "deleted": deleted,
    }


# --------------------------------------------------------------------------- #
# Sync operations.
# --------------------------------------------------------------------------- #
def sync_row(row_id: str) -> str:
    """Sync a single databank row by id. Returns 'upsert' | 'skip' | 'delete'."""
    rows = _fetch_rows([row_id])
    if not rows:
        # Row is gone (deleted / never existed) → drop its vector.
        vectorstore.delete_ids([chroma_id(row_id)])
        sync_summaries()
        return "delete"

    doc = build_document(rows[0])
    new_hash = doc.metadata.get("content_hash")
    existing = vectorstore.get_meta(doc.id)

    if existing and existing.get("content_hash") == new_hash:
        # Embedded text unchanged — refresh metadata only, skip the embedding.
        vectorstore.update_meta(doc.id, doc.metadata)
        return "skip"

    vectorstore.ingest([doc])
    sync_summaries()
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

    summaries = sync_summaries()

    logger.info(
        "databank sync_all: upserted=%d skipped=%d deleted=%d rows=%d summaries=%s",
        upserted, skipped, deleted, len(rows), summaries,
    )
    return {
        "upserted": upserted,
        "skipped": skipped,
        "deleted": deleted,
        "total_in_collection": vectorstore.count(),
    }
