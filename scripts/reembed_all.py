"""Force a clean, complete re-embed of every databank row.

Unlike `sync_databank` (which skips rows whose content_hash is unchanged), this
deletes all existing startup vectors first, then re-embeds every row from
Supabase. Use it when the collection is incomplete and you want a guaranteed
full dump.

    python -m scripts.reembed_all

Requires SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and OPENAI_API_KEY in .env.
"""

from __future__ import annotations

import logging

from app import databank, vectorstore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 1. Pull every row from Supabase (paginated).
    rows = databank._fetch_rows(None)
    print(f"Fetched {len(rows)} databank row(s) from Supabase.")

    # 2. Drop all existing startup vectors so nothing is skipped.
    existing = vectorstore.list_ids(where={"type": "startup"})
    deleted = vectorstore.delete_ids(existing)
    print(f"Cleared {deleted} existing startup vector(s).")

    # 3. Re-embed every row, unconditionally.
    docs = [databank.build_document(r) for r in rows]
    if docs:
        vectorstore.ingest(docs)

    total = vectorstore.count()
    print(
        f"Done. re-embedded={len(docs)} "
        f"rows={len(rows)} total_in_collection={total}"
    )
    if len(rows) != len(docs):
        print("WARNING: doc count != row count — investigate build_document.")


if __name__ == "__main__":
    main()
