"""Backfill / reconcile the Pasha databank into the RAG vector store.

Fetches every `databank` row from Supabase, upserts each as a startup document
(skipping rows whose embedded text is unchanged), and prunes vectors for rows
that no longer exist. Idempotent — safe to re-run any time the live webhook may
have missed events.

Usage:
    python -m scripts.sync_databank

Requires SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and OPENAI_API_KEY in .env.
"""

from __future__ import annotations

import logging

from app import databank


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Syncing databank → ChromaDB …")
    result = databank.sync_all()
    print(
        "Done. "
        f"upserted={result['upserted']} "
        f"skipped={result['skipped']} "
        f"deleted={result['deleted']} "
        f"total_in_collection={result['total_in_collection']}"
    )


if __name__ == "__main__":
    main()
