"""Print the nearest vector matches + distances for a query, ignoring the
relevance gate. Use it to see where a specific record lands and pick a sane
MAX_DISTANCE.

    python -m scripts.debug_query "tell me about smart soap"
    python -m scripts.debug_query "smart soap" 20
"""

from __future__ import annotations

import sys

from app.embeddings import embed_query
from app import vectorstore
from app.config import get_settings


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.debug_query "your question" [top_k]')
        sys.exit(1)

    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    settings = get_settings()

    print(f"Query: {question!r}")
    print(f"Current MAX_DISTANCE={settings.max_distance}  TOP_K={settings.top_k}")
    print(f"Collection size: {vectorstore.count()}\n")

    emb = embed_query(question)
    chunks = vectorstore.query(emb, k)

    if not chunks:
        print("No chunks returned at all — collection may be empty.")
        return

    for c in chunks:
        name = c.metadata.get("startup_name", "?")
        gate = "PASS" if c.distance <= settings.max_distance else "drop"
        snippet = c.text.replace("\n", " ")[:80]
        print(f"  dist={c.distance:.4f}  [{gate}]  {name}  | {snippet}")


if __name__ == "__main__":
    main()
