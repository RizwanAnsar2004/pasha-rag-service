"""Ingest an arbitrary JSON file into the RAG store.

If the file already matches the API schema ({"documents": [...]}) it is used
as-is. Otherwise the JSON is flattened into one document per top-level key,
with the nested content rendered as readable "path: value" lines (good for
retrieval).

Usage:
    python -m scripts.ingest_file data/data.json
"""

from __future__ import annotations

import json
import sys

from app.schemas import Document
from app import vectorstore


def _flatten(obj, prefix=""):
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines.extend(_flatten(v, f"{prefix}{k}: " if not prefix else f"{prefix}.{k}: "))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines.extend(_flatten(v, f"{prefix}[{i}] "))
    else:
        lines.append(f"{prefix}{obj}".strip())
    return lines


def build_documents(data) -> list[Document]:
    if isinstance(data, dict) and "documents" in data:
        return [Document(**d) for d in data["documents"]]

    docs: list[Document] = []
    if isinstance(data, dict):
        for key, value in data.items():
            text = "\n".join(_flatten(value, f"{key}: ")) if isinstance(value, (dict, list)) else f"{key}: {value}"
            text = text.strip()
            if text:
                docs.append(Document(id=key, text=text, metadata={"section": key}))
    else:
        docs.append(Document(id="root", text=json.dumps(data), metadata={}))
    return docs


def main(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = build_documents(data)
    print(f"Prepared {len(docs)} documents from {path}")
    ingested, total = vectorstore.ingest(docs)
    print(f"Ingested {ingested}. Collection now holds {total} documents.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/data.json")
