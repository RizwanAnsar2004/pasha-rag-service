# Secure RAG Service

A FastAPI service that ingests JSON data, embeds it locally, stores it in
ChromaDB, and answers questions **strictly from that context** using Google
Gemini — with guardrails against prompt injection and out-of-context answering.

## Features

- **JSON ingest → embeddings → Chroma.** POST documents as JSON; they are
  embedded with Google Gemini's embedding model (task-type aware:
  `RETRIEVAL_DOCUMENT` for ingest, `RETRIEVAL_QUERY` for search) and upserted
  into a persistent ChromaDB collection.
- **Grounded answers only.** Retrieval uses a cosine-distance relevance gate
  (`MAX_DISTANCE`). If nothing relevant is found, the service refuses instead of
  hallucinating.
- **Prompt-injection defense (defense in depth):**
  1. Input guardrails reject common override/jailbreak patterns before the model
     is ever called (`app/guardrails.py`).
  2. Retrieved context is neutralized (control chars + forged delimiters
     stripped) and wrapped in a `<context>` block.
  3. A strict system prompt treats all context and user text as untrusted data,
     never as instructions, and refuses to reveal itself or answer beyond
     context.
- **Optional API-key auth** on the service endpoints (`SERVICE_API_KEY`).

## Setup

```bash
cd rag-service
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env            # then edit .env and set GOOGLE_API_KEY
```

Get a Gemini API key at https://aistudio.google.com/apikey.

## Run

```bash
uvicorn app.main:app --reload
```

Interactive docs at http://127.0.0.1:8000/docs.

## Endpoints

### `POST /ingest`

```json
{
  "documents": [
    { "id": "doc-1", "text": "Some fact to remember.", "metadata": {"category": "policy"} }
  ]
}
```

The `data/sample_documents.json` file is in the same shape and can be posted
directly.

### `POST /query`

```json
{ "question": "What is the refund policy?", "top_k": 4 }
```

Response:

```json
{
  "answer": "...",
  "grounded": true,
  "refused": false,
  "reason": null,
  "sources": [{ "id": "...", "text": "...", "metadata": {}, "distance": 0.21 }]
}
```

- `grounded` — the answer came from retrieved context.
- `refused` — request blocked by guardrails or no relevant context found.

### `GET /health`

Returns service status and the document count.

## Configuration

All settings come from environment variables / `.env` (see `.env.example`):

| Variable           | Default              | Purpose                                   |
| ------------------ | -------------------- | ----------------------------------------- |
| `GOOGLE_API_KEY`   | —                    | Gemini API key (required)                 |
| `GENERATION_MODEL` | `gemini-2.5-flash`   | Gemini model for generation               |
| `EMBEDDING_MODEL`  | `gemini-embedding-001` | Gemini embedding model                  |
| `CHROMA_PATH`      | `./data/chroma`      | Persistent Chroma directory               |
| `COLLECTION_NAME`  | `documents`          | Chroma collection name                    |
| `TOP_K`            | `4`                  | Chunks retrieved per query                |
| `MAX_DISTANCE`     | `0.75`               | Cosine-distance relevance cutoff          |
| `SERVICE_API_KEY`  | — (auth off)         | If set, require `X-API-Key` header        |

## Tests

```bash
pytest
```

`tests/test_guardrails.py` covers the injection-detection and context-
neutralization logic (no API key required).
