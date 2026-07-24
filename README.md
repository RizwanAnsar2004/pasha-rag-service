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
- **Aggregate startup summaries.** Per-startup vectors can never answer count
  questions ("how many categories are there?") because top-k retrieval only
  surfaces a few profiles. After every databank sync, roll-up documents
  (categories, cities, incubation centers, product stages, overview) are
  rebuilt from the startup metadata in Chroma (`app/databank.py:sync_summaries`)
  so those totals are retrievable like any other fact.

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

### `POST /query/voice`

Voice variant of `/query`. Send a recorded clip as `multipart/form-data`
(`audio` file field, plus optional `top_k` / `session_id` / `request_id` form
fields). The clip is transcribed (`TRANSCRIPTION_MODEL`, English/Urdu-hinted)
and the transcript runs through the exact same guardrails, retrieval gate, and
rate limits as a typed question:

```bash
curl -X POST http://127.0.0.1:8000/query/voice \
  -F "audio=@question.webm;type=audio/webm" -F "session_id=abc123"
```

The response is the `/query` shape plus a `transcription` field so the UI can
show what was heard. Uploads are capped at `MAX_AUDIO_BYTES` (default 10 MB).

In the browser, record with `MediaRecorder` and post the blob:

```js
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const rec = new MediaRecorder(stream);
const chunks = [];
rec.ondataavailable = (e) => chunks.push(e.data);
rec.onstop = async () => {
  const form = new FormData();
  form.append("audio", new Blob(chunks, { type: rec.mimeType }), "q.webm");
  form.append("session_id", sessionId);
  const res = await fetch("/query/voice", { method: "POST", body: form });
  const { transcription, answer } = await res.json();
};
rec.start();            // …then rec.stop() when the mic button is tapped again
```

### `GET /health`

Returns service status and the document count.

## Scraping pasha.org.pk

`scripts/scrape_site.py` builds the website corpus. It runs in two phases so a
crawl can be reviewed before it costs anything to embed:

```bash
python -m scripts.scrape_site              # crawl -> data/pasha_site.json + data/pasha_pages/
python -m scripts.scrape_site --ingest     # crawl, then embed into Chroma
python -m scripts.scrape_site --ingest-only  # embed an existing crawl
python -m scripts.scrape_site --limit 20   # small trial run
```

URLs come from the site's Yoast sitemap index (452 of them). Three sources are
combined, because the HTML alone is not the whole site:

1. **Page HTML** — the main content of every page, chunked with overlap.
2. **AJAX rosters** — the Central Executive Committee (per term), Secretariat
   and Sub Committee listings are client-rendered; their pages arrive as empty
   shells, so the crawler posts to `admin-ajax.php` exactly as the browser does.
   Without this the entire leadership roster is missing from the corpus.
3. **WordPress REST API** — fills remaining gaps and builds an index document
   per post type (partners, Pulse editions, galleries) whose detail pages carry
   no server-rendered text.

`data/pasha_pages/*.txt` holds the complete text of each page, one file per
page — the readable record of what was captured, since the vector store only
ever holds chunks.

## Scraping startups.pasha.org.pk (the Startup Hub)

`scripts/scrape_startup_hub.py` builds the Startup Hub corpus — what the Hub is,
the directory, applying and getting verified, the Startups & Entrepreneurship
Committee, events and contact details. Same two phases:

```bash
python -m scripts.scrape_startup_hub                    # crawl -> data/pasha_startup_hub.json + data/hub_pages/
python -m scripts.scrape_startup_hub --ingest           # crawl, then embed into Chroma
python -m scripts.scrape_startup_hub --ingest --replace # …and prune Hub chunks the crawl no longer produces
```

The Hub is a Next.js app, so its server HTML is only part of the content. Three
sources are combined:

1. **Page HTML** — the seven public routes, chunked at 600 chars (half the
   pasha.org.pk size: these are landing pages of many short unrelated sections,
   and a bigger chunk averages them into one unfindable vector).
2. **The page bundle** — the FAQ is an accordion whose collapsed answers never
   reach the HTML, so the ten Q/A pairs are read from the JS chunk that defines
   the component. They become one `hub_faq` document each.
3. **The RSC flight payload** — the committee roster renders as unlabelled
   cards but ships as JSON, which parses cleanly and lets the scraper keep the
   public fields (name, role, organisation) and drop the members' emails.

Startup profiles under `/directory/<slug>` are deliberately not crawled: those
rows come from Supabase via `app/databank.py`, which is their source of truth.
For the same reason the `/directory` listing grid is dropped — it is a stale
copy of twelve of those rows plus a rounded total that contradicts the databank
summaries.

Documents are typed `hub_page` / `hub_faq`, so `scrape_site.py --replace`
(which clears `webpage` vectors) and this script never delete each other's work.
`data/hub_pages/*.txt` is the readable record of the crawl.

## Configuration

All settings come from environment variables / `.env` (see `.env.example`):

| Variable           | Default              | Purpose                                   |
| ------------------ | -------------------- | ----------------------------------------- |
| `GOOGLE_API_KEY`   | —                    | Gemini API key (required)                 |
| `GENERATION_MODEL` | `gemini-2.5-flash`   | Gemini model for generation               |
| `EMBEDDING_MODEL`  | `gemini-embedding-001` | Gemini embedding model                  |
| `CHROMA_PATH`      | `./data/chroma`      | Persistent Chroma directory               |
| `COLLECTION_NAME`  | `documents`          | Chroma collection name                    |
| `TRANSCRIPTION_MODEL` | `gpt-4o-mini-transcribe` | Speech-to-text model for /query/voice |
| `MAX_AUDIO_BYTES`  | `10000000`           | Upload cap for /query/voice               |
| `TOP_K`            | `4`                  | Chunks retrieved per query                |
| `MAX_DISTANCE`     | `0.75`               | Cosine-distance relevance cutoff          |
| `SERVICE_API_KEY`  | — (auth off)         | If set, require `X-API-Key` header        |

## Tests

```bash
pytest
```

`tests/test_guardrails.py` covers the injection-detection and context-
neutralization logic (no API key required).
