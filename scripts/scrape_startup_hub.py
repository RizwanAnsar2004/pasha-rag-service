"""Scrape startups.pasha.org.pk (the P@SHA Startup Hub) into RAG documents.

Same two-phase shape as `scripts.scrape_site` — crawl to JSON first, ingest
second — but the site is a different animal. pasha.org.pk is WordPress; the Hub
is a Next.js app whose server HTML is only part of the story:

  * The FAQ is an accordion. Only the OPEN item's answer is in the HTML; the
    other nine answers live in the page's JS bundle, so they are read from
    there (`fetch_faqs`).
  * The committee roster renders as unlabelled cards, but the same records sit
    in the RSC flight payload as JSON — cleaner to parse, and it lets us drop
    the members' email addresses instead of embedding them (`fetch_committee`).
  * Every page renders its whole body twice, once inside the closed nav
    overlay, so the second copy is cut off (`_undo_duplicate_render`).

Startup profiles under /directory/<slug> are deliberately NOT crawled: those
rows already reach the store from Supabase via `app.databank`, which is their
source of truth. The /directory listing grid is dropped for the same reason
(`_trim_listing`).

Documents are typed `hub_page` and `hub_faq`, which keeps them separate from the
`webpage` vectors that `scripts.scrape_site --replace` deletes.

Usage:
    python -m scripts.scrape_startup_hub                  # crawl -> data/pasha_startup_hub.json
    python -m scripts.scrape_startup_hub --ingest         # crawl, then ingest
    python -m scripts.scrape_startup_hub --ingest-only    # ingest existing JSON
    python -m scripts.scrape_startup_hub --ingest --replace   # mirror the crawl exactly
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.schemas import Document
from scripts.scrape_site import chunk_text

BASE_URL = "https://startups.pasha.org.pk"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
DEFAULT_OUT = "data/pasha_startup_hub.json"
DEFAULT_PAGES_DIR = "data/hub_pages"

USER_AGENT = "PashaRAGBot/1.0 (+https://startups.pasha.org.pk; RAG ingestion)"

TODAY = datetime.now().strftime("%Y-%m-%d")

# Half of what `scripts.scrape_site` uses. These pages are landing-page copy —
# a dozen unrelated 2-3 line sections stacked on one route — not articles, so a
# 1200-char chunk averages several topics into one vector and every one of them
# lands far from the query. Measured with "how long does the startup application
# review take?": the chunk holding the answer sits at distance 0.66 — past the
# 0.60 gate, so the question was refused — at 1200 chars, and at 0.54 at 600.
CHUNK_CHARS = 600
CHUNK_OVERLAP = 100

# Routes the sitemap omits but the nav links to.
EXTRA_PATHS = ("/committee", "/events")

# Auth screens and the admin area: a sign-in form has no facts to retrieve, and
# /admin, /api and /launch are Disallow:ed in robots.txt.
SKIP_RE = re.compile(r"^/(admin|api|launch|apply/(login|success))|^/directory/", re.I)

# A line on this share of pages is the shared shell (nav overlay, footer). Set
# high because the corpus is only seven pages and the same startup can legitimately
# appear on three of them (home, directory, committee) — at 0.5 the threshold ate
# a committee member's company name.
BOILERPLATE_RATIO = 0.7

STRIP_SELECTORS = ("script", "style", "noscript", "template", "svg", "form", "iframe")


# --------------------------------------------------------------------------- #
# Fetching.
# --------------------------------------------------------------------------- #
def _client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml"},
        timeout=timeout,
        follow_redirects=True,
    )


def _get(client: httpx.Client, url: str) -> str | None:
    try:
        response = client.get(url)
    except httpx.HTTPError:
        return None
    if response.status_code >= 400:
        return None
    return response.text


def discover_urls(client: httpx.Client) -> list[str]:
    """Sitemap URLs plus the nav-only routes, in a stable order."""
    urls: list[str] = []
    xml = _get(client, SITEMAP_URL)
    if xml:
        urls = [u.strip() for u in re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, re.I | re.S)]
    else:
        print("  ! sitemap unavailable — using the known route list", file=sys.stderr)
        urls = [f"{BASE_URL}/", f"{BASE_URL}/about", f"{BASE_URL}/directory", f"{BASE_URL}/contact"]

    for path in EXTRA_PATHS:
        urls.append(urljoin(BASE_URL, path))

    seen: set[str] = set()
    keep: list[str] = []
    for url in urls:
        if urlparse(url).netloc != urlparse(BASE_URL).netloc:
            continue
        if SKIP_RE.match(urlparse(url).path or "/"):
            continue
        norm = url.rstrip("/") or BASE_URL
        if norm not in seen:
            seen.add(norm)
            keep.append(url)
    return keep


# --------------------------------------------------------------------------- #
# Page text.
# --------------------------------------------------------------------------- #
def _clean_text(node) -> str:
    for tag in node.find_all(["br"]):
        tag.replace_with("\n")
    for tag in node.find_all(["p", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "div"]):
        tag.append("\n")
    text = node.get_text(" ")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def _undo_duplicate_render(text: str) -> str:
    """Collapse the Hub's double render of every page.

    The closed nav overlay contains a second copy of the whole page, so a naive
    text extraction returns everything twice. Deduping line by line would be
    wrong — it also eats the legitimately repeated cells of the committee cards
    ("CEO" under four different names) — so instead we find where the copy
    starts and cut it off there.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 10:
        return "\n".join(lines)

    for k in range(5, len(lines) - 4):
        if lines[k] != lines[0]:
            continue
        head, tail = lines[:k], lines[k:]
        span = min(len(head), len(tail))
        same = sum(1 for a, b in zip(head, tail[:span]) if a == b)
        # Near-identical rather than identical: the two renders differ in a few
        # interactive bits (the overlay's own close/label text).
        if same / span >= 0.9:
            return "\n".join(head + tail[span:])
    return "\n".join(lines)


# Where a rendered result list begins on a listing page. Everything from here
# down is one page of database rows plus the paging/filter chrome around it.
_LISTING_START_RE = re.compile(r"^(Filters\b|Showing\s+\d|[\d,]+\s+startups$)", re.I)

# Paths whose body is mostly such a list.
LISTING_PATHS = {"/directory"}


def _trim_listing(text: str) -> str:
    """Cut the rendered startup cards off a listing page.

    /directory renders its first twelve rows (twice — once per breakpoint), and
    those rows are a thin, stale copy of records the store already holds from
    Supabase, where they are the source of truth. Worse, the grid's own header
    reads "2,500 startups" — a rounded figure that contradicts the databank
    summary's exact total whenever both reach the model. Only the page's own
    copy above the grid is worth keeping.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if _LISTING_START_RE.match(line.strip()):
            return "\n".join(lines[:i]).strip()
    return text


def extract_page(html: str, url: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    elif soup.title:
        title = soup.title.get_text(strip=True)
    title = re.sub(r"\s*[·|\-–]\s*PASHA Startup Hub\s*$", "", title).strip()
    title = title or "P@SHA Startup Hub"
    # Every chunk carries the site name for context, but the home page's own
    # title already is the site name — don't say it twice.
    if not re.search(r"startup hub", title, re.I):
        title = f"P@SHA Startup Hub — {title}"

    desc = ""
    md = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
    if md and md.get("content"):
        desc = md["content"].strip()

    for selector in STRIP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    if soup.body is None:
        return None

    text = _undo_duplicate_render(_clean_text(soup.body))
    if (urlparse(url).path.rstrip("/") or "/") in LISTING_PATHS:
        text = _trim_listing(text)
    if not text:
        return None
    return {"url": url, "title": title, "description": desc, "text": text}


def strip_boilerplate(pages: list[dict]) -> list[dict]:
    """Drop the shared shell (nav list, footer columns, cookie strip)."""
    if len(pages) < 3:
        return pages

    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({line for line in page["text"].split("\n") if line.strip()})

    threshold = max(2, int(len(pages) * BOILERPLATE_RATIO))
    # Long lines are prose worth keeping even when every page carries them (the
    # footer's one-line description of what the Hub is, for instance).
    common = {line for line, n in counts.items() if n >= threshold and len(line) < 120}
    if common:
        print(f"  dropping {len(common)} shell lines seen on >={threshold} pages")

    for page in pages:
        kept = [ln for ln in page["text"].split("\n") if ln.strip() and ln not in common]
        page["text"] = "\n".join(kept).strip()
    return pages


# --------------------------------------------------------------------------- #
# FAQ — answers live in the JS bundle, not the HTML.
# --------------------------------------------------------------------------- #
_FAQ_ENTRY_RE = re.compile(r'\{\s*q\s*:\s*(".*?[^\\]")\s*,\s*a\s*:\s*(".*?[^\\]")\s*\}', re.S)


def _js_string(literal: str) -> str:
    """Decode a JS double-quoted string literal. JSON covers every escape the
    bundler emits except \\' , which it rejects outright."""
    try:
        return json.loads(literal.replace("\\'", "'"))
    except ValueError:
        return literal.strip('"')


def fetch_faqs(client: httpx.Client, html: str) -> list[dict]:
    """Read the Q/A pairs out of the page's client bundle.

    The accordion ships its content as a `[{q, a}, ...]` array inside the chunk
    that defines the FAQ component, so only that array — not the rendered
    markup — has all ten answers.
    """
    faqs: list[dict] = []
    seen: set[str] = set()
    for path in sorted(set(re.findall(r'/_next/static/chunks/[^"\']+\.js', html))):
        js = _get(client, urljoin(BASE_URL, path))
        if not js or '{q:"' not in js:
            continue
        for q_lit, a_lit in _FAQ_ENTRY_RE.findall(js):
            question = _js_string(q_lit).strip()
            answer = _js_string(a_lit).strip()
            if question and answer and question not in seen:
                seen.add(question)
                faqs.append({"question": question, "answer": answer})
    print(f"  faq: {len(faqs)} question/answer pairs")
    return faqs


def build_faq_documents(faqs: list[dict]) -> list[Document]:
    """One document per FAQ entry.

    Small and single-topic on purpose: `app.rag` injects the whole set for
    founder-help questions, and a ten-answer blob would crowd the context while
    embedding as the average of ten unrelated topics.
    """
    documents: list[Document] = []
    for i, faq in enumerate(faqs):
        slug = re.sub(r"[^a-z0-9]+", "-", faq["question"].lower()).strip("-")[:60]
        text = (
            f"P@SHA Startup Hub FAQ — {faq['question']}\n"
            f"(Source: {BASE_URL}/#faq)\n"
            f"(Status: CURRENT — live listing retrieved {TODAY})\n\n"
            f"Question: {faq['question']}\n"
            f"Answer: {faq['answer']}"
        )
        documents.append(
            Document(
                id=f"hubfaq:{slug or i}",
                text=text,
                metadata={
                    "type": "hub_faq",
                    "source": "startups.pasha.org.pk",
                    "url": f"{BASE_URL}/#faq",
                    "title": f"P@SHA Startup Hub FAQ — {faq['question']}",
                    "question": faq["question"],
                    "position": i,
                    "is_current": True,
                    "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                },
            )
        )
    return documents


# --------------------------------------------------------------------------- #
# Committee — records come from the RSC flight payload.
# --------------------------------------------------------------------------- #
# Bookkeeping accounts in the members table, not people on the committee.
COMMITTEE_SKIP_TYPES = {"admin"}

COMMITTEE_ROLE_LABELS = {
    "chairman": "Chair",
    "chair": "Chair",
    "secretariat": "Secretariat",
    "member": "Committee Member",
}

COMMITTEE_NAME = "P@SHA Startups & Entrepreneurship Committee"


def _flight_text(html: str) -> str:
    """Reassemble the RSC payload from its `self.__next_f.push` fragments."""
    parts: list[str] = []
    for m in re.finditer(r'self\.__next_f\.push\(\[\d+\s*,\s*(".*?")\]\)', html, re.S):
        try:
            parts.append(json.loads(m.group(1)))
        except ValueError:
            continue
    return "".join(parts)


def _json_array_at(text: str, key: str) -> list | None:
    """Extract the JSON array that follows `"key":` by matching brackets."""
    marker = f'"{key}":['
    start = text.find(marker)
    if start < 0:
        return None
    start = text.index("[", start)
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except ValueError:
                    return None
    return None


def fetch_committee(client: httpx.Client) -> list[dict]:
    """Committee members as {name, role, org, type}.

    Only public-profile fields are kept — the payload also carries each member's
    email address, which has no business in a retrieval corpus.
    """
    html = _get(client, f"{BASE_URL}/committee")
    if not html:
        return []
    members = _json_array_at(_flight_text(html), "members") or []

    people: list[dict] = []
    for member in members:
        if not isinstance(member, dict):
            continue
        kind = str(member.get("type") or "").lower()
        name = str(member.get("name") or "").strip()
        if not name or kind in COMMITTEE_SKIP_TYPES:
            continue
        people.append(
            {
                "name": name,
                "role": str(member.get("role") or "").strip(),
                "org": str(member.get("org") or "").strip(),
                "kind": COMMITTEE_ROLE_LABELS.get(kind, "Committee Member"),
            }
        )
    print(f"  committee: {len(people)} members")
    return people


def build_committee_documents(people: list[dict]) -> list[Document]:
    """A roster document plus one document per member.

    The roster answers "who is on the committee"; the per-person documents keep
    a name and its position as the dominant signal, so "who chairs the startup
    committee?" isn't beaten by a page that merely mentions the committee.
    """
    if not people:
        return []

    url = f"{BASE_URL}/committee"
    header = (
        f"{COMMITTEE_NAME}\n(Source: {url})\n"
        f"(Status: CURRENT — live listing retrieved {TODAY})"
    )
    lines = [
        f"{p['kind']}: {p['name']}"
        + (f" — {p['role']}" if p["role"] else "")
        + (f", {p['org']}" if p["org"] else "")
        for p in people
    ]
    roster_text = (
        f"{header}\n\n"
        f"The {COMMITTEE_NAME} has {len(people)} members. It steers the P@SHA "
        f"Startup Hub, its verification standards and its ecosystem "
        f"partnerships.\n" + "\n".join(lines)
    )

    def _meta(extra: dict) -> dict:
        return {
            "type": "hub_page",
            "source": "startups.pasha.org.pk",
            "url": url,
            "is_current": True,
            "date": TODAY,
            **extra,
        }

    documents = [
        Document(
            id="hubcommittee:roster",
            text=roster_text,
            metadata=_meta(
                {
                    "title": COMMITTEE_NAME,
                    "content_hash": hashlib.sha256(roster_text.encode("utf-8")).hexdigest(),
                }
            ),
        )
    ]

    for person in people:
        at_org = f" at {person['org']}" if person["org"] else ""
        role = f"{person['role']}{at_org}" if person["role"] else person["org"]
        unique = person["kind"] != "Committee Member"
        article = "the" if unique else "a"
        sentences = [f"{person['name']} is {article} {person['kind']} of the {COMMITTEE_NAME}."]
        if unique:
            sentences.append(f"The {person['kind']} of the {COMMITTEE_NAME} is {person['name']}.")
        if role:
            sentences.append(f"{person['name']} is {role}.")
        text = f"{header}\n\n" + " ".join(sentences)
        slug = re.sub(r"[^a-z0-9]+", "-", person["name"].lower()).strip("-")[:50]
        documents.append(
            Document(
                id=f"hubcommittee:{slug}",
                text=text,
                metadata=_meta(
                    {
                        "title": f"{person['name']} — {person['kind']}, {COMMITTEE_NAME}",
                        "person": person["name"],
                        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                ),
            )
        )
    return documents


# --------------------------------------------------------------------------- #
# Page documents.
# --------------------------------------------------------------------------- #
def _slug(url: str) -> str:
    path = urlparse(url).path.strip("/") or "home"
    return re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")[:60]


# "browse 2,481 startups" in a meta description, where the databank summary
# says 2,485. Both are the same data counted moments apart, so the second figure
# buys nothing and costs an answer that changes with the phrasing of the
# question. The totals belong to the summary documents; the description keeps
# its sentence without the number.
_LIVE_COUNT_RE = re.compile(r"\b[\d,]+\s+(?=startups\b)", re.I)


def build_page_documents(pages: list[dict]) -> list[Document]:
    documents: list[Document] = []
    seen_hashes: set[str] = set()

    for page in pages:
        body = page["text"]
        description = _LIVE_COUNT_RE.sub("", page["description"])
        if description and description not in body:
            body = f"{description}\n{body}"

        chunks = chunk_text(body, CHUNK_CHARS, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            # Title, URL and freshness ride inside the embedded text: only chunk
            # text reaches the answer prompt, so a chunk that loses them loses
            # every trace of where it came from and how current it is.
            text = (
                f"{page['title']}\n(Source: {page['url']})\n"
                f"(Status: CURRENT — live page retrieved {TODAY})\n\n{chunk}"
            )
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            documents.append(
                Document(
                    id=f"hub:{_slug(page['url'])}:{i}",
                    text=text,
                    metadata={
                        "type": "hub_page",
                        "source": "startups.pasha.org.pk",
                        "url": page["url"],
                        "title": page["title"],
                        "chunk": i,
                        "chunks": len(chunks),
                        "date": TODAY,
                        "is_current": True,
                        "content_hash": content_hash,
                    },
                )
            )
    return documents


def dump_pages(pages: list[dict], directory: str) -> None:
    """The readable record of what the crawl actually captured (the store only
    ever holds chunks)."""
    os.makedirs(directory, exist_ok=True)
    for existing in os.listdir(directory):
        if existing.endswith(".txt"):
            os.remove(os.path.join(directory, existing))
    for page in pages:
        with open(os.path.join(directory, f"{_slug(page['url'])}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{page['title']}\nURL: {page['url']}\n\n{page['text']}\n")
    print(f"Dumped {len(pages)} full page texts to {directory}/")


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def scrape() -> tuple[list[dict], list[Document]]:
    with _client() as client:
        urls = discover_urls(client)
        print(f"Fetching {len(urls)} Hub pages...")

        pages: list[dict] = []
        home_html = ""
        for url in urls:
            html = _get(client, url)
            if not html:
                print(f"  failed: {url}", file=sys.stderr)
                continue
            if urlparse(url).path in ("", "/"):
                home_html = html
            page = extract_page(html, url)
            if page:
                pages.append(page)
                print(f"  {url} ({len(page['text'])} chars)")

        pages = strip_boilerplate(pages)

        print("\nReading the FAQ out of the page bundle...")
        faqs = fetch_faqs(client, home_html) if home_html else []

        print("Reading the committee roster...")
        committee = fetch_committee(client)

    documents = build_page_documents(pages)
    documents += build_faq_documents(faqs)
    documents += build_committee_documents(committee)
    print(f"\nBuilt {len(documents)} documents from {len(pages)} pages.")
    return pages, documents


def save(documents: list[Document], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"documents": [d.model_dump() for d in documents]}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(documents)} documents to {path}")


def load(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        return [Document(**d) for d in json.load(f)["documents"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape startups.pasha.org.pk into the RAG store.")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"JSON output path (default: {DEFAULT_OUT})")
    parser.add_argument("--pages-dir", default=DEFAULT_PAGES_DIR,
                        help=f"Where to dump full page text (default: {DEFAULT_PAGES_DIR})")
    parser.add_argument("--ingest", action="store_true", help="Embed and upsert into Chroma after scraping.")
    parser.add_argument("--ingest-only", action="store_true", help="Skip the crawl; ingest an existing JSON file.")
    parser.add_argument("--replace", action="store_true",
                        help="Delete existing Hub vectors first, so the collection exactly "
                             "mirrors this crawl (no orphaned chunks).")
    args = parser.parse_args()

    if args.ingest_only:
        documents = load(args.out)
        print(f"Loaded {len(documents)} documents from {args.out}")
    else:
        pages, documents = scrape()
        if not documents:
            print("Nothing scraped — aborting.", file=sys.stderr)
            sys.exit(1)
        save(documents, args.out)
        dump_pages(pages, args.pages_dir)

    if args.ingest or args.ingest_only:
        # Imported lazily so a crawl-only run never needs an OpenAI key.
        from app import vectorstore

        if args.replace:
            # Upsert alone leaves orphans behind when a page yields fewer chunks
            # than last time. Only Hub vectors are cleared — the pasha.org.pk
            # `webpage` docs and the Supabase `startup` rows are untouched.
            fresh = {d.id for d in documents}
            stale = [
                doc_id
                for doc_type in ("hub_page", "hub_faq")
                for doc_id in vectorstore.list_ids(where={"type": doc_type})
                if doc_id not in fresh
            ]
            print(f"\nCleared {vectorstore.delete_ids(stale)} stale Hub vector(s).")

        print("\nEmbedding and upserting into Chroma...")
        ingested, total = vectorstore.ingest(documents)
        print(f"Ingested {ingested}. Collection now holds {total} documents.")


if __name__ == "__main__":
    main()
