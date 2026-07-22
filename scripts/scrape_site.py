"""Scrape www.pasha.org.pk and turn it into RAG documents.

Two phases, deliberately separate:

  1. `scrape`  — crawl the site and write `{"documents": [...]}` JSON. Needs no
                 OpenAI key, so a crawl can be inspected/diffed before it costs
                 anything.
  2. `--ingest` — embed and upsert that JSON into Chroma.

URLs come from the site's Yoast sitemap index (it is a WordPress site), which is
both complete and cheaper than a BFS link crawl. If the sitemap is unreachable
we fall back to following same-host links from the homepage.

Usage:
    python -m scripts.scrape_site                        # crawl -> data/pasha_site.json
    python -m scripts.scrape_site --ingest               # crawl, then ingest
    python -m scripts.scrape_site --ingest-only          # ingest existing JSON
    python -m scripts.scrape_site --limit 20 --workers 4 # small trial run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.schemas import Document

BASE_URL = "https://www.pasha.org.pk"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
DEFAULT_OUT = "data/pasha_site.json"
DEFAULT_PAGES_DIR = "data/pasha_pages"

# Identify the crawler honestly and leave a contact path, per crawling etiquette.
USER_AGENT = "PashaRAGBot/1.0 (+https://www.pasha.org.pk; RAG ingestion)"

# Chunk sizing. ~1200 chars (~300 tokens) keeps a chunk topically tight enough
# that cosine distance stays meaningful against the 0.85 `max_distance` gate,
# while the overlap stops a fact from being split across a boundary.
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150

# Pages with less than this much real text are navigation/taxonomy stubs; they
# add retrieval noise without adding facts.
MIN_PAGE_CHARS = 200

# A line appearing on more than this share of pages is site chrome (menus,
# footer addresses, cookie notices) that survived tag stripping. Dropping it
# stops every chunk from carrying the same boilerplate.
BOILERPLATE_RATIO = 0.30

# Containers holding the actual article body, best first.
CONTENT_SELECTORS = (
    "main",
    "article",
    ".entry-content",
    ".elementor-location-single",
    "#content",
    ".site-main",
)

# Chrome to remove before reading text.
STRIP_SELECTORS = (
    "script", "style", "noscript", "template", "svg", "form", "iframe",
    "nav", "header", "footer", "aside",
    ".elementor-location-header", ".elementor-location-footer",
    ".menu", ".breadcrumb", ".breadcrumbs", ".cookie", ".cookie-notice",
    ".screen-reader-text", ".skip-link", ".social-share", ".sharedaddy",
)

# Non-HTML endpoints that occasionally appear in sitemaps.
BINARY_RE = re.compile(r"\.(pdf|jpe?g|png|gif|webp|svg|zip|docx?|xlsx?|pptx?|mp4|mp3)$", re.I)


# --------------------------------------------------------------------------- #
# Fetching.
# --------------------------------------------------------------------------- #
def _client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml"},
        timeout=timeout,
        follow_redirects=True,
    )


def _get(client: httpx.Client, url: str, retries: int = 2) -> str | None:
    """Fetch a URL, returning HTML/XML text or None. Retries transient failures
    with a backoff so one flaky response doesn't drop a page from the corpus."""
    for attempt in range(retries + 1):
        try:
            response = client.get(url)
            if response.status_code >= 400:
                # A 4xx won't change on retry; a 5xx might.
                if response.status_code < 500 or attempt == retries:
                    return None
            else:
                ctype = response.headers.get("content-type", "")
                if "html" not in ctype and "xml" not in ctype:
                    return None
                return response.text
        except httpx.HTTPError:
            if attempt == retries:
                return None
        time.sleep(1.5 * (attempt + 1))
    return None


# --------------------------------------------------------------------------- #
# URL discovery.
# --------------------------------------------------------------------------- #
def _sitemap_locs(xml: str) -> list[str]:
    return re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, re.I | re.S)


def discover_from_sitemap(client: httpx.Client) -> dict[str, dict]:
    """Walk the sitemap index and return {url: {content_type, lastmod}}.

    `content_type` is taken from the sub-sitemap name (post, page, publications,
    press_releases, ...) — a free, accurate label for filtering later.
    """
    index = _get(client, SITEMAP_URL)
    if not index:
        return {}

    sub_sitemaps = _sitemap_locs(index) if "<sitemapindex" in index else [SITEMAP_URL]
    found: dict[str, dict] = {}

    for sub in sub_sitemaps:
        xml = _get(client, sub)
        if not xml:
            print(f"  ! could not read {sub}", file=sys.stderr)
            continue
        label = re.sub(r"-sitemap\.xml$", "", sub.rsplit("/", 1)[-1])
        # Pair each <loc> with the <lastmod> in the same <url> block, if present.
        for block in re.findall(r"<url>(.*?)</url>", xml, re.S) or []:
            loc = re.search(r"<loc>\s*(.*?)\s*</loc>", block, re.S)
            if not loc:
                continue
            mod = re.search(r"<lastmod>\s*(.*?)\s*</lastmod>", block, re.S)
            url = loc.group(1).strip()
            if BINARY_RE.search(urlparse(url).path):
                continue
            found[url] = {
                "content_type": label,
                "lastmod": (mod.group(1).strip() if mod else ""),
            }
        print(f"  {label}: {len(_sitemap_locs(xml))} urls")

    return found


def discover_by_crawl(client: httpx.Client, max_pages: int = 500) -> dict[str, dict]:
    """Fallback: breadth-first crawl of same-host links from the homepage."""
    host = urlparse(BASE_URL).netloc
    seen: set[str] = set()
    queue = [BASE_URL]
    found: dict[str, dict] = {}

    while queue and len(found) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        html = _get(client, url)
        if not html:
            continue
        found[url] = {"content_type": "page", "lastmod": ""}
        for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
            link = urljoin(url, a["href"]).split("#")[0].rstrip("/") or BASE_URL
            if urlparse(link).netloc == host and link not in seen:
                if not BINARY_RE.search(urlparse(link).path):
                    queue.append(link)
    return found


# --------------------------------------------------------------------------- #
# Extraction.
# --------------------------------------------------------------------------- #
def _clean_text(node) -> str:
    """Readable text from a soup node: block tags become line breaks so list
    items and table cells don't run together into one unsearchable blob."""
    for tag in node.find_all(["br"]):
        tag.replace_with("\n")
    for tag in node.find_all(["p", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "div"]):
        tag.append("\n")
    text = node.get_text(" ")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def extract_page(html: str, url: str) -> dict | None:
    """Pull title, description and body text out of one page."""
    soup = BeautifulSoup(html, "lxml")

    title = ""
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    elif soup.title:
        title = soup.title.get_text(strip=True)
    # Strip the site-name suffix WordPress appends to every <title>, including
    # the bare separator Yoast leaves behind when the site name is empty.
    title = re.sub(r"\s*[|\-–]\s*(P@SHA)?\s*$", "", title).strip()

    desc = ""
    md = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
    if md and md.get("content"):
        desc = md["content"].strip()

    for selector in STRIP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    body = None
    for selector in CONTENT_SELECTORS:
        candidate = soup.select_one(selector)
        if candidate and len(candidate.get_text(strip=True)) > 100:
            body = candidate
            break
    body = body or soup.body
    if body is None:
        return None

    text = _clean_text(body)
    if not text:
        return None
    return {"url": url, "title": title or url, "description": desc, "text": text}


def strip_boilerplate(pages: list[dict]) -> list[dict]:
    """Drop lines that repeat across most pages.

    Tag-level stripping misses chrome rendered inside the content container
    (Elementor puts menus and footers there). Frequency is the reliable signal:
    a line on 30%+ of pages is navigation, not content.
    """
    if len(pages) < 10:
        return pages

    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({line for line in page["text"].split("\n") if line.strip()})

    threshold = max(3, int(len(pages) * BOILERPLATE_RATIO))
    # Keep long lines regardless — a repeated paragraph of real prose (a mission
    # statement, a standard disclaimer) is content worth retrieving.
    common = {line for line, n in counts.items() if n >= threshold and len(line) < 200}
    if common:
        print(f"  dropping {len(common)} boilerplate lines seen on >={threshold} pages")

    for page in pages:
        kept = [ln for ln in page["text"].split("\n") if ln.strip() and ln not in common]
        page["text"] = "\n".join(kept).strip()
    return pages


# --------------------------------------------------------------------------- #
# Supplements: content the server-rendered HTML does not contain.
#
# Parts of the site are client-rendered — the detail pages for committee
# members, partners, Pulse editions etc. ship as empty shells and fill
# themselves in over AJAX. Scraping the HTML alone silently loses the entire
# leadership roster, so we go to the same endpoints the browser does.
# --------------------------------------------------------------------------- #
AJAX_URL = f"{BASE_URL}/wp-admin/admin-ajax.php"
TODAY = datetime.now().strftime("%Y-%m-%d")

# (action, field, values, page the content belongs to, label). `values` of None
# means the option list is read off the page's own <select>.
AJAX_ROSTERS = (
    ("myfilter", "peoplefilter", None, "/central-exective-committee/", "Central Executive Committee"),
    ("secretarialfilter", "secretarialfilter", ["all"], "/secretariat/", "P@SHA Secretariat"),
    ("subCommitteefilter", "subCommitteefilter", ["all"], "/sub-committees/", "P@SHA Sub Committees"),
)

# Post types whose REST records are plumbing, not content.
SKIP_REST_TYPES = {
    "attachment", "nav_menu_item", "wp_block", "wp_template",
    "wp_template_part", "wp_navigation",
}


def _select_options(client: httpx.Client, path: str, field: str) -> list[str]:
    """Read the option values of a filter <select> so we ask for every slice
    (e.g. all committee terms), not just the one rendered by default."""
    html = _get(client, urljoin(BASE_URL, path))
    if not html:
        return []
    select = BeautifulSoup(html, "lxml").find("select", attrs={"name": field})
    if not select:
        return []
    return [o.get("value") for o in select.find_all("option") if o.get("value")]


def _parse_people(html: str) -> list[tuple[str, str, str]]:
    """Extract (name, role, section) from roster card markup.

    Cards pair a name with a designation; the headings above them group people
    into terms ("Members (2024-2026)") or departments ("Skills Development").
    """
    soup = BeautifulSoup(html, "lxml")
    people: list[tuple[str, str, str]] = []
    section = ""

    for node in soup.select("h1, h2, h3, h4, h5, .__card"):
        if "__card" not in (node.get("class") or []):
            section = node.get_text(" ", strip=True) or section
            continue
        name_el = node.select_one(".name")
        role_el = node.select_one(".designation")
        name = name_el.get_text(" ", strip=True) if name_el else ""
        role = role_el.get_text(" ", strip=True) if role_el else ""
        if name:
            people.append((name, role, section))
    return people


def _roster_text(people: list[tuple[str, str, str]]) -> str:
    """The whole roster as one document — answers "list the committee"."""
    lines: list[str] = []
    section = None
    for name, role, sec in people:
        if sec != section:
            section = sec
            lines.append(f"\n{sec}" if sec else "")
        lines.append(f"{role}: {name}" if role else name)
    return "\n".join(lines).strip()


def _person_text(name: str, role: str, section: str, label: str) -> str:
    """One person as their own document.

    A 12-person roster in a single chunk embeds as an average of everyone, so
    "Who is the Senior Vice Chairman?" loses to any press release that repeats
    the phrase in prose. A short document per person keeps the role and the name
    as the dominant signal.
    """
    parts = [f"{name} is the {role} of P@SHA (Pakistan Software Houses Association)." if role else name]
    if role:
        parts.append(f"The {role} of P@SHA is {name}.")
    if section and section.lower() not in (role or "").lower():
        parts.append(f"Part of {label} — {section}.")
    else:
        parts.append(f"Part of {label}.")
    return " ".join(parts)


def fetch_ajax_rosters(client: httpx.Client) -> list[dict]:
    """Pull the people/committee listings the front end loads over AJAX."""
    pages: list[dict] = []
    for action, field, values, path, label in AJAX_ROSTERS:
        options = values if values is not None else _select_options(client, path, field)
        for position, option in enumerate(options or []):
            try:
                response = client.post(
                    AJAX_URL,
                    data={"action": action, field: option},
                    headers={"X-Requested-With": "XMLHttpRequest"},
                )
            except httpx.HTTPError:
                continue
            if response.status_code != 200 or not response.text.strip():
                continue
            people = _parse_people(response.text)
            if not people:
                continue

            title = label if option in ("all", "") else f"{label} ({option})"
            url = urljoin(BASE_URL, path)
            # A roster is whatever the site serves right now, so it outranks any
            # historical press release about the same position. Past committee
            # terms are the exception: the year <select> is ordered newest-first,
            # so only its leading option is the sitting committee.
            historical = position > 0 and bool(re.match(r"^\d{4}-\d{4}$", option or ""))
            common = {
                "url": url,
                "description": "",
                "_date": TODAY,
                "_current": not historical,
            }

            # The full roster answers "list the committee"...
            pages.append(
                {**common, "title": title, "text": _roster_text(people),
                 "_key": f"{action}-{option}"}
            )
            # ...and one document per person answers "who is the <role>?".
            for name, role, section in people:
                pages.append(
                    {
                        **common,
                        "title": f"{name} — {role}" if role else name,
                        "text": _person_text(name, role, section, title),
                        "_key": f"{action}-{option}-{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}",
                    }
                )
            print(f"  roster: {title} ({len(people)} people)")
    return pages


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    return _clean_text(BeautifulSoup(html, "lxml"))


def fetch_rest_items(client: httpx.Client, covered: set[str]) -> list[dict]:
    """Fill gaps from the WordPress REST API.

    Items with real body text become their own page (when the HTML crawl didn't
    already cover that URL). Items whose body lives in fields REST doesn't
    expose — partners, Pulse editions, committee terms — are rolled up into one
    index document per post type, so "which partners does P@SHA have?" is at
    least answerable from titles and dates.
    """
    try:
        response = client.get(f"{BASE_URL}/wp-json/wp/v2/types")
        types = response.json() if response.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return []

    pages: list[dict] = []
    for slug, info in types.items():
        rest_base = info.get("rest_base")
        if not rest_base or slug in SKIP_REST_TYPES:
            continue

        items: list[dict] = []
        page_num = 1
        while page_num <= 20:                      # hard stop; nothing is that big
            try:
                r = client.get(
                    f"{BASE_URL}/wp-json/wp/v2/{rest_base}",
                    params={"per_page": 100, "page": page_num},
                )
            except httpx.HTTPError:
                break
            if r.status_code != 200:
                break
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            items.extend(batch)
            if page_num >= int(r.headers.get("X-WP-TotalPages", 1)):
                break
            page_num += 1

        if not items:
            continue

        type_label = info.get("name") or slug
        stubs: list[str] = []
        added = 0

        for item in items:
            link = (item.get("link") or "").strip()
            title = _html_to_text(item.get("title", {}).get("rendered", ""))
            date = (item.get("date") or "")[:10]
            body = _html_to_text(item.get("content", {}).get("rendered", ""))
            if not body:
                body = _html_to_text(item.get("excerpt", {}).get("rendered", ""))

            if len(body) >= MIN_PAGE_CHARS:
                if link and link.rstrip("/") not in covered:
                    pages.append(
                        {
                            "url": link,
                            "title": title or type_label,
                            "description": "",
                            "text": f"{type_label} published {date}.\n{body}" if date else body,
                            "_key": f"rest-{slug}-{item.get('id')}",
                            "_date": date,
                        }
                    )
                    added += 1
            elif title:
                stubs.append(f"{title}" + (f" — published {date}" if date else "") + (f" ({link})" if link else ""))

        if stubs:
            listing = "\n".join(stubs)
            pages.append(
                {
                    "url": f"{BASE_URL}/wp-json/wp/v2/{rest_base}",
                    "title": f"P@SHA {type_label} — full list",
                    "description": f"Complete list of {type_label.lower()} published by P@SHA.",
                    "text": f"P@SHA {type_label} ({len(stubs)} entries):\n{listing}",
                    "_key": f"rest-index-{slug}",
                    "_date": TODAY,
                    "_current": True,           # a live snapshot of what exists
                }
            )
        if added or stubs:
            print(f"  rest {slug}: {added} full pages, {len(stubs)} index entries")

    return pages


# --------------------------------------------------------------------------- #
# Chunking -> Documents.
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on line boundaries into ~`size` char chunks with a tail overlap."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for line in lines:
        # A single oversized line (a wall-of-text paragraph) is hard-split so it
        # can never exceed the chunk budget on its own.
        if len(line) > size:
            if current:
                chunks.append("\n".join(current))
                current, length = [], 0
            for i in range(0, len(line), size - overlap):
                chunks.append(line[i : i + size])
            continue
        if length + len(line) + 1 > size and current:
            chunks.append("\n".join(current))
            tail, tail_len = [], 0
            for prev in reversed(current):          # carry a little context over
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            current, length = tail, tail_len
        current.append(line)
        length += len(line) + 1

    if current:
        chunks.append("\n".join(current))
    return [c.strip() for c in chunks if c.strip()]


def _slug(url: str, key: str | None = None) -> str:
    """Readable, stable id stem: path slug + short URL hash to avoid collisions
    once long paths are truncated. `key` distinguishes several documents that
    share one URL (the AJAX roster slices)."""
    if key:
        return f"{re.sub(r'[^a-z0-9]+', '-', key.lower()).strip('-')[:60]}"
    path = urlparse(url).path.strip("/") or "home"
    slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")[:60]
    return f"{slug}-{hashlib.sha1(url.encode()).hexdigest()[:6]}"


def build_documents(pages: list[dict], meta_by_url: dict[str, dict]) -> list[Document]:
    """Turn extracted pages into chunk-level Documents, deduped by content."""
    documents: list[Document] = []
    seen_hashes: set[str] = set()

    for page in pages:
        info = meta_by_url.get(page["url"], {})
        # The description is a hand-written summary; keep it as the first chunk's
        # lead-in rather than a separate document.
        body = page["text"]
        if page["description"] and page["description"] not in body:
            body = f"{page['description']}\n{body}"

        # The corpus spans 2017-2026, so a 2023 press release announcing an
        # appointment competes with today's roster for "who is the X?". Metadata
        # is invisible to the model — only chunk text reaches the prompt — so the
        # date has to ride inside the text for the recency rule in the system
        # prompt to have anything to work with.
        date = page.get("_date") or (info.get("lastmod") or "")[:10]
        if page.get("_current"):
            provenance = f"(Status: CURRENT — live listing retrieved {date})"
        elif date:
            provenance = f"(Last updated: {date})"
        else:
            provenance = ""

        chunks = chunk_text(body)
        for i, chunk in enumerate(chunks):
            # Title + URL ride inside the embedded text: a chunk from deep in a
            # page otherwise loses all trace of what page it came from, and the
            # answer prompt only ever sees chunk text.
            header = f"{page['title']}\n(Source: {page['url']})"
            if provenance:
                header = f"{header}\n{provenance}"
            text = f"{header}\n\n{chunk}"
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue                      # identical chunk on another page
            seen_hashes.add(content_hash)

            metadata = {
                "type": "webpage",
                "source": "pasha.org.pk",
                "url": page["url"],
                "title": page["title"],
                "content_type": info.get("content_type", "page"),
                "chunk": i,
                "chunks": len(chunks),
                "content_hash": content_hash,
                "is_current": bool(page.get("_current")),
            }
            if date:
                metadata["date"] = date
            if info.get("lastmod"):
                metadata["lastmod"] = info["lastmod"]

            documents.append(
                Document(
                    id=f"web:{_slug(page['url'], page.get('_key'))}:{i}",
                    text=text,
                    metadata=metadata,
                )
            )
    return documents


def dump_pages(pages: list[dict], directory: str) -> None:
    """Write each page's complete text to its own .txt file.

    The vector store only ever holds chunks, which makes "did we actually
    capture this page?" hard to answer. These dumps are the readable record of
    exactly what was scraped, one file per page.
    """
    os.makedirs(directory, exist_ok=True)
    for existing in os.listdir(directory):                 # keep it a clean mirror
        if existing.endswith(".txt"):
            os.remove(os.path.join(directory, existing))

    for page in pages:
        name = _slug(page["url"], page.get("_key"))
        with open(os.path.join(directory, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{page['title']}\nURL: {page['url']}\n\n{page['text']}\n")
    print(f"Dumped {len(pages)} full page texts to {directory}/")


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def scrape(limit: int | None, workers: int, delay: float) -> list[Document]:
    with _client() as client:
        print("Discovering URLs from sitemap...")
        meta_by_url = discover_from_sitemap(client)
        if not meta_by_url:
            print("Sitemap unavailable — falling back to link crawl.", file=sys.stderr)
            meta_by_url = discover_by_crawl(client)

        urls = sorted(meta_by_url)
        if limit:
            urls = urls[:limit]
        print(f"\nFetching {len(urls)} pages with {workers} workers...")

        pages: list[dict] = []
        failed: list[str] = []

        def fetch_one(url: str) -> dict | None:
            # Stagger requests so a small pool still behaves politely.
            if delay:
                time.sleep(delay)
            html = _get(client, url)
            if not html:
                failed.append(url)
                return None
            return extract_page(html, url)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n, page in enumerate(pool.map(fetch_one, urls), start=1):
                if page:
                    pages.append(page)
                if n % 50 == 0:
                    print(f"  {n}/{len(urls)}")

    print(f"\nFetched {len(pages)} pages ({len(failed)} failed).")
    if failed:
        for url in failed[:10]:
            print(f"  failed: {url}", file=sys.stderr)

    pages = strip_boilerplate(pages)

    thin = [p for p in pages if len(p["text"]) < MIN_PAGE_CHARS]
    pages = [p for p in pages if len(p["text"]) >= MIN_PAGE_CHARS]
    print(f"Kept {len(pages)} pages with content ({len(thin)} client-rendered shells).")

    # Recover what the HTML shells were hiding.
    with _client() as client:
        print("\nFetching AJAX-loaded rosters...")
        rosters = fetch_ajax_rosters(client)
        print(f"\nFilling gaps from the WordPress REST API...")
        covered = {p["url"].rstrip("/") for p in pages}
        rest = fetch_rest_items(client, covered)

    pages.extend(rosters)
    pages.extend(rest)
    print(f"\nTotal pages of content: {len(pages)} "
          f"({len(rosters)} rosters + {len(rest)} from REST).")

    documents = build_documents(pages, meta_by_url)
    print(f"Built {len(documents)} chunks.")
    return pages, documents


def save(documents: list[Document], path: str) -> None:
    payload = {"documents": [d.model_dump() for d in documents]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(documents)} documents to {path}")


def load(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(**d) for d in data["documents"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape pasha.org.pk into the RAG store.")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"JSON output path (default: {DEFAULT_OUT})")
    parser.add_argument("--pages-dir", default=DEFAULT_PAGES_DIR,
                        help=f"Where to dump full page text (default: {DEFAULT_PAGES_DIR})")
    parser.add_argument("--ingest", action="store_true", help="Embed and upsert into Chroma after scraping.")
    parser.add_argument("--ingest-only", action="store_true", help="Skip the crawl; ingest an existing JSON file.")
    parser.add_argument("--replace", action="store_true",
                        help="Delete existing webpage vectors before ingesting, so the "
                             "collection exactly mirrors this crawl (no orphaned chunks).")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N URLs (for a trial run).")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent fetches (default: 6).")
    parser.add_argument("--delay", type=float, default=0.3, help="Per-request delay in seconds (default: 0.3).")
    args = parser.parse_args()

    if args.ingest_only:
        documents = load(args.out)
        print(f"Loaded {len(documents)} documents from {args.out}")
    else:
        pages, documents = scrape(args.limit, args.workers, args.delay)
        if not documents:
            print("Nothing scraped — aborting.", file=sys.stderr)
            sys.exit(1)
        save(documents, args.out)
        dump_pages(pages, args.pages_dir)

    if args.ingest or args.ingest_only:
        # Imported lazily so a crawl-only run never needs an OpenAI key.
        from app import vectorstore

        if args.replace:
            # Upsert alone leaves orphans: when a page yields fewer chunks than
            # last time, the surplus ids keep their stale text and stay
            # retrievable. Clearing the website's own vectors first — and only
            # those, so databank rows are untouched — keeps the collection an
            # exact mirror of the crawl.
            stale = vectorstore.list_ids(where={"type": "webpage"})
            removed = vectorstore.delete_ids(stale)
            print(f"\nCleared {removed} existing webpage vector(s).")

        print("\nEmbedding and upserting into Chroma...")
        ingested, total = vectorstore.ingest(documents)
        print(f"Ingested {ingested}. Collection now holds {total} documents.")


if __name__ == "__main__":
    main()
