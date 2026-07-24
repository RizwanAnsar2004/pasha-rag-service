"""FastAPI application exposing the secure RAG service."""

from __future__ import annotations

import logging
import math
import time

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import databank, vectorstore
from .config import get_settings
from .rag import FRIENDLY_REFUSAL, answer_question, transcribe_audio
from .schemas import (
    DatabankEvent,
    DatabankSyncResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    VoiceQueryResponse,
)

logger = logging.getLogger(__name__)


#: Bound on an untrusted, caller-supplied value that becomes a key in the
#: limiter's store — without a cap, a caller could grow it without limit.
MAX_SESSION_ID_LEN = 100


def _client_key(request: Request) -> str:
    """Identify the caller by network address, for rate limiting.

    nginx sits in front of this service (deploy/nginx.conf), so `request.client`
    is always 127.0.0.1 — using it directly would put every caller in the world
    into one shared bucket. X-Forwarded-For's first entry is the original client.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


async def bind_session_id(request: Request) -> None:
    """Read `session_id` out of the request body and stash it on the request.

    This is a route dependency, not middleware, and that matters. FastAPI
    resolves dependencies before invoking the (slowapi-wrapped) endpoint, so the
    value is in place by the time the limiter's key function runs. Reading the
    body here also caches it on this same Request, so the endpoint's own
    parsing still sees it — whereas consuming the stream inside a
    BaseHTTPMiddleware would leave the endpoint with an empty body.

    Never raises: a malformed body is the endpoint's problem to report, and the
    limiter just falls back to the IP bucket.

    Handles both body shapes the service accepts: JSON (/query) and multipart
    form (/query/voice). Starlette caches whichever one is parsed here, so the
    endpoint's own body/File/Form parsing still works afterwards.
    """
    session_id: str | None = None
    content_type = request.headers.get("content-type", "")
    try:
        if content_type.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
            payload = dict(await request.form())
        else:
            payload = await request.json()
    except Exception:  # noqa: BLE001 - unparseable body, handled downstream
        payload = None

    if isinstance(payload, dict):
        raw = payload.get("session_id")
        if isinstance(raw, str):
            candidate = raw.strip()
            # An over-long id is ignored rather than truncated: truncating would
            # let crafted ids collide into one another's buckets. Ignoring drops
            # the caller to the IP bucket, which is the stricter outcome.
            if candidate and len(candidate) <= MAX_SESSION_ID_LEN:
                session_id = candidate

    request.state.session_id = session_id


def _session_key(request: Request) -> str:
    """Rate-limit bucket for a caller's conversation.

    Prefixed so a session id can never collide with an IP bucket. Falls back to
    the network identity for callers that send no session id (direct API use,
    older clients), which keeps them limited rather than unlimited.
    """
    session_id = getattr(request.state, "session_id", None)
    if session_id:
        return f"session:{session_id}"
    return f"ip:{_client_key(request)}"


def _ip_key(request: Request) -> str:
    """Backstop bucket, namespaced to match `_session_key`'s IP fallback."""
    return f"ip:{_client_key(request)}"


limiter = Limiter(key_func=_session_key)


def _retry_after_seconds(request: Request, default: int = 60) -> int:
    """Seconds until the tripped limit's window rolls over."""
    current_limit = getattr(request.state, "view_rate_limit", None)
    if not current_limit:
        return default
    try:
        reset_at, _remaining = limiter.limiter.get_window_stats(
            current_limit[0], *current_limit[1]
        )
    except Exception:  # noqa: BLE001 - a broken limiter store must not mask the 429
        return default
    # At least a second, so a client never reads "retry in 0" and hammers.
    return max(1, math.ceil(reset_at - time.time()))


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """429 carrying an accurate Retry-After.

    slowapi's stock handler only attaches Retry-After when `headers_enabled` is
    on, and that mode requires every endpoint to return a starlette Response —
    which /query doesn't; it returns a pydantic model for FastAPI to serialise.
    Building the header here keeps the endpoints as they are, and this is the
    only response that needs it.

    The value matters: our web client locks its composer for exactly this long,
    so an absent or wrong number strands the user for far longer than the real
    cooldown.
    """
    retry_after = _retry_after_seconds(request)
    return JSONResponse(
        {
            "error": f"Rate limit exceeded: {exc.detail}",
            "retry_after": retry_after,
        },
        status_code=429,
        headers={"Retry-After": str(retry_after)},
    )

app = FastAPI(
    title="Secure RAG Service",
    version="0.1.0",
    description=(
        "Ingest JSON documents into ChromaDB and answer questions strictly from "
        "that context, with guardrails against prompt injection and "
        "out-of-context answering."
    ),
)


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Simple API-key gate. Disabled when SERVICE_API_KEY is unset."""
    settings = get_settings()
    if not settings.auth_enabled:
        return
    if x_api_key != settings.service_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "documents": vectorstore.count()}


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest(req: IngestRequest) -> IngestResponse:
    settings = get_settings()
    ingested, total = vectorstore.ingest(req.documents)
    return IngestResponse(
        ingested=ingested,
        collection=settings.collection_name,
        total_in_collection=total,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_api_key), Depends(bind_session_id)],
)
# Two buckets, both must pass. The session limit is the real quota — one
# visitor's conversation, counted on its own. The IP limit is what stops that
# quota being sidestepped by minting a fresh session id per question; it's set
# high enough that shared/NAT'd addresses don't trip it in normal use.
#
# Limits are read lazily so QUERY_RATE_LIMIT / QUERY_IP_RATE_LIMIT can be tuned
# by environment without touching the code.
@limiter.limit(lambda: get_settings().query_rate_limit, key_func=_session_key)
@limiter.limit(lambda: get_settings().query_ip_rate_limit, key_func=_ip_key)
def query(request: Request, req: QueryRequest) -> QueryResponse:
    logger.info(
        "query received",
        extra={
            "session_id": getattr(request.state, "session_id", None),
            # Clamped: it's caller-supplied and goes straight into the log.
            "request_id": (req.request_id or "")[:MAX_SESSION_ID_LEN] or None,
        },
    )
    return answer_question(req.question, req.top_k)


@app.post(
    "/query/voice",
    response_model=VoiceQueryResponse,
    dependencies=[Depends(require_api_key), Depends(bind_session_id)],
)
# Same two buckets as /query — a voice question spends the same quota as a
# typed one (slowapi counts each route's window separately, so the shared knob
# names keep the *rates* aligned, not one combined counter).
@limiter.limit(lambda: get_settings().query_rate_limit, key_func=_session_key)
@limiter.limit(lambda: get_settings().query_ip_rate_limit, key_func=_ip_key)
def query_voice(
    request: Request,
    audio: UploadFile = File(..., description="Recorded question (webm/ogg/mp3/wav)."),
    top_k: int | None = Form(default=None, ge=1, le=20),
    session_id: str | None = Form(default=None),
    request_id: str | None = Form(default=None),
) -> VoiceQueryResponse:
    """Voice variant of /query: transcribe the clip, then run the exact same
    guarded pipeline on the transcript. The transcript is echoed back so the UI
    can display what was heard."""
    settings = get_settings()
    audio_bytes = audio.file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio upload."
        )
    if len(audio_bytes) > settings.max_audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Audio exceeds the {settings.max_audio_bytes} byte limit.",
        )

    try:
        transcript = transcribe_audio(
            audio_bytes, audio.filename, audio.content_type
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a gateway error
        logger.exception("voice transcription failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Audio transcription failed.",
        ) from exc

    logger.info(
        "voice query received",
        extra={
            "session_id": getattr(request.state, "session_id", None),
            "request_id": (request_id or "")[:MAX_SESSION_ID_LEN] or None,
        },
    )

    if not transcript:
        return VoiceQueryResponse(
            answer=FRIENDLY_REFUSAL,
            grounded=False,
            refused=True,
            reason="No speech was recognized in the audio.",
            sources=[],
            transcription="",
        )

    result = answer_question(transcript, top_k)
    return VoiceQueryResponse(**result.model_dump(), transcription=transcript)


@app.post("/databank/event", dependencies=[Depends(require_api_key)])
def databank_event(event: DatabankEvent) -> dict[str, object]:
    """Databank change sink — re-syncs the affected row from the source of truth
    (the payload is never trusted for content).

    Returns 200 on success and on an unactionable payload (no id — retrying
    won't help). On a *genuine* sync failure (e.g. Supabase/OpenAI hiccup) it
    raises 500 so the caller's bounded retry kicks in and the row still lands —
    keeping ingestion reliable on every approve/edit."""
    row_id = event.row_id()
    if not row_id:
        return {"ok": False, "reason": "no row id in payload"}
    try:
        action = databank.sync_row(row_id)
        return {"ok": True, "id": row_id, "action": action}
    except Exception as exc:  # noqa: BLE001
        logger.exception("databank event sync failed for %s", row_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"sync failed for {row_id}: {exc}",
        ) from exc


@app.post(
    "/databank/sync",
    response_model=DatabankSyncResponse,
    dependencies=[Depends(require_api_key)],
)
def databank_sync() -> DatabankSyncResponse:
    """Full backfill + reconcile of the databank table into the vector store."""
    result = databank.sync_all()
    return DatabankSyncResponse(**result)
