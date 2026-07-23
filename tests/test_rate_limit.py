"""Rate-limiting behaviour for /query.

The quota is keyed on the caller's `session_id` (sent in the request body), with
the client IP as a backstop. Retrieval and generation are stubbed out — these
tests are about who gets counted against which bucket, nothing else.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from limits import parse_many

from app.schemas import QueryResponse


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    """Build a TestClient with the two limits set to whatever the test needs."""

    def _build(session_limit: str = "3/hour", ip_limit: str = "100/hour") -> TestClient:
        monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
        monkeypatch.setenv("COLLECTION_NAME", "rate_limit_tests")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("QUERY_RATE_LIMIT", session_limit)
        monkeypatch.setenv("QUERY_IP_RATE_LIMIT", ip_limit)

        from app.config import get_settings

        get_settings.cache_clear()

        from app import main as app_main

        # Skip retrieval + generation entirely; only the limiter is under test.
        monkeypatch.setattr(
            app_main,
            "answer_question",
            lambda question, top_k=None: QueryResponse(answer="ok", grounded=True),
        )
        app_main.limiter.reset()

        return TestClient(app_main.app)

    yield _build

    from app.config import get_settings

    get_settings.cache_clear()


def _ask(client: TestClient, session_id: str | None = None, ip: str = "203.0.113.10"):
    payload: dict[str, object] = {"question": "What is PASHA?"}
    if session_id is not None:
        payload["session_id"] = session_id
    return client.post("/query", json=payload, headers={"x-forwarded-for": ip})


def test_shipped_default_limits_parse(monkeypatch):
    """The limits are strings read from the environment, so a typo would only
    surface on a live request. Pin the shipped defaults and prove they parse."""
    for var in ("QUERY_RATE_LIMIT", "QUERY_IP_RATE_LIMIT"):
        monkeypatch.delenv(var, raising=False)

    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()

    assert settings.query_rate_limit == "5/minute"
    assert [str(limit) for limit in parse_many(settings.query_rate_limit)] == [
        "5 per 1 minute"
    ]

    # The IP backstop needs both windows: a burst allowance and an hourly cap.
    assert [str(limit) for limit in parse_many(settings.query_ip_rate_limit)] == [
        "60 per 1 minute",
        "500 per 1 hour",
    ]


def test_session_is_limited_on_its_own_quota(make_client):
    client = make_client(session_limit="3/hour")

    for _ in range(3):
        assert _ask(client, session_id="session-a").status_code == 200

    assert _ask(client, session_id="session-a").status_code == 429


def test_rate_limited_response_says_when_to_retry(make_client):
    """The web client locks its composer for exactly the Retry-After it's given,
    so an absent header would strand the user on the caller's fallback guess
    rather than the real, sub-minute cooldown."""
    client = make_client(session_limit="2/minute")

    for _ in range(2):
        assert _ask(client, session_id="session-a").status_code == 200

    resp = _ask(client, session_id="session-a")
    assert resp.status_code == 429

    retry_after = int(resp.headers["retry-after"])
    assert 0 < retry_after <= 60, "a per-minute limit can't need more than a minute"
    # Same value in the body, for clients that don't surface headers.
    assert resp.json()["retry_after"] == retry_after


def test_sessions_do_not_share_a_bucket(make_client):
    """The point of the whole change: one visitor's questions must not consume
    another's quota. Both sessions arrive from the same IP."""
    client = make_client(session_limit="3/hour")

    for _ in range(3):
        assert _ask(client, session_id="session-a").status_code == 200
    assert _ask(client, session_id="session-a").status_code == 429

    # A different conversation from the same address starts fresh.
    assert _ask(client, session_id="session-b").status_code == 200


def test_ip_backstop_catches_session_rotation(make_client):
    """A caller minting a new session id per question escapes the session
    bucket, so the IP limit has to hold the line."""
    client = make_client(session_limit="100/hour", ip_limit="3/hour")

    for i in range(3):
        assert _ask(client, session_id=f"rotating-{i}").status_code == 200

    assert _ask(client, session_id="rotating-3").status_code == 429


def test_missing_session_id_falls_back_to_the_ip_bucket(make_client):
    """Direct API callers and older clients send no session id — they must still
    be limited, not waved through."""
    client = make_client(session_limit="3/hour")

    for _ in range(3):
        assert _ask(client).status_code == 200

    assert _ask(client).status_code == 429


def test_ip_fallback_is_per_caller(make_client):
    """The fallback keys on the forwarded client address, so two visitors
    without session ids don't share one quota."""
    client = make_client(session_limit="2/hour")

    for _ in range(2):
        assert _ask(client, ip="203.0.113.10").status_code == 200
    assert _ask(client, ip="203.0.113.10").status_code == 429

    assert _ask(client, ip="198.51.100.7").status_code == 200


def test_session_id_and_ip_buckets_do_not_collide(make_client):
    """Bucket keys are namespaced, so a session id that happens to look like an
    IP address can't drain that IP's quota."""
    client = make_client(session_limit="2/hour")

    for _ in range(2):
        assert _ask(client, session_id="203.0.113.10", ip="203.0.113.10").status_code == 200
    assert _ask(client, session_id="203.0.113.10", ip="203.0.113.10").status_code == 429

    # Same address, no session id -> untouched fallback bucket.
    assert _ask(client, ip="203.0.113.10").status_code == 200


def test_oversized_session_id_is_ignored_not_rejected(make_client):
    """An overlong id can't be allowed to bloat the limiter's key space, but it
    also mustn't cost the caller their answer. It's dropped, and they fall back
    to the IP bucket."""
    client = make_client(session_limit="2/hour")

    long_id = "x" * 5000
    assert _ask(client, session_id=long_id).status_code == 200
    # Counted against the IP bucket, so one more plain request exhausts it.
    assert _ask(client).status_code == 200
    assert _ask(client).status_code == 429
