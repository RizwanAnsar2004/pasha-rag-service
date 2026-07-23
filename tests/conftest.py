"""Shared test fixtures."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Give every test a clean rate-limit ledger.

    The limiter's storage hangs off the module-level `Limiter` in app.main,
    which outlives any single test's app fixture. Without this, /query calls
    accumulate across the whole session and unrelated tests start failing with
    429 once the hourly quota is spent — a failure that depends on test order
    and is miserable to diagnose.

    Looked up through sys.modules rather than imported, so this fixture never
    forces app.main to load before a test's own fixtures have set up the
    environment it reads.
    """

    def _reset() -> None:
        module = sys.modules.get("app.main")
        if module is not None:
            module.limiter.reset()

    _reset()
    yield
    _reset()
