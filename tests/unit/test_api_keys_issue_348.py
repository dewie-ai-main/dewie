"""Tests for Issue #348 — API keys should not render on the query page (app.html)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from dewie.api.middleware_base import InProcessLimiter, _rate_limit_key

# ── Minimal limiter fixture ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _disable_rate_limiting(monkeypatch):
    from dewie.api.middleware import limiter
    monkeypatch.setattr(limiter, "enabled", False)


def _fresh_limiter():
    return InProcessLimiter(key_func=_rate_limit_key)


# ── Fixtures ────────────────────────────────────────────────────────────────────

_TEST_USER_ID = "00000000-0000-0000-0000-000000000001"
_TEST_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


def _make_user_app(pg=None, authenticated=True):
    from fastapi import FastAPI

    from dewie.api.routes.user import router
    app = FastAPI()
    app.state.limiter = _fresh_limiter()
    app.include_router(router)

    if pg is None:
        pg = MagicMock()
    app.state.postgres = pg
    app.state.processor = AsyncMock()

    if authenticated:
        @app.middleware("http")
        async def _inject_session(request: Request, call_next):
            request.state.user_id = _TEST_USER_ID
            request.state.tenant_id = _TEST_TENANT_ID
            request.state.key_id = None
            return await call_next(request)

    return app


# ── Frontend HTML tests ────────────────────────────────────────────────────────


class TestAppHtmlNoApiKeys:
    """Verify app.html does NOT contain API key rendering logic."""

    def test_no_loadKeys_call(self):
        html_path = "static/app.html"
        with open(html_path) as f:
            content = f.read()
        assert "loadKeys()" not in content, "app.html should not call loadKeys()"

    def test_no_loadKeys_function(self):
        html_path = "static/app.html"
        with open(html_path) as f:
            content = f.read()
        assert "async function loadKeys(" not in content, "app.html should not define loadKeys()"

    def test_no_createKey_function(self):
        html_path = "static/app.html"
        with open(html_path) as f:
            content = f.read()
        assert "async function createKey(" not in content, "app.html should not define createKey()"

    def test_no_revokeKey_function(self):
        html_path = "static/app.html"
        with open(html_path) as f:
            content = f.read()
        assert "async function revokeKey(" not in content, "app.html should not define revokeKey()"

    def test_no_api_keys_section(self):
        html_path = "static/app.html"
        with open(html_path) as f:
            content = f.read()
        assert "api-keys" not in content, "app.html should not contain an api-keys section"


class TestAccountHtmlHasApiKeys:
    """Verify account.html DOES contain API key management."""

    def test_has_api_keys_section(self):
        html_path = "static/account.html"
        with open(html_path) as f:
            content = f.read()
        assert "api-keys" in content, "account.html should contain an api-keys section"

    def test_has_loadKeys_call(self):
        html_path = "static/account.html"
        with open(html_path) as f:
            content = f.read()
        assert "loadKeys()" in content, "account.html should call loadKeys()"


# ── Backend API key endpoint tests ─────────────────────────────────────────────


class TestApiKeysEndpoint:
    """Verify GET /user/api-keys works correctly."""

    def _make_engine_mock(self, fetchall_result=None):
        mock_result = MagicMock()
        mock_result.fetchall.return_value = fetchall_result or []

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_conn)
        cm.__aexit__ = AsyncMock(return_value=None)

        pg = AsyncMock()
        pg._engine = MagicMock()
        pg._engine.begin.return_value = cm
        pg._engine.connect.return_value = cm

        return pg, mock_conn, mock_result

    def test_authenticated_returns_200(self):
        pg, _, _ = self._make_engine_mock()
        app = _make_user_app(pg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/user/api-keys")
        assert resp.status_code == 200

    def test_unauthenticated_returns_401(self):
        app = _make_user_app(authenticated=False)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/user/api-keys")
        assert resp.status_code == 401
