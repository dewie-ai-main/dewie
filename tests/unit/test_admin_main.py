"""Tests for dewie.admin_main — internal admin FastAPI app."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_admin_app(pg: MagicMock | None = None, cache: MagicMock | None = None) -> object:
    """Create a fresh admin_app with state pre-populated (bypasses lifespan)."""
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse

    from dewie.admin_main import _admin_session_middleware
    from dewie.api.routes.health import router as health_router

    pg = pg or AsyncMock()
    cache = cache or AsyncMock()

    app = FastAPI()
    app.middleware("http")(_admin_session_middleware)
    app.include_router(health_router)

    @app.get("/health")
    async def health():  # type: ignore[override]
        return {"status": "ok"}

    @app.get("/admin/ping")
    async def ping():
        return {"ping": "pong"}

    @app.get("/")
    async def root():
        return RedirectResponse(url="/ui/admin.html", status_code=302)

    app.state.postgres = pg
    app.state.cache = cache
    app.state.processor = None
    return app


# ── Middleware: health always passes ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_no_auth_required():
    """GET /health should return 200 without any credentials."""
    app = _make_admin_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200


# ── Middleware: X-Admin-Key ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_admin_key_header_grants_access():
    """X-Admin-Key with the correct value grants access."""
    app = _make_admin_app()
    with patch.dict(os.environ, {"ADMIN_KEY": "secret-test-key"}):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/ping", headers={"X-Admin-Key": "secret-test-key"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_admin_key_wrong_value_returns_401():
    """X-Admin-Key with wrong value is rejected."""
    app = _make_admin_app()
    with patch.dict(os.environ, {"ADMIN_KEY": "secret-test-key"}):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/ping", headers={"X-Admin-Key": "wrong-key"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_no_auth_returns_401():
    """Request without any credentials returns 401."""
    app = _make_admin_app()
    with patch.dict(os.environ, {"ADMIN_KEY": ""}):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/ping")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_no_auth_browser_gets_401():
    """Browser request without auth gets 401 (no redirect)."""
    app = _make_admin_app()
    with patch.dict(os.environ, {"ADMIN_KEY": ""}):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", follow_redirects=False
        ) as client:
            resp = await client.get(
                "/admin/ping",
                headers={"Accept": "text/html,application/xhtml+xml"},
            )
    assert resp.status_code == 401


# ── Middleware: static paths pass through ─────────────────────────────────────


@pytest.mark.asyncio
async def test_ui_html_requires_auth_redirect():
    """Requests to /ui/*.html without auth are redirected to /ui/login.html.
    Server-side enforcement replaces the old client-side-JS-only redirect.
    """
    app = _make_admin_app()

    @app.get("/ui/test.html")
    async def ui_page():
        return {"page": "admin"}

    with patch.dict(os.environ, {"ADMIN_KEY": ""}):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", follow_redirects=False
        ) as client:
            resp = await client.get("/ui/test.html")
    assert resp.status_code == 302
    assert resp.headers["location"] == "/ui/login.html"


@pytest.mark.asyncio
async def test_ui_non_html_static_passes_without_auth():
    """Non-HTML static assets (/ui/*.js, /ui/*.css) pass through without auth
    so the login page can load its own assets.
    """
    app = _make_admin_app()

    @app.get("/ui/app.js")
    async def ui_js():
        return {"js": True}

    with patch.dict(os.environ, {"ADMIN_KEY": ""}):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ui/app.js")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_session_cookie_admin_grants_access():
    """Valid dewie_session cookie with is_admin=True grants access."""
    from dewie.local_auth import create_session_token

    app = _make_admin_app()
    token = create_session_token(
        user_id="test-user-id",
        email="admin@example.com",
        is_admin=True,
    )
    with patch.dict(os.environ, {"ADMIN_KEY": ""}):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies={"dewie_session": token},
        ) as client:
            resp = await client.get("/admin/ping")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_session_cookie_non_admin_returns_401():
    """Valid dewie_session cookie with is_admin=False is rejected."""
    from dewie.local_auth import create_session_token

    app = _make_admin_app()
    token = create_session_token(
        user_id="test-user-id",
        email="user@example.com",
        is_admin=False,
    )
    with patch.dict(os.environ, {"ADMIN_KEY": ""}):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies={"dewie_session": token},
        ) as client:
            resp = await client.get("/admin/ping")
    assert resp.status_code == 401


# ── admin_app integration ─────────────────────────────────────────────────────


def test_admin_app_has_correct_metadata():
    """admin_app should have the expected title and no public docs."""
    from dewie.admin_main import admin_app

    assert admin_app.title == "Dewie Admin"
    assert admin_app.docs_url is None
    assert admin_app.redoc_url is None
    assert admin_app.openapi_url is None
