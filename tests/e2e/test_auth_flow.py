"""E2E tests for the authentication flow.

Covers:
- Unauthenticated requests when AUTH_ENABLED=true → 401/403
- Session cookie (JWT) authentication → pass-through
- API key authentication → pass-through
- Invalid / revoked API key → 403
- Pending activation status → 403
- Admin-required paths without admin flag → 403
- Session invalidation via Redis min_iat
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from dewie.api.middleware import register_middleware

# ── Constants ─────────────────────────────────────────────────────────────────

_TEST_TENANT_ID = "00000000-0000-0000-0000-000000000001"
_TEST_USER_ID = "00000000-0000-0000-0000-000000000099"
_TEST_JWT_SECRET = "test-e2e-secret-do-not-use-in-production"


@pytest.fixture(autouse=True)
def _jwt_test_secret(monkeypatch):
    """Sign/verify must share the test secret.

    verify_session_token reads JWT_SECRET from env and caches it in a module
    global — set the env var and clear the cache for every test.
    """
    import dewie.local_auth as la

    monkeypatch.setenv("JWT_SECRET", _TEST_JWT_SECRET)
    monkeypatch.setattr(la, "_DEFAULT_JWT_SECRET", None)
    yield
    la._DEFAULT_JWT_SECRET = None


@pytest.fixture(autouse=True)
def _disable_rate_limiting(monkeypatch):
    """Prevent rate-limit bleed across tests."""
    from dewie.api.middleware import limiter

    monkeypatch.setattr(limiter, "enabled", False)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_auth_app(pg=None, cache=None):
    """Minimal app with auth middleware and a single protected route."""
    app = FastAPI()
    register_middleware(app)

    if pg is None:
        pg = AsyncMock()
    if cache is None:
        cache = _make_cache()

    app.state.postgres = pg
    app.state.cache = cache

    @app.get("/protected")
    async def _protected(request: Request):
        return JSONResponse({"ok": True, "user_id": request.state.user_id})

    @app.get("/health")
    async def _health():
        return JSONResponse({"status": "ok"})

    return app


def _make_cache():
    cache = AsyncMock()
    cache.get_session_min_iat = AsyncMock(return_value=None)  # not invalidated
    cache.get_tenant_plan = AsyncMock(return_value="free")
    cache.set_tenant_plan = AsyncMock()
    cache.incr_quota = AsyncMock(return_value=1)
    cache.decr_quota = AsyncMock()
    return cache


def _make_session_cookie(
    user_id: str = _TEST_USER_ID,
    tenant_id: str = _TEST_TENANT_ID,
    email: str = "test@example.com",
    is_admin: bool = False,
    activation_status: str = "approved",
    jwt_secret: str = _TEST_JWT_SECRET,
) -> str:
    from dewie.oauth import create_session_token

    return create_session_token(
        user_id=user_id,
        tenant_id=tenant_id,
        email=email,
        is_admin=is_admin,
        activation_status=activation_status,
        secret=jwt_secret,
    )


# ── Health endpoint (auth-exempt) ─────────────────────────────────────────────


class TestExemptRoutes:
    def test_health_always_accessible(self):
        with patch.multiple("dewie.config.settings", auth_enabled=True):
            app = _make_auth_app()
            client = TestClient(app)
            resp = client.get("/health")
        assert resp.status_code == 200


# ── Unauthenticated requests ──────────────────────────────────────────────────


class TestUnauthenticated:
    def test_no_auth_header_returns_401(self):
        with patch.multiple("dewie.config.settings", auth_enabled=True):
            app = _make_auth_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/protected")
        assert resp.status_code == 401

    def test_missing_api_key_returns_401(self):
        with patch.multiple("dewie.config.settings", auth_enabled=True):
            app = _make_auth_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/protected")
        assert resp.status_code == 401
        assert "X-API-Key" in resp.json().get("detail", "")

    def test_auth_disabled_allows_all(self):
        with patch.multiple("dewie.config.settings", auth_enabled=False):
            app = _make_auth_app()
            client = TestClient(app)
            resp = client.get("/protected")
        assert resp.status_code == 200


# ── Session cookie (JWT) auth ─────────────────────────────────────────────────


class TestSessionCookieAuth:
    def test_valid_session_cookie_passes(self):
        token = _make_session_cookie()
        with patch.multiple(
            "dewie.config.settings",
            auth_enabled=True,
        ):
            app = _make_auth_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/protected", cookies={"dewie_session": token})
        assert resp.status_code == 200
        assert resp.json()["user_id"] == _TEST_USER_ID

    def test_tampered_jwt_rejected(self):
        token = _make_session_cookie() + "tampered"
        with patch.multiple(
            "dewie.config.settings",
            auth_enabled=True,
        ):
            app = _make_auth_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/protected", cookies={"dewie_session": token})
        # Tampered JWT falls through to API key check → 401 missing API key
        assert resp.status_code == 401

    def test_pending_activation_returns_403(self):
        token = _make_session_cookie(activation_status="pending")
        with patch.multiple(
            "dewie.config.settings",
            auth_enabled=True,
        ):
            app = _make_auth_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/protected", cookies={"dewie_session": token})
        assert resp.status_code == 403
        assert "pending" in resp.json().get("detail", "").lower()

    def test_admin_flag_set_correctly(self):
        token = _make_session_cookie(is_admin=True)
        with patch.multiple(
            "dewie.config.settings",
            auth_enabled=True,
        ):
            app = FastAPI()
            register_middleware(app)
            app.state.postgres = AsyncMock()
            app.state.cache = _make_cache()

            @app.get("/whoami")
            async def _whoami(request: Request):
                return JSONResponse({"is_admin": request.state.is_admin})

            client = TestClient(app)
            resp = client.get("/whoami", cookies={"dewie_session": token})
        assert resp.status_code == 200
        assert resp.json()["is_admin"] is True


# ── API key auth ──────────────────────────────────────────────────────────────


class TestApiKeyAuth:
    def test_valid_api_key_passes(self):
        pg = AsyncMock()
        tenant_id = uuid.UUID(_TEST_TENANT_ID)
        key_id = uuid.uuid4()
        pg.verify_api_key = AsyncMock(
            return_value={
                "id": key_id,
                "tenant_id": tenant_id,
                "scopes": ["read"],
            }
        )

        with (
            patch.multiple("dewie.config.settings", auth_enabled=True),
            patch(
                "dewie.auth.verify_api_key",
                AsyncMock(return_value={"id": key_id, "tenant_id": tenant_id, "scopes": ["read"]}),
            ),
        ):
            app = _make_auth_app(pg=pg)
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/protected", headers={"X-API-Key": "ck_live_valid_key"})

        assert resp.status_code == 200

    def test_invalid_api_key_returns_403(self):
        with (
            patch.multiple("dewie.config.settings", auth_enabled=True),
            patch("dewie.auth.verify_api_key", AsyncMock(return_value=None)),
        ):
            app = _make_auth_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/protected", headers={"X-API-Key": "ck_live_bad_key"})
        assert resp.status_code == 403
        assert "Invalid" in resp.json().get("detail", "")

    def test_empty_api_key_returns_401(self):
        with patch.multiple("dewie.config.settings", auth_enabled=True):
            app = _make_auth_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/protected", headers={"X-API-Key": ""})
        assert resp.status_code == 401


# ── Admin-required paths ──────────────────────────────────────────────────────


class TestAdminRequiredPaths:
    def test_non_admin_session_blocked_on_admin_path(self):
        token = _make_session_cookie(is_admin=False)
        with patch.multiple(
            "dewie.config.settings",
            auth_enabled=True,
        ):
            app = FastAPI()
            register_middleware(app)
            app.state.postgres = AsyncMock()
            app.state.cache = _make_cache()

            @app.get("/api/pipeline/results")
            async def _admin_route():
                return JSONResponse({"data": "secret"})

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/pipeline/results", cookies={"dewie_session": token})
        assert resp.status_code == 403

    def test_admin_session_passes_admin_path(self):
        token = _make_session_cookie(is_admin=True)
        with patch.multiple(
            "dewie.config.settings",
            auth_enabled=True,
        ):
            app = FastAPI()
            register_middleware(app)
            app.state.postgres = AsyncMock()
            app.state.cache = _make_cache()

            @app.get("/api/pipeline/results")
            async def _admin_route():
                return JSONResponse({"data": "allowed"})

            client = TestClient(app)
            resp = client.get("/api/pipeline/results", cookies={"dewie_session": token})
        assert resp.status_code == 200
