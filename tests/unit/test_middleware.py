"""Tests for dewie.api.middleware — pure helper functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_request(headers=None, client_host="1.2.3.4"):
    req = MagicMock()
    req.headers = headers or {}
    req.client = MagicMock()
    req.client.host = client_host
    return req


# ── _real_ip ──────────────────────────────────────────────────────────────────


def test_real_ip_uses_forwarded_for():
    from dewie.api.middleware import _real_ip

    # XFF is trusted only when connecting client is a private/loopback proxy
    req = _make_request({"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}, client_host="127.0.0.1")
    assert _real_ip(req) == "10.0.0.1"


def test_real_ip_ignores_forwarded_for_from_public_ip():
    from dewie.api.middleware import _real_ip

    # XFF from a public client must NOT be trusted (spoofing prevention)
    req = _make_request({"X-Forwarded-For": "1.2.3.4"}, client_host="5.6.7.8")
    assert _real_ip(req) == "5.6.7.8"


def test_real_ip_falls_back_to_remote_addr():
    from dewie.api.middleware import _real_ip

    req = _make_request({})
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "dewie.api.middleware.get_remote_address", return_value="5.6.7.8"
    ):
        ip = _real_ip(req)
    assert ip == "5.6.7.8"


def test_real_ip_single_forwarded_for():
    from dewie.api.middleware import _real_ip

    req = _make_request({"X-Forwarded-For": "203.0.113.1"}, client_host="127.0.0.1")
    assert _real_ip(req) == "203.0.113.1"


# ── _rate_limit_key ───────────────────────────────────────────────────────────


def test_rate_limit_key_uses_api_key_prefix():
    from dewie.api.middleware import _rate_limit_key

    req = _make_request({"X-API-Key": "ck_live_abcdefghijklmnopqrstuvwxyz"})
    key = _rate_limit_key(req)
    assert key.startswith("key:")
    assert "ck_live_abcdefg" in key  # first 16 chars


def test_rate_limit_key_uses_ip_when_no_api_key():
    from dewie.api.middleware import _rate_limit_key

    req = _make_request({})
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "dewie.api.middleware._real_ip", return_value="9.9.9.9"
    ):
        key = _rate_limit_key(req)
    assert key == "9.9.9.9"


def test_rate_limit_key_empty_api_key_falls_back_to_ip():
    from dewie.api.middleware import _rate_limit_key

    req = _make_request({"X-API-Key": ""})
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "dewie.api.middleware._real_ip", return_value="1.1.1.1"
    ):
        key = _rate_limit_key(req)
    assert key == "1.1.1.1"


# ── _api_key_middleware helpers (unit) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_api_key_middleware_no_auth_disabled_passes():
    """When auth_enabled=False and no API key, request passes through."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from starlette.responses import Response as StarletteResponse

    from dewie.api.middleware import _api_key_middleware

    req = MagicMock()
    req.url.path = "/search"
    req.headers.get.return_value = ""
    req.state = MagicMock()
    next_response = StarletteResponse(content="OK", status_code=200)
    call_next = AsyncMock(return_value=next_response)

    with patch("dewie.api.middleware.settings") as mock_settings:
        mock_settings.auth_enabled = False
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_api_key_middleware_local_auth_enabled_bypasses_key():
    """When local_auth_enabled=True, requests pass and local identity is injected."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from starlette.responses import Response as StarletteResponse

    from dewie.api.middleware import _api_key_middleware

    req = MagicMock()
    req.url.path = "/api/query"
    req.headers.get.return_value = ""
    req.state = MagicMock()
    next_response = StarletteResponse(content="OK", status_code=200)
    call_next = AsyncMock(return_value=next_response)

    with patch("dewie.api.middleware.settings") as mock_settings:
        mock_settings.local_auth_enabled = True
        mock_settings.local_auth_is_admin = True
        mock_settings.local_auth_user_id = "00000000-0000-0000-0000-000000000099"
        mock_settings.local_auth_email = "local@test"
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 200
    assert req.state.user_id == "00000000-0000-0000-0000-000000000099"
    assert req.state.email == "local@test"
    assert req.state.is_admin is True


@pytest.mark.asyncio
async def test_api_key_middleware_missing_key_returns_401():
    """When auth_enabled=True and no X-API-Key, returns 401."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.middleware import _api_key_middleware

    req = MagicMock()
    req.url.path = "/search"
    req.headers.get = lambda k, default="": default
    req.state = MagicMock()
    call_next = AsyncMock()

    with patch("dewie.api.middleware.settings") as mock_settings:
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_api_key_middleware_docs_path_requires_key_when_auth_enabled():
    """/docs should no longer be auth-exempt in secure-by-default mode."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.middleware import _api_key_middleware

    req = MagicMock()
    req.url.path = "/docs"
    req.headers.get = lambda k, default="": default
    req.state = MagicMock()
    call_next = AsyncMock()

    with patch("dewie.api.middleware.settings") as mock_settings:
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_api_key_middleware_invalid_key_returns_403():
    """Invalid X-API-Key returns 403."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.middleware import _api_key_middleware

    req = MagicMock()
    req.url.path = "/search"
    req.headers.get = lambda k, default="": "bad-key" if k == "X-API-Key" else default
    req.state = MagicMock()
    req.app.state.postgres = MagicMock()
    call_next = AsyncMock()

    with (
        patch("dewie.api.middleware.settings") as mock_settings,
        patch("dewie.auth.verify_api_key", AsyncMock(return_value=None)),
    ):
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_api_key_middleware_valid_key_passes_through():
    """Valid X-API-Key sets workspace_ids state and allows the request."""
    import uuid
    from unittest.mock import AsyncMock, MagicMock, patch

    from starlette.responses import Response as StarletteResponse

    from dewie.api.middleware import _api_key_middleware

    ws_id = uuid.uuid4()
    key_record = {"workspace_ids": [ws_id], "id": "key-123"}

    req = MagicMock()
    req.url.path = "/search"
    req.headers.get = lambda k, default="": "ck_live_validkey" if k == "X-API-Key" else default
    req.state = MagicMock()
    req.app.state.postgres = MagicMock()

    next_response = StarletteResponse(content="OK", status_code=200)
    call_next = AsyncMock(return_value=next_response)

    with (
        patch("dewie.api.middleware.settings") as mock_settings,
        patch("dewie.auth.verify_api_key", AsyncMock(return_value=key_record)),
    ):
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 200
    assert req.state.workspace_ids == [ws_id]
    assert req.state.key_id == "key-123"


# ── Authorization: Bearer fallback (needed for MCP clients) ──────────────────


@pytest.mark.asyncio
async def test_api_key_middleware_bearer_fallback_when_no_x_api_key():
    """Authorization: Bearer <key> authenticates when X-API-Key is absent."""
    import uuid
    from unittest.mock import AsyncMock, MagicMock, patch

    from starlette.responses import Response as StarletteResponse

    from dewie.api.middleware import _api_key_middleware

    ws_id = uuid.uuid4()
    key_record = {"workspace_ids": [ws_id], "id": "key-bearer"}

    req = MagicMock()
    req.url.path = "/search"
    req.headers.get = lambda k, default="": (
        "Bearer ck_live_validkey" if k == "Authorization" else default
    )
    req.state = MagicMock()
    req.app.state.postgres = MagicMock()

    next_response = StarletteResponse(content="OK", status_code=200)
    call_next = AsyncMock(return_value=next_response)
    captured_raw_key = {}

    async def _fake_verify(raw_key, pg):
        captured_raw_key["value"] = raw_key
        return key_record

    with (
        patch("dewie.api.middleware.settings") as mock_settings,
        patch("dewie.auth.verify_api_key", _fake_verify),
    ):
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 200
    assert captured_raw_key["value"] == "ck_live_validkey"
    assert req.state.key_id == "key-bearer"


@pytest.mark.asyncio
async def test_api_key_middleware_x_api_key_wins_over_bearer():
    """When both headers are present, X-API-Key takes precedence."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from starlette.responses import Response as StarletteResponse

    from dewie.api.middleware import _api_key_middleware

    req = MagicMock()
    req.url.path = "/search"

    def _headers_get(k, default=""):
        if k == "X-API-Key":
            return "ck_live_from_header"
        if k == "Authorization":
            return "Bearer ck_live_from_bearer"
        return default

    req.headers.get = _headers_get
    req.state = MagicMock()
    req.app.state.postgres = MagicMock()

    next_response = StarletteResponse(content="OK", status_code=200)
    call_next = AsyncMock(return_value=next_response)
    captured_raw_key = {}

    async def _fake_verify(raw_key, pg):
        captured_raw_key["value"] = raw_key
        return {"workspace_ids": [], "id": "key-x"}

    with (
        patch("dewie.api.middleware.settings") as mock_settings,
        patch("dewie.auth.verify_api_key", _fake_verify),
    ):
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 200
    assert captured_raw_key["value"] == "ck_live_from_header"


@pytest.mark.asyncio
async def test_api_key_middleware_invalid_bearer_returns_403():
    """An invalid Authorization: Bearer key still gets rejected with 403."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.middleware import _api_key_middleware

    req = MagicMock()
    req.url.path = "/search"
    req.headers.get = lambda k, default="": (
        "Bearer not-a-real-key" if k == "Authorization" else default
    )
    req.state = MagicMock()
    req.app.state.postgres = MagicMock()
    call_next = AsyncMock()

    with (
        patch("dewie.api.middleware.settings") as mock_settings,
        patch("dewie.auth.verify_api_key", AsyncMock(return_value=None)),
    ):
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_api_key_middleware_no_headers_still_401():
    """Missing both X-API-Key and Authorization still returns 401."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.middleware import _api_key_middleware

    req = MagicMock()
    req.url.path = "/search"
    req.headers.get = lambda k, default="": default
    req.state = MagicMock()
    call_next = AsyncMock()

    with patch("dewie.api.middleware.settings") as mock_settings:
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_api_key_middleware_ingest_requires_ingest_scope():
    """/api/ingest requires ingest (or admin) scope when auth is enabled."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.middleware import _api_key_middleware

    key_record = {"workspace_ids": [], "id": "key-1", "scopes": ["read"]}

    req = MagicMock()
    req.url.path = "/api/ingest"
    req.headers.get = lambda k, default="": "ck_live_valid" if k == "X-API-Key" else default
    req.state = MagicMock()
    req.app.state.postgres = MagicMock()
    call_next = AsyncMock()

    with (
        patch("dewie.api.middleware.settings") as mock_settings,
        patch("dewie.auth.verify_api_key", AsyncMock(return_value=key_record)),
    ):
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_api_key_middleware_query_requires_read_scope():
    """/api/query requires read (or admin) scope when auth is enabled."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.middleware import _api_key_middleware

    key_record = {"workspace_ids": [], "id": "key-2", "scopes": ["ingest"]}

    req = MagicMock()
    req.url.path = "/api/query"
    req.headers.get = lambda k, default="": "ck_live_valid" if k == "X-API-Key" else default
    req.state = MagicMock()
    req.app.state.postgres = MagicMock()
    call_next = AsyncMock()

    with (
        patch("dewie.api.middleware.settings") as mock_settings,
        patch("dewie.auth.verify_api_key", AsyncMock(return_value=key_record)),
    ):
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_api_key_middleware_traverse_requires_read_scope():
    """/api/traverse requires read (or admin) scope when auth is enabled."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.middleware import _api_key_middleware

    key_record = {"workspace_ids": [], "id": "key-4", "scopes": ["ingest"]}

    req = MagicMock()
    req.url.path = "/api/traverse"
    req.headers.get = lambda k, default="": "ck_live_valid" if k == "X-API-Key" else default
    req.state = MagicMock()
    req.app.state.postgres = MagicMock()
    call_next = AsyncMock()

    with (
        patch("dewie.api.middleware.settings") as mock_settings,
        patch("dewie.auth.verify_api_key", AsyncMock(return_value=key_record)),
    ):
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_api_key_middleware_capabilities_requires_read_scope():
    """/api/capabilities requires read (or admin) scope when auth is enabled."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.middleware import _api_key_middleware

    key_record = {"workspace_ids": [], "id": "key-5", "scopes": ["ingest"]}

    req = MagicMock()
    req.url.path = "/api/capabilities/probe"
    req.headers.get = lambda k, default="": "ck_live_valid" if k == "X-API-Key" else default
    req.state = MagicMock()
    req.app.state.postgres = MagicMock()
    call_next = AsyncMock()

    with (
        patch("dewie.api.middleware.settings") as mock_settings,
        patch("dewie.auth.verify_api_key", AsyncMock(return_value=key_record)),
    ):
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_api_key_middleware_sets_is_admin_from_scope():
    """Admin scope should mark request.state.is_admin=True."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from starlette.responses import Response as StarletteResponse

    from dewie.api.middleware import _api_key_middleware

    key_record = {"workspace_ids": [], "id": "key-3", "scopes": ["admin"]}

    req = MagicMock()
    req.url.path = "/api/query"
    req.headers.get = lambda k, default="": "ck_live_admin" if k == "X-API-Key" else default
    req.state = MagicMock()
    req.app.state.postgres = MagicMock()
    call_next = AsyncMock(return_value=StarletteResponse(content="OK", status_code=200))

    with (
        patch("dewie.api.middleware.settings") as mock_settings,
        patch("dewie.auth.verify_api_key", AsyncMock(return_value=key_record)),
    ):
        mock_settings.auth_enabled = True
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 200
    assert req.state.is_admin is True
