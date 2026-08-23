"""Tests for auth route exemptions in middleware.

These tests guard against regressions where the /api prefix is added to
routers but the middleware exempt list is not updated — causing the login
page to show 'Missing X-API-Key header' before the user can even sign in.
"""

from __future__ import annotations

import pytest

# ── Exempt path list ─────────────────────────────────────────────────────────

AUTH_PATHS_THAT_MUST_BE_EXEMPT = [
    # Legacy (no prefix) — kept for backwards compat
    "/auth/login",
    "/auth/signup",
    "/auth/me",
    "/auth/logout",
    "/auth/signout",
    "/auth/google",
    "/auth/google/callback",
    "/auth/apple",
    "/auth/apple/callback",
    # /api prefixed — added when routers got prefix="/api"
    "/api/auth/login",
    "/api/auth/signup",
    "/api/auth/me",
    "/api/auth/logout",
    "/api/auth/signout",
    "/api/auth/google",
    "/api/auth/google/callback",
    "/api/auth/apple",
    "/api/auth/apple/callback",
]

PATHS_THAT_MUST_NOT_BE_EXEMPT = [
    "/api/query",
    "/api/ingest",
    "/api/admin",
    "/api/documents",
]


@pytest.mark.parametrize("path", AUTH_PATHS_THAT_MUST_BE_EXEMPT)
def test_auth_path_is_in_exempt_list(path):
    """Every auth path must appear in _AUTH_EXEMPT_PREFIXES."""
    from dewie.api.middleware import _AUTH_EXEMPT_PREFIXES

    assert any(path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES), (
        f"{path!r} is not covered by _AUTH_EXEMPT_PREFIXES — "
        "login page will show 'Missing X-API-Key header'"
    )


@pytest.mark.parametrize("path", PATHS_THAT_MUST_NOT_BE_EXEMPT)
def test_protected_path_is_not_exempt(path):
    """API paths must NOT be in the exempt list."""
    from dewie.api.middleware import _AUTH_EXEMPT_PREFIXES

    assert not any(path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES), (
        f"{path!r} is incorrectly exempt from auth"
    )


# ── Middleware behaviour: auth routes pass without API key ───────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/auth/login", "/api/auth/signup", "/auth/login"])
async def test_auth_routes_pass_without_api_key(path):
    """Auth endpoints must not return 401/403 when no X-API-Key is provided."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from starlette.responses import Response as StarletteResponse

    from dewie.api.middleware import _api_key_middleware

    req = MagicMock()
    req.url.path = path
    req.headers.get = lambda k, default="": default  # no API key
    req.state = MagicMock()
    req.cookies.get = lambda k, default="": default  # no session cookie

    next_response = StarletteResponse(content="{}", status_code=200)
    call_next = AsyncMock(return_value=next_response)

    with patch("dewie.api.middleware.settings") as s:
        s.auth_enabled = True
        s.local_auth_enabled = False
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code == 200, (
        f"Login path {path!r} blocked with status {resp.status_code} — "
        "middleware is not exempting auth routes properly"
    )


@pytest.mark.asyncio
async def test_non_auth_route_blocked_without_key():
    """/api/query must be blocked when auth_enabled=True and no key provided."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.api.middleware import _api_key_middleware

    req = MagicMock()
    req.url.path = "/api/query"
    req.headers.get = lambda k, default="": default
    req.state = MagicMock()
    req.cookies.get = lambda k, default="": default

    call_next = AsyncMock()

    with patch("dewie.api.middleware.settings") as s:
        s.auth_enabled = True
        s.local_auth_enabled = False
        resp = await _api_key_middleware(req, call_next)

    assert resp.status_code in (401, 403)


# ── Live server smoke test (requires running dev server) ─────────────────────

@pytest.mark.integration
def test_login_page_no_api_key_error():
    """Login page must not show 'Missing X-API-Key header' on initial load."""
    import requests

    base = "http://localhost:10946"
    # The login page itself (static HTML) should load fine
    r = requests.get(f"{base}/ui/login.html", timeout=5)
    assert r.status_code == 200
    assert "Missing X-API-Key" not in r.text


@pytest.mark.integration
def test_login_endpoint_accessible_without_api_key():
    """POST /api/auth/login must not be blocked by API key middleware."""
    import requests

    r = requests.post(
        "http://localhost:10946/api/auth/login",
        json={"username": "admin", "password": "admin"},
        timeout=5,
    )
    # Should be 200 (success) or 401 (wrong creds) — NOT 403 Missing API key
    assert r.status_code != 403, f"Login blocked by API key middleware: {r.text}"
    assert "X-API-Key" not in r.text


@pytest.mark.integration
def test_admin_login_succeeds():
    """admin/admin must successfully authenticate on a fresh DB."""
    import requests

    r = requests.post(
        "http://localhost:10946/api/auth/login",
        json={"username": "admin", "password": "admin"},
        timeout=5,
    )
    assert r.status_code == 200, f"admin/admin login failed: {r.text}"
    data = r.json()
    assert data.get("ok") is True
    assert "user_id" in data


@pytest.mark.integration
def test_login_bad_password_returns_401_not_api_key_error():
    """Wrong password returns 401, not 403 API key error."""
    import requests

    r = requests.post(
        "http://localhost:10946/api/auth/login",
        json={"username": "admin", "password": "wrongpassword"},
        timeout=5,
    )
    assert r.status_code == 401
    assert "X-API-Key" not in r.text


@pytest.mark.integration
def test_signup_endpoint_accessible_without_api_key():
    """POST /api/auth/signup must not be blocked by API key middleware."""
    # Use a unique username to avoid conflicts
    import time

    import requests
    username = f"testuser_{int(time.time())}"
    r = requests.post(
        "http://localhost:10946/api/auth/signup",
        json={"username": username, "password": "testpass123"},
        timeout=5,
    )
    # 200 = created, 409 = already exists — both mean middleware passed
    assert r.status_code != 403, f"Signup blocked by API key middleware: {r.text}"
    assert "X-API-Key" not in r.text
