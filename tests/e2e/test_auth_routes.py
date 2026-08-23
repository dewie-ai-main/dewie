"""E2E-style tests for auth route lifecycle endpoints.

These tests exercise request/response behavior for /auth routes with
application wiring, while mocking storage/backends.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dewie.api.middleware_base import limiter
from dewie.api.routes.auth import router


def _make_app(pg: AsyncMock | None = None) -> FastAPI:
    app = FastAPI()
    app.state.limiter = limiter
    app.state.postgres = pg or AsyncMock()
    app.include_router(router)
    return app


def test_signup_success_sets_cookie_and_returns_api_key():
    pg = AsyncMock()

    with (
        patch(
            "dewie.api.routes.auth.create_local_user",
            AsyncMock(
                return_value={
                    "id": "00000000-0000-0000-0000-000000000101",
                    "email": "new@example.com",
                    "name": "New User",
                    "is_admin": False,
                }
            ),
        ),
        patch("dewie.api.routes.auth.create_session_token", return_value="session-token"),
        patch(
            "dewie.auth.create_api_key",
            AsyncMock(return_value=("ck_live_test_key", {"key_prefix": "ck_live_"})),
        ),
    ):
        app = _make_app(pg)
        client = TestClient(app)
        resp = client.post(
            "/auth/signup",
            json={
                "email": "new@example.com",
                "username": "new@example.com",
                "password": "strongpass123",
            },
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["ok"] is True
    assert body["user_id"] == "00000000-0000-0000-0000-000000000101"
    assert body["api_key"] == "ck_live_test_key"
    assert "dewie_session" in resp.cookies


def test_signup_short_password_returns_400():
    app = _make_app()
    client = TestClient(app)

    resp = client.post(
        "/auth/signup",
        json={"email": "new@example.com", "username": "new@example.com", "password": "short"},
    )

    assert resp.status_code == 400
    assert "at least 8 characters" in resp.json().get("detail", "")


def test_login_invalid_credentials_returns_401():
    with patch("dewie.api.routes.auth.verify_local_user", AsyncMock(return_value=None)):
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/auth/login",
            json={"email": "new@example.com", "password": "wrongpass"},
        )

    assert resp.status_code == 401


def test_login_success_sets_session_cookie():
    with (
        patch(
            "dewie.api.routes.auth.verify_local_user",
            AsyncMock(
                return_value={
                    "id": "00000000-0000-0000-0000-000000000101",
                    "email": "new@example.com",
                    "name": "New User",
                    "is_admin": False,
                }
            ),
        ),
        patch("dewie.api.routes.auth.create_session_token", return_value="session-token"),
    ):
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/auth/login",
            json={"email": "new@example.com", "password": "strongpass123"},
        )

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert "dewie_session" in resp.cookies


def test_logout_revokes_session_and_clears_cookie():
    revoke_mock = AsyncMock()

    with patch("dewie.local_auth.revoke_session", revoke_mock):
        app = _make_app()
        client = TestClient(app)
        client.cookies.set("dewie_session", "session-token")
        resp = client.post("/auth/logout")

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    set_cookie = resp.headers.get("set-cookie", "")
    assert "dewie_session=" in set_cookie
    assert "Max-Age=0" in set_cookie
    revoke_mock.assert_awaited_once()


def test_signout_alias_revokes_session_and_clears_cookie():
    revoke_mock = AsyncMock()

    with patch("dewie.local_auth.revoke_session", revoke_mock):
        app = _make_app()
        client = TestClient(app)
        client.cookies.set("dewie_session", "session-token")
        resp = client.post("/auth/signout")

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    set_cookie = resp.headers.get("set-cookie", "")
    assert "dewie_session=" in set_cookie
    assert "Max-Age=0" in set_cookie
    revoke_mock.assert_awaited_once()


def test_change_password_requires_authentication():
    app = _make_app()
    client = TestClient(app)

    resp = client.post(
        "/auth/change-password",
        json={"current_password": "oldpass123", "new_password": "newpass123"},
    )

    assert resp.status_code == 401
    assert "Not authenticated" in resp.json().get("detail", "")
