"""Tests for authentication: login endpoint, cookie-based session, /auth/me."""
from __future__ import annotations

import os

import httpx

BASE_URL = "http://localhost:10946"
API_KEY = os.environ.get("BROWSER_TEST_API_KEY", "")
LOGIN_EMAIL = "dev@dewie.ai"
LOGIN_PASSWORD = "dewie-admin-2026"


def test_login_returns_200_with_cookie():
    """POST /auth/login succeeds and sets dewie_session cookie."""
    r = httpx.post(
        f"{BASE_URL}/auth/login",
        json={"email": LOGIN_EMAIL, "password": LOGIN_PASSWORD},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["ok"] is True
    assert data["email"] == LOGIN_EMAIL
    assert "dewie_session" in r.cookies, "Session cookie not set"


def test_auth_me_with_cookie_returns_real_user():
    """/auth/me with session cookie returns the real user, not 'Local mode'."""
    # Login first
    login = httpx.post(
        f"{BASE_URL}/auth/login",
        json={"email": LOGIN_EMAIL, "password": LOGIN_PASSWORD},
    )
    assert login.status_code == 200
    cookie = login.cookies["dewie_session"]

    # Use cookie for /auth/me
    me = httpx.get(
        f"{BASE_URL}/auth/me",
        cookies={"dewie_session": cookie},
    )
    assert me.status_code == 200
    data = me.json()
    assert data["email"] == LOGIN_EMAIL, f"Expected {LOGIN_EMAIL}, got {data['email']}"
    assert data["is_admin"] is True
    # Should NOT be the synthetic local-mode identity
    assert data["email"] != "local@dewie.ai", "Got synthetic local user instead of real user"


def test_auth_me_without_cookie_returns_local_mode():
    """/auth/me without credentials returns the local-mode identity (no redirect)."""
    r = httpx.get(f"{BASE_URL}/auth/me")
    assert r.status_code == 200
    data = r.json()
    assert data["is_admin"] is True  # local mode is always admin
    # Email is either the real user (if local_auth_enabled) or "Dewie Local Catalog"
    assert data["email"] is not None


def test_login_wrong_password_returns_401():
    r = httpx.post(
        f"{BASE_URL}/auth/login",
        json={"email": LOGIN_EMAIL, "password": "wrong-password"},
    )
    assert r.status_code == 401


def test_api_key_auth_works():
    """Requests with X-API-Key header are accepted."""
    r = httpx.get(
        f"{BASE_URL}/admin/users",
        headers={"X-API-Key": API_KEY},
    )
    assert r.status_code == 200


def test_missing_api_key_returns_401():
    """Requests without auth are rejected."""
    r = httpx.get(f"{BASE_URL}/admin/users")
    assert r.status_code == 401
