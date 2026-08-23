"""Tests for dewie.oauth — JWT session management and OAuth helpers."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dewie.oauth import (
    DEV_TENANT_ID,
    DEV_USER_ID,
    OAuthTokenPayload,
    create_session_token,
    decode_apple_id_token,
    get_apple_auth_url,
    get_google_auth_url,
    verify_bearer_token,
    verify_jwt,
    verify_session_token,
)

SECRET = "test-secret-key-for-unit-tests"


# ── create_session_token / verify_session_token ───────────────────────────────


def test_create_and_verify_session_token():
    token = create_session_token(
        user_id="user-123",
        tenant_id="tenant-456",
        email="test@example.com",
        is_admin=False,
        secret=SECRET,
    )
    payload = verify_session_token(token, SECRET)
    assert payload is not None
    assert payload["user_id"] == "user-123"
    assert payload["tenant_id"] == "tenant-456"
    assert payload["email"] == "test@example.com"
    assert payload["is_admin"] is False


def test_create_session_token_admin():
    token = create_session_token("u", "t", "a@b.com", True, SECRET)
    payload = verify_session_token(token, SECRET)
    assert payload["is_admin"] is True


def test_create_session_token_activation_status():
    token = create_session_token("u", "t", "a@b.com", False, SECRET, activation_status="approved")
    payload = verify_session_token(token, SECRET)
    assert payload["activation_status"] == "approved"


def test_verify_session_token_wrong_secret():
    token = create_session_token("u", "t", "a@b.com", False, SECRET)
    result = verify_session_token(token, "wrong-secret")
    assert result is None


def test_verify_session_token_invalid():
    result = verify_session_token("not.a.token", SECRET)
    assert result is None


def test_verify_session_token_expired():
    import jwt as _jwt

    now = int(time.time())
    payload = {
        "sub": "u",
        "user_id": "u",
        "tenant_id": "t",
        "email": "a@b.com",
        "is_admin": False,
        "activation_status": "approved",
        "iat": now - 7200,
        "exp": now - 3600,  # expired 1 hour ago
    }
    token = _jwt.encode(payload, SECRET, algorithm="HS256")
    result = verify_session_token(token, SECRET)
    assert result is None


def test_session_token_contains_sub():
    token = create_session_token("user-789", "t", "x@y.com", False, SECRET)
    payload = verify_session_token(token, SECRET)
    assert payload["sub"] == "user-789"


def test_dev_constants():
    assert DEV_USER_ID == "00000000-0000-0000-0000-000000000002"
    assert DEV_TENANT_ID == "00000000-0000-0000-0000-000000000001"


# ── get_google_auth_url ───────────────────────────────────────────────────────


def test_get_google_auth_url_contains_params():
    url = get_google_auth_url("my-client-id", "https://example.com/callback", "mystate")
    assert "accounts.google.com" in url
    assert "my-client-id" in url
    assert "mystate" in url
    assert "openid" in url


def test_get_google_auth_url_redirect_uri_encoded():
    url = get_google_auth_url("cid", "https://example.com/cb?x=1", "s")
    assert "accounts.google.com" in url


# ── get_apple_auth_url ────────────────────────────────────────────────────────


def test_get_apple_auth_url_contains_params():
    url = get_apple_auth_url("my-apple-client", "https://example.com/apple/cb", "applestate")
    assert "appleid.apple.com" in url
    assert "my-apple-client" in url
    assert "applestate" in url
    assert "form_post" in url


# ── decode_apple_id_token ─────────────────────────────────────────────────────


def test_decode_apple_id_token():
    import jwt as _jwt

    payload = {
        "sub": "apple-user-123",
        "email": "user@privaterelay.appleid.com",
        "iat": int(time.time()),
    }
    # Sign with HS256 just to create a valid JWT structure; decode without verify
    token = _jwt.encode(payload, "dummy", algorithm="HS256")
    result = decode_apple_id_token(token)
    assert result["sub"] == "apple-user-123"


# ── OAuthTokenPayload ─────────────────────────────────────────────────────────


def test_oauth_token_payload():
    import uuid

    tid = uuid.uuid4()
    p = OAuthTokenPayload("sub-123", tid, ["read"], int(time.time()) + 3600)
    assert p.sub == "sub-123"
    assert p.tenant_id == tid
    assert "read" in p.scopes


# ── verify_jwt (legacy) ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_jwt_returns_none_when_disabled():
    settings = MagicMock()
    settings.oauth_enabled = False
    result = await verify_jwt("any-token", settings)
    assert result is None


# ── verify_bearer_token (legacy) ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_bearer_token_returns_none_when_no_bearer():
    settings = MagicMock()
    settings.oauth_enabled = False
    result = await verify_bearer_token("Basic abc123", settings)
    assert result is None


@pytest.mark.asyncio
async def test_verify_bearer_token_returns_none_when_disabled():
    settings = MagicMock()
    settings.oauth_enabled = False
    result = await verify_bearer_token("Bearer some-token", settings)
    assert result is None


# ── exchange_google_code (mocked HTTP) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_exchange_google_code_success():
    from dewie.oauth import exchange_google_code

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"access_token": "tok123"}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_cm):
        result = await exchange_google_code("code", "cid", "csecret", "https://cb.com")

    assert result["access_token"] == "tok123"


@pytest.mark.asyncio
async def test_get_google_userinfo_success():
    from dewie.oauth import get_google_userinfo

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"email": "user@example.com", "name": "Test User"}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_cm):
        result = await get_google_userinfo("access_tok_123")

    assert result["email"] == "user@example.com"
