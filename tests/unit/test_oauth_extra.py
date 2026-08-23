"""Additional tests for dewie.oauth — functions not covered by test_oauth.py."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── verify_jwt / verify_bearer_token (legacy) ─────────────────────────────────


@pytest.mark.asyncio
async def test_verify_jwt_returns_none_when_oauth_disabled():
    from dewie.oauth import verify_jwt

    settings = MagicMock()
    settings.oauth_enabled = False
    result = await verify_jwt("any-token", settings)
    assert result is None


@pytest.mark.asyncio
async def test_verify_bearer_token_no_bearer_prefix():
    from dewie.oauth import verify_bearer_token

    settings = MagicMock()
    settings.oauth_enabled = False
    result = await verify_bearer_token("not-a-bearer-token", settings)
    assert result is None


@pytest.mark.asyncio
async def test_verify_bearer_token_valid_prefix_but_disabled():
    from dewie.oauth import verify_bearer_token

    settings = MagicMock()
    settings.oauth_enabled = False
    result = await verify_bearer_token("Bearer some-token", settings)
    assert result is None


# ── decode_apple_id_token ─────────────────────────────────────────────────────


def test_decode_apple_id_token():
    import jwt as _jwt

    from dewie.oauth import decode_apple_id_token

    # Create a minimal unsigned JWT-like payload using HS256 (for the decode-only test)
    payload = {"sub": "apple-user-123", "email": "user@example.com", "aud": "com.example.app"}
    token = _jwt.encode(payload, "fake-secret", algorithm="HS256")

    result = decode_apple_id_token(token)
    assert result["sub"] == "apple-user-123"


# ── OAuthTokenPayload ─────────────────────────────────────────────────────────


def test_oauth_token_payload_attributes():
    import uuid

    from dewie.oauth import OAuthTokenPayload

    tenant_id = uuid.uuid4()
    p = OAuthTokenPayload(
        sub="user-123",
        tenant_id=tenant_id,
        scopes=["read", "ingest"],
        exp=int(time.time()) + 3600,
    )
    assert p.sub == "user-123"
    assert p.tenant_id == tenant_id
    assert "read" in p.scopes


# ── get_google_auth_url ────────────────────────────────────────────────────────


def test_get_google_auth_url_contains_params():
    from dewie.oauth import get_google_auth_url

    url = get_google_auth_url(
        client_id="my-client", redirect_uri="https://example.com/callback", state="my-state"
    )
    assert "accounts.google.com" in url
    assert "my-client" in url
    assert "my-state" in url
    assert "response_type=code" in url


# ── get_apple_auth_url ────────────────────────────────────────────────────────


def test_get_apple_auth_url_contains_params():
    from dewie.oauth import get_apple_auth_url

    url = get_apple_auth_url(
        client_id="my.app.id", redirect_uri="https://example.com/callback", state="apple-state"
    )
    assert "appleid.apple.com" in url
    assert "my.app.id" in url
    assert "apple-state" in url
    assert "form_post" in url


# ── exchange_google_code ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exchange_google_code_success():
    from dewie.oauth import exchange_google_code

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"access_token": "tok", "id_token": "id"})

    client = AsyncMock()
    client.post = AsyncMock(return_value=resp)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=cm):
        result = await exchange_google_code("code", "client_id", "secret", "https://example.com/cb")

    assert result["access_token"] == "tok"


# ── get_google_userinfo ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_google_userinfo_success():
    from dewie.oauth import get_google_userinfo

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"sub": "12345", "email": "user@gmail.com"})

    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=cm):
        result = await get_google_userinfo("my-access-token")

    assert result["email"] == "user@gmail.com"
