# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
dewie/oauth.py — Google + Apple OAuth 2.0 and JWT web session management.

Session flow:
  Browser → GET /auth/google → Google consent → GET /auth/google/callback
         → upsert user in DB → issue HttpOnly JWT cookie → redirect to app

  Programmatic access: X-API-Key header (unchanged)
  Browser access: dewie_session cookie containing signed HS256 JWT

JWT payload (session):
  { user_id, tenant_id, email, is_admin, iat, exp }

Apple Sign In notes:
  - client_secret is a short-lived JWT signed with your .p8 private key
  - Apple uses POST for the callback (not GET)
  - User info (name, email) is only sent by Apple on the FIRST sign-in
  - Requires: APPLE_OAUTH_CLIENT_ID, APPLE_OAUTH_TEAM_ID,
              APPLE_OAUTH_KEY_ID, APPLE_OAUTH_PRIVATE_KEY
"""

from __future__ import annotations

import time
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Session JWT — HS256, issued by us, stored in HttpOnly cookie
# ---------------------------------------------------------------------------

_ALGORITHM = "HS256"

DEV_USER_ID = "00000000-0000-0000-0000-000000000002"
DEV_TENANT_ID = "00000000-0000-0000-0000-000000000001"


def create_session_token(
    user_id: str,
    tenant_id: str,
    email: str,
    is_admin: bool,
    secret: str,
    expire_days: int = 30,
    activation_status: str = "pending",
) -> str:
    """Issue a signed HS256 JWT for a browser session cookie."""
    import jwt as _jwt  # PyJWT

    now = int(time.time())
    payload = {
        "sub": user_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "email": email,
        "is_admin": is_admin,
        "activation_status": activation_status,
        "iat": now,
        "exp": now + expire_days * 86400,
    }
    return _jwt.encode(payload, secret, algorithm=_ALGORITHM)


def verify_session_token(token: str, secret: str) -> dict[str, Any] | None:
    """
    Validate a session JWT from a cookie.

    Returns the decoded payload dict or None if invalid/expired.
    """
    try:
        import jwt as _jwt

        payload = _jwt.decode(token, secret, algorithms=[_ALGORITHM])
        return payload
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Google OAuth 2.0
# ---------------------------------------------------------------------------

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
_GOOGLE_SCOPES = "openid email profile"


def get_google_auth_url(
    client_id: str,
    redirect_uri: str,
    state: str,
) -> str:
    """Build the Google OAuth consent URL."""
    from urllib.parse import urlencode

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": _GOOGLE_SCOPES,
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{_GOOGLE_AUTH_URL}?{urlencode(params)}"


async def exchange_google_code(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """Exchange an auth code for tokens. Returns the token response dict."""
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def get_google_userinfo(access_token: str) -> dict[str, Any]:
    """Fetch the authenticated user's profile from Google."""
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            _GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Apple Sign In (OAuth 2.0)
# ---------------------------------------------------------------------------

_APPLE_AUTH_URL = "https://appleid.apple.com/auth/authorize"
_APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
_APPLE_SCOPES = "name email"


def _build_apple_client_secret(
    client_id: str,
    team_id: str,
    key_id: str,
    private_key_pem: str,
) -> str:
    """
    Generate a short-lived Apple client_secret JWT (valid 6 months max).

    Apple requires the client_secret to be a JWT signed with your private key.
    See: https://developer.apple.com/documentation/sign_in_with_apple/generate_and_validate_tokens
    """
    import jwt as _jwt

    now = int(time.time())
    # Normalise PEM: allow \\n (literal backslash-n) from env vars
    pem = private_key_pem.replace("\\n", "\n").strip()

    payload = {
        "iss": team_id,
        "iat": now,
        "exp": now + 86400 * 180,  # 6 months
        "aud": "https://appleid.apple.com",
        "sub": client_id,
    }
    headers = {"kid": key_id, "alg": "ES256"}
    return _jwt.encode(payload, pem, algorithm="ES256", headers=headers)


def get_apple_auth_url(
    client_id: str,
    redirect_uri: str,
    state: str,
) -> str:
    """Build the Apple Sign In consent URL."""
    from urllib.parse import urlencode

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": _APPLE_SCOPES,
        "response_mode": "form_post",
        "state": state,
    }
    return f"{_APPLE_AUTH_URL}?{urlencode(params)}"


async def exchange_apple_code(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """Exchange Apple auth code for tokens."""
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _APPLE_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        resp.raise_for_status()
        return resp.json()


def decode_apple_id_token(id_token: str) -> dict[str, Any]:
    """
    Decode Apple's id_token without signature verification.

    The id_token is a JWT issued by Apple containing sub, email (if granted).
    We decode without verification here since we already exchanged the code
    with Apple's token endpoint — a successful exchange confirms authenticity.
    """
    import jwt as _jwt

    return _jwt.decode(id_token, options={"verify_signature": False})


# ---------------------------------------------------------------------------
# Legacy Clerk scaffold — kept for backwards compatibility with middleware
# (OAUTH_ENABLED=false means this is never called)
# ---------------------------------------------------------------------------


class OAuthTokenPayload:
    """Legacy: parsed Clerk JWT payload (unused with Google/Apple)."""

    def __init__(self, sub: str, tenant_id: uuid.UUID, scopes: list[str], exp: int) -> None:
        self.sub = sub
        self.tenant_id = tenant_id
        self.scopes = scopes
        self.exp = exp


async def verify_jwt(token: str, settings: Any) -> OAuthTokenPayload | None:
    """Legacy Clerk JWT validation. Returns None unless OAUTH_ENABLED=true."""
    if not settings.oauth_enabled:
        return None
    # Clerk integration removed — use Google/Apple SSO instead
    return None


async def verify_bearer_token(authorization: str, settings: Any) -> OAuthTokenPayload | None:
    """Legacy Clerk Bearer token check. Returns None unless OAUTH_ENABLED=true."""
    if not authorization.startswith("Bearer "):
        return None
    token = authorization[len("Bearer ") :]
    return await verify_jwt(token, settings)
