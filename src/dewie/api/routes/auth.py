# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""Lightweight auth/session endpoints for browser UI compatibility."""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
import time
from datetime import UTC
from typing import Any

log = logging.getLogger("dewie.api")

# ── Helpers ────────────────────────────────────────────────────────────────────────

def _extract_request_id(request: Request) -> str:  # noqa: F821
    """Extract request_id from request state, falling back to 'unknown'."""
    return getattr(request.state, "request_id", "unknown")


def _redact(value: str | None) -> str | None:
    """Redact a potentially sensitive string value for logging."""
    if value is None:
        return None
    if len(value) > 1000:
        value = value[:1000] + "... [truncated]"
    sensitive_fields = ("api_key", "password", "token", "secret", "authorization")
    for field in sensitive_fields:
        if field in value.lower():
            return "***REDACTED***"
    return value

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from dewie.api.middleware_base import limiter, rate_limit
from dewie.config import settings
from dewie.local_auth import (
    create_local_user,
    create_session_token,
    hash_password,
    verify_local_user,
)
from dewie.utils.email import send_password_reset_email

router = APIRouter(prefix="/auth", tags=["auth"])

# ── Bot detection patterns ──────────────────────────────────────────────────

# Patterns that indicate automated/bot signups (e.g. testuser_{unix_timestamp})
_SUSPICIOUS_EMAIL_PATTERNS: tuple[str, ...] = (
    r"^testuser_",           # testuser_{anything} — was used by AI reviewer run
    r"^test_\d+$",           # test_{digits} only
    r"^\d{10,}$",            # 10+ consecutive digits (looks like unix timestamp)
)
_re_suspicious = re.compile("|".join(f"(?:{p})" for p in _SUSPICIOUS_EMAIL_PATTERNS), re.IGNORECASE)


# ── Request/Response schemas ──────────────────────────────────────────────────


class SignupRequest(BaseModel):
    email: str = Field(default="", description="Username or email address for login")
    username: str = Field(default="", description="Alias for email — accepted from frontend")

    @model_validator(mode="before")
    @classmethod
    def _coerce_username(cls, data: Any) -> Any:
        """Accept 'username' as alias for 'email' so the frontend doesn't need updating."""
        if isinstance(data, dict) and not data.get("email") and data.get("username"):
            data = dict(data)
            data["email"] = data["username"]
        return data

    @model_validator(mode="after")
    def _reject_bot_emails(self) -> SignupRequest:
        """Reject bot-like signup patterns (e.g. testuser_{unix_timestamp})."""
        email = (self.email or "").strip().lower()
        if _re_suspicious.search(email):
            raise ValueError(
                "This email address pattern is not allowed. Please use a valid email address."
            )
        return self

    password: str = Field(description="Password (min 8 characters)")
    name: str | None = Field(default=None, description="Optional display name")


class LoginRequest(BaseModel):
    email: str = Field(default="", description="Username or email address")
    username: str = Field(default="", description="Alias for email — accepted from frontend")

    @model_validator(mode="before")
    @classmethod
    def _coerce_username(cls, data: Any) -> Any:
        """Accept 'username' as alias for 'email' so the frontend doesn't need updating."""
        if isinstance(data, dict) and not data.get("email") and data.get("username"):
            data = dict(data)
            data["email"] = data["username"]
        return data
    password: str = Field(description="Password")


class ForgotPasswordRequest(BaseModel):
    username: str = Field(description="Username or email address")


class ResetPasswordRequest(BaseModel):
    reset_token: str = Field(description="Reset token from email")
    password: str = Field(description="New password (min 8 characters)")


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(description="Current password")
    new_password: str = Field(description="New password (min 8 characters)")


class AuthResponse(BaseModel):
    ok: bool
    user_id: str | None = None
    email: str | None = None
    name: str | None = None


@router.get("/me")
async def auth_me(request: Request) -> dict:  # type: ignore[type-arg]
    """Return current session identity for UI pages.

    If no authenticated session is present (common in local OSS mode), return
    a synthetic local profile so UI pages remain usable.

    The ``auth_method`` field in the response uses the following priority order:

    1. ``password`` — user has a bcrypt password_hash set (wins even if OAuth
       identifiers are also present, e.g. dev@dewie.ai seeded with both).
    2. ``google``  — user authenticated via Google OAuth (google_sub set).
    3. ``apple``   — user authenticated via Apple Sign In (apple_sub set).

    This priority was introduced to fix issue #244 where dev@dewie.ai was
    incorrectly reported as ``auth_method='google'`` after a password was
    added in a schema migration.
    """
    request_id = _extract_request_id(request)
    log.info("session_verify started request_id=%s", request_id)
    t0 = time.monotonic()

    user_id = getattr(request.state, "user_id", None)
    email = getattr(request.state, "email", None)
    is_admin = bool(getattr(request.state, "is_admin", False))
    activation_status = getattr(request.state, "activation_status", "approved")

    # /auth/me is on the exempt list so middleware skips cookie-reading.
    # Read the session cookie directly here so the browser admin panel works.
    if not user_id:
        session_token = request.cookies.get("dewie_session", "").strip()
        if session_token:
            log.info("session_verify cookie_found request_id=%s", request_id)
            from dewie.local_auth import verify_session_token
            payload = verify_session_token(session_token)
            if payload:
                user_id = payload.get("sub")
                email = payload.get("email")
                is_admin = bool(payload.get("is_admin", False))
                activation_status = "approved"
                log.info("session_verify succeeded request_id=%s user_id=%s elapsed_ms=%.2f", request_id, user_id, round((time.monotonic() - t0) * 1000, 2))
            else:
                log.info("session_verify cookie_invalid request_id=%s", request_id)
        else:
            log.info("session_verify no_cookie request_id=%s", request_id)
    elif not user_id and not getattr(request.state, "user_id", None):
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info("session_verify succeeded request_id=%s authenticated=false elapsed_ms=%.2f", request_id, elapsed)

    if user_id:
        # Fetch real user name and auth method from DB
        name = None
        auth_method = "password"  # default for session-cookie (local) users
        try:
            pg = request.app.state.postgres
            async with pg._engine.connect() as conn:
                from sqlalchemy import text as _text
                row = (await conn.execute(
                    _text(
                        "SELECT name, is_admin, google_sub, apple_sub, "
                        "CASE WHEN password_hash IS NOT NULL AND password_hash != '' "
                        "THEN true ELSE false END AS has_password "
                        "FROM users WHERE id = :id LIMIT 1"
                    ),
                    {"id": str(user_id)}
                )).mappings().fetchone()
                if row:
                    name = row["name"]
                    # Always read is_admin from DB — the JWT value can be stale
                    # (e.g. user created via signup before seed_default_admin ran,
                    # or promoted to admin after their last login).
                    is_admin = bool(row["is_admin"])
                    # Priority: explicit password > Google OAuth > Apple Sign In
                    # A user can have OAuth identifiers (google_sub/apple_sub) AND a
                    # password (e.g. dev@dewie.ai seeded with both). Show the active
                    # credential — password wins if set, then check OAuth providers.
                    if row["has_password"]:
                        auth_method = "password"
                    elif row["google_sub"]:
                        auth_method = "google"
                    elif row["apple_sub"]:
                        auth_method = "apple"
        except Exception as exc:  # noqa: BLE001
            log.warning("auth_me: failed to fetch user auth details for %s: %s", user_id, exc)
        return {
            "user_id": str(user_id),
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "email": email or "user@local",
            "name": name,
            "picture": None,
            "is_admin": is_admin,
            "plan": "free",
            "activation_status": activation_status or "approved",
            "auth_method": auth_method,
            "authenticated": True,
        }

    # When auth is enabled and the request has no valid session, return an
    # explicit not-authenticated marker instead of a synthetic local user.
    # The synthetic local-mode identity is only appropriate in open/dev mode
    # (AUTH_ENABLED=false or LOCAL_AUTH_ENABLED=true), where every request
    # already passes through middleware as an authenticated local user.
    # Returning a fake identity here caused issue #219: unauthenticated users
    # appeared logged-in on the home page and were let past the /ui/ guard
    # into app.html, where all API calls then failed with 401 because no real
    # session cookie was present.
    from dewie.config import settings as _settings
    if _settings.auth_enabled and not _settings.local_auth_enabled:
        return {
            "authenticated": False,
            "user_id": None,
            "email": None,
            "name": None,
            "picture": None,
            "is_admin": False,
            "plan": None,
            "activation_status": None,
        }

    return {
        "user_id": "00000000-0000-0000-0000-000000000002",
        "tenant_id": "00000000-0000-0000-0000-000000000001",
        "email": getattr(_settings, "local_auth_email", "Dewie Local Catalog"),
        "name": "Local mode",
        "picture": None,
        "is_admin": True,
        "plan": "local",
        "activation_status": "approved",
        "auth_method": "local",
        "authenticated": True,
    }


@router.post("/signup", status_code=201)
@limiter.limit(rate_limit(5))
async def auth_signup(body: SignupRequest, request: Request) -> JSONResponse:
    """Create a new local user account with username/password.

    Password must be at least 8 characters.
    The username field accepts any non-empty string and does not need to be an email address.
    Returns HTTP 201 + JWT session cookie on success.
    """
    request_id = _extract_request_id(request)
    log.info("signup started request_id=%s email=%s", request_id, body.email)
    t0 = time.monotonic()
    pg = request.app.state.postgres

    if len(body.password) < 8:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.warning("signup failed request_id=%s reason=short_password elapsed_ms=%.2f", request_id, elapsed)
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    if not body.username:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.warning("signup failed request_id=%s reason=missing_username elapsed_ms=%.2f", request_id, elapsed)
        raise HTTPException(status_code=400, detail="Username is required")

    try:
        user = await create_local_user(pg, body.email, body.password, body.name)
    except ValueError as e:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.warning("signup failed request_id=%s reason=email_exists email=%s elapsed_ms=%.2f", request_id, body.email, elapsed)
        raise HTTPException(status_code=409, detail=str(e)) from e

    # Create session token
    token = create_session_token(user["id"], user["email"], user["is_admin"])

    # Auto-generate default API key
    import uuid

    from dewie.auth import create_api_key

    raw_key, key_record = await create_api_key(
        pg,
        user_id=uuid.UUID(user["id"]),
        name="default",
        scopes=["read"],
    )

    elapsed = round((time.monotonic() - t0) * 1000, 2)
    log.info("signup succeeded request_id=%s user_id=%s email=%s elapsed_ms=%.2f", request_id, user["id"], user["email"], elapsed)

    resp = JSONResponse(
        status_code=201,
        content={
            "ok": True,
            "user_id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "api_key": raw_key,
            "api_key_prefix": key_record["key_prefix"],
        },
    )
    resp.set_cookie(
        "dewie_session", token,
        max_age=14 * 24 * 3600,
        path="/",
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
    )
    return resp


@router.post("/login")
async def auth_login(body: LoginRequest, request: Request) -> JSONResponse:
    """Authenticate with username/password and return JWT session cookie.

    Returns HTTP 200 + JWT session cookie on success.
    Returns HTTP 401 if credentials invalid.
    """
    request_id = _extract_request_id(request)
    log.info("login started request_id=%s email=%s", request_id, body.email)
    t0 = time.monotonic()
    pg = request.app.state.postgres

    user = await verify_local_user(pg, body.email, body.password)
    if not user:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.warning("login failed request_id=%s reason=invalid_credentials elapsed_ms=%.2f", request_id, elapsed)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Create session token
    token = create_session_token(user["id"], user["email"], user["is_admin"])
    elapsed = round((time.monotonic() - t0) * 1000, 2)
    log.info("login succeeded request_id=%s user_id=%s email=%s elapsed_ms=%.2f", request_id, user["id"], user["email"], elapsed)

    resp = JSONResponse(
        content={"ok": True, "user_id": user["id"], "email": user["email"], "name": user["name"]},
    )
    resp.set_cookie(
        "dewie_session", token,
        max_age=14 * 24 * 3600,
        path="/",
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
    )
    return resp


@router.post("/forgot-password", status_code=202)
@limiter.limit(rate_limit(5))
async def auth_forgot_password(body: ForgotPasswordRequest, request: Request) -> dict:  # type: ignore[type-arg]
    """Request a password reset link (email-based).

    Generates a reset token valid for 24 hours.
    In production, this would send an email with the reset link.
    For now, returns the token so user can reset password.
    """
    pg = request.app.state.postgres

    if not body.username:
        raise HTTPException(status_code=400, detail="Username is required")

    # Check if user exists
    from sqlalchemy import text
    
    async with pg._engine.connect() as conn:
        result = await conn.execute(
            text("SELECT id, password_hash, email FROM users WHERE email = :email LIMIT 1"),
            {"email": body.username},
        )
        row = result.mappings().fetchone()

    if not row:
        # Don't reveal if email exists (security best practice)
        return {"ok": True, "message": "If email exists, a reset link will be sent"}

    # OAuth users don't have passwords, can't reset
    if not row["password_hash"]:
        return {"ok": True, "message": "If email exists, a reset link will be sent"}

    # Generate reset token (24 hour expiry) — store only hash, never plaintext
    reset_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(reset_token.encode()).hexdigest()

    from datetime import datetime, timedelta

    expires_at = datetime.now(UTC) + timedelta(hours=24)

    async with pg._engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE users SET password_reset_token = :token, password_reset_expires = :expires "
                "WHERE id = :id"
            ),
            {"token": token_hash, "expires": expires_at, "id": str(row["id"])},
        )

    # Send email with reset link — best-effort, don't expose SMTP errors to caller
    reset_link = f"{settings.base_url}/ui/reset-password.html?token={reset_token}"
    try:
        await send_password_reset_email(row["email"], reset_link)
    except Exception as _email_exc:
        import logging as _log
        _log.getLogger(__name__).warning("Failed to send reset email to %s: %s", row["email"], _email_exc)

    return {
        "ok": True,
        "message": "If that email is registered you will receive a reset link shortly",
    }


@router.post("/reset-password", response_model=dict)  # type: ignore[misc]
@limiter.limit(rate_limit(10))
async def auth_reset_password(body: ResetPasswordRequest, request: Request) -> dict:  # type: ignore[type-arg]
    """Reset password using a valid reset token."""
    pg = request.app.state.postgres

    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")


    from sqlalchemy import text

    token_hash = hashlib.sha256(body.reset_token.encode()).hexdigest()

    # Find user with valid reset token (compare against stored hash)
    async with pg._engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT id FROM users "
                "WHERE password_reset_token = :token "
                "AND password_reset_expires > now() "
                "LIMIT 1"
            ),
            {"token": token_hash},
        )
        row = result.mappings().fetchone()

    if not row:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user_id = str(row["id"])

    # Update password and clear reset token
    async with pg._engine.begin() as conn:
        password_hash = hash_password(body.password)
        await conn.execute(
            text(
                "UPDATE users "
                "SET password_hash = :hash, "
                "password_reset_token = NULL, password_reset_expires = NULL "
                "WHERE id = :id"
            ),
            {"hash": password_hash, "id": user_id},
        )

    return {"ok": True, "message": "Password reset successful. Please log in."}


@router.post("/logout")
async def auth_logout(response: Response, request: Request) -> dict:  # type: ignore[type-arg]
    """Logout: revoke session in DB and clear session cookie."""
    request_id = _extract_request_id(request)
    log.info("logout started request_id=%s", request_id)
    t0 = time.monotonic()

    # Revoke the session token server-side
    session_token = request.cookies.get("dewie_session", "").strip()
    if session_token:
        pg = request.app.state.postgres
        try:
            from dewie.local_auth import revoke_session
            await revoke_session(pg, session_token)
            log.info("logout session_revoked request_id=%s", request_id)
        except Exception as exc:
            log.warning("logout revoke_failed request_id=%s error=%s", request_id, exc)

    response.delete_cookie("dewie_session", path="/")
    elapsed = round((time.monotonic() - t0) * 1000, 2)
    log.info("logout succeeded request_id=%s elapsed_ms=%.2f", request_id, elapsed)
    return {"ok": True}


@router.post("/signout")
async def auth_signout(response: Response, request: Request) -> dict:  # type: ignore[type-arg]
    """Alias for logout used by legacy static pages. Revokes session server-side."""
    request_id = _extract_request_id(request)
    log.info("signout started request_id=%s", request_id)
    t0 = time.monotonic()

    # Revoke the session token server-side
    session_token = request.cookies.get("dewie_session", "").strip()
    if session_token:
        pg = request.app.state.postgres
        try:
            from dewie.local_auth import revoke_session
            await revoke_session(pg, session_token)
            log.info("signout session_revoked request_id=%s", request_id)
        except Exception as exc:
            log.warning("signout revoke_failed request_id=%s error=%s", request_id, exc)

    response.delete_cookie("dewie_session", path="/")
    elapsed = round((time.monotonic() - t0) * 1000, 2)
    log.info("signout succeeded request_id=%s elapsed_ms=%.2f", request_id, elapsed)
    return {"ok": True}


@router.post("/change-password")
async def auth_change_password(
    body: ChangePasswordRequest, request: Request
) -> dict:  # type: ignore[type-arg]
    """Change the current user's password. Verifies the current password first."""
    pg = request.app.state.postgres

    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        session_token = request.cookies.get("dewie_session", "").strip()
        if session_token:
            from dewie.local_auth import verify_session_token
            payload = verify_session_token(session_token)
            if payload:
                user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Fetch current password hash and email
    from sqlalchemy import text
    async with pg._engine.connect() as conn:
        result = await conn.execute(
            text("SELECT password_hash, email FROM users WHERE id = :id LIMIT 1"),
            {"id": str(user_id)},
        )
        row = result.mappings().fetchone()

    if not row or not row["password_hash"]:
        raise HTTPException(status_code=400, detail="This account does not have a password set (OAuth-only)")

    # Verify current password
    if not await verify_local_user(pg, row["email"], body.current_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    # Update password
    password_hash = hash_password(body.new_password)
    async with pg._engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE users SET password_hash = :hash WHERE id = :id"
            ),
            {"hash": password_hash, "id": str(user_id)},
        )

    return {"ok": True, "message": "Password changed successfully"}
