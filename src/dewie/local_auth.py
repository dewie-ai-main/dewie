# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Local password authentication — username/password login for self-hosted deployments.

Uses bcrypt (rounds=12) for password hashing + JWT for session tokens.
"""

from __future__ import annotations

import asyncio as _asyncio
import hashlib
import os
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt
from sqlalchemy import text

# Session JWT settings
_DEFAULT_JWT_SECRET = None

def _get_jwt_secret() -> str:
    """Get or generate JWT secret (32 bytes for HMAC-SHA256)."""
    global _DEFAULT_JWT_SECRET
    if _DEFAULT_JWT_SECRET:
        return _DEFAULT_JWT_SECRET
    
    import logging as _log
    _logger = _log.getLogger(__name__)

    secret = os.environ.get("JWT_SECRET")
    if secret:
        if len(secret) < 32:
            _logger.warning(
                "JWT_SECRET is %d bytes; recommend 32+ bytes for security", len(secret)
            )
        _DEFAULT_JWT_SECRET = secret
        return secret

    # Generate a secure default (only for dev/local environments)
    _DEFAULT_JWT_SECRET = secrets.token_urlsafe(32)
    _logger.warning(
        "No JWT_SECRET env var set. Using ephemeral secret — sessions will be invalidated "
        "on every restart. Set JWT_SECRET for persistent sessions."
    )
    return _DEFAULT_JWT_SECRET


JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24


def hash_password(password: str) -> str:
    """Hash a password using bcrypt (cost=12, same as API keys)."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode(), salt).decode()


async def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a bcrypt hash (runs in thread pool)."""
    return await _asyncio.to_thread(
        bcrypt.checkpw, password.encode(), password_hash.encode()
    )


def create_session_token(user_id: str, email: str, is_admin: bool = False) -> str:
    """Create a JWT session token (24h expiry by default)."""
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "email": email,
        "is_admin": is_admin,
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def verify_session_token(token: str) -> dict[str, Any] | None:
    """Verify and decode a JWT session token."""
    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.InvalidTokenError:
        return None


def _hash_session_token(token: str) -> str:
    """SHA-256 hash of a session token for storage in the revoked_sessions table."""
    return hashlib.sha256(token.encode()).hexdigest()


async def revoke_session(pg: Any, token: str) -> bool:
    """Record a session token as revoked in the DB.

    Idempotent — calling multiple times for the same token is safe.
    Returns True if a row was inserted (first revocation), False if already present.
    """
    token_hash = _hash_session_token(token)
    async with pg._engine.begin() as conn:
        result = await conn.execute(
            text(
                """
                INSERT INTO revoked_session_tokens (token_hash, revoked_at)
                VALUES (:token_hash, CURRENT_TIMESTAMP)
                ON CONFLICT DO NOTHING
                """
            ),
            {"token_hash": token_hash},
        )
        return result.rowcount > 0


async def is_session_revoked(pg: Any, token: str) -> bool:
    """Check whether a session token has been revoked.

    Returns True if the token (or its SHA-256 hash) appears in the
    revoked_session_tokens table.
    """
    token_hash = _hash_session_token(token)
    async with pg._engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT 1 FROM revoked_session_tokens
                WHERE token_hash = :token_hash
                LIMIT 1
                """
            ),
            {"token_hash": token_hash},
        )
        return result.fetchone() is not None


def _default_display_name(identifier: str) -> str:
    """Derive a display name from a login identifier.

    If the identifier looks like an email address, returns the local part
    (everything before the '@').  Otherwise returns the identifier itself.
    Either way, leading/trailing whitespace is stripped.
    """
    identifier = identifier.strip()
    if "@" in identifier:
        return identifier.split("@", 1)[0] or identifier
    return identifier


async def create_local_user(
    pg: Any, email: str, password: str, name: str | None = None
) -> dict[str, Any]:
    """
    Create a new local user. The `email` parameter is used as the login identifier
    and does not need to be a valid email address — any unique non-empty string works.

    Returns user record dict with keys: id, email, name, created_at, is_admin.
    Raises ValueError if the identifier already exists.
    """
    password_hash = hash_password(password)

    async with pg._engine.begin() as conn:
        # Check if email already exists
        existing = await conn.execute(
            text("SELECT id FROM users WHERE email = :email LIMIT 1"),
            {"email": email},
        )
        if existing.fetchone():
            raise ValueError(f"Email already exists: {email}")

        # Insert new user
        result = await conn.execute(
            text(
                """
                INSERT INTO users (
                    id, tenant_id, email, name, password_hash,
                    is_admin, plan, activation_status, created_at
                )
                VALUES (
                    :id, :tenant_id, :email, :name, :password_hash,
                    :is_admin, :plan, :activation_status, CURRENT_TIMESTAMP
                )
                RETURNING id, email, name, created_at, is_admin
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "email": email,
                "name": name or _default_display_name(email),
                "password_hash": password_hash,
                "is_admin": False,
                "plan": "free",
                "activation_status": "approved",
            },
        )
        row = result.mappings().fetchone()

    return {
        "id": str(row["id"]),
        "email": row["email"],
        "name": row["name"],
        "created_at": row["created_at"],
        "is_admin": row["is_admin"],
    }


async def verify_local_user(
    pg: Any, email: str, password: str
) -> dict[str, Any] | None:
    """
    Verify login identifier/password credentials.

    Returns user record dict with keys: id, email, name, is_admin.
    Returns None if identifier not found or password invalid.
    """
    async with pg._engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT id, email, name, password_hash, is_admin "
                "FROM users WHERE email = :email LIMIT 1"
            ),
            {"email": email},
        )
        row = result.mappings().fetchone()

    if not row:
        return None

    # Verify password (runs in thread pool to avoid blocking)
    if not row["password_hash"]:
        return None  # OAuth-only user, no password

    password_valid = await verify_password(password, row["password_hash"])
    if not password_valid:
        return None

    # Update last_login_at
    try:
        async with pg._engine.begin() as conn:
            await conn.execute(
                text("UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = :id"),
                {"id": row["id"]},
            )
    except Exception:
        pass  # Best-effort

    return {
        "id": str(row["id"]),
        "email": row["email"],
        "name": row["name"],
        "is_admin": row["is_admin"],
    }


async def update_user_password(pg: Any, user_id: str, password: str) -> bool:
    """Update a user's password (idempotent — returns True if successful)."""
    password_hash = hash_password(password)

    async with pg._engine.begin() as conn:
        result = await conn.execute(
            text("UPDATE users SET password_hash = :hash WHERE id = :id"),
            {"hash": password_hash, "id": user_id},
        )
        return result.rowcount > 0


async def seed_default_admin(
    pg: Any, email: str, password: str
) -> dict[str, Any] | None:
    """Seed a default admin user if the users table is empty.

    Returns the created user dict on success, or None if users already exist.
    """
    async with pg._engine.begin() as conn:
        count_result = await conn.execute(
            text("SELECT COUNT(*) AS cnt FROM users"),
        )
        row = count_result.mappings().fetchone()
        user_count = int(row["cnt"])

    if user_count > 0:
        return None

    password_hash = hash_password(password)

    async with pg._engine.begin() as conn:
        result = await conn.execute(
            text(
                """
                INSERT INTO users (
                    id, tenant_id, email, name, password_hash,
                    is_admin, plan, activation_status, created_at
                )
                VALUES (
                    :id, :tenant_id, :email, :name, :password_hash,
                    :is_admin, :plan, :activation_status, CURRENT_TIMESTAMP
                )
                RETURNING id, email, name, created_at, is_admin
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "email": email,
                "name": _default_display_name(email),
                "password_hash": password_hash,
                "is_admin": True,
                "plan": "enterprise",
                "activation_status": "approved",
            },
        )
        row = result.mappings().fetchone()

    return {
        "id": str(row["id"]),
        "email": row["email"],
        "name": row["name"],
        "created_at": row["created_at"],
        "is_admin": row["is_admin"],
    }
