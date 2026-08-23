# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
dewie/auth.py — API key generation, verification, and management.

API key format:
  ck_live_<32-char-urlsafe-token>  — production keys
  ck_test_<32-char-urlsafe-token>  — test keys

Keys are stored as bcrypt hashes. The first 12 characters of the plaintext
key are stored as key_prefix for efficient lookup before bcrypt comparison.

Usage:
    raw_key, hashed = generate_api_key(live=True)
    # give raw_key to the user, store hashed in DB

    record = await verify_api_key(raw_key, pg)
    if record:
        workspace_ids = record["workspace_ids"]  # empty list = all workspaces
"""

from __future__ import annotations

import json
import secrets
import uuid
from typing import Any

import bcrypt

# Default tenant ID used when auth is disabled or for system-owned objects
DEFAULT_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

# Scopes
SCOPE_READ = "read"
SCOPE_INGEST = "ingest"
SCOPE_ADMIN = "admin"
ALL_SCOPES = [SCOPE_READ, SCOPE_INGEST, SCOPE_ADMIN]


def generate_api_key(live: bool = True) -> tuple[str, str]:
    """
    Generate a new API key.

    Returns:
        (plaintext_key, bcrypt_hash) — store the hash, give the user the plaintext.
    """
    prefix = "ck_live_" if live else "ck_test_"
    raw = prefix + secrets.token_urlsafe(32)
    hashed = bcrypt.hashpw(raw.encode(), bcrypt.gensalt(rounds=12)).decode()
    return raw, hashed


def key_prefix(raw_key: str) -> str:
    """Return the first 12 characters of a key for DB lookup."""
    return raw_key[:12]


async def verify_api_key(raw_key: str, pg: Any) -> dict[str, Any] | None:
    """
    Verify an API key against stored hashes.

    Looks up candidates by key_prefix (first 12 chars), then bcrypt-checks
    each candidate. Returns the key record dict or None if invalid/revoked.

    Record dict keys: id, workspace_ids, scopes, name, key_prefix
    """
    if not raw_key or not raw_key.startswith(("ck_live_", "ck_test_")):
        return None

    prefix = key_prefix(raw_key)

    async with pg._engine.connect() as conn:
        from sqlalchemy import text as _text

        rows = await conn.execute(
            _text(
                "SELECT id, user_id, workspace_ids, key_hash, scopes, name, key_prefix, revoked_at "
                "FROM api_keys WHERE key_prefix = :prefix"
            ),
            {"prefix": prefix},
        )
        candidates = rows.mappings().fetchall()

    for row in candidates:
        if row["revoked_at"] is not None:
            continue
        # bcrypt.checkpw is CPU-bound (~300ms at rounds=12). Run in thread pool
        # to avoid blocking the uvicorn event loop and degrading all concurrent requests.
        import asyncio as _asyncio

        try:
            match = await _asyncio.to_thread(bcrypt.checkpw, raw_key.encode(), row["key_hash"].encode())
        except (ValueError, Exception):
            # Corrupted or invalid hash — skip this candidate
            continue
        if match:
            # Update last_used_at asynchronously — best effort.
            # SQLite has no now(); use CURRENT_TIMESTAMP there.
            is_sqlite = bool(getattr(pg, "_is_sqlite", False))
            now_expr = "CURRENT_TIMESTAMP" if is_sqlite else "now()"
            try:
                async with pg._engine.begin() as conn:
                    from sqlalchemy import text as _text2

                    await conn.execute(
                        _text2(f"UPDATE api_keys SET last_used_at = {now_expr} WHERE id = :id"),
                        {"id": row["id"]},
                    )
            except Exception:
                pass

            # On SQLite the array columns come back as JSON TEXT; Postgres
            # returns real lists. Decode by actual type so this is robust to
            # either backend.
            def _decode_array(v: Any) -> list:
                if isinstance(v, str):
                    return json.loads(v) if v else []
                return list(v or [])

            workspace_ids = [uuid.UUID(str(w)) for w in _decode_array(row["workspace_ids"])]
            scopes = _decode_array(row["scopes"])
            return {
                "id": row["id"],
                "user_id": str(row["user_id"]) if row["user_id"] else None,
                "workspace_ids": workspace_ids,
                "scopes": scopes,
                "name": row["name"],
                "key_prefix": row["key_prefix"],
            }

    return None


async def create_api_key(
    pg: Any,
    *,
    workspace_ids: list[uuid.UUID] | None = None,
    name: str | None = None,
    scopes: list[str] | None = None,
    live: bool = True,
    user_id: uuid.UUID | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Create and persist a new API key.

    workspace_ids: restrict access to specific workspaces; empty/None = all workspaces.
    user_id: the user who owns this key (None for admin/system keys).

    Returns:
        (plaintext_key, record_dict) — plaintext shown once, never stored.
    """
    if scopes is None:
        scopes = [SCOPE_READ]
    if workspace_ids is None:
        workspace_ids = []

    raw, hashed = generate_api_key(live=live)
    prefix = key_prefix(raw)

    # SQLite stores the array columns as JSON TEXT; Postgres uses native arrays.
    is_sqlite = bool(getattr(pg, "_is_sqlite", False))
    ws_list = [str(w) for w in workspace_ids]
    ws_param = json.dumps(ws_list) if is_sqlite else ws_list
    scopes_param = json.dumps(scopes) if is_sqlite else scopes

    async with pg._engine.begin() as conn:
        from sqlalchemy import text as _text

        row = await conn.execute(
            _text(
                "INSERT INTO api_keys (user_id, workspace_ids, key_hash, key_prefix, scopes, name) "
                "VALUES (:user_id, :workspace_ids, :key_hash, :key_prefix, :scopes, :name) "
                "RETURNING id, created_at"
            ),
            {
                "user_id": str(user_id) if user_id else None,
                "workspace_ids": ws_param,
                "key_hash": hashed,
                "key_prefix": prefix,
                "scopes": scopes_param,
                "name": name,
            },
        )
        result = row.mappings().fetchone()

    return raw, {
        "id": result["id"],
        "workspace_ids": workspace_ids,
        "key_prefix": prefix,
        "scopes": scopes,
        "name": name,
        "created_at": result["created_at"],
    }


async def revoke_api_key(pg: Any, key_id: uuid.UUID) -> bool:
    """Revoke a key by ID. Returns True if a row was updated."""
    async with pg._engine.begin() as conn:
        from sqlalchemy import text as _text

        r = await conn.execute(
            _text(
                "UPDATE api_keys SET revoked_at = now() "
                "WHERE id = :id AND revoked_at IS NULL"
            ),
            {"id": str(key_id)},
        )
        return r.rowcount > 0
