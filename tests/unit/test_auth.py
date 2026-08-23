"""Tests for dewie.auth — API key generation and verification."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from dewie.auth import (
    ALL_SCOPES,
    SCOPE_ADMIN,
    SCOPE_INGEST,
    SCOPE_READ,
    generate_api_key,
    key_prefix,
)

# ── generate_api_key ──────────────────────────────────────────────────────────


def test_generate_live_key_format():
    raw, hashed = generate_api_key(live=True)
    assert raw.startswith("ck_live_")
    assert len(raw) > 8


def test_generate_test_key_format():
    raw, hashed = generate_api_key(live=False)
    assert raw.startswith("ck_test_")


def test_generated_hash_is_bcrypt():
    raw, hashed = generate_api_key()
    import bcrypt

    assert bcrypt.checkpw(raw.encode(), hashed.encode())


def test_two_keys_are_unique():
    raw1, _ = generate_api_key()
    raw2, _ = generate_api_key()
    assert raw1 != raw2


# ── key_prefix ───────────────────────────────────────────────────────────────


def test_key_prefix_returns_first_12_chars():
    raw = "ck_live_ABCDEFGHIJKLMNOP"
    assert key_prefix(raw) == raw[:12]


def test_key_prefix_live():
    raw, _ = generate_api_key(live=True)
    assert key_prefix(raw) == raw[:12]


# ── constants ─────────────────────────────────────────────────────────────────


def test_all_scopes_contains_all():
    assert SCOPE_READ in ALL_SCOPES
    assert SCOPE_INGEST in ALL_SCOPES
    assert SCOPE_ADMIN in ALL_SCOPES


# ── verify_api_key ─────────────────────────────────────────────────────────────


def _make_pg_with_candidates(candidates):
    """Build a pg mock where _engine.connect() returns candidates as row mappings."""
    mock_result = MagicMock()
    mock_result.mappings.return_value.fetchall.return_value = candidates

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)

    mock_connect_ctx = MagicMock()
    mock_connect_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_connect_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_begin_ctx.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._engine = MagicMock()
    pg._engine.connect.return_value = mock_connect_ctx
    pg._engine.begin.return_value = mock_begin_ctx
    return pg


@pytest.mark.asyncio
async def test_verify_rejects_invalid_prefix():
    from dewie.auth import verify_api_key

    pg = MagicMock()
    result = await verify_api_key("not_a_key", pg)
    assert result is None


@pytest.mark.asyncio
async def test_verify_rejects_empty_key():
    from dewie.auth import verify_api_key

    pg = MagicMock()
    result = await verify_api_key("", pg)
    assert result is None


@pytest.mark.asyncio
async def test_verify_returns_none_on_no_candidates():
    from dewie.auth import verify_api_key

    pg = _make_pg_with_candidates([])
    raw, _ = generate_api_key()
    result = await verify_api_key(raw, pg)
    assert result is None


@pytest.mark.asyncio
async def test_verify_returns_none_for_revoked_key():

    from dewie.auth import verify_api_key

    raw, hashed = generate_api_key()
    workspace_ids = [str(uuid.uuid4())]
    candidate = {
        "id": str(uuid.uuid4()),
        "workspace_ids": workspace_ids,
        "key_hash": hashed,
        "scopes": ["read"],
        "name": "test key",
        "key_prefix": key_prefix(raw),
        "revoked_at": "2026-01-01T00:00:00",
    }
    pg = _make_pg_with_candidates([candidate])
    result = await verify_api_key(raw, pg)
    assert result is None


@pytest.mark.asyncio
async def test_verify_returns_record_for_valid_key():
    from dewie.auth import verify_api_key

    raw, hashed = generate_api_key()
    workspace_ids = [str(uuid.uuid4())]
    candidate = {
        "id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "workspace_ids": workspace_ids,
        "key_hash": hashed,
        "scopes": ["read", "ingest"],
        "name": "my api key",
        "key_prefix": key_prefix(raw),
        "revoked_at": None,
    }
    pg = _make_pg_with_candidates([candidate])
    result = await verify_api_key(raw, pg)
    assert result is not None
    assert isinstance(result["workspace_ids"], list)
    assert result["scopes"] == ["read", "ingest"]
    assert result["name"] == "my api key"


@pytest.mark.asyncio
async def test_verify_returns_none_for_wrong_key():
    from dewie.auth import verify_api_key

    raw, hashed = generate_api_key()
    wrong_raw, _ = generate_api_key()
    workspace_ids = [str(uuid.uuid4())]
    candidate = {
        "id": str(uuid.uuid4()),
        "workspace_ids": workspace_ids,
        "key_hash": hashed,
        "scopes": ["read"],
        "name": "test",
        "key_prefix": key_prefix(raw),
        "revoked_at": None,
    }
    # Feed wrong_raw but the hash is for raw — bcrypt check will fail
    pg = _make_pg_with_candidates([candidate])
    result = await verify_api_key(wrong_raw, pg)
    assert result is None


# ── create_api_key ────────────────────────────────────────────────────────────


def _make_pg_for_create(return_id="key-id-1", return_created_at="2026-01-01"):
    mock_row = MagicMock()
    mock_row.mappings.return_value.fetchone.return_value = {
        "id": return_id,
        "created_at": return_created_at,
    }
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_row)
    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_begin_ctx.__aexit__ = AsyncMock(return_value=False)
    pg = MagicMock()
    pg._engine = MagicMock()
    pg._engine.begin.return_value = mock_begin_ctx
    return pg


@pytest.mark.asyncio
async def test_create_api_key_returns_raw_and_record():
    from dewie.auth import create_api_key

    pg = _make_pg_for_create()
    raw, record = await create_api_key(pg, name="test key")
    assert raw.startswith("ck_live_")
    assert record["name"] == "test key"
    assert record["scopes"] == [SCOPE_READ]
    assert isinstance(record["workspace_ids"], list)


@pytest.mark.asyncio
async def test_create_api_key_custom_scopes():
    from dewie.auth import create_api_key

    pg = _make_pg_for_create()
    raw, record = await create_api_key(pg, scopes=[SCOPE_READ, SCOPE_INGEST])
    assert record["scopes"] == [SCOPE_READ, SCOPE_INGEST]


@pytest.mark.asyncio
async def test_create_api_key_test_mode():
    from dewie.auth import create_api_key

    pg = _make_pg_for_create()
    raw, _ = await create_api_key(pg, live=False)
    assert raw.startswith("ck_test_")


@pytest.mark.asyncio
async def test_create_api_key_includes_user_id_in_insert():
    from dewie.auth import create_api_key

    test_user_id = uuid.uuid4()
    pg = _make_pg_for_create()
    raw, _ = await create_api_key(pg, user_id=test_user_id)

    call_args = pg._engine.begin.return_value.__aenter__.return_value.execute.call_args
    sql = call_args[0][0].text
    params = call_args[0][1]

    assert "user_id" in sql
    assert params["user_id"] == str(test_user_id)


@pytest.mark.asyncio
async def test_create_api_key_user_id_null():
    """Admin create (user_id=None) → NULL in DB, not zero UUID."""
    from dewie.auth import create_api_key

    pg = _make_pg_for_create()
    await create_api_key(pg)

    call_args = pg._engine.begin.return_value.__aenter__.return_value.execute.call_args
    params = call_args[0][1]

    assert params["user_id"] is None


# ── revoke_api_key ────────────────────────────────────────────────────────────


def _make_pg_for_revoke(rowcount=1):
    mock_result = MagicMock()
    mock_result.rowcount = rowcount
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_begin_ctx.__aexit__ = AsyncMock(return_value=False)
    pg = MagicMock()
    pg._engine = MagicMock()
    pg._engine.begin.return_value = mock_begin_ctx
    return pg


@pytest.mark.asyncio
async def test_revoke_api_key_returns_true_when_revoked():
    from dewie.auth import revoke_api_key

    pg = _make_pg_for_revoke(rowcount=1)
    key_id = uuid.uuid4()
    result = await revoke_api_key(pg, key_id)
    assert result is True


@pytest.mark.asyncio
async def test_revoke_api_key_returns_false_when_not_found():
    from dewie.auth import revoke_api_key

    pg = _make_pg_for_revoke(rowcount=0)
    key_id = uuid.uuid4()
    result = await revoke_api_key(pg, key_id)
    assert result is False
