"""
tests/test_auth.py — API key auth + tenant isolation tests.

All tests are unit tests with mocked DB — no live Postgres required.
"""

from __future__ import annotations

import uuid
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt
import pytest

from dewie.auth import (
    DEFAULT_TENANT_ID,
    SCOPE_ADMIN,
    SCOPE_INGEST,
    SCOPE_READ,
    generate_api_key,
    key_prefix,
    verify_api_key,
)

# ---------------------------------------------------------------------------
# generate_api_key
# ---------------------------------------------------------------------------


def test_generate_live_key_prefix():
    raw, hashed = generate_api_key(live=True)
    assert raw.startswith("ck_live_")


def test_generate_test_key_prefix():
    raw, hashed = generate_api_key(live=False)
    assert raw.startswith("ck_test_")


def test_generate_key_hash_is_bcrypt():
    raw, hashed = generate_api_key()
    assert hashed.startswith("$2b$")  # bcrypt signature
    assert bcrypt.checkpw(raw.encode(), hashed.encode())


def test_generate_keys_are_unique():
    keys = {generate_api_key()[0] for _ in range(20)}
    assert len(keys) == 20  # all unique


def test_key_prefix_length():
    raw, _ = generate_api_key()
    assert key_prefix(raw) == raw[:12]
    assert len(key_prefix(raw)) == 12


def test_default_tenant_id_is_valid_uuid():
    assert isinstance(DEFAULT_TENANT_ID, uuid.UUID)
    assert str(DEFAULT_TENANT_ID) == "00000000-0000-0000-0000-000000000001"


# ---------------------------------------------------------------------------
# verify_api_key
# ---------------------------------------------------------------------------


def _make_pg_with_keys(key_rows: list[dict]):
    """Build a mock pg where api_keys lookup returns key_rows."""
    mock_conn = MagicMock()
    mock_rows = MagicMock()
    mock_rows.mappings.return_value.fetchall.return_value = key_rows
    mock_conn.execute = AsyncMock(return_value=mock_rows)

    # context manager for connect()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    # context manager for begin() (used by last_used_at update)
    mock_begin_conn = MagicMock()
    mock_begin_conn.execute = AsyncMock()
    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__aenter__ = AsyncMock(return_value=mock_begin_conn)
    mock_begin_ctx.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._engine = MagicMock()
    pg._engine.connect = MagicMock(return_value=mock_ctx)
    pg._engine.begin = MagicMock(return_value=mock_begin_ctx)
    return pg


@pytest.mark.asyncio
async def test_verify_valid_key_returns_record():
    raw, hashed = generate_api_key(live=True)
    tenant_id = uuid.uuid4()
    key_id = uuid.uuid4()

    row = {
        "id": key_id,
        "key_hash": hashed,
        "key_prefix": key_prefix(raw),
        "workspace_ids": [],
        "scopes": [SCOPE_READ],
        "name": "test key",
        "revoked_at": None,
    }
    pg = _make_pg_with_keys([row])

    result = await verify_api_key(raw, pg)

    assert result is not None
    assert result["id"] == key_id
    assert SCOPE_READ in result["scopes"]


@pytest.mark.asyncio
async def test_verify_wrong_key_returns_none():
    raw, hashed = generate_api_key(live=True)
    wrong_key = "ck_live_" + "x" * 43  # valid format, wrong secret

    row = {
        "id": uuid.uuid4(),
        "tenant_id": uuid.uuid4(),
        "key_hash": hashed,
        "key_prefix": key_prefix(raw),
        "scopes": [SCOPE_READ],
        "name": None,
        "revoked_at": None,
    }
    pg = _make_pg_with_keys([row])

    result = await verify_api_key(wrong_key, pg)
    assert result is None


@pytest.mark.asyncio
async def test_verify_revoked_key_returns_none():
    from datetime import datetime

    raw, hashed = generate_api_key(live=True)

    row = {
        "id": uuid.uuid4(),
        "tenant_id": uuid.uuid4(),
        "key_hash": hashed,
        "key_prefix": key_prefix(raw),
        "scopes": [SCOPE_READ],
        "name": None,
        "revoked_at": datetime.now(UTC),  # revoked
    }
    pg = _make_pg_with_keys([row])

    result = await verify_api_key(raw, pg)
    assert result is None


@pytest.mark.asyncio
async def test_verify_non_ck_key_returns_none():
    pg = _make_pg_with_keys([])
    result = await verify_api_key("Bearer some-jwt-token", pg)
    assert result is None


@pytest.mark.asyncio
async def test_verify_empty_key_returns_none():
    pg = _make_pg_with_keys([])
    result = await verify_api_key("", pg)
    assert result is None


# ---------------------------------------------------------------------------
# AUTH_ENABLED=false default behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_enabled_by_default():
    """With AUTH_ENABLED=true (default for production), auth is enforced."""
    from dewie.config import Settings

    assert Settings(_env_file=None).auth_enabled is True


@pytest.mark.asyncio
async def test_auth_middleware_sets_default_tenant_when_disabled():
    """Middleware sets default tenant_id on request.state when auth is off."""
    from dewie.api.middleware import _api_key_middleware
    from dewie.auth import DEFAULT_TENANT_ID

    request = MagicMock()
    request.state = MagicMock()
    request.url.path = "/query"

    async def fake_next(req):
        return MagicMock(status_code=200)

    with patch("dewie.api.middleware.settings") as mock_settings:
        mock_settings.auth_enabled = False
        mock_settings.access_log_enabled = False
        await _api_key_middleware(request, fake_next)

    assert request.state.tenant_id == DEFAULT_TENANT_ID


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------


def test_scope_constants():
    assert SCOPE_READ == "read"
    assert SCOPE_INGEST == "ingest"
    assert SCOPE_ADMIN == "admin"
