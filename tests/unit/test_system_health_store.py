"""Tests for dewie.storage.system_health and llm_cache."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest


def _make_pg_with_session(row=None, rowcount=0):
    pg = MagicMock()
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    mappings_result = MagicMock()
    mappings_result.first.return_value = row
    execute_result = MagicMock()
    execute_result.mappings.return_value = mappings_result
    execute_result.rowcount = rowcount
    session.execute = AsyncMock(return_value=execute_result)

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._session_factory = MagicMock(return_value=cm)
    return pg, session


# ── write_health_kv ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_health_kv_success():
    from dewie.storage.system_health import write_health_kv

    pg, session = _make_pg_with_session()
    await write_health_kv(pg, "test_key", "test_value")
    session.execute.assert_called_once()
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_write_health_kv_never_raises():
    from dewie.storage.system_health import write_health_kv

    pg, session = _make_pg_with_session()
    session.execute = AsyncMock(side_effect=Exception("DB down"))
    # Should not raise
    await write_health_kv(pg, "key", "value")


# ── read_health_kv ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_health_kv_hit():
    from datetime import datetime

    from dewie.storage.system_health import read_health_kv

    row = {"value": "active", "updated_at": datetime(2024, 1, 1)}
    pg, _ = _make_pg_with_session(row=row)
    result = await read_health_kv(pg, "pipeline_status")
    assert result["value"] == "active"
    assert result["updated_at"] is not None


@pytest.mark.asyncio
async def test_read_health_kv_miss():
    from dewie.storage.system_health import read_health_kv

    pg, _ = _make_pg_with_session(row=None)
    result = await read_health_kv(pg, "nonexistent_key")
    assert result is None


@pytest.mark.asyncio
async def test_read_health_kv_db_error_returns_none():
    from dewie.storage.system_health import read_health_kv

    pg, session = _make_pg_with_session()
    session.execute = AsyncMock(side_effect=Exception("connection failed"))
    result = await read_health_kv(pg, "key")
    assert result is None


# ── llm_cache ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_cached_miss():
    from dewie.storage.llm_cache import get_cached

    pg, _ = _make_pg_with_session(row=None)
    result = await get_cached(
        pg, UUID("00000000-0000-0000-0000-000000000001"), "step", "gpt-4o", "prompt"
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_cached_hit():
    from dewie.storage.llm_cache import get_cached

    row = {"raw_response": '{"summary": "test"}'}
    pg, _ = _make_pg_with_session(row=row)
    result = await get_cached(
        pg, UUID("00000000-0000-0000-0000-000000000001"), "step", "gpt-4o", "prompt"
    )
    assert result == '{"summary": "test"}'


@pytest.mark.asyncio
async def test_set_cached():
    from dewie.storage.llm_cache import set_cached

    pg, session = _make_pg_with_session()
    await set_cached(
        pg, UUID("00000000-0000-0000-0000-000000000001"), "step", "gpt-4o", "prompt", '{"ok": true}'
    )
    session.execute.assert_called_once()
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_bust_cache_all_steps():
    from dewie.storage.llm_cache import bust_cache

    pg, session = _make_pg_with_session(rowcount=3)
    count = await bust_cache(pg, UUID("00000000-0000-0000-0000-000000000001"))
    assert count == 3
    session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_bust_cache_specific_step():
    from dewie.storage.llm_cache import bust_cache

    pg, session = _make_pg_with_session(rowcount=1)
    count = await bust_cache(
        pg, UUID("00000000-0000-0000-0000-000000000001"), step="llm_extraction"
    )
    assert count == 1
    # Verify the sql used includes the step filter
    call_args = session.execute.call_args[0]
    assert "step" in str(call_args[1])
