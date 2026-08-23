"""Tests for dewie.storage.system_health."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_pg():
    pg = MagicMock()
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pg._session_factory = MagicMock(return_value=ctx)
    return pg, session


# ── write_health_kv ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_health_kv_calls_execute_and_commit():
    from dewie.storage.system_health import write_health_kv

    pg, session = _make_pg()

    await write_health_kv(pg, "last_enrich", "2026-04-27T12:00:00")
    session.execute.assert_called_once()
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_write_health_kv_passes_correct_params():
    from dewie.storage.system_health import write_health_kv

    pg, session = _make_pg()

    await write_health_kv(pg, "my_key", "my_value")
    params = session.execute.call_args[0][1]
    assert params["key"] == "my_key"
    assert params["value"] == "my_value"


@pytest.mark.asyncio
async def test_write_health_kv_swallows_exception():
    from dewie.storage.system_health import write_health_kv

    pg, session = _make_pg()
    session.execute = AsyncMock(side_effect=Exception("db down"))

    # Should not raise
    await write_health_kv(pg, "key", "value")


# ── read_health_kv ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_health_kv_returns_value():
    from dewie.storage.system_health import read_health_kv

    pg, session = _make_pg()

    now = datetime(2026, 4, 27, 12, 0, 0)
    mock_row = {"value": "ok", "updated_at": now}
    mock_result = MagicMock()
    mock_result.mappings.return_value.first.return_value = mock_row
    session.execute = AsyncMock(return_value=mock_result)

    result = await read_health_kv(pg, "status")
    assert result is not None
    assert result["value"] == "ok"
    assert "2026" in result["updated_at"]


@pytest.mark.asyncio
async def test_read_health_kv_returns_none_on_miss():
    from dewie.storage.system_health import read_health_kv

    pg, session = _make_pg()

    mock_result = MagicMock()
    mock_result.mappings.return_value.first.return_value = None
    session.execute = AsyncMock(return_value=mock_result)

    result = await read_health_kv(pg, "missing_key")
    assert result is None


@pytest.mark.asyncio
async def test_read_health_kv_handles_null_updated_at():
    from dewie.storage.system_health import read_health_kv

    pg, session = _make_pg()

    mock_row = {"value": "test", "updated_at": None}
    mock_result = MagicMock()
    mock_result.mappings.return_value.first.return_value = mock_row
    session.execute = AsyncMock(return_value=mock_result)

    result = await read_health_kv(pg, "key")
    assert result["updated_at"] is None


@pytest.mark.asyncio
async def test_read_health_kv_swallows_exception():
    from dewie.storage.system_health import read_health_kv

    pg, session = _make_pg()
    session.execute = AsyncMock(side_effect=Exception("db error"))

    result = await read_health_kv(pg, "key")
    assert result is None
