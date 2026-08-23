"""Tests for dewie.storage.pipeline_errors."""

from __future__ import annotations

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


# ── classify_error ────────────────────────────────────────────────────────────


def test_classify_error_429():
    from dewie.storage.pipeline_errors import classify_error

    exc = Exception("HTTP 429: Too Many Requests")
    assert classify_error(exc) == "429"


def test_classify_error_rate_limit():
    from dewie.storage.pipeline_errors import classify_error

    exc = Exception("rate limit exceeded")
    assert classify_error(exc) == "429"


def test_classify_error_timeout():
    from dewie.storage.pipeline_errors import classify_error

    exc = Exception("request timed out after 30s")
    assert classify_error(exc) == "timeout"


def test_classify_error_parse():
    from dewie.storage.pipeline_errors import classify_error

    exc = Exception("failed to parse JSON response")
    assert classify_error(exc) == "parse"


def test_classify_error_unknown():
    from dewie.storage.pipeline_errors import classify_error

    exc = Exception("some other error")
    assert classify_error(exc) == "unknown"


def test_classify_error_validation():
    from dewie.enrichment.validators import StepValidationError
    from dewie.storage.pipeline_errors import classify_error

    exc = StepValidationError("extraction", "doc-id", "validation failed")
    assert classify_error(exc) == "validation"


# ── write_error ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_error_calls_execute_and_commit():
    from dewie.storage.pipeline_errors import write_error

    pg, session = _make_pg()

    await write_error(pg, "doc-id-123", "llm_extraction", "parse", "JSON parse failed")
    session.execute.assert_called_once()
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_write_error_passes_correct_params():
    from dewie.storage.pipeline_errors import write_error

    pg, session = _make_pg()

    await write_error(pg, "doc-123", "embedding", "timeout", "timed out", retry_count=2)
    params = session.execute.call_args[0][1]
    assert params["doc_id"] == "doc-123"
    assert params["step"] == "embedding"
    assert params["error_type"] == "timeout"
    assert params["message"] == "timed out"
    assert params["retry_count"] == 2


@pytest.mark.asyncio
async def test_write_error_swallows_exception():
    from dewie.storage.pipeline_errors import write_error

    pg, session = _make_pg()
    session.execute = AsyncMock(side_effect=Exception("db down"))

    # Should not raise
    await write_error(pg, None, "step", "unknown", "error")


@pytest.mark.asyncio
async def test_write_error_with_none_doc_id():
    from dewie.storage.pipeline_errors import write_error

    pg, session = _make_pg()

    await write_error(pg, None, "load_body", "unknown", "no doc id")
    params = session.execute.call_args[0][1]
    assert params["doc_id"] is None


# ── mark_resolved ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_resolved_empty_ids():
    from dewie.storage.pipeline_errors import mark_resolved

    pg, session = _make_pg()

    resolved, requeued = await mark_resolved(pg, [])
    assert resolved == 0
    assert requeued == 0
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_mark_resolved_without_requeue():
    from dewie.storage.pipeline_errors import mark_resolved

    pg, session = _make_pg()

    resolved, requeued = await mark_resolved(pg, [1, 2, 3], requeue=False)
    assert resolved == 3
    assert requeued == 0
    session.execute.assert_called_once()  # Only the UPDATE pipeline_errors call


@pytest.mark.asyncio
async def test_mark_resolved_with_requeue():
    from dewie.storage.pipeline_errors import mark_resolved

    pg, session = _make_pg()

    mock_result = MagicMock()
    mock_result.rowcount = 2
    session.execute = AsyncMock(side_effect=[MagicMock(), mock_result])

    resolved, requeued = await mark_resolved(pg, [1, 2, 3], requeue=True)
    assert resolved == 3
    assert requeued == 2


@pytest.mark.asyncio
async def test_mark_resolved_swallows_exception():
    from dewie.storage.pipeline_errors import mark_resolved

    pg, session = _make_pg()
    session.execute = AsyncMock(side_effect=Exception("db error"))

    resolved, requeued = await mark_resolved(pg, [1, 2])
    assert resolved == 0
    assert requeued == 0


# ── get_error_stats ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_error_stats_basic():
    from dewie.storage.pipeline_errors import get_error_stats

    pg, session = _make_pg()

    call_count = [0]

    async def mock_execute(sql, params=None):
        call_count[0] += 1
        r = MagicMock()
        if call_count[0] == 1:
            r.scalar.return_value = 10  # successful
        elif call_count[0] == 2:
            r.scalar.return_value = 2  # failed
        elif call_count[0] == 3:
            r.mappings.return_value.all.return_value = [{"step": "embedding", "n": 2}]
        elif call_count[0] == 4:
            r.mappings.return_value.all.return_value = [{"error_type": "timeout", "n": 2}]
        elif call_count[0] == 5:
            r.scalar.return_value = 2  # unresolved count
        else:
            r.mappings.return_value.all.return_value = []
        return r

    session.execute = mock_execute

    stats = await get_error_stats(pg)
    assert "error_rate" in stats
    assert "failed_docs" in stats
    assert stats["total_docs_attempted"] == 12
    assert stats["failed_docs"] == 2


@pytest.mark.asyncio
async def test_get_error_stats_swallows_exception():
    from dewie.storage.pipeline_errors import get_error_stats

    pg, session = _make_pg()
    session.execute = AsyncMock(side_effect=Exception("db error"))

    stats = await get_error_stats(pg)
    assert stats["error_rate"] == 0.0
    assert stats["failed_docs"] == 0
    assert "step_breakdown" in stats


# ── ERROR_RATE_THRESHOLD constant ────────────────────────────────────────────


def test_error_rate_threshold():
    from dewie.storage.pipeline_errors import ERROR_RATE_THRESHOLD

    assert ERROR_RATE_THRESHOLD == 0.05
