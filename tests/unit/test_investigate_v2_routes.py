"""Extra unit tests for dewie.api.routes.investigate_v2 helpers.

The main endpoint tests (create_investigate_job, get_investigate_job)
are already covered in test_investigate_routes.py via TestClient.
This file focuses on the lower-level helpers and Pydantic model validation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── DB mock helper ────────────────────────────────────────────────────────────


def _make_pg(fetchone_return=None):
    """Return a minimal pg mock matching the pattern used in investigate_v2."""
    mock_result = MagicMock()
    mock_result.fetchone.return_value = fetchone_return

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=None)

    pg = AsyncMock()
    pg._engine = MagicMock()
    pg._engine.begin.return_value = cm
    pg._engine.connect.return_value = cm

    return pg, mock_conn


# ── _update_job ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_job_no_kwargs_noop():
    """_update_job is a no-op when no kwargs given."""
    from dewie.api.routes.investigate_v2 import _update_job

    pg, _ = _make_pg()
    await _update_job(pg, "some-job-id")
    pg._engine.begin.assert_not_called()


@pytest.mark.asyncio
async def test_update_job_scalar_value():
    """_update_job builds a correct SET clause for scalar values."""
    from dewie.api.routes.investigate_v2 import _update_job

    pg, mock_conn = _make_pg()
    await _update_job(pg, "abc-123", status="running")
    pg._engine.begin.assert_called_once()
    mock_conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_update_job_dict_value():
    """_update_job handles dict values (JSONB CAST) without errors."""
    from dewie.api.routes.investigate_v2 import _update_job

    pg, _ = _make_pg()
    await _update_job(pg, "abc-123", plan={"sub_questions": ["What is X?"]})
    pg._engine.begin.assert_called_once()


@pytest.mark.asyncio
async def test_update_job_list_value():
    """_update_job handles list values (JSONB CAST) without errors."""
    from dewie.api.routes.investigate_v2 import _update_job

    pg, _ = _make_pg()
    await _update_job(pg, "abc-123", result=["item1", "item2"])
    pg._engine.begin.assert_called_once()


@pytest.mark.asyncio
async def test_update_job_multiple_kwargs():
    """_update_job handles multiple kwargs in one call."""
    from dewie.api.routes.investigate_v2 import _update_job

    pg, mock_conn = _make_pg()
    await _update_job(pg, "abc-123", status="done", result={"report": "x"})
    pg._engine.begin.assert_called_once()
    mock_conn.execute.assert_called_once()


# ── _fetch_job ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_job_not_found():
    """_fetch_job returns None when no row matches."""
    from dewie.api.routes.investigate_v2 import _fetch_job

    pg, _ = _make_pg(fetchone_return=None)
    result = await _fetch_job(pg, "nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_job_found_with_datetime_timestamps():
    """_fetch_job returns a dict with ISO-formatted timestamps when row exists."""
    from dewie.api.routes.investigate_v2 import _fetch_job

    now = datetime.now(UTC)
    job_id = str(uuid.uuid4())
    row = (job_id, "What is AI?", "matrix", "done", None, {"report": "x"}, None, now, now, now)
    pg, _ = _make_pg(fetchone_return=row)
    result = await _fetch_job(pg, job_id)
    assert result is not None
    assert result["id"] == job_id
    assert result["query"] == "What is AI?"
    assert result["status"] == "done"
    assert isinstance(result["created_at"], str)
    assert "T" in result["created_at"]  # ISO format


@pytest.mark.asyncio
async def test_fetch_job_string_timestamps_preserved():
    """_fetch_job preserves timestamps that are already strings."""
    from dewie.api.routes.investigate_v2 import _fetch_job

    row = (
        "job-456",
        "Test query",
        "subquestion",
        "pending",
        None,
        None,
        None,
        "2026-05-13T12:00:00+00:00",
        None,
        None,
    )
    pg, _ = _make_pg(fetchone_return=row)
    result = await _fetch_job(pg, "job-456")
    assert result["created_at"] == "2026-05-13T12:00:00+00:00"
    assert result["started_at"] is None
    assert result["completed_at"] is None


@pytest.mark.asyncio
async def test_fetch_job_none_timestamps_stay_none():
    """_fetch_job maps all-None timestamps to None in output."""
    from dewie.api.routes.investigate_v2 import _fetch_job

    row = ("j1", "q", "plan", "pending", None, None, None, None, None, None)
    pg, _ = _make_pg(fetchone_return=row)
    result = await _fetch_job(pg, "j1")
    assert result["started_at"] is None
    assert result["completed_at"] is None


# ── InvestigateJobRequest validation ──────────────────────────────────────────


def test_investigate_job_request_defaults():
    from dewie.api.routes.investigate_v2 import InvestigateJobRequest

    req = InvestigateJobRequest(query="test query")
    assert req.strategy == "matrix"
    assert req.num_sources == 5
    assert req.ingest is True
    assert req.context is None


def test_investigate_job_request_empty_query_raises():
    from pydantic import ValidationError

    from dewie.api.routes.investigate_v2 import InvestigateJobRequest

    with pytest.raises(ValidationError):
        InvestigateJobRequest(query="")


def test_investigate_job_request_invalid_strategy_raises():
    from pydantic import ValidationError

    from dewie.api.routes.investigate_v2 import InvestigateJobRequest

    with pytest.raises(ValidationError):
        InvestigateJobRequest(query="valid", strategy="nonsense")


def test_investigate_job_request_valid_strategies():
    from dewie.api.routes.investigate_v2 import InvestigateJobRequest

    for strategy in ("matrix", "subquestion", "plan"):
        req = InvestigateJobRequest(query="some query", strategy=strategy)
        assert req.strategy == strategy


def test_investigate_job_request_custom_num_sources():
    from dewie.api.routes.investigate_v2 import InvestigateJobRequest

    req = InvestigateJobRequest(query="some query", num_sources=20)
    assert req.num_sources == 20


# ── _now helper ───────────────────────────────────────────────────────────────


def test_now_returns_utc_datetime():
    from dewie.api.routes.investigate_v2 import _now

    result = _now()
    assert result.tzinfo is not None
    delta = abs((datetime.now(UTC) - result).total_seconds())
    assert delta < 5
