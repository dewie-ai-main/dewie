"""Tests for dewie.api.routes.health — _get_git_info and helpers."""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── _get_git_info ─────────────────────────────────────────────────────────────


def test_get_git_info_success():
    from dewie.api.routes.health import _get_git_info

    with patch("dewie.api.routes.health.subprocess.check_output") as mock_co:
        mock_co.side_effect = ["abc1234def\n", "v1.0.0-3-gabc1234\n"]
        result = _get_git_info()
    assert result["git_sha"] == "abc1234def"
    assert result["git_describe"] == "v1.0.0-3-gabc1234"


def test_get_git_info_failure_returns_unknown():
    from dewie.api.routes.health import _get_git_info

    with patch(
        "dewie.api.routes.health.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "git"),
    ):
        result = _get_git_info()
    assert result["git_sha"] == "unknown"
    assert result["git_describe"] == "unknown"


def test_get_git_info_timeout_returns_unknown():
    from dewie.api.routes.health import _get_git_info

    with patch(
        "dewie.api.routes.health.subprocess.check_output",
        side_effect=subprocess.TimeoutExpired("git", 5),
    ):
        result = _get_git_info()
    assert result["git_sha"] == "unknown"
    assert result["git_describe"] == "unknown"


def test_get_git_info_file_not_found_returns_unknown():
    from dewie.api.routes.health import _get_git_info

    with patch(
        "dewie.api.routes.health.subprocess.check_output",
        side_effect=FileNotFoundError("git not found"),
    ):
        result = _get_git_info()
    assert result["git_sha"] == "unknown"
    assert result["git_describe"] == "unknown"


# ── RecordE2ERequest model ─────────────────────────────────────────────────────


def test_record_e2e_request_defaults():
    from dewie.api.routes.health import RecordE2ERequest

    r = RecordE2ERequest(doc_id="d", git_sha="sha", status="ok")
    assert r.has_embedding is False
    assert r.aq_count == 0
    assert r.elapsed_seconds == 0.0
    assert r.error is None
    assert r.enriched_at is None


def test_record_e2e_request_with_values():
    from dewie.api.routes.health import RecordE2ERequest

    r = RecordE2ERequest(
        doc_id="abc",
        git_sha="sha123",
        status="ok",
        has_embedding=True,
        aq_count=5,
        elapsed_seconds=1.5,
    )
    assert r.has_embedding is True
    assert r.aq_count == 5


# ── record_heartbeat ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_heartbeat():
    from dewie.api.routes.health import record_heartbeat

    pg = MagicMock()
    req = MagicMock()
    req.app.state.postgres = pg

    with patch("dewie.storage.system_health.write_health_kv", AsyncMock()):
        result = await record_heartbeat(req)

    assert result["ok"] is True
    assert "timestamp" in result


# ── record_e2e ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_e2e():
    from dewie.api.routes.health import RecordE2ERequest, record_e2e

    pg = MagicMock()
    req = MagicMock()
    req.app.state.postgres = pg
    body = RecordE2ERequest(doc_id="doc-1", git_sha="abc", status="ok")

    with patch("dewie.storage.system_health.write_health_kv", AsyncMock()):
        result = await record_e2e(req, body)

    assert result["ok"] is True
    assert result["record"]["status"] == "ok"
    assert result["record"]["doc_id"] == "doc-1"


# ── get_pipeline_health ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_pipeline_health_no_heartbeat():
    from dewie.api.routes.health import get_pipeline_health

    conn = AsyncMock()
    row_result = MagicMock()
    row_map = {
        "ready": 0,
        "pending": 0,
        "failed": 0,
        "with_embedding": 0,
        "enriched_30min": 0,
        "enriched_5min": 0,
    }
    row_result.mappings.return_value.one.return_value = row_map
    row_result.mappings.return_value.__iter__ = MagicMock(return_value=iter([]))
    conn.execute = AsyncMock(return_value=row_result)
    begin_cm = AsyncMock()
    begin_cm.__aenter__ = AsyncMock(return_value=conn)
    begin_cm.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._engine.begin.return_value = begin_cm
    cache = MagicMock()
    cache._redis.get = AsyncMock(return_value=None)
    req = MagicMock()
    req.app.state.postgres = pg
    req.app.state.cache = cache

    error_stats = {
        "total_docs_attempted": 0,
        "failed_docs": 0,
        "error_rate": 0.0,
        "above_threshold": False,
        "any_step_above_threshold": False,
        "step_breakdown": {},
    }

    with (
        patch("dewie.storage.system_health.read_health_kv", AsyncMock(return_value=None)),
        patch(
            "dewie.storage.pipeline_errors.get_error_stats", AsyncMock(return_value=error_stats)
        ),
        patch(
            "dewie.api.routes.health._get_git_info",
            return_value={"git_sha": "abc", "git_describe": "v1"},
        ),
    ):
        result = await get_pipeline_health(req)

    assert result["last_heartbeat"]["ok"] is False
    assert result["last_heartbeat"]["value"] is None
    assert "current_version" in result
