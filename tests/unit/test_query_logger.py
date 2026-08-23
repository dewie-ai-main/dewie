"""Tests for dewie.storage.query_logger."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dewie.storage.query_logger import QueryLogEntry, log_query

# ── QueryLogEntry dataclass ───────────────────────────────────────────────────


def test_query_log_entry_defaults():
    entry = QueryLogEntry(question="What is AI?")
    assert entry.question == "What is AI?"
    assert entry.model is None
    assert entry.source == "api"
    assert entry.session_id is None
    assert entry.tenant_id is None
    assert entry.user_id is None
    assert entry.hops == 0
    assert entry.hop_trace == []
    assert entry.docs_returned == []
    assert entry.full_results is None
    assert entry.answer is None
    assert entry.correct is None
    assert entry.input_tokens == 0
    assert entry.output_tokens == 0
    assert entry.cost_usd == 0.0
    assert entry.elapsed_ms == 0


def test_query_log_entry_custom():
    entry = QueryLogEntry(
        question="test",
        model="gpt-4",
        source="benchmark",
        hops=3,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.01,
        elapsed_ms=500,
        correct=True,
        answer="42",
    )
    assert entry.model == "gpt-4"
    assert entry.source == "benchmark"
    assert entry.hops == 3
    assert entry.input_tokens == 100
    assert entry.correct is True
    assert entry.answer == "42"


def test_query_log_entry_mutable_defaults_are_independent():
    a = QueryLogEntry(question="a")
    b = QueryLogEntry(question="b")
    a.hop_trace.append({"step": 1})
    assert b.hop_trace == []


# ── log_query ─────────────────────────────────────────────────────────────────


def _make_session_mock():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, session


@pytest.mark.asyncio
async def test_log_query_calls_execute_and_commit():
    ctx, session = _make_session_mock()
    factory = MagicMock(return_value=ctx)

    with patch("dewie.storage.query_logger._get_factory", return_value=factory):
        await log_query(QueryLogEntry(question="What is ML?", model="gpt-4"))

    session.execute.assert_called_once()
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_log_query_passes_correct_params():
    ctx, session = _make_session_mock()
    factory = MagicMock(return_value=ctx)

    entry = QueryLogEntry(
        question="test question",
        model="claude",
        source="benchmark",
        hops=2,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.002,
        elapsed_ms=300,
    )

    with patch("dewie.storage.query_logger._get_factory", return_value=factory):
        await log_query(entry)

    _, call_kwargs = session.execute.call_args
    params = session.execute.call_args[0][1]
    assert params["question"] == "test question"
    assert params["model"] == "claude"
    assert params["source"] == "benchmark"
    assert params["hops"] == 2
    assert params["input_tokens"] == 10
    assert params["cost_usd"] == 0.002


@pytest.mark.asyncio
async def test_log_query_full_results_none_when_setting_off():
    ctx, session = _make_session_mock()
    factory = MagicMock(return_value=ctx)

    entry = QueryLogEntry(question="q", full_results={"key": "val"})

    with (
        patch("dewie.storage.query_logger._get_factory", return_value=factory),
        patch("dewie.storage.query_logger.settings") as mock_settings,
    ):
        mock_settings.query_log_save_full_results = False
        await log_query(entry)

    params = session.execute.call_args[0][1]
    assert params["full_results"] is None


@pytest.mark.asyncio
async def test_log_query_full_results_serialized_when_setting_on():
    import json

    ctx, session = _make_session_mock()
    factory = MagicMock(return_value=ctx)

    entry = QueryLogEntry(question="q", full_results={"key": "val"})

    with (
        patch("dewie.storage.query_logger._get_factory", return_value=factory),
        patch("dewie.storage.query_logger.settings") as mock_settings,
    ):
        mock_settings.query_log_save_full_results = True
        await log_query(entry)

    params = session.execute.call_args[0][1]
    assert params["full_results"] == json.dumps({"key": "val"})


@pytest.mark.asyncio
async def test_log_query_swallows_exceptions():
    factory = MagicMock(side_effect=Exception("db is down"))

    with patch("dewie.storage.query_logger._get_factory", return_value=factory):
        # Should not raise
        await log_query(QueryLogEntry(question="q"))


@pytest.mark.asyncio
async def test_log_query_swallows_execute_error():
    ctx, session = _make_session_mock()
    session.execute = AsyncMock(side_effect=Exception("table not found"))
    factory = MagicMock(return_value=ctx)

    with patch("dewie.storage.query_logger._get_factory", return_value=factory):
        await log_query(QueryLogEntry(question="q"))
