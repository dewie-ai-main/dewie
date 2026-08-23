"""Tests for dewie.storage.pipeline_errors — classify_error and helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# ── classify_error ────────────────────────────────────────────────────────────


def test_classify_error_rate_limit_429():
    from dewie.storage.pipeline_errors import classify_error

    assert classify_error(Exception("HTTP 429 rate limit")) == "429"


def test_classify_error_too_many_requests():
    from dewie.storage.pipeline_errors import classify_error

    assert classify_error(Exception("Too many requests")) == "429"


def test_classify_error_timeout():
    from dewie.storage.pipeline_errors import classify_error

    assert classify_error(Exception("Connection timed out")) == "timeout"


def test_classify_error_timeout_variant():
    from dewie.storage.pipeline_errors import classify_error

    assert classify_error(Exception("timeout after 30s")) == "timeout"


def test_classify_error_parse_json():
    from dewie.storage.pipeline_errors import classify_error

    assert classify_error(Exception("Failed to parse json response")) == "parse"


def test_classify_error_parse_variant():
    from dewie.storage.pipeline_errors import classify_error

    assert classify_error(Exception("parse error in extraction")) == "parse"


def test_classify_error_unknown():
    from dewie.storage.pipeline_errors import classify_error

    assert classify_error(Exception("something weird happened")) == "unknown"


def test_classify_error_validation():
    from dewie.enrichment.validators import StepValidationError
    from dewie.storage.pipeline_errors import classify_error

    exc = StepValidationError("load_body", "doc-1", "Body is empty")
    assert classify_error(exc) == "validation"


# ── ERROR_RATE_THRESHOLD ──────────────────────────────────────────────────────


def test_error_rate_threshold():
    from dewie.storage.pipeline_errors import ERROR_RATE_THRESHOLD

    assert 0 < ERROR_RATE_THRESHOLD <= 1.0


# ── write_error — never raises ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_error_never_raises():
    from dewie.storage.pipeline_errors import write_error

    pg = MagicMock()
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=Exception("DB down"))
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._session_factory = MagicMock(return_value=cm)
    # Should not raise even when DB fails
    await write_error(pg, "doc-1", "load_body", "timeout", "timed out")


@pytest.mark.asyncio
async def test_write_error_success():
    from dewie.storage.pipeline_errors import write_error

    pg = MagicMock()
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._session_factory = MagicMock(return_value=cm)
    await write_error(pg, "doc-1", "llm_extraction", "parse", "bad json")
    session.execute.assert_called_once()
    session.commit.assert_called_once()
