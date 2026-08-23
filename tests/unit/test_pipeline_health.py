"""
Unit tests for the pipeline health monitor (Issue #19).

Covers:
- Each step validator: passing and failing
- classify_error for each error type
- write_error: non-fatal, calls DB
- get_error_stats: correct rate calculation, above_threshold logic, 429 exclusion
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from dewie.enrichment.validators import (
    StepValidationError,
    validate_db_upsert,
    validate_embedding,
    validate_field_population,
    validate_llm_extraction,
    validate_load_body,
    validate_relationships,
)
from dewie.models.content import ContentDocument, ContentStatus
from dewie.storage.pipeline_errors import classify_error, get_error_stats, write_error

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_doc(**kwargs) -> ContentDocument:
    defaults = {
        "url": "https://example.com/test",
        "title": "Test Document",
        "source": "example.com",
        "status": ContentStatus.READY,
        "enriched_at": datetime.utcnow(),
    }
    defaults.update(kwargs)
    return ContentDocument(**defaults)


def _make_extraction_result(embed_summary: str = "Dense retrieval summary.", keywords=None):
    """Return a minimal real ExtractionResult."""
    from dewie.enrichment.base import ExtractionResult

    return ExtractionResult(
        document_type="blog_post",
        author=None,
        tone="informative",
        reading_level="standard",
        keywords=keywords if keywords is not None else ["ai", "graph"],
        themes=["knowledge graphs"],
        entities=[],
        summary="A short summary.",
        quality_score=80,
        sentiment=0.5,
        language="en",
        answers_questions=["What is AI?"],
        missing_coverage=[],
        embed_summary=embed_summary,
    )


def _make_pg_for_stats(
    successful: int,
    failed: int,
    by_step_rows=None,
    by_type_rows=None,
    unresolved_count: int = 0,
    unresolved_errors=None,
):
    """
    Build a mock PostgresClient whose _session_factory returns a session
    that answers 6 execute() calls in order:
      1. successful_in_window scalar  (SELECT COUNT(*) FROM documents WHERE status='ready' ...)
      2. failed_docs scalar           (SELECT COUNT(DISTINCT doc_id) FROM pipeline_errors ...)
      3. by_step mappings
      4. by_type mappings
      5. unresolved_count scalar (all-time)
      6. unresolved_errors mappings (all-time)
    total_docs_attempted = successful + failed (fixes the denominator bug)
    """
    if by_step_rows is None:
        by_step_rows = []
    if by_type_rows is None:
        by_type_rows = []
    if unresolved_errors is None:
        unresolved_errors = []

    session = AsyncMock()

    successful_result = MagicMock()
    successful_result.scalar.return_value = successful

    failed_result = MagicMock()
    failed_result.scalar.return_value = failed

    step_result = MagicMock()
    step_result.mappings.return_value.all.return_value = by_step_rows

    type_result = MagicMock()
    type_result.mappings.return_value.all.return_value = by_type_rows

    unresolved_count_result = MagicMock()
    unresolved_count_result.scalar.return_value = unresolved_count

    unresolved_rows_result = MagicMock()
    unresolved_rows_result.mappings.return_value.all.return_value = unresolved_errors

    session.execute.side_effect = [
        successful_result,
        failed_result,
        step_result,
        type_result,
        unresolved_count_result,
        unresolved_rows_result,
    ]

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._session_factory.return_value = cm
    return pg


def _make_pg_for_write():
    """Build a mock PostgresClient for write_error."""
    session = AsyncMock()
    exec_result = MagicMock()
    session.execute.return_value = exec_result

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._session_factory.return_value = cm
    return pg, session


# ── validate_load_body ─────────────────────────────────────────────────────────


def test_validate_load_body_passes():
    validate_load_body("doc-1", "some body text")  # must not raise


def test_validate_load_body_fails_none():
    with pytest.raises(StepValidationError) as exc_info:
        validate_load_body("doc-1", None)
    assert exc_info.value.step == "load_body"


def test_validate_load_body_fails_empty():
    with pytest.raises(StepValidationError):
        validate_load_body("doc-1", "")


def test_validate_load_body_fails_non_string():
    with pytest.raises(StepValidationError):
        validate_load_body("doc-1", 42)


# ── validate_llm_extraction ────────────────────────────────────────────────────


def test_validate_llm_extraction_passes():
    result = _make_extraction_result()
    validate_llm_extraction(result)  # must not raise


def test_validate_llm_extraction_fails_not_extraction_result():
    with pytest.raises(StepValidationError) as exc_info:
        validate_llm_extraction({"embed_summary": "x", "keywords": ["a"]})
    assert exc_info.value.step == "llm_extraction"


def test_validate_llm_extraction_fails_empty_embed_summary():
    result = _make_extraction_result(embed_summary="")
    with pytest.raises(StepValidationError) as exc_info:
        validate_llm_extraction(result)
    assert "embed_summary" in str(exc_info.value)


def test_validate_llm_extraction_fails_empty_keywords():
    result = _make_extraction_result(keywords=[])
    with pytest.raises(StepValidationError) as exc_info:
        validate_llm_extraction(result)
    assert "keywords" in str(exc_info.value)


# ── validate_field_population ──────────────────────────────────────────────────


def test_validate_field_population_passes():
    doc = _make_doc()
    validate_field_population(doc)  # must not raise


def test_validate_field_population_fails_wrong_status():
    doc = _make_doc(status=ContentStatus.PENDING)
    with pytest.raises(StepValidationError) as exc_info:
        validate_field_population(doc)
    assert exc_info.value.step == "field_population"


def test_validate_field_population_fails_no_enriched_at():
    doc = _make_doc(enriched_at=None)
    with pytest.raises(StepValidationError) as exc_info:
        validate_field_population(doc)
    assert "enriched_at" in str(exc_info.value)


# ── validate_db_upsert ─────────────────────────────────────────────────────────


def test_validate_db_upsert_passes():
    validate_db_upsert("doc-1", None)  # no exception → must not raise


def test_validate_db_upsert_fails_with_exception():
    exc = RuntimeError("DB connection failed")
    with pytest.raises(StepValidationError) as exc_info:
        validate_db_upsert("doc-1", exc)
    assert exc_info.value.step == "db_upsert"


# ── validate_embedding ─────────────────────────────────────────────────────────


def test_validate_embedding_passes():
    vector = [0.1] * 1536
    validate_embedding("doc-1", vector)  # must not raise


def test_validate_embedding_fails_none():
    with pytest.raises(StepValidationError) as exc_info:
        validate_embedding("doc-1", None)
    assert exc_info.value.step == "embedding"


def test_validate_embedding_fails_wrong_length():
    with pytest.raises(StepValidationError) as exc_info:
        validate_embedding("doc-1", [0.1] * 512)
    assert "1536" in str(exc_info.value)


def test_validate_embedding_fails_contains_none():
    vector = [0.1] * 1535 + [None]  # type: ignore[list-item]
    with pytest.raises(StepValidationError) as exc_info:
        validate_embedding("doc-1", vector)
    assert "None" in str(exc_info.value)


def test_validate_embedding_fails_not_list():
    with pytest.raises(StepValidationError):
        validate_embedding("doc-1", "not a list")


# ── validate_relationships ─────────────────────────────────────────────────────


def test_validate_relationships_passes_zero():
    validate_relationships("doc-1", 0)  # must not raise


def test_validate_relationships_passes_positive():
    validate_relationships("doc-1", 42)


def test_validate_relationships_fails_negative():
    with pytest.raises(StepValidationError) as exc_info:
        validate_relationships("doc-1", -1)
    assert exc_info.value.step == "relationships"


def test_validate_relationships_fails_non_int():
    with pytest.raises(StepValidationError):
        validate_relationships("doc-1", "three")


# ── classify_error ─────────────────────────────────────────────────────────────


def test_classify_error_429_by_status_code():
    assert classify_error(Exception("HTTP 429 Too Many Requests")) == "429"


def test_classify_error_429_by_rate_limit():
    assert classify_error(Exception("rate limit exceeded")) == "429"


def test_classify_error_429_by_too_many():
    assert classify_error(Exception("too many requests from your IP")) == "429"


def test_classify_error_timeout():
    assert classify_error(Exception("Request timeout after 30s")) == "timeout"


def test_classify_error_timed_out():
    assert classify_error(Exception("Connection timed out")) == "timeout"


def test_classify_error_parse_json():
    assert classify_error(Exception("Failed to parse JSON response")) == "parse"


def test_classify_error_parse_keyword():
    assert classify_error(Exception("parse error in response body")) == "parse"


def test_classify_error_parse_extraction_result():
    assert classify_error(Exception("ExtractionResult validation failed")) == "parse"


def test_classify_error_validation():
    exc = StepValidationError("embedding", "doc-1", "vector is None")
    assert classify_error(exc) == "validation"


def test_classify_error_unknown():
    assert classify_error(Exception("something else entirely")) == "unknown"


# ── write_error ────────────────────────────────────────────────────────────────


async def test_write_error_calls_db():
    pg, session = _make_pg_for_write()
    await write_error(pg, "doc-1", "embedding", "timeout", "Request timed out", retry_count=1)
    session.execute.assert_called_once()
    session.commit.assert_called_once()


async def test_write_error_never_raises_on_db_failure():
    """write_error must swallow DB exceptions."""
    pg = MagicMock()
    pg._session_factory.side_effect = Exception("DB is down")
    # Must not raise
    await write_error(pg, "doc-1", "embedding", "unknown", "boom")


# ── get_error_stats ────────────────────────────────────────────────────────────


async def test_get_error_stats_above_threshold_true():
    """
    Rate = 6/100 = 0.06 > 0.05 → above_threshold=True.
    successful=94, failed=6 → total_docs_attempted=100, rate=0.06
    """
    by_type_rows = [{"error_type": "parse", "n": 6}]
    by_step_rows = [{"step": "llm_extraction", "n": 6}]
    pg = _make_pg_for_stats(
        successful=94, failed=6, by_step_rows=by_step_rows, by_type_rows=by_type_rows
    )

    stats = await get_error_stats(pg, window_minutes=60)

    assert stats["total_docs_attempted"] == 100  # 94 successful + 6 failed
    assert stats["failed_docs"] == 6
    assert stats["error_rate"] == pytest.approx(0.06, abs=1e-6)
    assert stats["above_threshold"] is True


async def test_get_error_stats_above_threshold_false():
    """
    Rate = 5/100 = 0.05 — not strictly above 0.05 → above_threshold=False.
    successful=95, failed=5 → total=100, rate=0.05
    """
    pg = _make_pg_for_stats(successful=95, failed=5)

    stats = await get_error_stats(pg, window_minutes=60)

    assert stats["error_rate"] == pytest.approx(0.05, abs=1e-6)
    assert stats["above_threshold"] is False


async def test_get_error_stats_zero_docs():
    """No docs attempted → rate=0.0, above_threshold=False."""
    pg = _make_pg_for_stats(successful=0, failed=0)

    stats = await get_error_stats(pg, window_minutes=60)

    assert stats["error_rate"] == 0.0
    assert stats["above_threshold"] is False


async def test_get_error_stats_429_counts_toward_rate():
    """
    429 errors count toward the error rate.
    93 total docs failed out of 100 → rate 0.93 → above_threshold=True.
    successful=7, failed=93 → total=100, rate=93/100=0.93
    """
    by_type_rows = [
        {"error_type": "429", "n": 90},
        {"error_type": "parse", "n": 3},
    ]
    by_step_rows = [{"step": "llm_extraction", "n": 3}]
    pg = _make_pg_for_stats(
        successful=7, failed=93, by_step_rows=by_step_rows, by_type_rows=by_type_rows
    )

    stats = await get_error_stats(pg, window_minutes=60)

    assert stats["failed_docs"] == 93
    assert stats["error_rate"] == pytest.approx(0.93, abs=1e-6)
    assert stats["above_threshold"] is True
    assert stats["by_type"].get("429") == 90
    assert stats["by_type"].get("parse") == 3


async def test_get_error_stats_by_step_and_type():
    """Verify by_step and by_type mappings are correctly returned."""
    by_step_rows = [
        {"step": "embedding", "n": 10},
        {"step": "llm_extraction", "n": 5},
    ]
    by_type_rows = [
        {"error_type": "timeout", "n": 8},
        {"error_type": "parse", "n": 7},
    ]
    pg = _make_pg_for_stats(
        successful=185, failed=15, by_step_rows=by_step_rows, by_type_rows=by_type_rows
    )

    stats = await get_error_stats(pg, window_minutes=60)

    assert stats["by_step"] == {"embedding": 10, "llm_extraction": 5}
    assert stats["by_type"] == {"timeout": 8, "parse": 7}


async def test_get_error_stats_returns_safe_default_on_db_error():
    """If the DB is down, get_error_stats must return safe zero values."""
    pg = MagicMock()
    pg._session_factory.side_effect = Exception("DB is down")

    stats = await get_error_stats(pg, window_minutes=60)

    assert stats["total_docs_attempted"] == 0
    assert stats["error_rate"] == 0.0
    assert stats["above_threshold"] is False
    assert "step_breakdown" in stats


async def test_get_error_stats_step_breakdown_included():
    """step_breakdown must include all 6 pipeline steps with error counts and pct."""
    by_step_rows = [
        {"step": "llm_extraction", "n": 10},
        {"step": "embedding", "n": 2},
    ]
    # successful=88, failed=12 → total=100
    pg = _make_pg_for_stats(successful=88, failed=12, by_step_rows=by_step_rows)

    stats = await get_error_stats(pg, window_minutes=60)

    sb = stats["step_breakdown"]
    assert set(sb.keys()) == {
        "load_body",
        "llm_extraction",
        "field_population",
        "db_upsert",
        "embedding",
        "relationships",
    }
    assert sb["llm_extraction"]["errors"] == 10
    assert sb["llm_extraction"]["pct_of_total_processed"] == pytest.approx(10.0, abs=0.1)
    assert sb["embedding"]["errors"] == 2
    assert sb["load_body"]["errors"] == 0
    assert sb["load_body"]["pct_of_total_processed"] == 0.0


# ── mark_resolved ──────────────────────────────────────────────────────────────


async def test_mark_resolved_requeue_true():
    """mark_resolved with requeue=True should update pipeline_errors AND documents."""
    session = AsyncMock()
    rowcount_result = MagicMock()
    rowcount_result.rowcount = 2
    session.execute.side_effect = [MagicMock(), rowcount_result]

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._session_factory.return_value = cm

    from dewie.storage.pipeline_errors import mark_resolved

    resolved, requeued = await mark_resolved(pg, [1, 2, 3], requeue=True)

    assert resolved == 3
    assert requeued == 2  # rowcount from docs UPDATE
    assert session.execute.call_count == 2  # errors UPDATE + docs UPDATE
    session.commit.assert_called_once()


async def test_mark_resolved_requeue_false():
    """mark_resolved with requeue=False should only update pipeline_errors."""
    session = AsyncMock()
    session.execute.return_value = MagicMock()

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._session_factory.return_value = cm

    from dewie.storage.pipeline_errors import mark_resolved

    resolved, requeued = await mark_resolved(pg, [1, 2], requeue=False)

    assert resolved == 2
    assert requeued == 0
    assert session.execute.call_count == 1  # only errors UPDATE


async def test_mark_resolved_empty_ids():
    """mark_resolved with empty list is a no-op."""
    pg = MagicMock()
    from dewie.storage.pipeline_errors import mark_resolved

    resolved, requeued = await mark_resolved(pg, [], requeue=True)
    assert resolved == 0
    assert requeued == 0
    pg._session_factory.assert_not_called()
