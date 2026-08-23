"""Tests for dewie.enrichment.validators."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import MagicMock

import pytest

# ── validate_load_body ────────────────────────────────────────────────────────


def test_validate_load_body_ok():
    from dewie.enrichment.validators import validate_load_body

    validate_load_body("doc-1", "some body text")


def test_validate_load_body_none_raises():
    from dewie.enrichment.validators import StepValidationError, validate_load_body

    with pytest.raises(StepValidationError, match="empty"):
        validate_load_body("doc-1", None)


def test_validate_load_body_empty_string_raises():
    from dewie.enrichment.validators import StepValidationError, validate_load_body

    with pytest.raises(StepValidationError):
        validate_load_body("doc-1", "")


# ── validate_llm_extraction ───────────────────────────────────────────────────


def _make_extraction(**kwargs):
    from dewie.enrichment.base import ExtractionResult

    defaults = dict(
        summary="A summary",
        embed_summary="Embed summary",
        answers_questions=["What is this?"],
        keywords=["ai"],
        entities=[],
        topics=[],
        sentiment=0.0,
        tone="neutral",
        document_type="blog_post",
        reading_level="intermediate",
        author=None,
        alternate_terms=[],
        enrichment_quality_score=50,
    )
    defaults.update(kwargs)
    return ExtractionResult(**defaults)


def test_validate_llm_extraction_ok():
    from dewie.enrichment.validators import validate_llm_extraction

    result = _make_extraction()
    validate_llm_extraction(result)


def test_validate_llm_extraction_empty_summary_raises():
    from dewie.enrichment.validators import StepValidationError, validate_llm_extraction

    result = _make_extraction(summary="")
    with pytest.raises(StepValidationError, match="summary"):
        validate_llm_extraction(result)


def test_validate_llm_extraction_not_extraction_result_raises():
    from dewie.enrichment.validators import StepValidationError, validate_llm_extraction

    with pytest.raises(StepValidationError):
        validate_llm_extraction({"summary": "x"})


def test_validate_llm_extraction_empty_aq_raises():
    from dewie.enrichment.validators import StepValidationError, validate_llm_extraction

    result = _make_extraction(answers_questions=[])
    with pytest.raises(StepValidationError, match="answers_questions"):
        validate_llm_extraction(result)


def test_validate_llm_extraction_empty_keywords_raises():
    from dewie.enrichment.validators import StepValidationError, validate_llm_extraction

    result = _make_extraction(keywords=[])
    with pytest.raises(StepValidationError, match="keywords"):
        validate_llm_extraction(result)


# ── validate_field_population ─────────────────────────────────────────────────


def test_validate_field_population_ok():
    from datetime import datetime

    from dewie.enrichment.validators import validate_field_population
    from dewie.models.content import ContentStatus

    doc = MagicMock()
    doc.status = ContentStatus.READY
    doc.enriched_at = datetime.now(UTC)
    validate_field_population(doc)


def test_validate_field_population_wrong_status_raises():
    from dewie.enrichment.validators import StepValidationError, validate_field_population
    from dewie.models.content import ContentStatus

    doc = MagicMock()
    doc.status = ContentStatus.PENDING
    doc.enriched_at = None
    with pytest.raises(StepValidationError, match="PENDING"):
        validate_field_population(doc)


def test_validate_field_population_no_enriched_at_raises():
    from dewie.enrichment.validators import StepValidationError, validate_field_population
    from dewie.models.content import ContentStatus

    doc = MagicMock()
    doc.status = ContentStatus.READY
    doc.enriched_at = None
    with pytest.raises(StepValidationError, match="enriched_at"):
        validate_field_population(doc)


# ── validate_db_upsert ────────────────────────────────────────────────────────


def test_validate_db_upsert_ok():
    from dewie.enrichment.validators import validate_db_upsert

    validate_db_upsert("doc-1", None)


def test_validate_db_upsert_raises_on_exception():
    from dewie.enrichment.validators import StepValidationError, validate_db_upsert

    with pytest.raises(StepValidationError, match="upsert raised"):
        validate_db_upsert("doc-1", ValueError("db error"))


# ── validate_embedding ────────────────────────────────────────────────────────


def test_validate_embedding_ok():
    from dewie.enrichment.validators import validate_embedding

    validate_embedding("doc-1", [0.1] * 1536)


def test_validate_embedding_none_raises():
    from dewie.enrichment.validators import StepValidationError, validate_embedding

    with pytest.raises(StepValidationError, match="None"):
        validate_embedding("doc-1", None)


def test_validate_embedding_wrong_length_raises():
    from dewie.enrichment.validators import StepValidationError, validate_embedding

    with pytest.raises(StepValidationError, match="768"):
        validate_embedding("doc-1", [0.1] * 768)


def test_validate_embedding_not_list_raises():
    from dewie.enrichment.validators import StepValidationError, validate_embedding

    with pytest.raises(StepValidationError):
        validate_embedding("doc-1", "not a list")


def test_validate_embedding_with_none_values_raises():
    from dewie.enrichment.validators import StepValidationError, validate_embedding

    vec = [0.1] * 1536
    vec[100] = None
    with pytest.raises(StepValidationError, match="None values"):
        validate_embedding("doc-1", vec)


# ── validate_relationships ────────────────────────────────────────────────────


def test_validate_relationships_ok():
    from dewie.enrichment.validators import validate_relationships

    validate_relationships("doc-1", 0)
    validate_relationships("doc-1", 5)


def test_validate_relationships_negative_raises():
    from dewie.enrichment.validators import StepValidationError, validate_relationships

    with pytest.raises(StepValidationError):
        validate_relationships("doc-1", -1)


def test_validate_relationships_non_int_raises():
    from dewie.enrichment.validators import StepValidationError, validate_relationships

    with pytest.raises(StepValidationError):
        validate_relationships("doc-1", "5")
