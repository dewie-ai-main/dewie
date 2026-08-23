# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Per-step output validators for the enrichment pipeline.

Each validator is called after its corresponding task completes.
On failure, raises StepValidationError with step name, doc_id, and reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class StepValidationError(Exception):
    """Raised when a pipeline step produces invalid output."""

    def __init__(self, step: str, doc_id: str | None, reason: str) -> None:
        self.step = step
        self.doc_id = doc_id
        self.reason = reason
        super().__init__(f"[{step}] doc={doc_id}: {reason}")


def validate_load_body(doc_id: str | None, body: object) -> None:
    """Body must be a non-empty string."""
    if not isinstance(body, str) or not body:
        raise StepValidationError("load_body", doc_id, "body is None or empty")


def validate_llm_extraction(result: object) -> None:
    """ExtractionResult must have meaningful content: summary, embed_summary, AQ, and keywords."""
    from dewie.enrichment.base import ExtractionResult

    if not isinstance(result, ExtractionResult):
        raise StepValidationError(
            "llm_extraction", None, f"result is not an ExtractionResult: {type(result)}"
        )
    if not result.summary or not result.summary.strip():
        raise StepValidationError("llm_extraction", None, "summary is empty")
    if not result.embed_summary or not result.embed_summary.strip():
        raise StepValidationError("llm_extraction", None, "embed_summary is empty")
    if not result.answers_questions:
        raise StepValidationError("llm_extraction", None, "answers_questions is empty")
    if not result.keywords:
        raise StepValidationError("llm_extraction", None, "keywords list is empty")


def validate_field_population(doc: object) -> None:
    """Document must have status=READY and enriched_at set."""
    from dewie.models.content import ContentStatus

    doc_id = str(getattr(doc, "id", None))
    status = getattr(doc, "status", None)
    enriched_at = getattr(doc, "enriched_at", None)

    if status != ContentStatus.READY:
        raise StepValidationError(
            "field_population", doc_id, f"status is {status!r}, expected READY"
        )
    if enriched_at is None:
        raise StepValidationError("field_population", doc_id, "enriched_at is not set")


def validate_db_upsert(doc_id: str | None, exc: BaseException | None) -> None:
    """Upsert must have completed without exception."""
    if exc is not None:
        raise StepValidationError("db_upsert", doc_id, f"upsert raised: {exc}")


def validate_embedding(doc_id: str | None, vector: object) -> None:
    """Vector must be a list of exactly 1536 floats with no None values."""
    if vector is None:
        raise StepValidationError("embedding", doc_id, "vector is None")
    if not isinstance(vector, list):
        raise StepValidationError("embedding", doc_id, f"vector is not a list: {type(vector)}")
    if len(vector) != 1536:
        raise StepValidationError(
            "embedding", doc_id, f"vector has {len(vector)} elements, expected 1536"
        )
    if any(v is None for v in vector):
        raise StepValidationError("embedding", doc_id, "vector contains None values")


def validate_relationships(doc_id: str | None, count: object) -> None:
    """Relationship count must be a non-negative integer."""
    if not isinstance(count, int) or count < 0:
        raise StepValidationError("relationships", doc_id, f"count must be int >= 0, got {count!r}")
