# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Pipeline error store: write, query, and classify enrichment pipeline errors.

Functions are intentionally non-fatal: write_error never raises so that
error logging cannot cause additional failures in already-failing tasks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from dewie.storage.postgres import PostgresClient

logger = logging.getLogger(__name__)

# Error rate threshold above which alerting should fire
ERROR_RATE_THRESHOLD = 0.05


def classify_error(exc: BaseException) -> str:
    """
    Infer an error type string from an exception.

    Returns one of: '429', 'timeout', 'parse', 'validation', 'unknown'.
    """
    from dewie.enrichment.validators import StepValidationError

    if isinstance(exc, StepValidationError):
        return "validation"

    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return "429"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "json" in msg or "parse" in msg or "extractionresult" in msg.lower():
        return "parse"
    return "unknown"


async def write_error(
    pg: PostgresClient,
    doc_id: str | None,
    step: str,
    error_type: str,
    message: str,
    retry_count: int = 0,
) -> None:
    """
    Insert a row into pipeline_errors. Never raises — errors are logged only.
    """
    try:
        sql = text("""
            INSERT INTO pipeline_errors (doc_id, step, error_type, message, retry_count)
            VALUES (CAST(:doc_id AS UUID), :step, :error_type, :message, :retry_count)
        """)
        async with pg._session_factory() as session:
            await session.execute(
                sql,
                {
                    "doc_id": doc_id,
                    "step": step,
                    "error_type": error_type,
                    "message": message,
                    "retry_count": retry_count,
                },
            )
            await session.commit()
    except Exception:
        logger.warning(
            "pipeline_errors.write_error failed (doc=%s step=%s)", doc_id, step, exc_info=True
        )


_PIPELINE_STEPS = [
    "load_body",
    "llm_extraction",
    "field_population",
    "db_upsert",
    "embedding",
    "relationships",
]


async def get_error_stats(pg: PostgresClient, window_minutes: int = 60) -> dict:
    """
    Return error statistics for the given time window plus all-time unresolved counts.

    error_rate = failed_docs / total_docs_attempted in the window.
    total_docs_attempted = successful_in_window + failed_in_window (fixes denominator bug
    where only failed docs were counted).

    Returns:
        {
            total_docs_attempted: int,
            failed_docs: int,
            error_rate: float,
            above_threshold: bool,
            by_step: {step: count},
            by_type: {error_type: count},
            step_breakdown: {step: {errors: int, pct_of_total_processed: float}},
            unresolved_count: int,          # all-time unresolved errors
            unresolved_errors: [...]         # full list of unresolved errors
        }
    """
    try:
        async with pg._session_factory() as session:
            # Successful docs enriched in window (documents table)
            # Uses enriched_at as the relevant timestamp for document readiness.
            success_row = await session.execute(
                text("""
                SELECT COUNT(DISTINCT id) AS n
                FROM documents
                WHERE status = 'ready'
                  AND enriched_at > NOW() - (:window * INTERVAL '1 minute')
            """),
                {"window": window_minutes},
            )
            successful_in_window = int(success_row.scalar() or 0)

            # Distinct doc_ids from pipeline_errors with resolved errors in window.
            # These represent documents that experienced errors but were recovered.
            failed_row = await session.execute(
                text("""
                SELECT COUNT(DISTINCT doc_id) AS n
                FROM pipeline_errors
                WHERE created_at > NOW() - (:window * INTERVAL '1 minute')
                  AND resolved = TRUE
            """),
                {"window": window_minutes},
            )
            failed_docs = int(failed_row.scalar() or 0)

            # Breakdown by step (window) — unresolved errors only.
            step_rows = await session.execute(
                text("""
                SELECT step, COUNT(*) AS n
                FROM pipeline_errors
                WHERE created_at > NOW() - (:window * INTERVAL '1 minute')
                  AND resolved = FALSE
                GROUP BY step
            """),
                {"window": window_minutes},
            )
            by_step = {r["step"]: int(r["n"]) for r in step_rows.mappings().all()}

            # Breakdown by type (window) — unresolved errors only.
            type_rows = await session.execute(
                text("""
                SELECT error_type, COUNT(*) AS n
                FROM pipeline_errors
                WHERE created_at > NOW() - (:window * INTERVAL '1 minute')
                  AND resolved = FALSE
                GROUP BY error_type
            """),
                {"window": window_minutes},
            )
            by_type = {r["error_type"]: int(r["n"]) for r in type_rows.mappings().all()}

            # All-time unresolved count
            unresolved_count_row = await session.execute(
                text("""
                SELECT COUNT(*) AS n FROM pipeline_errors WHERE resolved = FALSE
            """)
            )
            unresolved_count = int(unresolved_count_row.scalar() or 0)

            # All-time unresolved errors (full list)
            unresolved_rows = await session.execute(
                text("""
                SELECT id, doc_id, step, error_type, message, retry_count, created_at
                FROM pipeline_errors
                WHERE resolved = FALSE
                ORDER BY created_at DESC
            """)
            )
            unresolved_errors = [
                {
                    "id": r["id"],
                    "doc_id": str(r["doc_id"]) if r["doc_id"] else None,
                    "step": r["step"],
                    "error_type": r["error_type"],
                    "message": r["message"],
                    "retry_count": r["retry_count"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in unresolved_rows.mappings().all()
            ]

        total_docs_attempted = successful_in_window + failed_docs
        error_rate = failed_docs / total_docs_attempted if total_docs_attempted > 0 else 0.0
        above_threshold = error_rate > ERROR_RATE_THRESHOLD

        # Per-step breakdown: error count + % of total processed + per-step threshold
        step_breakdown = {}
        any_step_above_threshold = False
        for step_name in _PIPELINE_STEPS:
            errors = by_step.get(step_name, 0)
            pct = (errors / total_docs_attempted * 100) if total_docs_attempted > 0 else 0.0
            step_above = pct > ERROR_RATE_THRESHOLD * 100
            if step_above:
                any_step_above_threshold = True
            step_breakdown[step_name] = {
                "errors": errors,
                "pct_of_total_processed": round(pct, 1),
                "above_threshold": step_above,
            }

        return {
            "total_docs_attempted": total_docs_attempted,
            "failed_docs": failed_docs,
            "error_rate": round(error_rate, 6),
            "above_threshold": above_threshold,
            "any_step_above_threshold": any_step_above_threshold,
            "by_step": by_step,
            "by_type": by_type,
            "step_breakdown": step_breakdown,
            "unresolved_count": unresolved_count,
            "unresolved_errors": unresolved_errors,
        }
    except Exception:
        logger.warning("pipeline_errors.get_error_stats failed", exc_info=True)
        return {
            "total_docs_attempted": 0,
            "failed_docs": 0,
            "error_rate": 0.0,
            "above_threshold": False,
            "any_step_above_threshold": False,
            "by_step": {},
            "by_type": {},
            "step_breakdown": {
                s: {"errors": 0, "pct_of_total_processed": 0.0, "above_threshold": False}
                for s in _PIPELINE_STEPS
            },
            "unresolved_count": 0,
            "unresolved_errors": [],
        }


async def mark_resolved(
    pg: PostgresClient,
    error_ids: list[int],
    requeue: bool = True,
) -> tuple[int, int]:
    """
    Mark pipeline_errors rows as resolved.

    If requeue=True, also resets the corresponding docs to status='pending'
    so they re-enter the enrichment queue.

    Returns (resolved_count, requeued_count).
    """
    if not error_ids:
        return 0, 0
    resolved_count = 0
    requeued_count = 0
    try:
        async with pg._session_factory() as session:
            await session.execute(
                text("UPDATE pipeline_errors SET resolved = TRUE WHERE id = ANY(:ids)"),
                {"ids": error_ids},
            )
            resolved_count = len(error_ids)

            if requeue:
                from dewie.config import settings as _settings
                result = await session.execute(
                    text("""
                    UPDATE documents
                    SET status = 'pending', enriched_at = NULL
                    WHERE id IN (
                        SELECT DISTINCT doc_id FROM pipeline_errors
                        WHERE id = ANY(:ids) AND doc_id IS NOT NULL
                    )
                    AND status != 'failed'
                    AND retry_count < :max_retries
                """),
                    {"ids": error_ids, "max_retries": _settings.max_enrichment_retries},
                )
                requeued_count = result.rowcount

            await session.commit()
    except Exception:
        logger.warning("pipeline_errors.mark_resolved failed", exc_info=True)
    return resolved_count, requeued_count
