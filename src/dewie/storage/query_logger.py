# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
query_logger.py — Structured query logging for Dewie.

Logs every query (API, benchmark, navigator) to the query_log table.
Call log_query() at the end of any navigation or search operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from dewie.config import settings


@dataclass
class QueryLogEntry:
    question: str
    model: str | None = None
    source: str = "api"  # 'api', 'benchmark', 'navigator'
    session_id: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    hops: int = 0
    hop_trace: list[dict] = field(default_factory=list)
    docs_returned: list[dict] = field(default_factory=list)
    full_results: dict | None = (
        None  # complete SearchResponse payload (if save_full_results enabled)
    )
    answer: str | None = None
    correct: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_ms: int = 0


_engine = None
_factory = None


def _get_factory():
    global _engine, _factory
    if _factory is None:
        _engine = create_async_engine(settings.postgres_dsn, pool_size=2, max_overflow=2)
        _factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _factory


async def log_query(entry: QueryLogEntry) -> None:
    """Persist a QueryLogEntry to the query_log table. Fire-and-forget safe."""
    import json as _json

    try:
        factory = _get_factory()
        full_results_json = (
            _json.dumps(entry.full_results)
            if entry.full_results is not None and settings.query_log_save_full_results
            else None
        )
        async with factory() as session:
            await session.execute(
                text("""
                INSERT INTO query_log
                  (source, question, model, session_id, tenant_id, user_id, hops, hop_trace,
                   docs_returned, full_results, answer, correct, input_tokens, output_tokens,
                   cost_usd, elapsed_ms)
                VALUES
                  (:source, :question, :model, :session_id, :tenant_id, :user_id, :hops,
                   cast(:hop_trace as jsonb), cast(:docs_returned as jsonb),
                   cast(:full_results as jsonb), :answer, :correct,
                   :input_tokens, :output_tokens, :cost_usd, :elapsed_ms)
            """),
                {
                    "source": entry.source,
                    "question": entry.question,
                    "model": entry.model,
                    "session_id": entry.session_id,
                    "tenant_id": entry.tenant_id,
                    "user_id": entry.user_id,
                    "hops": entry.hops,
                    "hop_trace": _json.dumps(entry.hop_trace),
                    "docs_returned": _json.dumps(entry.docs_returned),
                    "full_results": full_results_json,
                    "answer": entry.answer,
                    "correct": entry.correct,
                    "input_tokens": entry.input_tokens,
                    "output_tokens": entry.output_tokens,
                    "cost_usd": entry.cost_usd,
                    "elapsed_ms": entry.elapsed_ms,
                },
            )
            await session.commit()
    except Exception as exc:
        # Never let logging crash the caller
        import logging

        logging.getLogger(__name__).warning("query_log write failed: %s", exc)
