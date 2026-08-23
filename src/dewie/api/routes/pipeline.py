# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Pipeline health API routes.

GET  /pipeline/errors?window=60      — error stats + recent/unresolved errors
POST /pipeline/errors/resolve        — mark errors resolved, optionally requeue docs
GET  /pipeline/workers/status        — enrichment worker status (running/paused + count)
POST /pipeline/workers/pause         — stop all enrichment workers via Docker
POST /pipeline/workers/resume        — start all enrichment workers via Docker
"""

from __future__ import annotations

import logging
import subprocess
import time
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
log = logging.getLogger("dewie.api")


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _extract_request_id(request: Request) -> str:
    """Extract request_id from request state, falling back to 'unknown'."""
    return getattr(request.state, "request_id", "unknown")


def _truncate(value: str, max_len: int = 1000) -> str:
    """Truncate a string to *max_len* characters for safe logging."""
    if len(value) > max_len:
        return value[:max_len] + f"... [truncated to {max_len} chars]"
    return value

# ── Docker helpers ─────────────────────────────────────────────────────────────

_DOCKER_WORKER_FILTER = "name=dewie-enrichment-worker"


def _docker(*args: str) -> tuple[int, str]:
    """Run a docker command and return (returncode, combined output)."""
    try:
        result = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode, output
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="docker not found in PATH") from None
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=503, detail="docker command timed out") from None
    except Exception as exc:
        log.exception("docker unexpected error")
        raise HTTPException(status_code=503, detail=f"docker error: {exc}") from exc


def _worker_containers() -> list[dict]:  # type: ignore[type-arg]
    """Return list of {name, status} for enrichment worker containers."""
    rc, output = _docker(
        "ps", "-a", "--filter", _DOCKER_WORKER_FILTER, "--format", "{{.Names}}\t{{.Status}}"
    )
    if rc != 0:
        return []
    containers = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        name = parts[0].strip()
        status = parts[1].strip() if len(parts) > 1 else "unknown"
        # Skip dev workers
        if "dev" in name:
            continue
        containers.append({"name": name, "status": status})
    return containers


def _worker_status() -> dict:  # type: ignore[type-arg]
    """Return {name: state} for all enrichment worker containers."""
    return {
        c["name"]: "RUNNING" if c["status"].startswith("Up") else "STOPPED"
        for c in _worker_containers()
    }


@router.get("/errors")
async def get_pipeline_errors(request: Request, window: int = 60) -> dict:  # type: ignore[type-arg]
    """
    Return pipeline error statistics for the given time window plus all-time
    unresolved errors.

    Query parameters:
        window (int): lookback window in minutes (default 60)
    """
    request_id = _extract_request_id(request)
    t0 = time.monotonic()

    log.info(
        "get_pipeline_errors started",
        extra={
            "request_id": request_id,
            "window": window,
        },
    )

    try:
        from dewie.storage.pipeline_errors import ERROR_RATE_THRESHOLD, get_error_stats

        pg = request.app.state.postgres
        stats = await get_error_stats(pg, window_minutes=window)

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "get_pipeline_errors succeeded",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
            },
        )
        return {
            "window_minutes": window,
            "threshold": ERROR_RATE_THRESHOLD,
            # Windowed stats
            "total_docs_attempted": stats["total_docs_attempted"],
            "failed": stats["failed_docs"],
            "error_rate": stats["error_rate"],
            "above_threshold": stats["above_threshold"],
            "by_step": stats["by_step"],
            "by_type": stats["by_type"],
            # All-time unresolved
            "unresolved_count": stats["unresolved_count"],
            "unresolved_errors": stats["unresolved_errors"],
        }
    except Exception:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.exception(
            "get_pipeline_errors failed",
            extra={
                "request_id": request_id,
                "window": window,
                "elapsed_ms": elapsed,
            },
        )
        raise


class PriorityInjectRequest(BaseModel):
    doc_id: UUID


class InjectBodyRequest(BaseModel):
    doc_id: UUID
    body_text: str


@router.post("/inject-body")
async def inject_body(request: Request, body: InjectBodyRequest) -> dict:  # type: ignore[type-arg]
    """
    Write body_text directly to a document's Postgres row and flat-file store.

    Used by the E2E test harness to inject known content before priority enrichment,
    allowing deterministic validation of the full pipeline against a specific body.
    """
    request_id = _extract_request_id(request)
    t0 = time.monotonic()

    log.info(
        "inject_body started",
        extra={
            "request_id": request_id,
            "doc_id": str(body.doc_id),
            "body_length": len(body.body_text),
        },
    )

    try:
        from sqlalchemy import text

        pg = request.app.state.postgres
        doc_id = str(body.doc_id)

        doc = await pg.get_by_id(body.doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        async with pg._session_factory() as session:
            await session.execute(
                text("UPDATE documents SET body_text = :body WHERE id = CAST(:doc_id AS UUID)"),
                {"body": body.body_text, "doc_id": doc_id},
            )
            await session.commit()

        # Also write to flat-file store for worker compatibility
        try:
            await pg.write_body_text(body.doc_id, body.body_text)
        except Exception:
            pass  # DB write is authoritative; flat-file is best-effort

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "inject_body succeeded",
            extra={
                "request_id": request_id,
                "doc_id": doc_id,
                "bytes_written": len(body.body_text.encode()),
                "elapsed_ms": elapsed,
            },
        )
        return {"doc_id": doc_id, "bytes_written": len(body.body_text.encode())}
    except HTTPException:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "inject_body http_error",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
            },
        )
        raise
    except Exception:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.exception(
            "inject_body failed",
            extra={
                "request_id": request_id,
                "doc_id": str(body.doc_id),
                "elapsed_ms": elapsed,
            },
        )
        raise


@router.post("/enrich/priority")
async def priority_enrich(request: Request, body: PriorityInjectRequest) -> dict:  # type: ignore[type-arg]
    """
    Push a document to the front of the enrichment queue.

    Resets the doc to pending, clears its LLM cache, resolves any existing
    pipeline errors, and sets priority=1 so the next drain cycle picks it up
    before all other pending documents.
    """
    request_id = _extract_request_id(request)
    t0 = time.monotonic()

    log.info(
        "priority_enrich started",
        extra={
            "request_id": request_id,
            "doc_id": str(body.doc_id),
        },
    )

    try:
        from sqlalchemy import text

        pg = request.app.state.postgres
        doc_id = str(body.doc_id)

        # Verify the document exists
        doc = await pg.get_by_id(body.doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        async with pg._session_factory() as session:
            await session.execute(
                text("DELETE FROM llm_cache WHERE doc_id = CAST(:doc_id AS UUID)"),
                {"doc_id": doc_id},
            )
            await session.execute(
                text(
                    "UPDATE pipeline_errors SET resolved = TRUE"
                    " WHERE doc_id = CAST(:doc_id AS UUID) AND resolved = FALSE"
                ),
                {"doc_id": doc_id},
            )
            await session.execute(
                text(
                    "UPDATE documents SET status = 'pending', enriched_at = NULL,"
                    " embedding = NULL, priority = 1"
                    " WHERE id = CAST(:doc_id AS UUID)"
                ),
                {"doc_id": doc_id},
            )
            await session.commit()

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "priority_enrich succeeded",
            extra={
                "request_id": request_id,
                "doc_id": doc_id,
                "elapsed_ms": elapsed,
            },
        )
        return {
            "doc_id": doc_id,
            "status": "queued",
            "message": "Doc will be picked up in next drain cycle (\u226430s)",
        }
    except HTTPException:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "priority_enrich http_error",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
            },
        )
        raise
    except Exception:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.exception(
            "priority_enrich failed",
            extra={
                "request_id": request_id,
                "doc_id": str(body.doc_id),
                "elapsed_ms": elapsed,
            },
        )
        raise


class ResolveRequest(BaseModel):
    error_ids: list[int]
    requeue: bool = True


@router.post("/errors/resolve")
async def resolve_pipeline_errors(request: Request, body: ResolveRequest) -> dict:  # type: ignore[type-arg]
    """
    Mark pipeline errors as resolved and optionally requeue their docs.

    If requeue=True (default), each doc referenced by the resolved errors is
    reset to status='pending' so it re-enters the enrichment queue.

    Returns { "resolved": N, "requeued": M }.
    """
    request_id = _extract_request_id(request)
    t0 = time.monotonic()

    log.info(
        "resolve_pipeline_errors started",
        extra={
            "request_id": request_id,
            "error_count": len(body.error_ids),
            "requeue": body.requeue,
        },
    )

    try:
        from dewie.storage.pipeline_errors import mark_resolved

        pg = request.app.state.postgres
        resolved, requeued = await mark_resolved(pg, body.error_ids, requeue=body.requeue)

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "resolve_pipeline_errors succeeded",
            extra={
                "request_id": request_id,
                "resolved": resolved,
                "requeued": requeued,
                "elapsed_ms": elapsed,
            },
        )
        return {"resolved": resolved, "requeued": requeued}
    except Exception:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.exception(
            "resolve_pipeline_errors failed",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
            },
        )
        raise


# ── Worker pause / resume ──────────────────────────────────────────────────────


@router.get("/workers/status")
async def workers_status(request: Request) -> dict:  # type: ignore[type-arg]
    """
    Return current enrichment worker states.

    Response: { "workers": {"dewie-enrichment-worker-2": "RUNNING", ...},
                "running": 2, "total": 2, "paused": false }
    """
    request_id = _extract_request_id(request)
    t0 = time.monotonic()

    log.info(
        "workers_status started",
        extra={"request_id": request_id},
    )

    try:
        statuses = _worker_status()
        if not statuses:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.info(
                "workers_status succeeded",
                extra={
                    "request_id": request_id,
                    "elapsed_ms": elapsed,
                    "running": 0,
                    "total": 0,
                },
            )
            return {
                "workers": {},
                "running": 0,
                "total": 0,
                "paused": True,
                "warning": "No enrichment worker containers found. Is Docker running?",
            }
        running = sum(1 for s in statuses.values() if s == "RUNNING")
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "workers_status succeeded",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
                "running": running,
                "total": len(statuses),
            },
        )
        return {
            "workers": statuses,
            "running": running,
            "total": len(statuses),
            "paused": running == 0,
        }
    except Exception:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.exception(
            "workers_status failed",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
            },
        )
        raise


@router.post("/workers/pause")
async def workers_pause(request: Request) -> dict:  # type: ignore[type-arg]
    """
    Stop all enrichment workers. In-flight jobs complete first (graceful stop).
    Returns updated worker status.
    """
    request_id = _extract_request_id(request)
    t0 = time.monotonic()

    log.info(
        "workers_pause started",
        extra={"request_id": request_id},
    )

    try:
        statuses = _worker_status()
        to_stop = [name for name, state in statuses.items() if state == "RUNNING"]
        if not to_stop:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.info(
                "workers_pause succeeded",
                extra={
                    "request_id": request_id,
                    "elapsed_ms": elapsed,
                    "action": "already_stopped",
                },
            )
            return {"ok": True, "message": "Workers already stopped", "workers": statuses}

        rc, output = _docker("stop", *to_stop)
        if rc != 0:
            raise HTTPException(status_code=500, detail=f"docker stop failed: {output}")

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "workers_pause succeeded",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
                "stopped_count": len(to_stop),
            },
        )
        return {"ok": True, "message": f"Stopped {len(to_stop)} worker(s)", "workers": _worker_status()}
    except HTTPException:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "workers_pause http_error",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
            },
        )
        raise
    except Exception:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.exception(
            "workers_pause failed",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
            },
        )
        raise


@router.post("/workers/resume")
async def workers_resume(request: Request) -> dict:  # type: ignore[type-arg]
    """
    Start all enrichment workers that are currently stopped.
    Returns updated worker status.
    """
    request_id = _extract_request_id(request)
    t0 = time.monotonic()

    log.info(
        "workers_resume started",
        extra={"request_id": request_id},
    )

    try:
        statuses = _worker_status()
        to_start = [name for name, state in statuses.items() if state == "STOPPED"]
        if not to_start:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.info(
                "workers_resume succeeded",
                extra={
                    "request_id": request_id,
                    "elapsed_ms": elapsed,
                    "action": "already_running",
                },
            )
            return {"ok": True, "message": "Workers already running", "workers": statuses}

        rc, output = _docker("start", *to_start)
        if rc != 0:
            raise HTTPException(status_code=500, detail=f"docker start failed: {output}")

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "workers_resume succeeded",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
                "started_count": len(to_start),
            },
        )
        return {
            "ok": True,
            "message": f"Started {len(to_start)} worker(s)",
            "workers": _worker_status(),
        }
    except HTTPException:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "workers_resume http_error",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
            },
        )
        raise
    except Exception:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.exception(
            "workers_resume failed",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
            },
        )
        raise


# ── Corpus source breakdown ────────────────────────────────────────────────────

_UNKNOWN_SOURCE = "(unknown)"


@router.get("/corpus/sources")
async def corpus_sources(request: Request) -> list:  # type: ignore[type-arg]
    """
    Return per-source status breakdown sorted by total count descending.

    Response: [{ "source": str, "ready": int, "pending": int, "failed": int, "total": int }]
    """
    request_id = _extract_request_id(request)
    t0 = time.monotonic()

    log.info(
        "corpus_sources started",
        extra={"request_id": request_id},
    )

    try:
        from sqlalchemy import text as sqlt

        pg = request.app.state.postgres
        is_sqlite = getattr(pg, "_is_sqlite", False)
        async with pg._engine.begin() as conn:
            if is_sqlite:
                rows = await conn.execute(
                    sqlt("""
                    SELECT
                      COALESCE(NULLIF(source, ''), :unknown) AS source,
                      SUM(CASE WHEN status = 'ready'   THEN 1 ELSE 0 END) AS ready,
                      SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                      SUM(CASE WHEN status = 'failed'  THEN 1 ELSE 0 END) AS failed,
                      COUNT(*) AS total
                    FROM documents
                    GROUP BY COALESCE(NULLIF(source, ''), :unknown)
                    ORDER BY total DESC
                """),
                    {"unknown": _UNKNOWN_SOURCE},
                )
            else:
                rows = await conn.execute(
                    sqlt("""
                    SELECT
                      COALESCE(NULLIF(source, ''), :unknown) AS source,
                      COUNT(*) FILTER (WHERE status = 'ready')   AS ready,
                      COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                      COUNT(*) FILTER (WHERE status = 'failed')  AS failed,
                      COUNT(*) AS total
                    FROM documents
                    GROUP BY COALESCE(NULLIF(source, ''), :unknown)
                    ORDER BY total DESC
                """),
                    {"unknown": _UNKNOWN_SOURCE},
                )
            result = [dict(r) for r in rows.mappings()]

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "corpus_sources succeeded",
            extra={
                "request_id": request_id,
                "source_count": len(result),
                "elapsed_ms": elapsed,
            },
        )
        return result
    except HTTPException:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "corpus_sources http_error",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
            },
        )
        raise
    except Exception:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.exception(
            "corpus_sources failed",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed,
            },
        )
        raise HTTPException(status_code=500, detail="Query failed") from None


@router.get("/corpus/quality")
async def corpus_quality(request: Request) -> dict:  # type: ignore[type-arg]
    """
    Return corpus-wide quality metrics and per-source breakdown.

    For PostgreSQL: reads from materialized views (corpus_quality_cache + corpus_sources_cache)
    which are refreshed concurrently every 5 minutes by a background task.
    For SQLite: runs live aggregate queries directly.

    To manually force a refresh (PostgreSQL only): POST /pipeline/corpus/quality/refresh
    """
    from sqlalchemy import text as sqlt

    pg = request.app.state.postgres
    is_sqlite = getattr(pg, "_is_sqlite", False)
    try:
        async with pg._engine.connect() as conn:
            if is_sqlite:
                # Live aggregate queries for SQLite (no materialized views)
                summary_row = (
                    (
                        await conn.execute(
                            sqlt("""
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN status = 'ready'   THEN 1 ELSE 0 END) AS ready,
                        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                        SUM(CASE WHEN status = 'failed'  THEN 1 ELSE 0 END) AS failed,
                        SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_embedding,
                        SUM(CASE WHEN length(COALESCE(embed_summary,'')) > 100 THEN 1 ELSE 0 END) AS embed_summary_good,
                        SUM(CASE WHEN length(COALESCE(embed_summary,'')) BETWEEN 1 AND 100 THEN 1 ELSE 0 END) AS embed_summary_stub,
                        SUM(CASE WHEN embed_summary IS NULL OR embed_summary = '' THEN 1 ELSE 0 END) AS embed_summary_none,
                        CAST(AVG(length(COALESCE(embed_summary,''))) AS INTEGER) AS avg_embed_summary_len,
                        CAST(AVG(length(COALESCE(body_text,''))) AS INTEGER) AS avg_body_len,
                        CAST(AVG(
                            CASE WHEN answers_questions IS NOT NULL AND answers_questions != '[]'
                                 THEN json_array_length(answers_questions) ELSE 0 END
                        ) AS REAL) AS avg_aqs,
                        SUM(CASE WHEN answers_questions IS NOT NULL AND answers_questions != '[]' THEN 1 ELSE 0 END) AS with_aqs,
                        SUM(CASE WHEN answers_questions IS NULL OR answers_questions = '[]' THEN 1 ELSE 0 END) AS empty_aqs,
                        SUM(CASE WHEN COALESCE(enrichment_quality_score, 0) >= 80 THEN 1 ELSE 0 END) AS quality_high,
                        SUM(CASE WHEN COALESCE(enrichment_quality_score, 0) BETWEEN 50 AND 79 THEN 1 ELSE 0 END) AS quality_medium,
                        SUM(CASE WHEN COALESCE(enrichment_quality_score, 0) BETWEEN 20 AND 49 THEN 1 ELSE 0 END) AS quality_low,
                        SUM(CASE WHEN COALESCE(enrichment_quality_score, 0) < 20 THEN 1 ELSE 0 END) AS quality_stub
                    FROM documents
                """)
                        )
                    )
                    .mappings()
                    .one()
                )
                source_rows = (
                    (
                        await conn.execute(
                            sqlt("""
                    SELECT
                        COALESCE(NULLIF(source,''), 'unknown') AS source,
                        SUM(CASE WHEN status='ready'   THEN 1 ELSE 0 END) AS ready,
                        SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
                        SUM(CASE WHEN status='failed'  THEN 1 ELSE 0 END) AS failed,
                        COUNT(*) AS total
                    FROM documents
                    GROUP BY 1
                    ORDER BY ready DESC
                    LIMIT 50
                """)
                        )
                    )
                    .mappings()
                    .all()
                )
                chunk_row = {"docs_with_chunks": 0, "total_chunks": 0, "chunks_with_embed": 0,
                             "chunks_with_aq_embed": 0, "avg_chunks_per_doc": 0,
                             "avg_chunk_tokens": 0, "avg_chunk_chars": 0, "docs_with_aq_embed": 0}
                chunk_status_row = {"status_none": 0, "status_chunked": 0, "status_skipped": 0, "status_failed": 0}
                s = dict(summary_row)
                return {
                    "summary": {
                        "total": s["total"],
                        "ready": s["ready"],
                        "pending": s["pending"],
                        "failed": s["failed"],
                        "with_embedding": s["with_embedding"],
                        "embed_summary_good": s["embed_summary_good"],
                        "embed_summary_stub": s["embed_summary_stub"],
                        "embed_summary_none": s["embed_summary_none"],
                        "avg_embed_summary_len": s["avg_embed_summary_len"],
                        "avg_body_len": s["avg_body_len"],
                        "avg_aqs": s["avg_aqs"],
                        "with_aqs": s["with_aqs"],
                        "empty_aqs": s["empty_aqs"],
                    },
                    "quality_distribution": {
                        "high": s["quality_high"],
                        "medium": s["quality_medium"],
                        "low": s["quality_low"],
                        "stub": s["quality_stub"],
                    },
                    "by_source": [dict(r) for r in source_rows],
                    "chunks": {**chunk_row, **chunk_status_row},
                    "refreshed_at": None,
                }

            # PostgreSQL: read from materialized views
            summary_row = (
                (await conn.execute(sqlt("SELECT * FROM corpus_quality_cache"))).mappings().one()
            )

            source_rows = (
                (
                    await conn.execute(
                        sqlt("SELECT * FROM corpus_sources_cache ORDER BY ready DESC LIMIT 50")
                    )
                )
                .mappings()
                .all()
            )

            chunk_row = (
                (
                    await conn.execute(
                        sqlt("""
                SELECT
                    COUNT(DISTINCT doc_id)                         AS docs_with_chunks,
                    COUNT(*)                                       AS total_chunks,
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL)  AS chunks_with_embed,
                    COUNT(*) FILTER (WHERE aq_embedding IS NOT NULL) AS chunks_with_aq_embed,
                    ROUND(AVG(chunks_per_doc))::int                AS avg_chunks_per_doc,
                    ROUND(AVG(avg_chunk_tokens))::int              AS avg_chunk_tokens,
                    ROUND(AVG(avg_chunk_chars))::int               AS avg_chunk_chars,
                    COUNT(*) FILTER (WHERE has_aq_embed)           AS docs_with_aq_embed
                FROM (
                    SELECT doc_id,
                           COUNT(*)             AS chunks_per_doc,
                           AVG(token_count)     AS avg_chunk_tokens,
                           AVG(length(text))    AS avg_chunk_chars,
                           BOOL_OR(aq_embedding IS NOT NULL) AS has_aq_embed
                    FROM document_chunks
                    GROUP BY doc_id
                ) s
                JOIN document_chunks dc USING (doc_id)
            """)
                    )
                )
                .mappings()
                .one()
            )

            # ── Chunk status breakdown ───────────────────────────────────────
            chunk_status_row = (
                (
                    await conn.execute(
                        sqlt("""
                SELECT
                    COUNT(*) FILTER (WHERE chunk_status = 'none')    AS status_none,
                    COUNT(*) FILTER (WHERE chunk_status = 'chunked') AS status_chunked,
                    COUNT(*) FILTER (WHERE chunk_status = 'skipped') AS status_skipped,
                    COUNT(*) FILTER (WHERE chunk_status = 'failed')  AS status_failed
                FROM documents
                WHERE status = 'ready'
            """)
                    )
                )
                .mappings()
                .one()
            )

        s = dict(summary_row)
        return {
            "summary": {
                "total": s["total"],
                "ready": s["ready"],
                "pending": s["pending"],
                "failed": s["failed"],
                "with_embedding": s["with_embedding"],
                "embed_summary_good": s["embed_summary_good"],
                "embed_summary_stub": s["embed_summary_stub"],
                "embed_summary_none": s["embed_summary_none"],
                "avg_embed_summary_len": s["avg_embed_summary_len"],
                "avg_body_len": s["avg_body_len"],
                "avg_aqs": s["avg_aqs"],
                "with_aqs": s["with_aqs"],
                "empty_aqs": s["empty_aqs"],
            },
            "quality_distribution": {
                "high": s["quality_high"],
                "medium": s["quality_medium"],
                "low": s["quality_low"],
                "stub": s["quality_stub"],
            },
            "by_source": [dict(r) for r in source_rows],
            "chunks": {**dict(chunk_row), **dict(chunk_status_row)},
            "refreshed_at": s.get("refreshed_at"),
        }
    except Exception as exc:
        log.warning("corpus_quality query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Query failed") from exc


@router.post("/corpus/quality/refresh")
async def corpus_quality_refresh(request: Request) -> dict:  # type: ignore[type-arg]
    """Force an immediate refresh of the corpus quality materialized views."""
    from sqlalchemy import text as sqlt

    pg = request.app.state.postgres
    if pg._is_sqlite:
        return {"ok": True, "message": "Materialized views refresh skipped (SQLite)"}

    try:
        async with pg._engine.begin() as conn:
            await conn.execute(sqlt("REFRESH MATERIALIZED VIEW CONCURRENTLY corpus_quality_cache"))
            await conn.execute(sqlt("REFRESH MATERIALIZED VIEW CONCURRENTLY corpus_sources_cache"))
        return {"ok": True, "message": "Materialized views refreshed"}
    except Exception as exc:
        log.warning("corpus_quality_refresh failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/corpus/sources/{source}/docs")
async def corpus_source_docs(
    request: Request,
    source: str,
    limit: int = 50,
) -> list:  # type: ignore[type-arg]
    """
    Return the most-recently-updated documents for a given source.

    Path: source — the source identifier (use "(unknown)" for docs with empty source)
    Query: limit — max docs to return (capped at 200)

    Response: [{ "id": str, "url": str, "title": str, "status": str, "enriched_at": str|null }]
    """
    from sqlalchemy import text as sqlt

    pg = request.app.state.postgres
    limit = min(max(1, limit), 200)

    try:
        async with pg._engine.begin() as conn:
            if source == _UNKNOWN_SOURCE:
                rows = await conn.execute(
                    sqlt("""
                    SELECT id::text, url, title, status, enriched_at
                    FROM documents
                    WHERE source IS NULL OR source = ''
                    ORDER BY COALESCE(enriched_at, ingested_at) DESC
                    LIMIT :limit
                """),
                    {"limit": limit},
                )
            else:
                rows = await conn.execute(
                    sqlt("""
                    SELECT id::text, url, title, status, enriched_at
                    FROM documents
                    WHERE source = :source
                    ORDER BY COALESCE(enriched_at, ingested_at) DESC
                    LIMIT :limit
                """),
                    {"source": source, "limit": limit},
                )
            return [dict(r) for r in rows.mappings()]
    except Exception as exc:
        log.warning("corpus_source_docs query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Query failed") from exc
