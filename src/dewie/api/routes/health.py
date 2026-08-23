# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Pipeline health dashboard routes.

GET  /pipeline/health              — full health dashboard
POST /pipeline/health/heartbeat    — write liveness timestamp
POST /pipeline/health/record-e2e   — write e2e test result (called by scripts/healthcheck.py)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)

# Heartbeat is stale after 20 minutes
_HEARTBEAT_OK_SECONDS = 20 * 60

# Git working directory for subprocess calls.
# Defaults to cwd so it works without any configuration; override with
# DEWIE_ROOT env var in environments where cwd differs from the repo root.
_GIT_CWD = os.environ.get("DEWIE_ROOT") or os.getcwd()


def _get_git_info() -> dict:  # type: ignore[type-arg]
    """Fetch live git SHA and describe. Never raises."""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_GIT_CWD, text=True, timeout=5
        ).strip()
        git_describe = subprocess.check_output(
            ["git", "describe", "--tags", "--always"], cwd=_GIT_CWD, text=True, timeout=5
        ).strip()
        return {"git_sha": git_sha, "git_describe": git_describe}
    except Exception:
        logger.debug("_get_git_info unavailable (no git in container or DEWIE_ROOT not set)")
        return {"git_sha": "unknown", "git_describe": "unknown"}


@router.get("/health")
async def get_pipeline_health(request: Request) -> dict:  # type: ignore[type-arg]
    """Full pipeline health dashboard."""
    from dewie.storage.pipeline_errors import get_error_stats
    from dewie.storage.system_health import read_health_kv

    pg = request.app.state.postgres

    # ── Heartbeat ─────────────────────────────────────────────────────────────
    hb_kv = await read_health_kv(pg, "last_heartbeat")
    if hb_kv and hb_kv.get("value"):
        try:
            ts = datetime.fromisoformat(hb_kv["value"])
            age_seconds = (datetime.utcnow() - ts).total_seconds()
            heartbeat_ok = age_seconds < _HEARTBEAT_OK_SECONDS
        except Exception:
            age_seconds = None
            heartbeat_ok = False
        last_heartbeat = {
            "value": hb_kv["value"],
            "updated_at": hb_kv["updated_at"],
            "age_seconds": round(age_seconds) if age_seconds is not None else None,
            "ok": heartbeat_ok,
        }
    else:
        last_heartbeat = {"value": None, "updated_at": None, "age_seconds": None, "ok": False}

    # ── E2E test ──────────────────────────────────────────────────────────────
    e2e_kv = await read_health_kv(pg, "last_e2e_test")
    last_e2e_test = None
    if e2e_kv and e2e_kv.get("value"):
        try:
            last_e2e_test = json.loads(e2e_kv["value"])
        except Exception:
            last_e2e_test = {"raw": e2e_kv["value"]}

    # ── Current version (always live from git) ────────────────────────────────
    current_version = _get_git_info()

    # ── Version match ─────────────────────────────────────────────────────────
    version_match = last_e2e_test is not None and (
        current_version["git_sha"] == "unknown"
        or last_e2e_test.get("git_sha") == current_version["git_sha"]
    )

    # ── Error case tests ──────────────────────────────────────────────────────
    ec_kv = await read_health_kv(pg, "last_error_case_tests")
    error_case_tests = None
    if ec_kv and ec_kv.get("value"):
        try:
            error_case_tests = json.loads(ec_kv["value"])
        except Exception:
            error_case_tests = {"raw": ec_kv["value"]}

    # ── Error stats with step breakdown ──────────────────────────────────────
    stats = await get_error_stats(pg, window_minutes=60)

    # ── Corpus counts ─────────────────────────────────────────────────────────
    corpus: dict = {}  # type: ignore[type-arg]
    # Try Redis cache first (fast path, written by health cron)
    try:
        import json as _json

        cache = request.app.state.cache
        cached_health = await cache._redis.get("health:pipeline_health")
        if cached_health:
            cached_data = _json.loads(cached_health)
            corpus = cached_data.get("corpus", {})
    except Exception:
        pass  # Redis unavailable — fall through to live query

    # Always fall back to live query if corpus is empty
    if not corpus:
        try:
            from sqlalchemy import text as sqlt

            is_sqlite = getattr(pg, '_is_sqlite', False)
            async with pg._engine.begin() as conn:
                if is_sqlite:
                    row = await conn.execute(
                        sqlt("""
                        SELECT
                          COUNT(*) AS total,
                          SUM(CASE WHEN status = 'ready' THEN 1 ELSE 0 END) AS ready,
                          SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                          SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                          SUM(CASE WHEN status = 'ready' AND embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_embedding,
                          SUM(CASE WHEN status = 'ready' AND enriched_at >= datetime('now', '-30 minutes') THEN 1 ELSE 0 END) AS enriched_30min,
                          SUM(CASE WHEN status = 'ready' AND enriched_at >= datetime('now', '-5 minutes') THEN 1 ELSE 0 END) AS enriched_5min
                        FROM documents
                    """)
                    )
                else:
                    row = await conn.execute(
                        sqlt("""
                        SELECT
                          COUNT(*) FILTER (WHERE status='ready')    AS ready,
                          COUNT(*) FILTER (WHERE status='pending')  AS pending,
                          COUNT(*) FILTER (WHERE status='failed')   AS failed,
                          COUNT(*) FILTER (WHERE status='ready' AND embedding IS NOT NULL) AS with_embedding,
                          COUNT(*) FILTER (WHERE status='ready' AND enriched_at > NOW() - INTERVAL '30 minutes') AS enriched_30min,
                          COUNT(*) FILTER (WHERE status='ready' AND enriched_at > NOW() - INTERVAL '5 minutes')  AS enriched_5min
                        FROM documents
                    """)
                    )
                r = row.mappings().one()
                corpus = dict(r)
        except Exception:
            logger.warning("corpus counts failed", exc_info=True)

    # ── Ingest metrics ────────────────────────────────────────────────────────
    ingest: dict = {}  # type: ignore[type-arg]
    try:
        from sqlalchemy import text as sqlt

        is_sqlite = getattr(pg, '_is_sqlite', False)
        async with pg._engine.begin() as conn:
            if is_sqlite:
                row = await conn.execute(
                    sqlt("""
                    SELECT
                      SUM(CASE WHEN ingested_at >= datetime('now', '-1 hour') THEN 1 ELSE 0 END) AS last_1h,
                      SUM(CASE WHEN ingested_at >= datetime('now', '-24 hours') THEN 1 ELSE 0 END) AS last_24h,
                      SUM(CASE WHEN ingested_at >= datetime('now', '-7 days') THEN 1 ELSE 0 END) AS last_7d
                    FROM documents
                """)
                )
                r = row.mappings().one()
                ingest = {
                    "last_1h": r["last_1h"],
                    "last_24h": r["last_24h"],
                    "last_7d": r["last_7d"],
                }
                src_rows = await conn.execute(
                    sqlt("""
                    SELECT source, COUNT(*) AS cnt
                    FROM documents
                    WHERE ingested_at >= datetime('now', '-24 hours')
                    GROUP BY source
                    ORDER BY cnt DESC
                    LIMIT 20
                """)
                )
                ingest["by_source_24h"] = {r["source"]: r["cnt"] for r in src_rows.mappings()}
            else:
                row = await conn.execute(
                    sqlt("""
                    SELECT
                      COUNT(*) FILTER (WHERE ingested_at > NOW() - INTERVAL '1 hour')   AS last_1h,
                      COUNT(*) FILTER (WHERE ingested_at > NOW() - INTERVAL '24 hours') AS last_24h,
                      COUNT(*) FILTER (WHERE ingested_at > NOW() - INTERVAL '7 days')   AS last_7d
                    FROM documents
                """)
                )
                r = row.mappings().one()
                ingest = {
                    "last_1h": r["last_1h"],
                    "last_24h": r["last_24h"],
                    "last_7d": r["last_7d"],
                }
                src_rows = await conn.execute(
                    sqlt("""
                    SELECT source, COUNT(*) AS cnt
                    FROM documents
                    WHERE ingested_at > NOW() - INTERVAL '24 hours'
                    GROUP BY source
                    ORDER BY cnt DESC
                    LIMIT 20
                """)
                )
                ingest["by_source_24h"] = {r["source"]: r["cnt"] for r in src_rows.mappings()}
    except Exception:
        logger.warning("ingest metrics query failed", exc_info=True)

    # ── Recent enriched samples (5 most recent ready docs) ───────────────────
    recent_docs: list = []  # type: ignore[type-arg]
    try:
        from sqlalchemy import text as sqlt

        async with pg._engine.begin() as conn:
            rows = await conn.execute(
                sqlt("""
                SELECT id::text, title, source, document_type, tone,
                  CAST(sentiment AS float) AS sentiment,
                  jsonb_array_length(COALESCE(answers_questions, '[]'::jsonb)) AS aq_count,
                  (embedding IS NOT NULL) AS has_embedding,
                  (embed_summary IS NOT NULL AND embed_summary <> '') AS has_embed_summary,
                  LEFT(embed_summary, 120) AS embed_summary_preview,
                  enriched_at::text AS enriched_at
                FROM documents
                WHERE status='ready' AND enriched_at IS NOT NULL
                ORDER BY enriched_at DESC LIMIT 5
            """)
            )
            for r in rows.mappings():
                recent_docs.append(dict(r))
    except Exception:
        logger.warning("recent_docs query failed", exc_info=True)

    return {
        "last_heartbeat": last_heartbeat,
        "last_e2e_test": last_e2e_test,
        "current_version": current_version,
        "version_match": version_match,
        "error_case_tests": error_case_tests,
        "corpus": corpus,
        "ingest": ingest,
        "recent_enriched": recent_docs,
        "step_breakdown": stats.get("step_breakdown", {}),
        "error_stats": {
            "window_minutes": 60,
            "total_docs_attempted": stats["total_docs_attempted"],
            "failed_docs": stats["failed_docs"],
            "error_rate": stats["error_rate"],
            "above_threshold": stats["above_threshold"],
            "any_step_above_threshold": stats.get("any_step_above_threshold", False),
        },
    }


@router.post("/health/heartbeat")
async def record_heartbeat(request: Request) -> dict:  # type: ignore[type-arg]
    """Write last_heartbeat timestamp — called by an external liveness monitor."""
    from dewie.storage.system_health import write_health_kv

    pg = request.app.state.postgres
    ts = datetime.utcnow().isoformat()
    await write_health_kv(pg, "last_heartbeat", ts)
    return {"ok": True, "timestamp": ts}


class RecordE2ERequest(BaseModel):
    doc_id: str
    git_sha: str
    status: str  # "ok" | "failed" | "timeout"
    enriched_at: str | None = None
    has_embedding: bool = False
    has_embed_summary: bool = False
    aq_count: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


@router.post("/health/record-e2e")
async def record_e2e(request: Request, body: RecordE2ERequest) -> dict:  # type: ignore[type-arg]
    """Write e2e test result to system_health — called by scripts/healthcheck.py."""
    from dewie.storage.system_health import write_health_kv

    pg = request.app.state.postgres
    record = {
        "last_run": datetime.utcnow().isoformat(),
        "git_sha": body.git_sha,
        "doc_id": body.doc_id,
        "status": body.status,
        "enriched_at": body.enriched_at,
        "has_embedding": body.has_embedding,
        "has_embed_summary": body.has_embed_summary,
        "aq_count": body.aq_count,
        "elapsed_seconds": body.elapsed_seconds,
        "error": body.error,
    }
    await write_health_kv(pg, "last_e2e_test", json.dumps(record))
    return {"ok": True, "record": record}
