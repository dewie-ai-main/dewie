# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Human-readable dashboard served at GET /dashboard.

A single-page app built with vanilla JS + Tailwind CDN — no build step,
no new Python dependencies.  All data is fetched from the JSON endpoints
defined in this same file.
"""

from __future__ import annotations

import contextlib
import pathlib

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse

from dewie._static import static_dir
from dewie.storage.postgres import PostgresClient

_STATIC_DIR = static_dir() or pathlib.Path(__file__).resolve().parents[4] / "static"

router = APIRouter(tags=["dashboard"])

# ── JSON data endpoints ────────────────────────────────────────────────────────


def _pg(request: Request) -> PostgresClient:
    return request.app.state.postgres


@router.get("/stats", include_in_schema=False)
async def stats(request: Request) -> dict:  # type: ignore[type-arg]
    import json as _json

    pg = _pg(request)

    # Fast path: return cached result if available (30s TTL)
    try:
        cache = request.app.state.cache
        cached = await cache._redis.get("dashboard:stats")
        if cached:
            return _json.loads(cached)
    except Exception:
        pass

    counts = await pg.count_by_status()
    sessions = await pg.list_crawl_sessions()
    total = sum(counts.values())

    result = {
        "total": total,
        "by_status": counts,
        "enrich_queue": 0,
        "enrich_workers": 0,
        "crawl_sessions": [
            {
                "session": str(s["crawl_session"]),
                "total": s["total"],
                "ready": s["ready"],
                "processing": s["processing"],
                "failed": s["failed"],
                "started_at": s["started_at"] if isinstance(s["started_at"], str) else (s["started_at"].isoformat() if s["started_at"] else None),
                "last_seen_at": s["last_seen_at"] if isinstance(s["last_seen_at"], str) else (s["last_seen_at"].isoformat() if s["last_seen_at"] else None),
            }
            for s in sessions
        ],
    }

    # Cache for 30 seconds to avoid repeated full-table scans
    with contextlib.suppress(Exception):
        await cache._redis.setex("dashboard:stats", 30, _json.dumps(result))

    return result

@router.get("/ingest-stats", include_in_schema=False)
async def ingest_stats(request: Request, response: Response) -> dict:  # type: ignore[type-arg]
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    import json as _json

    from sqlalchemy import text as sa_text

    pg = _pg(request)

    # Server-side caching (30s) prevents redundant full-table scans on dashboard polls.
    try:
        cache = request.app.state.cache
        cached = await cache._redis.get("dashboard:ingest_stats")
        if cached:
            return _json.loads(cached)
    except Exception:
        cache = None

    is_sqlite = getattr(pg, "_is_sqlite", False)
    conn = pg._engine

    async with conn.connect() as c:
        # --- Single-pass aggregate counts ---
        if is_sqlite:
            agg_row = (await c.execute(sa_text("""
                SELECT
                  COUNT(*)                                                                              AS total,
                  MAX(ingested_at)                                                                      AS newest_at,
                  COUNT(DISTINCT source)                                                                AS distinct_sources,
                  SUM(CASE WHEN answers_questions IS NOT NULL AND answers_questions != '[]' THEN 1 ELSE 0 END) AS enriched_aq,
                  SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END)                               AS has_embedding,
                  SUM(CASE WHEN search_vec IS NOT NULL THEN 1 ELSE 0 END)                              AS has_search_vec
                FROM documents
            """))).mappings().one()
            newest_at_val = agg_row["newest_at"]
            time_row = (await c.execute(sa_text("""
                SELECT
                  SUM(CASE WHEN ingested_at > datetime(:max_at, '-10 minutes') THEN 1 ELSE 0 END) AS last_10min,
                  SUM(CASE WHEN ingested_at > datetime(:max_at, '-1 hours')    THEN 1 ELSE 0 END) AS last_1h,
                  SUM(CASE WHEN ingested_at > datetime(:max_at, '-24 hours')   THEN 1 ELSE 0 END) AS last_24h
                FROM documents
            """), {"max_at": newest_at_val})).mappings().one()
        else:
            agg_row = (await c.execute(sa_text("""
                SELECT
                  COUNT(*)                                                                                    AS total,
                  MAX(ingested_at)                                                                            AS newest_at,
                  COUNT(DISTINCT source)                                                                      AS distinct_sources,
                  COUNT(*) FILTER (WHERE answers_questions IS NOT NULL AND answers_questions != '[]'::jsonb) AS enriched_aq,
                  COUNT(*) FILTER (WHERE embedding IS NOT NULL)                                               AS has_embedding,
                  COUNT(*) FILTER (WHERE search_vec IS NOT NULL)                                              AS has_search_vec
                FROM documents
            """))).mappings().one()
            newest_at_val = agg_row["newest_at"]
            time_row = (await c.execute(sa_text("""
                SELECT
                  COUNT(*) FILTER (WHERE ingested_at > :max_at - INTERVAL '10 minutes') AS last_10min,
                  COUNT(*) FILTER (WHERE ingested_at > :max_at - INTERVAL '1 hour')     AS last_1h,
                  COUNT(*) FILTER (WHERE ingested_at > :max_at - INTERVAL '24 hours')   AS last_24h
                FROM documents
            """), {"max_at": newest_at_val})).mappings().one()

        edges = (await c.execute(sa_text("SELECT COUNT(*) FROM document_edges"))).scalar()
        by_source = (
            await c.execute(
                sa_text(
                    "SELECT source, COUNT(*) as n FROM documents GROUP BY source ORDER BY n DESC LIMIT 50"
                )
            )
        ).fetchall()
        recent_docs = (
            await c.execute(
                sa_text(
                    "SELECT id, title, source, ingested_at FROM documents ORDER BY ingested_at DESC LIMIT 20"
                )
            )
        ).fetchall()

    newest_at = agg_row["newest_at"]
    result = {
        "total_docs": agg_row["total"],
        "total_edges": edges,
        "distinct_sources": agg_row["distinct_sources"],
        "enriched_aq": agg_row["enriched_aq"],
        "has_embedding": agg_row["has_embedding"],
        "has_search_vec": agg_row["has_search_vec"],
        "added_last_10min": time_row["last_10min"],
        "added_last_1h": time_row["last_1h"],
        "added_last_24h": time_row["last_24h"],
        "newest_doc_at": newest_at if isinstance(newest_at, str) else (newest_at.isoformat() if newest_at else None),
        "by_source": [{"source": r[0], "count": r[1]} for r in by_source],
        "recent": [
            {
                "id": str(r[0]),
                "title": r[1],
                "source": r[2],
                "ingested_at": r[3] if isinstance(r[3], str) else (r[3].isoformat() if r[3] else None),
            }
            for r in recent_docs
        ],
    }

    # Cache for 30 seconds to reduce DB load from dashboard polling
    try:
        if cache is not None:
            await cache._redis.setex("dashboard:ingest_stats", 30, _json.dumps(result))
    except Exception:
        pass

    return result

@router.get("/ingest-tool-stats", include_in_schema=False)
async def ingest_tool_stats(request: Request, response: Response) -> dict:  # type: ignore[type-arg]
    """Per-tool ingestion metrics: reddit, youtube, huggingface, rss/other."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    from sqlalchemy import text as sa_text

    pg = _pg(request)
    is_sqlite = getattr(pg, "_is_sqlite", False)
    async with pg._engine.connect() as conn:
        if is_sqlite:
            rows = (
                await conn.execute(
                    sa_text("""
                SELECT
                  CASE
                    WHEN url LIKE '%reddit.com%'                       THEN 'reddit'
                    WHEN url LIKE '%youtube.com%' OR url LIKE '%youtu.be%' THEN 'youtube'
                    WHEN url LIKE '%huggingface.co%'                   THEN 'huggingface'
                    ELSE 'rss'
                  END AS tool,
                  COUNT(*)                                                              AS total,
                  SUM(CASE WHEN status = 'ready'   THEN 1 ELSE 0 END)                  AS ready,
                  SUM(CASE WHEN status = 'failed'  THEN 1 ELSE 0 END)                  AS failed,
                  SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END)                  AS pending,
                  SUM(CASE WHEN ingested_at >= datetime('now', '-1 hour')   THEN 1 ELSE 0 END) AS last_1h,
                  SUM(CASE WHEN ingested_at >= datetime('now', '-24 hours') THEN 1 ELSE 0 END) AS last_24h,
                  MAX(ingested_at)                                                      AS last_seen
                FROM documents
                GROUP BY 1
                ORDER BY total DESC
            """)
                )
            ).fetchall()
        else:
            rows = (
                await conn.execute(
                    sa_text("""
                SELECT
                  CASE
                    WHEN url LIKE '%reddit.com%'                       THEN 'reddit'
                    WHEN url LIKE '%youtube.com%' OR url LIKE '%youtu.be%' THEN 'youtube'
                    WHEN url LIKE '%huggingface.co%'                   THEN 'huggingface'
                    ELSE 'rss'
                  END AS tool,
                  COUNT(*)                                                          AS total,
                  COUNT(*) FILTER (WHERE status = 'ready')                         AS ready,
                  COUNT(*) FILTER (WHERE status = 'failed')                        AS failed,
                  COUNT(*) FILTER (WHERE status = 'pending')                       AS pending,
                  COUNT(*) FILTER (WHERE ingested_at > NOW() - INTERVAL '1 hour')  AS last_1h,
                  COUNT(*) FILTER (WHERE ingested_at > NOW() - INTERVAL '24 hours') AS last_24h,
                  MAX(ingested_at)                                                  AS last_seen
                FROM documents
                GROUP BY 1
                ORDER BY total DESC
            """)
                )
            ).fetchall()

        # Per-tool error breakdown from pipeline_errors (joined via doc_id)
        error_rows = (
            await conn.execute(
                sa_text("""
            SELECT
              CASE
                WHEN d.url LIKE '%reddit.com%'                             THEN 'reddit'
                WHEN d.url LIKE '%youtube.com%' OR d.url LIKE '%youtu.be%' THEN 'youtube'
                WHEN d.url LIKE '%huggingface.co%'                         THEN 'huggingface'
                ELSE 'rss'
              END AS tool,
              e.step,
              e.error_type,
              COUNT(*) AS n
            FROM pipeline_errors e
            JOIN documents d ON d.id = e.doc_id
            WHERE e.resolved = false
            GROUP BY 1, 2, 3
            ORDER BY n DESC
        """)
            )
        ).fetchall()

    tools = []
    for r in rows:
        last_seen = r[7]
        tools.append(
            {
                "tool": r[0],
                "total": r[1],
                "ready": r[2],
                "failed": r[3],
                "pending": r[4],
                "last_1h": r[5],
                "last_24h": r[6],
                "last_seen": last_seen if isinstance(last_seen, str) else (last_seen.isoformat() if last_seen else None),
                "fail_rate": round(r[3] / r[1] * 100, 1) if r[1] else 0,
            }
        )

    errors: dict = {}  # type: ignore[type-arg]
    for r in error_rows:
        tool = r[0]
        if tool not in errors:
            errors[tool] = []
        errors[tool].append({"step": r[1], "error_type": r[2], "count": r[3]})

    return {"tools": tools, "errors": errors}


@router.get("/documents", include_in_schema=False)
async def list_documents(request: Request, limit: int = 50, offset: int = 0) -> dict:  # type: ignore[type-arg]
    pg = _pg(request)
    docs = await pg.list_recent(limit=min(limit, 200), offset=offset)
    return {
        "docs": [
            {
                "id": str(d.id),
                "url": d.url,
                "title": d.title or d.url,
                "source": d.source,
                "status": d.status.value,
                "topics": (d.topics or [])[:5],
                "entities": (d.entities or [])[:5],
                "sentiment": round(d.sentiment, 2) if d.sentiment is not None else None,
                "ingested_at": d.ingested_at.isoformat() if d.ingested_at else None,
                "crawl_session": str(d.crawl_session) if d.crawl_session else None,
            }
            for d in docs
        ]
    }


# ── HTML shell (served from static/ directory) ──────────────────────────────


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    html = (_STATIC_DIR / "dashboard.html").read_text()
    return HTMLResponse(html)


# ── Ingest status UI (served from static/ directory) ────────────────────────


@router.get("/ui/ingest-status", response_class=HTMLResponse, include_in_schema=False)
async def ingest_status_ui() -> HTMLResponse:
    html = (_STATIC_DIR / "ingest-status.html").read_text()
    return HTMLResponse(html)


@router.get("/service-status", include_in_schema=False)
async def service_status(request: Request) -> dict:  # type: ignore[type-arg]
    """Return green/red status for key services: API, Database, MCP, Web Panel."""
    import httpx

    base_url = str(request.base_url).rstrip("/")
    results: dict = {}

    # 1. API — self (we're serving this response, so always up)
    results["api"] = {"ok": True, "label": "API", "detail": "responding"}

    # 2. Web Panel — check /dashboard route
    results["web_panel"] = {"ok": True, "label": "Web Panel", "detail": "responding"}

    # 3. Database — quick query
    try:
        from sqlalchemy import text as sqlt
        pg = request.app.state.postgres
        async with pg._engine.connect() as conn:
            await conn.execute(sqlt("SELECT 1"))
        results["database"] = {"ok": True, "label": "Database", "detail": "connected"}
    except Exception as exc:
        results["database"] = {"ok": False, "label": "Database", "detail": str(exc)[:120]}

    # 4. MCP — check /mcp/tools endpoint reachability
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(f"{base_url}/mcp/tools")
        # 404 = no MCP sidecar running (degraded, not error)
        if resp.status_code == 404:
            results["mcp"] = {"ok": True, "label": "MCP", "detail": "Not running (dev mode)"}
        else:
            results["mcp"] = {
                "ok": resp.status_code < 500,
                "label": "MCP",
                "detail": f"HTTP {resp.status_code}",
            }
    except Exception as exc:
        results["mcp"] = {"ok": False, "label": "MCP", "detail": str(exc)[:120]}

    # 5. Ingestion — Wikipedia doc pipeline test
    try:
        from sqlalchemy import text as sqlt
        pg = request.app.state.postgres
        async with pg._engine.begin() as conn:
            result = await conn.execute(
                sqlt(
                    "SELECT COUNT(*) as total, "
                    "COUNT(*) FILTER (WHERE enriched_at IS NOT NULL) as enriched "
                    "FROM documents "
                    "WHERE source LIKE 'en.wikipedia.org%%' "
                    "AND ingested_at > NOW() - INTERVAL '24 hours'"
                )
            )
            row = result.fetchone()
            total = row[0] if row else 0
            enriched = row[1] if row else 0
        if total > 0 and enriched > 0:
            results["ingestion"] = {
                "ok": True, "label": "Ingestion Pipeline",
                "detail": f"{enriched}/{total} Wikipedia docs enriched (24h)",
            }
        elif total > 0:
            results["ingestion"] = {
                "ok": True, "label": "Ingestion Pipeline",
                "detail": f"{total} Wikipedia docs ingested, {total - enriched} pending enrichment",
            }
        else:
            results["ingestion"] = {
                "ok": True, "label": "Ingestion Pipeline",
                "detail": "No Wikipedia docs in last 24h — submit a URL via Corpus page or wait for ingestion cycle",
            }
    except Exception as exc:
        results["ingestion"] = {
            "ok": False, "label": "Ingestion Pipeline",
            "detail": str(exc)[:120],
        }

    overall_ok = all(v["ok"] for v in results.values())

    # Normalise each service entry so status.html can consume it:
    # status.html expects {status: "ok"|"error", message: str}
    # dashboard callers expect {ok: bool, label: str, detail: str}
    # Emit both so both consumers work.
    from datetime import datetime as _dt
    for v in results.values():
        v["status"] = "ok" if v["ok"] else "error"
        v["message"] = v.get("detail", "")

    return {
        "ok": overall_ok,
        "overall": "ok" if overall_ok else "error",
        "checked_at": _dt.utcnow().isoformat() + "Z",
        "services": results,
    }


@router.get("/ui/status", response_class=HTMLResponse, include_in_schema=False)
async def status_ui() -> HTMLResponse:
    html = (_STATIC_DIR / "status.html").read_text()
    return HTMLResponse(html)


# ── MCP Config UI (served from static/ with dynamic API URL) ─────────────────


@router.get("/ui/mcp-config", response_class=HTMLResponse, include_in_schema=False)
async def mcp_config_ui(request: Request) -> HTMLResponse:
    base_url = str(request.base_url).rstrip("/")
    html = (_STATIC_DIR / "mcp-config.html").read_text()
    return HTMLResponse(html.replace("__API_URL__", base_url))

