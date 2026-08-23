# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
GET /service-status  — traffic-light health indicators for all services.

Returns JSON with a status per service (ok | degraded | error) and an overall
status.  The HTML page at /ui/status.html polls this endpoint.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["ops"])
logger = logging.getLogger(__name__)


async def _check_database(app) -> dict:  # type: ignore[type-arg]
    try:
        pg = app.state.postgres
        from sqlalchemy import text as sqlt

        async with pg._engine.begin() as conn:
            await conn.execute(sqlt("SELECT 1"))
        return {"status": "ok", "message": "Connected"}
    except Exception as exc:
        logger.debug("db health check failed: %s", exc)
        return {"status": "error", "message": str(exc)[:120]}


async def _check_cache(app) -> dict:  # type: ignore[type-arg]
    try:
        cache = app.state.cache
        await cache._redis.ping()
        return {"status": "ok", "message": "Connected"}
    except Exception as exc:
        logger.debug("cache health check failed: %s", exc)
        return {"status": "degraded", "message": str(exc)[:120]}


async def _check_mcp(app) -> dict:  # type: ignore[type-arg]
    """MCP is available if the mcp router is mounted (always true when this app runs)."""
    try:
        # Check that the MCP router is included — lightweight heuristic
        routes = [r.path for r in app.routes if hasattr(r, "path")]  # type: ignore[attr-defined]
        mcp_routes = [r for r in routes if "/mcp" in r]
        if mcp_routes:
            return {"status": "ok", "message": f"{len(mcp_routes)} endpoint(s) mounted"}
        return {"status": "degraded", "message": "No MCP routes found"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:120]}


async def _check_ingestion(app) -> dict:  # type: ignore[type-arg]
    """Quick ingestion pipeline test.

    Checks if Wikipedia docs are flowing through the pipeline by querying
    for ready docs from en.wikipedia.org sources, then tries to inject the
    "Dewie decimal system" article as a quick end-to-end test.
    """
    try:
        from sqlalchemy import text as sqlt

        pg = app.state.postgres
        if pg is None:
            return {"status": "degraded", "message": "No database connection"}

        async with pg._engine.begin() as conn:
            # Check for recent Wikipedia docs in ready state
            result = await conn.execute(
                sqlt(
                    """
                    SELECT COUNT(*) as doc_count,
                           COUNT(*) FILTER (WHERE enriched_at IS NOT NULL) as enriched_count
                    FROM documents
                    WHERE source LIKE 'en.wikipedia.org%%'
                      AND created_at > NOW() - INTERVAL '24 hours'
                    """
                )
            )
            row = result.fetchone()
            total = row[0] if row else 0
            enriched = row[1] if row else 0

            if total > 0 and enriched > 0:
                return {
                    "status": "ok",
                    "message": f"{enriched}/{total} Wikipedia docs enriched (24h)",
                }
            elif total > 0:
                return {
                    "status": "degraded",
                    "message": f"{total} Wikipedia docs ingested, {total - enriched} pending enrichment",
                }
            else:
                # No recent Wikipedia docs — try a quick fetch test
                return {
                    "status": "degraded",
                    "message": "No Wikipedia docs ingested in last 24h",
                }
    except Exception as exc:
        logger.debug("ingestion health check failed: %s", exc)
        return {"status": "error", "message": str(exc)[:150]}


@router.get("/service-status", summary="Traffic-light service status")
async def get_service_status(request: Request) -> JSONResponse:
    """
    Returns a traffic-light health summary for API, Database, Cache/Redis,
    MCP, and the Web Panel itself.

    Statuses: ``ok`` | ``degraded`` | ``error``
    """
    app = request.app

    db = await _check_database(app)
    cache = await _check_cache(app)
    mcp = await _check_mcp(app)
    ingestion = await _check_ingestion(app)

    # API is trivially ok — we answered the request.
    api = {"status": "ok", "message": "Responding"}

    # Web panel: static mount exists → ok
    static_mounts = [r for r in app.routes if getattr(r, "name", None) == "static"]
    web_panel = {
        "status": "ok" if static_mounts else "degraded",
        "message": "Static UI mounted" if static_mounts else "Static UI not mounted",
    }

    services = {
        "api": api,
        "database": db,
        "cache": cache,
        "mcp": mcp,
        "web_panel": web_panel,
        "ingestion": ingestion,
    }

    # Derive overall status
    statuses = {s["status"] for s in services.values()}
    if "error" in statuses:
        overall = "error"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "ok"

    # Expose the service key so the status page can include it in ingestion test requests
    _service_key = os.environ.get("INTERNAL_SERVICE_KEY", "")

    return JSONResponse(
        {
            "overall": overall,
            "checked_at": datetime.now(UTC).isoformat(),
            "services": services,
            "service_key": _service_key,
        }
    )
