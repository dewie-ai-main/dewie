# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""Graph API — neighbors, intersection, bridge path."""

from __future__ import annotations

import logging
import time as _time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger("dewie.api")

router = APIRouter(prefix="/graph", tags=["graph"])


def _pg(request: Request):
    return request.app.state.postgres


def _is_sqlite(request: Request) -> bool:
    return getattr(_pg(request), "_is_sqlite", False)


def _in_clause(ids: list[str], prefix: str = "id") -> tuple[str, dict]:
    """Build IN-clause params for SQLite (no ANY/uuid[] support)."""
    params = {f"{prefix}{i}": v for i, v in enumerate(ids)}
    placeholders = ", ".join(f":{prefix}{i}" for i in range(len(ids)))
    return placeholders, params


async def _neighbors_raw(session: AsyncSession, doc_id: str, limit: int = 20, sqlite: bool = False) -> list[dict]:
    if sqlite:
        rows = (
            (
                await session.execute(
                    text("""
        SELECT doc_id, title, summary, keywords, entities, answers_questions, weight
        FROM (
            SELECT e.target_id AS doc_id, d.title, d.summary,
                   d.keywords, d.entities, d.answers_questions, e.weight
            FROM document_edges e
            JOIN documents d ON d.id = e.target_id
            WHERE e.source_id = :doc_id
            UNION ALL
            SELECT e.source_id AS doc_id, d.title, d.summary,
                   d.keywords, d.entities, d.answers_questions, e.weight
            FROM document_edges e
            JOIN documents d ON d.id = e.source_id
            WHERE e.target_id = :doc_id
        ) combined
        ORDER BY weight DESC
        LIMIT :limit
    """),
                    {"doc_id": doc_id, "limit": limit},
                )
            )
            .mappings()
            .all()
        )
        return [{k: v for k, v in dict(r).items() if k != "answers_questions"} for r in rows]
    rows = (
        (
            await session.execute(
                text("""
        SELECT doc_id, title, summary, keywords, entities, answers_questions, weight
        FROM (
            SELECT e.target_id::text AS doc_id, d.title, d.summary,
                   d.keywords, d.entities, d.answers_questions, e.weight
            FROM document_edges e
            JOIN documents d ON d.id = e.target_id
            WHERE e.source_id = cast(:doc_id as uuid)
            UNION ALL
            SELECT e.source_id::text AS doc_id, d.title, d.summary,
                   d.keywords, d.entities, d.answers_questions, e.weight
            FROM document_edges e
            JOIN documents d ON d.id = e.source_id
            WHERE e.target_id = cast(:doc_id as uuid)
        ) combined
        ORDER BY weight DESC
        LIMIT :limit
    """),
                {"doc_id": doc_id, "limit": limit},
            )
        )
        .mappings()
        .all()
    )
    return [{k: v for k, v in dict(r).items() if k != "answers_questions"} for r in rows]


@router.get("/neighbors/{doc_id}")
async def get_neighbors(doc_id: UUID, request: Request, limit: int = 20) -> list[dict[str, Any]]:
    """Return neighbor docs sorted by edge weight."""
    request_id = getattr(request.state, "request_id", "unknown")
    log.info("graph_neighbors started", extra={"request_id": request_id, "doc_id": str(doc_id), "limit": limit})
    pg = _pg(request)
    workspace_ids = getattr(request.state, "workspace_ids", [])
    try:
        async with pg._session_factory() as session:
            result = await _neighbors_raw(session, str(doc_id), limit, sqlite=_is_sqlite(request))
        log.info("graph_neighbors succeeded", extra={"request_id": request_id, "count": len(result)})
        return result
    except Exception:
        log.exception("graph_neighbors failed", extra={"request_id": request_id})
        raise


@router.post("/intersection")
async def intersection(request: Request, body: dict) -> dict[str, Any]:
    """
    Given a list of doc_ids, return docs that are neighbors of
    ALL (or most) of them — the overlap zone.
    min_overlap: how many pinned docs must share a neighbor (default = all).
    """
    doc_ids: list[str] = [str(d) for d in body.get("doc_ids", [])]
    limit: int = body.get("limit", 30)
    min_overlap: int = body.get("min_overlap", len(doc_ids))  # default = strict intersection

    request_id = getattr(request.state, "request_id", "unknown")
    log.info("graph_intersection started", extra={"request_id": request_id, "doc_count": len(doc_ids), "min_overlap": min_overlap})
    if len(doc_ids) < 2:
        return {"docs": [], "error": "Need at least 2 doc_ids"}

    pg = _pg(request)
    workspace_ids = getattr(request.state, "workspace_ids", [])
    sqlite = _is_sqlite(request)
    async with pg._session_factory() as session:
        if sqlite:
            placeholders, params = _in_clause(doc_ids, prefix="sid")
            not_placeholders, not_params = _in_clause(doc_ids, prefix="nid")
            all_params = {**params, **not_params, "min_overlap": min_overlap, "limit": limit}
            rows = (
                (
                    await session.execute(
                        text(f"""
                SELECT e.target_id AS doc_id,
                       d.title, d.summary, d.keywords, d.entities, d.answers_questions,
                       COUNT(*) AS overlap_count,
                       AVG(e.weight) AS avg_weight
                FROM document_edges e
                JOIN documents d ON d.id = e.target_id
                WHERE e.source_id IN ({placeholders})
                  AND e.target_id NOT IN ({not_placeholders})
                GROUP BY e.target_id, d.title, d.summary, d.keywords, d.entities, d.answers_questions
                HAVING COUNT(*) >= :min_overlap
                ORDER BY COUNT(*) DESC, AVG(e.weight) DESC
                LIMIT :limit
            """),
                        all_params,
                    )
                )
                .mappings()
                .all()
            )
        else:
            # Count how many of the pinned docs each candidate appears as neighbor of
            rows = (
                (
                    await session.execute(
                        text("""
                SELECT e.target_id::text AS doc_id,
                       d.title, d.summary, d.keywords, d.entities, d.answers_questions,
                       COUNT(*) AS overlap_count,
                       AVG(e.weight) AS avg_weight,
                       ARRAY_AGG(e.source_id::text) AS from_docs
                FROM document_edges e
                JOIN documents d ON d.id = e.target_id
                WHERE e.source_id = ANY(cast(:ids as uuid[]))
                  AND e.target_id != ALL(cast(:ids as uuid[]))
                GROUP BY e.target_id, d.title, d.summary, d.keywords, d.entities, d.answers_questions
                HAVING COUNT(*) >= :min_overlap
                ORDER BY COUNT(*) DESC, AVG(e.weight) DESC
                LIMIT :limit
            """),
                        {"ids": doc_ids, "min_overlap": min_overlap, "limit": limit},
                    )
                )
                .mappings()
                .all()
            )

    result = {
        "docs": [{k: v for k, v in dict(r).items() if k != "answers_questions"} for r in rows],
        "pinned_count": len(doc_ids),
        "min_overlap": min_overlap,
    }
    log.info("graph_intersection succeeded", extra={"request_id": request_id, "doc_count": len(result["docs"])})
    return result


@router.post("/bridge")
async def bridge_path(request: Request, body: dict) -> dict[str, Any]:
    """
    Find the shortest path between two doc_ids through the graph.
    Uses BFS over document_edges up to max_depth hops.
    """
    t0 = _time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    source: str = str(body.get("source_id", ""))
    target: str = str(body.get("target_id", ""))
    max_depth: int = min(body.get("max_depth", 5), 8)
    log.info("graph_bridge started", extra={"request_id": request_id, "source": source, "target": target, "max_depth": max_depth})

    if not source or not target:
        return {"path": [], "error": "Need source_id and target_id"}
    if source == target:
        return {"path": [source], "hops": 0}

    pg = _pg(request)
    workspace_ids = getattr(request.state, "workspace_ids", [])
    sqlite = _is_sqlite(request)

    async with pg._session_factory() as session:

        # BFS
        visited = {source: None}  # node -> parent
        frontier = [source]
        found = False

        for _depth in range(max_depth):
            if not frontier:
                break
            # Get all neighbors of current frontier in one query
            if sqlite:
                placeholders, params = _in_clause(frontier, prefix="fid")
                bfs_sql = text(f"""
                    SELECT source_id, target_id, weight
                    FROM document_edges
                    WHERE source_id IN ({placeholders})
                    ORDER BY weight DESC
                """)
                bfs_params = params
            else:
                bfs_sql = text("""
                    SELECT source_id::text, target_id::text, weight
                    FROM document_edges
                    WHERE source_id = ANY(cast(:ids as uuid[]))
                    ORDER BY weight DESC
                """)
                bfs_params = {"ids": frontier}
            rows = (
                (
                    await session.execute(bfs_sql, bfs_params)
                )
                .mappings()
                .all()
            )

            next_frontier = []
            for r in rows:
                nid = r["target_id"]
                if nid not in visited:
                    visited[nid] = r["source_id"]
                    if nid == target:
                        found = True
                        break
                    next_frontier.append(nid)
            if found:
                break
            frontier = next_frontier

        if not found:
            return {"path": [], "hops": -1, "error": f"No path found within {max_depth} hops"}

        # Reconstruct path
        path_ids = []
        cur = target
        while cur is not None:
            path_ids.append(cur)
            cur = visited.get(cur)
        path_ids.reverse()

        # Fetch titles
        if sqlite:
            placeholders, params = _in_clause(path_ids, prefix="pid")
            title_sql = text(f"SELECT id, title, summary, keywords, entities FROM documents WHERE id IN ({placeholders})")
            title_params = params
        else:
            title_sql = text("SELECT id::text, title, summary, keywords, entities FROM documents WHERE id = ANY(cast(:ids as uuid[]))")
            title_params = {"ids": path_ids}
        rows = (
            (
                await session.execute(title_sql, title_params)
            )
            .mappings()
            .all()
        )

    by_id = {r["id"]: dict(r) for r in rows}
    path = [by_id.get(pid, {"id": pid, "title": pid}) for pid in path_ids]

    result = {"path": path, "hops": len(path) - 1}
    log.info("graph_bridge succeeded", extra={"request_id": request_id, "hops": result["hops"], "elapsed_s": round(_time.time() - t0, 3)})

    elapsed = _time.time() - t0
    try:
        from dewie.storage.query_logger import QueryLogEntry
        from dewie.storage.query_logger import log_query as _log_query

        await _log_query(
            QueryLogEntry(
                question=f"bridge:{source}->{target}",
                source="api",
                workspace_ids=getattr(request.state, "workspace_ids", []),
                user_id=getattr(request.state, "user_id", None),
                elapsed_ms=int(elapsed * 1000),
            )
        )
    except Exception:
        pass

    return result
