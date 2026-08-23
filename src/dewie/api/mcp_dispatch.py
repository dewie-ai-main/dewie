# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Transport-agnostic MCP tool dispatch — extracted from api/routes/mcp.py so the
same business logic can be reused by both the REST `/api/mcp` endpoint and the
in-process Streamable HTTP transport (api/mcp_streamable.py), without either
one making an HTTP call back into the other.

`HTTPException` is used as the cross-transport error type. It's just a plain
exception class (status_code + detail) — works fine raised outside a live
FastAPI request context. The REST route lets it propagate natively (FastAPI
already handles HTTPException raised anywhere during request processing); the
MCP transport lets it propagate too — the `mcp` SDK's lowlevel Server.call_tool
handler catches any Exception and converts it to an MCP isError result
automatically, so no special translation is needed on that side either.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import HTTPException

log = logging.getLogger(__name__)

# Fields that must be redacted from log output
_SENSITIVE_FIELDS = {"api_key", "password", "token", "secret", "authorization"}


def _redact(value: str | None) -> str | None:
    """Redact a potentially sensitive string value for logging."""
    if value is None:
        return None
    if len(value) > 1000:
        value = value[:1000] + "... [truncated]"
    for field in _SENSITIVE_FIELDS:
        if field in value.lower():
            return "***REDACTED***"
    return value


def _log_mcp_call(
    tool_name: str,
    input_params: dict[str, Any],
    user_id: str | None,
    model: str | None,
    elapsed_ms: float,
    result_summary: dict[str, Any] | None = None,
) -> None:
    """Log an MCP tool call to the query_log table with source=mcp.

    Transport-agnostic: takes plain values instead of a Request, so callers on
    either transport can supply request_id/model however is natural for them.
    """
    from dewie.storage.query_logger import QueryLogEntry, log_query

    question_parts = [f"mcp:{tool_name}"]

    if tool_name in ("search_corpus", "browse", "research"):
        q = str(input_params.get("query", "")).strip()
        if q:
            question_parts.append(q[:200])
    elif tool_name in ("ingest_url", "dewie_ingest"):
        u = str(input_params.get("url", "")).strip()
        if u:
            question_parts.append(f"ingest:{u[:200]}")
    elif tool_name in ("expand", "read", "intersect", "bridge"):
        did = str(input_params.get("doc_id", "") or input_params.get("source_id", "")).strip()
        if did:
            question_parts.append(did[:200])

    question = " | ".join(question_parts)

    entry = QueryLogEntry(
        question=question,
        source="mcp",
        model=model,
        user_id=str(user_id) if user_id else None,
        elapsed_ms=int(round(elapsed_ms)),
        docs_returned=[result_summary] if result_summary else [],
    )

    import asyncio

    asyncio.create_task(log_query(entry))


# ── Background-task helpers (fire-and-forget, transport-agnostic) ──────────────


async def _bg_ingest(doc, pg) -> None:
    """Fire-and-forget corpus save for dewie_fetch."""
    try:
        await pg.upsert(doc)
        from dewie.storage.body_store import save_body

        if doc.body:
            save_body(doc.id, doc.body)
            await pg.write_body_text(doc.id, doc.body)
    except Exception as exc:
        log.warning("_bg_ingest failed for %s: %s", getattr(doc, "id", "?"), exc)


# ── Graph helpers ────────────────────────────────────────────────────────────


async def _neighbors(pg, doc_id: str, limit: int = 20, *, workspace_ids: list | None = None) -> list[dict]:
    """Get graph neighbours for a document (mirrors graph.get_neighbors)."""
    from sqlalchemy import text as sa_text

    sqlite = getattr(pg, "_is_sqlite", False)
    async with pg._session_factory() as session:
        if sqlite:
            rows = (
                (
                    await session.execute(
                        sa_text("""
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
        else:
            rows = (
                (
                    await session.execute(
                        sa_text("""
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


async def _read_body(pg, doc_id: str) -> str | None:
    """Read the full text body of a document (mirrors documents.get_content)."""
    import uuid as _uuid

    from sqlalchemy import text as sa_text

    from dewie.storage.body_store import load_body

    try:
        doc = await pg.get_by_id(_uuid.UUID(doc_id))
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Document not found.") from None

    body: str | None = None
    try:
        async with pg._session_factory() as session:
            row = (
                (
                    await session.execute(
                        sa_text("SELECT body_text FROM documents WHERE id = CAST(:id AS UUID)"),
                        {"id": doc_id},
                    )
                )
                .mappings()
                .first()
            )
        if row and row["body_text"]:
            body = row["body_text"]
    except Exception:
        pass  # fall through to file store

    if not body:
        body = load_body(doc_id)

    return body


async def _intersection(
    pg, doc_ids: list[str], min_overlap: int | None = None, limit: int = 10, *, workspace_ids: list | None = None
) -> dict:
    """Find document overlap (mirrors graph.intersection)."""
    from sqlalchemy import text as sa_text

    sqlite = getattr(pg, "_is_sqlite", False)
    if min_overlap is None:
        min_overlap = len(doc_ids)

    async with pg._session_factory() as session:
        if sqlite:

            def _in_clause(ids: list[str], prefix: str = "id") -> tuple[str, dict]:
                params = {f"{prefix}{i}": v for i, v in enumerate(ids)}
                placeholders = ", ".join(f":{prefix}{i}" for i in range(len(ids)))
                return placeholders, params

            placeholders, params = _in_clause(doc_ids, prefix="sid")
            not_placeholders, not_params = _in_clause(doc_ids, prefix="nid")
            all_params = {**params, **not_params, "min_overlap": min_overlap, "limit": limit}
            rows = (
                (
                    await session.execute(
                        sa_text(f"""
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
            rows = (
                (
                    await session.execute(
                        sa_text("""
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

    return {
        "docs": [{k: v for k, v in dict(r).items() if k != "answers_questions"} for r in rows],
        "pinned_count": len(doc_ids),
        "min_overlap": min_overlap,
    }


async def _bridge_path(
    pg, source_id: str, target_id: str, max_depth: int = 5, *, workspace_ids: list | None = None
) -> dict:
    """Find shortest path between two documents (mirrors graph.bridge_path)."""
    from sqlalchemy import text as sa_text

    sqlite = getattr(pg, "_is_sqlite", False)

    if source_id == target_id:
        return {"path": [source_id], "hops": 0}

    async with pg._session_factory() as session:
        # BFS
        visited = {source_id: None}
        frontier = [source_id]
        found = False

        def _in_clause(ids: list[str], prefix: str = "id") -> tuple[str, dict]:
            params = {f"{prefix}{i}": v for i, v in enumerate(ids)}
            placeholders = ", ".join(f":{prefix}{i}" for i in range(len(ids)))
            return placeholders, params

        for _depth in range(max_depth):
            if not frontier:
                break
            if sqlite:
                placeholders, params = _in_clause(frontier, prefix="fid")
                bfs_rows = (
                    (
                        await session.execute(
                            sa_text(f"""
                        SELECT source_id, target_id, weight
                        FROM document_edges
                        WHERE source_id IN ({placeholders})
                        ORDER BY weight DESC
                    """),
                            params,
                        )
                    )
                    .mappings()
                    .all()
                )
            else:
                bfs_rows = (
                    (
                        await session.execute(
                            sa_text("""
                        SELECT source_id::text, target_id::text, weight
                        FROM document_edges
                        WHERE source_id = ANY(cast(:ids as uuid[]))
                        ORDER BY weight DESC
                    """),
                            {"ids": frontier},
                        )
                    )
                    .mappings()
                    .all()
                )

            next_frontier = []
            for r in bfs_rows:
                nid = r["target_id"]
                if nid not in visited:
                    visited[nid] = r["source_id"]
                    if nid == target_id:
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
        cur = target_id
        while cur is not None:
            path_ids.append(cur)
            cur = visited.get(cur)
        path_ids.reverse()

        # Fetch titles
        if sqlite:
            placeholders, params = _in_clause(path_ids, prefix="pid")
            title_rows = (
                (
                    await session.execute(
                        sa_text(f"SELECT id, title, summary, keywords, entities FROM documents WHERE id IN ({placeholders})"),
                        params,
                    )
                )
                .mappings()
                .all()
            )
        else:
            title_rows = (
                (
                    await session.execute(
                        sa_text(
                            "SELECT id::text, title, summary, keywords, entities FROM documents "
                            "WHERE id = ANY(cast(:ids as uuid[]))"
                        ),
                        {"ids": path_ids},
                    )
                )
                .mappings()
                .all()
            )

    by_id = {str(r["id"]): dict(r) for r in title_rows}
    path = [by_id.get(pid, {"id": pid, "title": pid}) for pid in path_ids]

    return {"path": path, "hops": len(path) - 1}


# ── Main dispatch ────────────────────────────────────────────────────────────


async def dispatch_mcp_tool(
    tool_name: str,
    input_data: dict[str, Any],
    *,
    pg,
    user_id: str | None,
    workspace_ids: list,
    is_admin: bool,
    key_id: str | None = None,
    model: str | None = None,
    request_id: str = "unknown",
    enqueue_background: Callable[[Coroutine], None],
) -> dict[str, Any]:
    """Dispatch one MCP tool call and return its `content` payload.

    Raises HTTPException on any auth/validation/not-found/internal failure —
    both the REST route and the MCP transport let it propagate natively.
    """
    t0 = time.monotonic()
    body_input = input_data

    # ── search_corpus ──────────────────────────────────────────────────────────
    if tool_name == "search_corpus":
        query = body_input.get("query", "").strip()
        if not query:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch search_corpus rejected request_id=%s reason=%s elapsed_ms=%.2f", request_id, "empty_query", elapsed)
            raise HTTPException(status_code=422, detail="'query' is required for search_corpus.")

        corpus_id = body_input.get("corpus_id")
        limit = min(int(body_input.get("limit", 10)), 25)
        source_filter_val = (body_input.get("source") or "").strip() or None

        # Default to user's personal corpus if no corpus_id given
        if not corpus_id and user_id:
            corpus_id = f"user:{user_id}"

        log.info(
            "dispatch search_corpus dispatching request_id=%s corpus_id=%s limit=%d source=%s",
            request_id, _redact(corpus_id), limit, source_filter_val,
        )

        try:
            results = await pg.search(
                query=query,
                limit=limit,
                ranker="rrf",
                workspace_ids=workspace_ids,
                source_filter=[source_filter_val] if source_filter_val else None,
            )
        except Exception as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.exception("dispatch search_corpus failed request_id=%s elapsed_ms=%.2f", request_id, elapsed)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "search_failed"})
            raise HTTPException(status_code=500, detail="Search failed.") from exc

        # Build safe result list — NEVER include answers_questions
        safe_results = []
        for doc, score in results:
            safe_results.append(
                {
                    "doc_id": str(doc.id),
                    "title": doc.title,
                    "url": str(doc.url) if doc.url else None,
                    "summary": doc.summary,
                    "document_type": doc.document_type.value if doc.document_type else None,
                    "source": doc.source,
                    "score": round(score, 4),
                    # answers_questions is intentionally excluded
                }
            )

        # Increment query usage if API key auth
        if key_id and user_id:
            try:
                from sqlalchemy import text

                async with pg._engine.begin() as conn:
                    await conn.execute(
                        text("""
                        INSERT INTO user_usage (user_id, day, queries_run)
                        VALUES (CAST(:uid AS UUID), CURRENT_DATE, 1)
                        ON CONFLICT (user_id, day)
                        DO UPDATE SET queries_run = user_usage.queries_run + 1
                    """),
                        {"uid": user_id},
                    )
            except Exception:
                pass

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info("dispatch search_corpus succeeded request_id=%s result_count=%d elapsed_ms=%.2f", request_id, len(safe_results), elapsed)
        _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"results": safe_results, "count": len(safe_results)})

        return {"results": safe_results, "count": len(safe_results)}

    # ── ingest_url / dewie_ingest ────────────────────────────────────────────────
    elif tool_name in ("ingest_url", "dewie_ingest"):
        url = body_input.get("url", "").strip()
        if not url:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch ingest_url rejected request_id=%s reason=%s elapsed_ms=%.2f", request_id, "empty_url", elapsed)
            raise HTTPException(status_code=422, detail="'url' is required for ingest_url.")

        if not user_id:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch ingest_url rejected request_id=%s reason=%s elapsed_ms=%.2f", request_id, "no_user_id", elapsed)
            raise HTTPException(status_code=403, detail="ingest_url requires user-level auth.")

        log.info("dispatch ingest_url dispatching request_id=%s url=%s", request_id, _redact(url))

        from dewie.ingestion.web import WebIngester

        async with WebIngester() as ingester:
            docs = [doc async for doc in ingester.fetch(url)]

        if not docs:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch ingest_url no content request_id=%s url=%s elapsed_ms=%.2f", request_id, _redact(url), elapsed)
            raise HTTPException(status_code=422, detail="No content found at URL.")

        doc = docs[0]
        doc.corpus_id = f"user:{user_id}"

        await pg.upsert(doc)

        from dewie.storage.body_store import save_body

        if hasattr(doc, "body") and doc.body:
            save_body(doc.id, doc.body)
            try:
                await pg.write_body_text(doc.id, doc.body)
            except Exception:
                pass

        # Tag with user_id
        try:
            from sqlalchemy import text

            async with pg._engine.begin() as conn:
                await conn.execute(
                    text("UPDATE documents SET user_id = CAST(:uid AS UUID) WHERE id = CAST(:doc_id AS UUID)"),
                    {"uid": user_id, "doc_id": str(doc.id)},
                )
        except Exception as exc:
            log.warning("dispatch ingest: failed to set user_id on doc %s: %s", doc.id, exc)

        # Increment usage
        try:
            from sqlalchemy import text

            async with pg._engine.begin() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO user_usage (user_id, day, docs_ingested)
                    VALUES (CAST(:uid AS UUID), CURRENT_DATE, 1)
                    ON CONFLICT (user_id, day)
                    DO UPDATE SET docs_ingested = user_usage.docs_ingested + 1
                """),
                    {"uid": user_id},
                )
        except Exception:
            pass

        # No eager enqueue_background(_enrich_one(...)) here — doc is already
        # status='pending', so run_enrichment_loop's poller (batch_size/sleep_secs
        # throttled) picks it up on its next tick. An eager call here bypassed
        # that throttle entirely: bulk ingests (many ingest_url calls in a row)
        # fired one unbounded concurrent enrichment task per doc, hammering the
        # configured LLM/embedding backend all at once.

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info("dispatch ingest_url succeeded request_id=%s doc_id=%s elapsed_ms=%.2f", request_id, str(doc.id), elapsed)
        _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"doc_id": str(doc.id), "status": "pending", "doc_count": 1})

        return {
            "doc_id": str(doc.id),
            "status": "pending",
            "message": "Accepted for processing.",
        }

    # ── expand (dewie_expand) ────────────────────────────────────────────────────
    elif tool_name == "expand":
        doc_id = body_input.get("doc_id", "").strip()
        if not doc_id:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch expand rejected request_id=%s reason=%s elapsed_ms=%.2f", request_id, "empty_doc_id", elapsed)
            raise HTTPException(status_code=422, detail="'doc_id' is required for expand.")

        limit = int(body_input.get("limit", 20))
        log.info("dispatch expand request_id=%s doc_id=%s limit=%d", request_id, _redact(doc_id), limit)

        try:
            result = await _neighbors(pg, doc_id, limit, workspace_ids=workspace_ids)
        except Exception as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.exception("dispatch expand failed request_id=%s elapsed_ms=%.2f", request_id, elapsed)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "expand_failed"})
            raise HTTPException(status_code=500, detail="Expand failed.") from exc

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info("dispatch expand succeeded request_id=%s result_count=%d elapsed_ms=%.2f", request_id, len(result), elapsed)
        _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"doc_count": len(result)})

        return {"neighbors": result, "count": len(result)}

    # ── read (dewie_read) ────────────────────────────────────────────────────────
    elif tool_name == "read":
        doc_id = body_input.get("doc_id", "").strip()
        if not doc_id:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch read rejected request_id=%s reason=%s elapsed_ms=%.2f", request_id, "empty_doc_id", elapsed)
            raise HTTPException(status_code=422, detail="'doc_id' is required for read.")

        log.info("dispatch read request_id=%s doc_id=%s", request_id, _redact(doc_id))

        try:
            body_text = await _read_body(pg, doc_id)
        except HTTPException:
            raise
        except Exception as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.exception("dispatch read failed request_id=%s elapsed_ms=%.2f", request_id, elapsed)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "read_failed"})
            raise HTTPException(status_code=500, detail="Read failed.") from exc

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info("dispatch read succeeded request_id=%s body_chars=%d elapsed_ms=%.2f", request_id, len(body_text or ""), elapsed)
        _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": True, "body_chars": len(body_text or "")})

        return {"body": body_text}

    # ── intersect (dewie_intersect) ──────────────────────────────────────────────
    elif tool_name == "intersect":
        doc_ids = body_input.get("doc_ids", [])
        if len(doc_ids) < 2:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch intersect rejected request_id=%s reason=%s elapsed_ms=%.2f", request_id, "need_2_plus_doc_ids", elapsed)
            raise HTTPException(status_code=422, detail="'doc_ids' requires at least 2 document IDs.")

        limit = int(body_input.get("limit", 10))
        min_overlap = body_input.get("min_overlap")
        if min_overlap is not None:
            min_overlap = int(min_overlap)

        log.info("dispatch intersect request_id=%s doc_count=%d limit=%d min_overlap=%s", request_id, len(doc_ids), limit, min_overlap)

        try:
            result = await _intersection(pg, doc_ids, min_overlap, limit, workspace_ids=workspace_ids)
        except Exception as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.exception("dispatch intersect failed request_id=%s elapsed_ms=%.2f", request_id, elapsed)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "intersect_failed"})
            raise HTTPException(status_code=500, detail="Intersection failed.") from exc

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info("dispatch intersect succeeded request_id=%s result_count=%d elapsed_ms=%.2f", request_id, len(result.get("docs", [])), elapsed)
        _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"doc_count": len(result.get("docs", []))})

        return result

    # ── bridge (dewie_bridge) ────────────────────────────────────────────────────
    elif tool_name == "bridge":
        source_id = body_input.get("source_id", "").strip()
        target_id = body_input.get("target_id", "").strip()
        if not source_id or not target_id:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch bridge rejected request_id=%s reason=%s elapsed_ms=%.2f", request_id, "need_source_and_target", elapsed)
            raise HTTPException(status_code=422, detail="'source_id' and 'target_id' are required for bridge.")

        max_depth = min(int(body_input.get("max_depth", 5)), 8)

        log.info(
            "dispatch bridge request_id=%s source=%s target=%s max_depth=%d",
            request_id, _redact(source_id), _redact(target_id), max_depth,
        )

        try:
            result = await _bridge_path(pg, source_id, target_id, max_depth, workspace_ids=workspace_ids)
        except Exception as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.exception("dispatch bridge failed request_id=%s elapsed_ms=%.2f", request_id, elapsed)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "bridge_failed"})
            raise HTTPException(status_code=500, detail="Bridge failed.") from exc

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info("dispatch bridge succeeded request_id=%s hops=%d elapsed_ms=%.2f", request_id, result.get("hops", -1), elapsed)
        _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"hops": result.get("hops", -1)})

        return result

    # ── browse (dewie_browse) ────────────────────────────────────────────────────
    elif tool_name == "browse":
        query = body_input.get("query", "").strip()
        if not query:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch browse rejected request_id=%s reason=%s elapsed_ms=%.2f", request_id, "empty_query", elapsed)
            raise HTTPException(status_code=422, detail="'query' is required for browse.")

        limit = min(int(body_input.get("limit", 10)), 15)
        ranker = str(body_input.get("ranker", "rrf_aq"))

        log.info("dispatch browse request_id=%s query=%s limit=%d ranker=%s", request_id, _redact(query), limit, ranker)

        try:
            results = await pg.search(
                query=query,
                limit=limit,
                ranker=ranker,
                workspace_ids=workspace_ids,
            )
        except Exception as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.exception("dispatch browse failed request_id=%s elapsed_ms=%.2f", request_id, elapsed)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "browse_failed"})
            raise HTTPException(status_code=500, detail="Browse search failed.") from exc

        # Build formatted output similar to mcp_server _browse
        lines = [f"## {len(results)} articles found for: \"{query}\"\n"]
        for i, (doc, score) in enumerate(results, 1):
            pub = ""
            if hasattr(doc, "published_at") and doc.published_at:
                pub = str(doc.published_at)[:10]
            source = doc.source or ""
            title = doc.title or "(no title)"
            summary = (doc.summary or "")[:200].strip()
            doc_id_val = str(doc.id)
            lines.append(
                f"**{i}. {title}**\n"
                f"   Source: {source}{' · ' + pub if pub else ''} · relevance: {score:.2f}\n"
                f"   {summary}{'…' if len(doc.summary or '') > 200 else ''}\n"
                f"   doc_id: `{doc_id_val}`\n"
            )

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info("dispatch browse succeeded request_id=%s result_count=%d elapsed_ms=%.2f", request_id, len(results), elapsed)
        _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"doc_count": len(results)})

        return {"formatted": "\n".join(lines), "count": len(results)}

    # ── research (dewie_research) ─────────────────────────────────────────────────
    elif tool_name == "research":
        query = body_input.get("query", "").strip()
        if not query:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch research rejected request_id=%s reason=%s elapsed_ms=%.2f", request_id, "empty_query", elapsed)
            raise HTTPException(status_code=422, detail="'query' is required for research.")

        mode = str(body_input.get("mode", "quick"))
        max_iterations = min(int(body_input.get("max_iterations", 3)), 8)
        max_iterations = max(1, max_iterations)
        web_fallback = bool(body_input.get("web_fallback", False))
        corpus_id = body_input.get("corpus_id")

        log.info("dispatch research request_id=%s mode=%s query=%s", request_id, mode, _redact(query))

        try:
            from dewie.api.routes.research_agent import (
                AgentResearchRequest,
                UsageAccumulator,
                _resolve_model,
                _run_deep,
                _run_quick,
            )

            ra_body = AgentResearchRequest(
                query=query,
                mode=mode,
                max_iterations=max_iterations,
                web_fallback=web_fallback,
                corpus_id=corpus_id,
            )
            model_name = _resolve_model(None)
            usage = UsageAccumulator(model=model_name)
            trace: list[str] = [f"mode={mode}, model={model_name}"]

            if mode == "quick":
                docs_used, discarded, gaps, answer, confidence, web_used = await _run_quick(
                    pg, ra_body, model_name, usage, trace
                )
                iterations = 1
            else:
                docs_used, discarded, gaps, answer, confidence, iterations, web_used = await _run_deep(
                    pg, ra_body, model_name, usage, trace
                )

            content = {
                "answer": answer,
                "confidence": round(confidence, 4),
                "mode": mode,
                "docs_used": [
                    {
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "url": d.url,
                        "source": d.source,
                        "score": d.score,
                        "relevance": d.relevance,
                    }
                    for d in docs_used
                ],
                "docs_discarded": len(discarded),
                "gaps": gaps,
                "web_results_used": web_used,
                "iterations": iterations,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "model": usage.model,
                    "estimated_cost_usd": round(usage.estimated_cost_usd, 6),
                },
                "trace": trace,
            }
        except HTTPException:
            raise
        except Exception as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.exception("dispatch research failed request_id=%s elapsed_ms=%.2f", request_id, elapsed)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "research_failed"})
            raise HTTPException(status_code=500, detail="Research failed.") from exc

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "dispatch research succeeded request_id=%s mode=%s docs_used=%d elapsed_ms=%.2f",
            request_id, mode, len(content["docs_used"]), elapsed,
        )
        _log_mcp_call(
            tool_name, body_input, user_id, model, elapsed,
            result_summary={"doc_count": len(content["docs_used"]), "mode": mode, "confidence": content["confidence"]},
        )

        return content

    # ── web_search ─────────────────────────────────────────────────────────────
    elif tool_name == "web_search":
        query = body_input.get("query", "").strip()
        if not query:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch web_search rejected request_id=%s reason=%s elapsed_ms=%.2f", request_id, "empty_query", elapsed)
            raise HTTPException(status_code=422, detail="'query' is required for web_search.")

        limit = min(int(body_input.get("limit", 5)), 10)
        force_web = bool(body_input.get("force_web", False))
        corpus_only = bool(body_input.get("corpus_only", False))

        from dewie.search.providers import get_search_provider
        from dewie.search.web_lookup import persist_document, web_lookup

        provider = get_search_provider()
        log.info(
            "dispatch web_search dispatching request_id=%s provider=%s force_web=%s corpus_only=%s",
            request_id, provider.name if provider else None, force_web, corpus_only,
        )

        try:
            lookup, new_doc = await web_lookup(
                query,
                pg=pg,
                provider=provider,
                limit=limit,
                workspace_ids=workspace_ids,
                force_web=force_web,
                corpus_only=corpus_only,
            )
        except Exception as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.exception("dispatch web_search failed request_id=%s elapsed_ms=%.2f", request_id, elapsed)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "web_search_failed"})
            raise HTTPException(status_code=500, detail="Web search failed.") from exc

        if new_doc is not None:
            if user_id:
                new_doc.corpus_id = new_doc.corpus_id or f"user:{user_id}"
            # persist_document upserts as status='pending' — the enrichment
            # poller picks it up on its own throttled schedule, no eager
            # enqueue_background(_enrich_one(...)) needed here (see ingest_url).
            enqueue_background(persist_document(new_doc, pg))

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "dispatch web_search succeeded request_id=%s source=%s corpus_hits=%d web_hits=%d elapsed_ms=%.2f",
            request_id, lookup.source, len(lookup.corpus_hits), len(lookup.web_hits), elapsed,
        )
        _log_mcp_call(
            tool_name, body_input, user_id, model, elapsed,
            result_summary={
                "source": lookup.source,
                "corpus_hit_count": len(lookup.corpus_hits),
                "web_hit_count": len(lookup.web_hits),
                "ingested_doc_id": lookup.ingested_doc_id,
            },
        )

        return lookup.to_content()

    # ── add_catalog ──────────────────────────────────────────────────────────────
    elif tool_name == "add_catalog":
        if not is_admin:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.warning("dispatch add_catalog forbidden request_id=%s elapsed_ms=%.2f", request_id, elapsed)
            raise HTTPException(status_code=403, detail="add_catalog requires admin privileges.")

        catalog_name = str(body_input.get("name", "")).strip()
        catalog_type = str(body_input.get("type", "")).strip()
        enabled = bool(body_input.get("enabled", True))

        if not catalog_name:
            raise HTTPException(status_code=422, detail="'name' is required for add_catalog.")

        _VALID = frozenset({"mcp", "postgres", "sqlite"})
        if catalog_type not in _VALID:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid catalog type {catalog_type!r}. Valid: {sorted(_VALID)}",
            )

        # Build type-specific config from flat input fields
        config: dict[str, Any] = {}
        if catalog_type == "mcp":
            if body_input.get("endpoint"):
                config["endpoint"] = str(body_input["endpoint"]).strip()
            if body_input.get("api_key"):
                config["api_key"] = str(body_input["api_key"]).strip()
        elif catalog_type == "postgres":
            if body_input.get("dsn"):
                config["dsn"] = str(body_input["dsn"]).strip()
        elif catalog_type == "sqlite":
            if body_input.get("filepath"):
                config["filepath"] = str(body_input["filepath"]).strip()
        # Allow caller to pass arbitrary extra config keys
        for k, v in body_input.items():
            if k not in {"name", "type", "enabled", "endpoint", "api_key", "dsn", "filepath"}:
                config[k] = v

        try:
            import inspect as _inspect

            src_id = uuid.uuid4()
            if _inspect.iscoroutinefunction(getattr(pg, "create_source", None)):
                source = await pg.create_source(
                    source_id=src_id,
                    name=catalog_name,
                    source_type=catalog_type,
                    config=config,
                    enabled=enabled,
                )
                source_id_out = str(source["id"])
            else:
                import json as _json

                from sqlalchemy import text as _text

                async with pg._engine.connect() as conn:
                    try:
                        await conn.execute(
                            _text(
                                "INSERT INTO dewie_sources "
                                "(id, name, type, config_json, enabled, created_at, updated_at) "
                                "VALUES (:id, :name, :type, :config_json, :enabled, "
                                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                            ),
                            {
                                "id": str(src_id),
                                "name": catalog_name,
                                "type": catalog_type,
                                "config_json": _json.dumps(config),
                                "enabled": enabled,
                            },
                        )
                        await conn.commit()
                    except Exception as exc:
                        await conn.rollback()
                        raise exc
                source_id_out = str(src_id)
        except Exception as exc:
            msg = str(exc)
            if "unique" in msg.lower() or "duplicate" in msg.lower():
                raise HTTPException(status_code=409, detail=f"Catalog '{catalog_name}' already exists.") from exc
            raise HTTPException(status_code=500, detail=f"Database error: {msg}") from exc

        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info("dispatch add_catalog succeeded request_id=%s id=%s elapsed_ms=%.2f", request_id, source_id_out, elapsed)

        return {"ok": True, "id": source_id_out, "name": catalog_name, "type": catalog_type}

    # ── list_sources ─────────────────────────────────────────────────────────────
    elif tool_name == "list_sources":
        try:
            from sqlalchemy import text as _text

            ws_ids = [str(w) for w in (workspace_ids or [])]
            if ws_ids:
                placeholders = ", ".join(f":ws{i}" for i in range(len(ws_ids)))
                ws_params = {f"ws{i}": v for i, v in enumerate(ws_ids)}
                sql = _text(
                    f"SELECT DISTINCT d.source FROM documents d "
                    f"JOIN corpora c ON d.corpus_id = c.id "
                    f"WHERE d.source IS NOT NULL AND d.source != '' "
                    f"AND (d.sharing_tier = 'public' OR c.workspace_id IN ({placeholders})) "
                    f"ORDER BY d.source"
                )
            else:
                ws_params = {}
                sql = _text("SELECT DISTINCT source FROM documents WHERE source IS NOT NULL AND source != '' ORDER BY source")
            async with pg._engine.connect() as conn:
                rows = (await conn.execute(sql, ws_params)).fetchall()
            sources = [r[0] for r in rows]
        except Exception as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False})
            raise HTTPException(status_code=500, detail="Failed to list sources.") from exc
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"count": len(sources)})
        return {"sources": sources, "count": len(sources)}

    # ── list_catalogs ────────────────────────────────────────────────────────────
    elif tool_name == "list_catalogs":
        from dewie.config import settings as _cfg

        if _cfg.auth_enabled and not is_admin:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "forbidden"})
            raise HTTPException(status_code=403, detail="list_catalogs requires admin privileges.")
        try:
            catalogs = await pg.list_sources()
        except Exception as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False})
            raise HTTPException(status_code=500, detail="Failed to list catalogs.") from exc
        safe = [{"id": c.get("id"), "name": c.get("name"), "type": c.get("type"), "enabled": c.get("enabled")} for c in catalogs]
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"count": len(safe)})
        return {"catalogs": safe, "count": len(safe)}

    # ── dewie_fetch ──────────────────────────────────────────────────────────────
    elif tool_name == "dewie_fetch":
        url = body_input.get("url", "").strip()
        if not url:
            raise HTTPException(status_code=422, detail="'url' is required for dewie_fetch.")
        save = bool(body_input.get("save", True))
        log.info("dispatch dewie_fetch dispatching request_id=%s url=%s save=%s", request_id, _redact(url), save)

        from dewie.ingestion.web import WebIngester

        try:
            async with WebIngester() as ingester:
                docs = [doc async for doc in ingester.fetch(url)]
        except ValueError as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "blocked"})
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not docs or not docs[0].body:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "no_content"})
            raise HTTPException(status_code=422, detail="No content found at URL.")
        doc = docs[0]
        if save and user_id:
            doc.corpus_id = f"user:{user_id}"
            enqueue_background(_bg_ingest(doc, pg))
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "dispatch dewie_fetch succeeded request_id=%s url=%s saved=%s elapsed_ms=%.2f",
            request_id, _redact(url), save and bool(user_id), elapsed,
        )
        _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"url": url, "saved": save and bool(user_id)})
        return {
            "url": url,
            "title": doc.title or "",
            "content": doc.body[:8000],
        }

    else:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.warning("dispatch unknown tool request_id=%s tool=%s elapsed_ms=%.2f", request_id, tool_name, elapsed)
        _log_mcp_call(tool_name, body_input, user_id, model, elapsed, result_summary={"success": False, "error": "unknown_tool"})
        raise HTTPException(status_code=422, detail=f"Unknown tool: {tool_name!r}.")
