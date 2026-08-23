# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""Corpus endpoints — sources listing, quality metrics, and gap analysis."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import text

log = logging.getLogger("dewie.api")

# Fields that must be redacted from log output
_SENSITIVE_FIELDS = {"api_key", "password", "token", "secret", "authorization"}


def _redact(value: str | None) -> str:
    """Redact a potentially sensitive string value for logging."""
    if value is None:
        return None
    if len(value) > 1000:
        value = value[:1000] + "... [truncated]"
    for field in _SENSITIVE_FIELDS:
        if field in value.lower():
            return "***REDACTED***"
    return value


def _extract_request_id(request: Request) -> str:
    """Extract request_id from request state, falling back to 'unknown'."""
    return getattr(request.state, "request_id", "unknown")


router = APIRouter(prefix="/corpus", tags=["corpus"])


def _get_pg(request: Request):
    return request.app.state.postgres


@router.get("/doc-types")
async def list_doc_types(
    request: Request,
    catalog_id: str | None = Query(None, description="Filter by catalog ID")
) -> list[str]:
    """Return distinct document types in the corpus, optionally filtered by catalog."""
    pg = _get_pg(request)
    async with pg._engine.connect() as conn:
        conditions = []
        params = {}
        if catalog_id and catalog_id != "__local__":
            conditions.append("d.corpus_id = :catalog_id")
            params["catalog_id"] = catalog_id
        
        query_parts = ["SELECT DISTINCT d.document_type FROM documents d"]
        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))
        query_parts.append("AND d.document_type IS NOT NULL")
        query_parts.append("ORDER BY 1")
        
        sql = text(" ".join(query_parts))
        rows = (await conn.execute(sql, params)).mappings().fetchall()
    return [r["document_type"] for r in rows]


@router.get("/sources")
async def list_corpus_sources(request: Request) -> list[dict]:
    """Return distinct document sources with ready/pending counts."""
    pg = _get_pg(request)
    is_sqlite = getattr(pg, "_is_sqlite", False)
    async with pg._engine.connect() as conn:
        if is_sqlite:
            sql = text("""
                SELECT
                    COALESCE(NULLIF(source,''), 'unknown') AS source,
                    SUM(CASE WHEN status='ready'   THEN 1 ELSE 0 END) AS ready,
                    SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
                    SUM(CASE WHEN status='failed'  THEN 1 ELSE 0 END) AS failed,
                    COUNT(*) AS total
                FROM documents
                GROUP BY 1
                ORDER BY total DESC
                LIMIT 200
            """)
        else:
            sql = text("""
                SELECT
                    COALESCE(NULLIF(source,''), 'unknown') AS source,
                    COUNT(*) FILTER (WHERE status='ready')   AS ready,
                    COUNT(*) FILTER (WHERE status='pending') AS pending,
                    COUNT(*) FILTER (WHERE status='failed')  AS failed,
                    COUNT(*) AS total
                FROM documents
                GROUP BY 1
                ORDER BY total DESC
                LIMIT 200
            """)
        rows = (await conn.execute(sql)).mappings().fetchall()
    return [dict(r) for r in rows]


@router.get("/quality")
async def corpus_quality(
    request: Request,
    source: str | None = Query(None, description="Filter by document source domain"),
    doc_type: str | None = Query(None, description="Filter by document type"),
    catalog_id: str | None = Query(None, description="Filter by catalog ID"),
) -> dict:
    """Return quality metrics for the corpus, optionally filtered by source, type, or catalog."""
    pg = _get_pg(request)
    is_sqlite = getattr(pg, "_is_sqlite", False)

    conditions = []
    params: dict = {}
    if doc_type:
        conditions.append("COALESCE(d.document_type, '') = :doc_type")
        params["doc_type"] = doc_type
    if source:
        conditions.append("COALESCE(NULLIF(d.source,''), 'unknown') = :source")
        params["source"] = source
    if catalog_id and catalog_id != "__local__":
        conditions.append("d.corpus_id = :catalog_id")
        params["catalog_id"] = catalog_id

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)


    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    if is_sqlite:
        agg_filter = lambda col, cond: f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END)"  # noqa: E731  # sqlite has no FILTER
        count_filter = lambda cond: agg_filter("1", cond)  # noqa: E731
        summary_sql = text(f"""
            SELECT
                COUNT(*) AS total,
                {count_filter("d.status='ready'")} AS ready,
                {count_filter("d.status='pending'")} AS pending,
                {count_filter("d.status='failed'")} AS failed,
                {count_filter("d.embedding IS NOT NULL")} AS with_embedding,
                {count_filter("length(COALESCE(d.embed_summary,''))>100")} AS embed_summary_good,
                CAST(AVG(length(COALESCE(d.embed_summary,''))) AS INTEGER) AS avg_embed_summary_len,
                CAST(AVG(length(COALESCE(d.body_text,''))) AS INTEGER) AS avg_body_len,
                CAST(AVG(CASE WHEN d.answers_questions IS NOT NULL AND d.answers_questions!='[]'
                    THEN json_array_length(d.answers_questions) ELSE 0 END) AS REAL) AS avg_aqs,
                {count_filter("COALESCE(d.enrichment_quality_score,0)>=80")} AS quality_high,
                {count_filter("COALESCE(d.enrichment_quality_score,0) BETWEEN 50 AND 79")} AS quality_medium,
                {count_filter("COALESCE(d.enrichment_quality_score,0) BETWEEN 20 AND 49")} AS quality_low,
                {count_filter("COALESCE(d.enrichment_quality_score,0)<20")} AS quality_stub
            FROM documents d {where}
        """)
        by_source_sql = text(f"""
            SELECT
                COALESCE(NULLIF(d.source,''), 'unknown') AS source,
                {count_filter("d.status='ready'")} AS ready,
                {count_filter("d.status='pending'")} AS pending,
                {count_filter("d.status='failed'")} AS failed,
                COUNT(*) AS total
            FROM documents d {where}
            GROUP BY 1 ORDER BY ready DESC LIMIT 50
        """)
    else:
        summary_sql = text(f"""
            SELECT
                COUNT(*)                                                        AS total,
                COUNT(*) FILTER (WHERE d.status='ready')                        AS ready,
                COUNT(*) FILTER (WHERE d.status='pending')                      AS pending,
                COUNT(*) FILTER (WHERE d.status='failed')                       AS failed,
                COUNT(*) FILTER (WHERE d.embedding IS NOT NULL)                 AS with_embedding,
                COUNT(*) FILTER (WHERE length(COALESCE(d.embed_summary,''))>100) AS embed_summary_good,
                ROUND(AVG(length(COALESCE(d.embed_summary,''))))::int           AS avg_embed_summary_len,
                ROUND(AVG(length(COALESCE(d.body_text,''))))::int               AS avg_body_len,
                ROUND(AVG(
                    CASE WHEN d.answers_questions IS NOT NULL
                              AND jsonb_array_length(d.answers_questions)>0
                         THEN jsonb_array_length(d.answers_questions) ELSE 0 END
                ), 1)                                                           AS avg_aqs,
                COUNT(*) FILTER (WHERE COALESCE(d.enrichment_quality_score,0)>=80)            AS quality_high,
                COUNT(*) FILTER (WHERE COALESCE(d.enrichment_quality_score,0) BETWEEN 50 AND 79) AS quality_medium,
                COUNT(*) FILTER (WHERE COALESCE(d.enrichment_quality_score,0) BETWEEN 20 AND 49) AS quality_low,
                COUNT(*) FILTER (WHERE COALESCE(d.enrichment_quality_score,0)<20)             AS quality_stub
            FROM documents d {where}
        """)
        by_source_sql = text(f"""
            SELECT
                COALESCE(NULLIF(d.source,''), 'unknown') AS source,
                COUNT(*) FILTER (WHERE d.status='ready')   AS ready,
                COUNT(*) FILTER (WHERE d.status='pending') AS pending,
                COUNT(*) FILTER (WHERE d.status='failed')  AS failed,
                COUNT(*) AS total,
                ROUND(AVG(length(COALESCE(d.embed_summary,''))))::int AS avg_embed_summary_len,
                ROUND(AVG(
                    CASE WHEN d.answers_questions IS NOT NULL
                              AND jsonb_array_length(d.answers_questions)>0
                         THEN jsonb_array_length(d.answers_questions) ELSE 0 END
                ), 1) AS avg_aqs,
                ROUND(AVG(COALESCE(d.enrichment_quality_score, 0)))::int AS quality_score
            FROM documents d {where}
            GROUP BY 1 ORDER BY ready DESC LIMIT 50
        """)

    async with pg._engine.connect() as conn:
        s = dict((await conn.execute(summary_sql, params)).mappings().one())
        source_rows = (await conn.execute(by_source_sql, params)).mappings().fetchall()

    return {
        "summary": {
            "total": s["total"],
            "ready": s["ready"],
            "pending": s["pending"],
            "failed": s["failed"],
            "with_embedding": s.get("with_embedding", 0),
            "embed_summary_good": s["embed_summary_good"],
            "avg_embed_summary_len": s["avg_embed_summary_len"],
            "avg_body_len": s.get("avg_body_len", 0),
            "avg_aqs": s.get("avg_aqs"),
        },
        "quality_distribution": {
            "high": s["quality_high"],
            "medium": s["quality_medium"],
            "low": s["quality_low"],
            "stub": s["quality_stub"],
        },
        "by_source": [dict(r) for r in source_rows],
        "source_filter": source,
    }


def _json_list(val) -> list:
    """Normalise a JSON column value to a list."""
    if isinstance(val, list):
        return val
    if val is None:
        return []
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


async def _generate_jsonl_export(
    pg, corpus_id: str, status: str, is_sqlite: bool
):
    """Generator that yields JSONL-encoded documents in batches."""
    batch_size = 1000
    offset = 0
    id_filter = "WHERE corpus_id = :corpus_id AND status = :status"
    id_cast = "" if is_sqlite else "cast(:offset as int)"

    while True:
        sql = text(f"""
            SELECT id, url, title, summary, topics, keywords, entities,
                   published_at, enriched_at, corpus_id
            FROM documents
            {id_filter}
            LIMIT :batch_size OFFSET {id_cast}
        """)
        async with pg._session_factory() as session:
            rows = (
                await session.execute(
                    sql,
                    {
                        "corpus_id": corpus_id,
                        "status": status,
                        "batch_size": batch_size,
                        "offset": offset,
                    },
                )
            ).mappings().all()

        if not rows:
            break

        for row in rows:
            doc = {
                "id": str(row["id"]),
                "url": row["url"],
                "title": row["title"],
                "summary": row["summary"] or "",
                "topics": _json_list(row["topics"]),
                "keywords": _json_list(row["keywords"]),
                "entities": _json_list(row["entities"]),
                "published_at": row["published_at"].isoformat() if row.get("published_at") else None,
                "enriched_at": row["enriched_at"].isoformat() if row.get("enriched_at") else None,
                "corpus_id": row["corpus_id"],
                "tags": [],
            }
            yield json.dumps(doc) + "\n"

        offset += batch_size


@router.get("/gap-report")
async def get_gap_report(
    request: Request,
    topic_filter: str | None = Query(None, description="Comma-separated topic filter"),
    min_gap_severity: str = Query("minor", description="Minimum severity: major|moderate|minor"),
    as_of: str | None = Query(None, description="ISO date filter (YYYY-MM-DD)"),
) -> dict:
    """Return proactive gap analysis of the entire corpus."""
    pg = _get_pg(request)
    topics = [t.strip() for t in topic_filter.split(",")] if topic_filter else None
    return await pg.get_gap_report(
        topic_filter=topics,
        min_gap_severity=min_gap_severity,
        as_of=as_of,
    )


@router.get("/export")
async def export_corpus(
    request: Request,
    corpus_id: str = Query(..., description="ID of the corpus to export"),
    format: str = Query("jsonl", pattern="^(jsonl|json)$", description="Export format"),
    status: str = Query("ready", description="Filter by document status"),
) -> StreamingResponse:
    """Export corpus documents as JSONL or JSON.

    Uses streaming to handle large datasets without OOM errors.
    """
    import time as _time

    request_id = _extract_request_id(request)
    redacted_corpus_id = _redact(corpus_id)
    log.info(
        "export_corpus started",
        request_id=request_id,
        corpus_id=redacted_corpus_id,
        format=format,
        status=status,
    )
    _start = _time.time()
    try:
        pg = _get_pg(request)
        is_sqlite = getattr(pg, "_is_sqlite", False)

        if format == "jsonl":
            return StreamingResponse(
                _generate_jsonl_export(pg, corpus_id, status, is_sqlite),
                media_type="application/jsonl",
                headers={"Content-Disposition": f"attachment; filename=corpus_{corpus_id}.jsonl"},
            )
        else:
            async def _generate_json_array():
                yield "["
                first = True
                async for chunk in _generate_jsonl_export(pg, corpus_id, status, is_sqlite):
                    if not first:
                        yield ","
                    else:
                        first = False
                    yield chunk.rstrip("\n")
                yield "]"

            return StreamingResponse(
                _generate_json_array(),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=corpus_{corpus_id}.json"},
            )
    except Exception:
        elapsed = _time.time() - _start
        log.exception(
            "export_corpus failed",
            request_id=request_id,
            corpus_id=redacted_corpus_id,
            elapsed_seconds=elapsed,
        )
        raise
