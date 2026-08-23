# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
/documents endpoints for accessing document content.

The /documents/{id}/content endpoint is the sanctioned way for agents to
access raw document text.  Postgres body_text is the canonical source;
flat-file store is the fallback; URL re-fetch is the last resort.
Redis is NOT used for body reads (issue #119).
"""

from __future__ import annotations

import logging
import uuid as _uuid
from urllib.parse import urlparse as _urlparse
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from dewie.api.middleware import limiter, rate_limit
from dewie.crawler.fetcher import StaticFetcher
from dewie.storage.postgres import PostgresClient


class _StructLogger:
    """Adapt structured-logging calls onto the stdlib ``dewie.api`` logger.

    Call sites in this module use ``log.info("msg", request_id=..., doc_id=...)``.
    The stdlib ``Logger`` rejects unknown keyword arguments, which would raise
    ``TypeError`` mid-request and surface as a 500. This wrapper folds any
    non-standard keywords into ``extra={}`` so the ``_RequestIDFilter`` and log
    formatter pick them up, mirroring the convention used in ``ingest.py``.
    """

    _STD = {"exc_info", "stack_info", "stacklevel", "extra"}

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _emit(self, level: int, msg: str, *args, **kwargs) -> None:
        std = {k: kwargs.pop(k) for k in list(kwargs) if k in self._STD}
        extra = dict(std.pop("extra", None) or {})
        extra.update(kwargs)
        self._logger.log(level, msg, *args, extra=extra or None, **std)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._emit(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._emit(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._emit(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._emit(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("exc_info", True)
        self._emit(logging.ERROR, msg, *args, **kwargs)


log = _StructLogger("dewie.api")

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


router = APIRouter(prefix="/documents", tags=["documents"])


def _get_pg(request: Request) -> PostgresClient:
    return request.app.state.postgres


async def _audit_doc(request: Request, action: str, resource_type: str, resource_id: str,
                     metadata: dict | None = None) -> None:
    """Record an audit event if audit logging is enabled."""
    actor_id = getattr(request.state, "actor_id", None) or "unknown"
    tenant_id_str = getattr(request.state, "tenant_id", None)
    if tenant_id_str:
        import uuid as _uuid
        tenant_id = _uuid.UUID(str(tenant_id_str))
    else:
        import uuid as _uuid
        tenant_id = _uuid.UUID("00000000-0000-0000-0000-000000000001")

    from dewie.compliance import audit_log
    from dewie.config import settings

    await audit_log(
        _get_pg(request),
        settings=settings,
        tenant_id=tenant_id,
        actor_id=actor_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        metadata=metadata,
    )


# ── Customer bring-your-own-ingest ────────────────────────────────────────────


class DocumentIngestRequest(BaseModel):
    """Request body for the customer-facing /documents/ingest endpoint.

    Supply either ``url`` (article or RSS URL to fetch) **or** ``text`` (raw
    body text) — at least one is required.  When both are given, ``text`` is
    used as the document body and ``url`` is stored as the canonical source URL.
    """

    url: str | None = Field(
        default=None,
        description="URL to ingest (article, RSS/Atom feed, or any crawlable page).",
    )
    text: str | None = Field(
        default=None,
        description="Raw document body text. When supplied, URL fetching is skipped.",
    )
    title: str | None = Field(
        default=None,
        description="Optional document title.",
    )
    corpus_id: str | None = Field(
        default=None,
        description="Opaque corpus identifier. Convention: 'customer:NAME'.",
    )
    visibility: str = Field(
        default="private",
        description="public | private. Defaults to private for customer-supplied docs.",
    )


class DocumentIngestResponse(BaseModel):
    """Returned immediately after the document is accepted for processing."""

    doc_id: str = Field(description="UUID of the ingested document.")
    status: str = Field(description="Always 'pending' — enrichment runs asynchronously.")
    message: str = Field(description="Human-readable confirmation.")


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Customer ingest — URL or raw text",
    description=(
        "Submit a document for ingestion into your corpus. "
        "Supply a URL to fetch, raw text, or both. "
        "Enrichment (topics, keywords, embeddings) happens asynchronously. "
        "Time-to-first-result is typically minutes to a few hours depending on queue depth."
    ),
)
@limiter.limit(rate_limit())
async def ingest_document(
    request: Request,
    body: DocumentIngestRequest,
    background_tasks: BackgroundTasks,
) -> DocumentIngestResponse:
    """Customer-facing ingest endpoint — URL or raw text body.

    Requires an API key with ``ingest`` scope (X-API-Key header).
    Enrichment runs in the background; returns immediately with a document ID.

    Raises:
        400: Neither url nor text supplied.
        422: URL supplied but no content could be retrieved.
    """
    import time as _time

    request_id = _extract_request_id(request)
    redacted_url = _redact(body.url)
    log.info(
        "ingest_document started",
        request_id=request_id,
        url=redacted_url,
        has_text=body.text is not None,
        corpus_id=_redact(body.corpus_id),
    )
    _start = _time.time()
    try:
        from dewie.models.content import ContentDocument, ContentStatus

        if not body.url and not body.text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide at least one of: url, text.",
            )

        pg: PostgresClient = _get_pg(request)
        processor = getattr(request.app.state, "processor", None)

        docs: list[ContentDocument] = []

        if body.text:
            # Raw text path — build a synthetic document without fetching anything
            canonical_url = body.url or f"text://{_uuid.uuid4()}"
            source = _urlparse(canonical_url).netloc or "customer"
            doc = ContentDocument(
                id=_uuid.uuid4(),
                url=canonical_url,
                title=body.title or canonical_url,
                body=body.text,
                source=source,
                status=ContentStatus.PENDING,
                corpus_id=body.corpus_id,
                visibility=body.visibility,
            )
            docs = [doc]
        else:
            # URL fetch path — delegate to existing WebIngester / RSSIngester
            url = body.url  # type: ignore[assignment]
            is_feed = any(url.endswith(ext) for ext in (".xml", ".rss", ".atom")) or "feed" in url

            if is_feed:
                from dewie.ingestion.rss import RSSIngester

                ingester = RSSIngester()
                async for fetched_doc in await ingester.fetch(url):
                    docs.append(fetched_doc)
            else:
                from dewie.ingestion.web import WebIngester

                async with WebIngester() as ingester:
                    docs = [d async for d in ingester.fetch(url)]

            if not docs:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No content found at URL (may be paywalled, rate-limited, or empty).",
                )

            # Apply caller metadata to fetched docs
            for doc in docs:
                if body.corpus_id and doc.corpus_id is None:
                    doc.corpus_id = body.corpus_id
                if body.title and not doc.title:
                    doc.title = body.title
                doc.visibility = body.visibility

        # Persist stubs so IDs are immediately available
        for doc in docs:
            await pg.upsert(doc)

        # Persist body text for worker pick-up
        from dewie.storage.body_store import save_body

        for doc in docs:
            if getattr(doc, "body", None):
                save_body(doc.id, doc.body)
                try:
                    await pg.write_body_text(doc.id, doc.body)
                except Exception as exc:
                    log.warning("Failed to write body_text for doc %s: %s", doc.id, exc)

        # Fire-and-forget enrichment
        if processor is not None and docs:
            background_tasks.add_task(_enrich_batch, docs, pg, processor)
        else:
            log.info("ingest_document: processor not available, skipping background enrichment")

        first_doc = docs[0]
        elapsed = _time.time() - _start
        log.info(
            "ingest_document succeeded",
            request_id=request_id,
            doc_id=str(first_doc.id),
            doc_count=len(docs),
            elapsed_seconds=elapsed,
        )
        await _audit_doc(request, "doc.ingest", "document", str(first_doc.id), {"doc_count": len(docs)})
        return DocumentIngestResponse(
            doc_id=str(first_doc.id),
            status="pending",
            message=f"Accepted {len(docs)} document(s) for enrichment.",
        )
    except HTTPException:
        raise
    except Exception:
        elapsed = _time.time() - _start
        log.exception(
            "ingest_document failed",
            request_id=request_id,
            url=redacted_url,
            elapsed_seconds=elapsed,
        )
        raise


async def _enrich_batch(docs: list, pg: PostgresClient, processor) -> None:  # type: ignore[type-arg]
    """Background enrichment task."""
    for doc in docs:
        try:
            await processor.enrich_and_persist(doc, pg)
        except Exception as exc:
            log.warning("Enrichment failed for doc %s: %s", doc.id, exc)



@router.get("/list", response_model=dict)  # type: ignore[misc]
@limiter.limit(rate_limit())
async def list_documents(
    request: Request,
    limit: int = 200,
    offset: int = 0,
) -> dict:  # type: ignore[type-arg]
    """List all documents with metadata."""
    import time as _time

    request_id = _extract_request_id(request)
    log.info(
        "list_documents started",
        request_id=request_id,
        limit=limit,
        offset=offset,
    )
    _start = _time.time()
    try:
        pg: PostgresClient = _get_pg(request)
        docs = await pg.list_recent(limit=limit, offset=offset)
        elapsed = _time.time() - _start
        log.info(
            "list_documents succeeded",
            request_id=request_id,
            result_count=len(docs),
            elapsed_seconds=elapsed,
        )
        return {
            "documents": [
                {
                    "id": str(doc.id),
                    "url": doc.url,
                    "title": doc.title,
                    "summary": doc.summary,
                    "source": doc.source,
                    "status": doc.status.value,
                    "topics": doc.topics,
                    "keywords": doc.keywords,
                    "entities": doc.entities,
                    "sentiment": doc.sentiment,
                }
                for doc in docs
            ]
        }
    except Exception:
        elapsed = _time.time() - _start
        log.exception(
            "list_documents failed",
            request_id=request_id,
            elapsed_seconds=elapsed,
        )
        raise


def _doc_type(doc):
    """Classify a document into a type bucket.

    Matches the client-side docType() in documents.html for consistent
    filtering between backend and frontend.
    """
    import re

    url = (doc.url or "").lower()
    source = (doc.source or "").lower()

    # YouTube
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"

    # Extract path portion only for extension checks (avoid false matches
    # from query params like "?file=mp3")
    parsed = _urlparse(url)
    path = (parsed.path or "").lower()

    # Podcast — detected from source or audio file extensions in path
    if source in ("podcast", "podcasts"):
        return "podcast"
    if re.search(r"\.(mp3|mp4|wav|m4a|ogg|podcast)$", path):
        return "podcast"

    # PDF
    if re.search(r"\.pdf$", path):
        return "pdf"

    # Documents (Word, Excel, PowerPoint, etc.) — uploaded files
    if source == "upload":
        return "document"
    if re.search(r"\.(docx?|xlsx?|pptx?|odt|ods|odp|pages|numbers|key)$", path):
        return "document"

    return "website"


@router.get("/my", response_model=dict)  # type: ignore[misc]
@limiter.limit(rate_limit())
async def list_my_documents(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
    status: str | None = None,
    type_filter: str | None = None,
) -> dict:  # type: ignore[type-arg]
    """UI compatibility endpoint for /ui/documents.html.

    In OSS/local mode this returns global recent docs (no per-user partition).
    Supports optional filtering by search, status, and document type.
    """
    import time as _time

    request_id = _extract_request_id(request)
    log.info(
        "list_my_documents started",
        request_id=request_id,
        limit=limit,
        offset=offset,
        search=_truncate(search) if search else None,
        status=status,
        type_filter=type_filter,
    )
    _start = _time.time()
    try:
        pg: PostgresClient = _get_pg(request)
        docs = await pg.list_recent(limit=max(limit * 3, 100), offset=0)

        q = (search or "").strip().lower()
        status_filter = (status or "").strip().lower()
        doc_type_filter = (type_filter or "").strip().lower()

        filtered = []
        for doc in docs:
            # Status filter
            doc_status = doc.status.value
            if status_filter and doc_status != status_filter:
                continue
            # Document type filter
            if doc_type_filter and _doc_type(doc) != doc_type_filter:
                continue
            # Search filter
            if q:
                hay = " ".join(
                    [
                        doc.title or "",
                        doc.url or "",
                        doc.summary or "",
                        " ".join(doc.topics or []),
                        " ".join(doc.keywords or []),
                        " ".join(doc.entities or []),
                    ]
                ).lower()
                if q not in hay:
                    continue
            filtered.append(doc)

        total = len(filtered)
        page = filtered[offset : offset + limit]
        elapsed = _time.time() - _start
        log.info(
            "list_my_documents succeeded",
            request_id=request_id,
            total=total,
            page_count=len(page),
            elapsed_seconds=elapsed,
        )
        return {
            "total": total,
            "documents": [
                {
                    "id": str(doc.id),
                    "url": doc.url,
                    "title": doc.title,
                    "summary": doc.summary,
                    "source": doc.source,
                    "status": doc.status.value,
                    "topics": doc.topics,
                    "keywords": doc.keywords,
                    "entities": doc.entities,
                    "visibility": "private",
                    "user_tags": {},
                    "ingested_at": doc.ingested_at.isoformat() if doc.ingested_at else None,
                }
                for doc in page
            ],
        }
    except Exception:
        elapsed = _time.time() - _start
        log.exception(
            "list_my_documents failed",
            request_id=request_id,
            elapsed_seconds=elapsed,
        )
        raise


def _truncate(value: str | None, maxlen: int = 1000) -> str | None:
    """Truncate a string for safe logging."""
    if value is None:
        return None
    if len(value) <= maxlen:
        return value
    return value[:maxlen] + f"... ({len(value)} chars total)"


@router.get("/{doc_id}/content", response_class=PlainTextResponse)
@limiter.limit(rate_limit())
async def get_content(doc_id: UUID, request: Request) -> PlainTextResponse:
    """
    Return the full plain-text body of a document.

    Read priority (issue #119 — Redis removed from body read path):
      1. Postgres body_text — canonical, written at ingest time
      2. Flat-file body store — legacy fallback for pre-#51 docs
      3. URL re-fetch via StaticFetcher — last resort for docs with no stored body

    This endpoint is the future billing/gating hook:
      - API key auth middleware slots in here via a Depends() injection.
      - slowapi rate limiting is applied globally.
      - Usage can be metered per doc_id or per caller.
    """
    import time as _time

    request_id = _extract_request_id(request)
    log.info("get_content started", request_id=request_id, doc_id=str(doc_id))
    _start = _time.time()
    try:
        from sqlalchemy import text as sa_text

        from dewie.storage.body_store import load_body

        pg: PostgresClient = _get_pg(request)

        doc = await pg.get_by_id(doc_id)
        if doc is None:
            elapsed = _time.time() - _start
            log.info(
                "get_content succeeded",
                request_id=request_id,
                doc_id=str(doc_id),
                status=404,
                elapsed_seconds=elapsed,
            )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

        # ── 1. Postgres body_text (canonical) ─────────────────────────────────────
        body: str | None = None
        try:
            async with pg._session_factory() as session:
                row = (
                    (
                        await session.execute(
                            sa_text(
                                "SELECT body_text FROM documents WHERE id = CAST(:id AS UUID)"
                            ),
                            {"id": str(doc_id)},
                        )
                    )
                    .mappings()
                    .first()
                )
            if row and row["body_text"]:
                body = row["body_text"]
        except Exception:
            pass  # fall through to file store

        # ── 2. Flat-file body store (pre-#51 fallback) ────────────────────────────
        if not body:
            body = load_body(str(doc_id))

        if body:
            elapsed = _time.time() - _start
            log.info(
                "get_content succeeded",
                request_id=request_id,
                doc_id=str(doc_id),
                body_chars=len(body),
                elapsed_seconds=elapsed,
            )
            return PlainTextResponse(content=body)

        # ── 3. URL re-fetch (last resort) ─────────────────────────────────────────
        async with StaticFetcher() as fetcher:
            try:
                fetched_doc, _ = await fetcher.fetch(doc.url)
                body = fetched_doc.body
            except Exception as exc:
                import httpx

                if isinstance(exc, httpx.HTTPStatusError):
                    status_code = exc.response.status_code
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Source URL returned {status_code}: {doc.url}",
                    ) from exc
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to fetch source document: {exc}",
                ) from exc

        elapsed = _time.time() - _start
        log.info(
            "get_content succeeded",
            request_id=request_id,
            doc_id=str(doc_id),
            body_chars=len(body) if body else 0,
            source="url_refetch",
            elapsed_seconds=elapsed,
        )
        return PlainTextResponse(content=body)
    except HTTPException:
        raise
    except Exception:
        elapsed = _time.time() - _start
        log.exception(
            "get_content failed",
            request_id=request_id,
            doc_id=str(doc_id),
            elapsed_seconds=elapsed,
        )
        raise


@router.get("/{doc_id}/inspect")
async def inspect_document(doc_id: UUID, request: Request):
    """
    Full document inspection — every field from ingest to embedding.
    Used by /ui/inspect.html for end-to-end data audit.
    """
    import asyncio
    import time as _time

    from sqlalchemy import text

    from dewie.storage.body_store import load_body

    request_id = _extract_request_id(request)
    log.info(
        "inspect_document started",
        request_id=request_id,
        doc_id=str(doc_id),
    )
    _start = _time.time()
    try:
        pg: PostgresClient = _get_pg(request)
        is_sqlite = getattr(pg, "_is_sqlite", False)
        id_filter = "WHERE id = :id" if is_sqlite else "WHERE id = cast(:id as uuid)"
        # Real embedding dimensions, computed per dialect (SQLite stores the
        # vector as a JSON array; Postgres uses pgvector). Avoids hardcoding a
        # provider-specific size that is wrong for non-OpenAI embedders.
        dims_expr = (
            "json_array_length(embedding)" if is_sqlite else "vector_dims(embedding)"
        )

        # Fetch all columns via raw SQL (embedding as dimension count, not raw vector).
        # Wrapped in try/except so that missing optional columns on older schema versions
        # (e.g. enrichment_quality_score, alternate_terms, embed_summary) do not cause a
        # 500 — instead we fall back to a minimal safe query (issue #242).
        row = None
        try:
            async with pg._session_factory() as session:
                row = (
                    (
                        await session.execute(
                            text(f"""
                SELECT
                    id, url, title, source, status,
                    ingested_at, enriched_at,
                    summary, embed_summary,
                    topics, keywords, entities,
                    answers_questions, alternate_terms,
                    sentiment, tone, document_type, reading_level, author,
                    enrichment_quality_score,
                    enrichment_version, embedding_model,
                    (embedding IS NOT NULL) as has_embedding,
                    CASE WHEN embedding IS NOT NULL
                         THEN {dims_expr}
                         ELSE NULL END as embedding_dims
                FROM documents
                {id_filter}
            """),
                            {"id": str(doc_id)},
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
        except Exception:
            # Schema may be missing optional columns — fall back to a minimal query
            # that only uses columns present since the initial schema.
            try:
                async with pg._session_factory() as session:
                    row = (
                        (
                            await session.execute(
                                text(f"""
                SELECT
                    id, url, title, source, status,
                    ingested_at, enriched_at,
                    summary,
                    topics, keywords, entities,
                    answers_questions,
                    sentiment, embedding_model,
                    (embedding IS NOT NULL) as has_embedding,
                    CASE WHEN embedding IS NOT NULL
                         THEN {dims_expr}
                         ELSE NULL END as embedding_dims
                FROM documents
                {id_filter}
            """),
                                {"id": str(doc_id)},
                            )
                        )
                        .mappings()
                        .one_or_none()
                    )
            except Exception:
                pass  # If even the minimal query fails, fall through to 404

        if row is None:
            raise HTTPException(status_code=404, detail="Document not found.")

        # Load body from flat-file store
        body = await asyncio.to_thread(load_body, str(doc_id))

        # Load LLM cache entries for this doc
        llm_id_filter = "WHERE doc_id = :id" if is_sqlite else "WHERE doc_id = cast(:id as uuid)"
        cache_rows: list = []
        try:
            async with pg._session_factory() as session:
                cache_rows = (
                    (
                        await session.execute(
                            text(f"""
                SELECT step, model, length(raw_response) as response_chars,
                       created_at, prompt_hash
                FROM llm_cache
                {llm_id_filter}
                ORDER BY created_at DESC
            """),
                            {"id": str(doc_id)},
                        )
                    )
                    .mappings()
                    .all()
                )
        except Exception:
            pass  # llm_cache table may not exist on older schema versions

        elapsed = _time.time() - _start
        log.info(
            "inspect_document succeeded",
            request_id=request_id,
            doc_id=str(doc_id),
            has_embedding=row.get("has_embedding", False),
            body_chars=len(body) if body else 0,
            llm_cache_count=len(cache_rows),
            elapsed_seconds=elapsed,
        )
        import json as _json

        def _parse_json_list(val) -> list:
            """SQLite returns JSON columns as strings; Postgres as parsed lists."""
            if val is None:
                return []
            if isinstance(val, str):
                try:
                    return _json.loads(val) or []
                except Exception:
                    return []
            return val or []

        _aq_list = _parse_json_list(row.get("answers_questions"))
        _has_emb = row.get("has_embedding")
        # SQLite returns 0/1 integers; coerce to bool
        _has_emb = bool(_has_emb) if _has_emb is not None else False

        return {
            "id": str(row["id"]),
            "url": row["url"],
            "title": row["title"],
            "source": row["source"],
            "status": row["status"],
            "ingested_at": row["ingested_at"] if isinstance(row["ingested_at"], str) else (row["ingested_at"].isoformat() if row["ingested_at"] else None),
            "enriched_at": row["enriched_at"] if isinstance(row["enriched_at"], str) else (row["enriched_at"].isoformat() if row["enriched_at"] else None),
            "body": {
                "chars": len(body) if body else 0,
                "preview": (body or "")[:500],
                "full": body or "",
                "available": bool(body),
            },
            "summary": row.get("summary", ""),
            "embed_summary": row.get("embed_summary", ""),
            "topics": _parse_json_list(row.get("topics")),
            "keywords": _parse_json_list(row.get("keywords")),
            "entities": _parse_json_list(row.get("entities")),
            "aq_count": len(_aq_list),
            "alternate_terms": _parse_json_list(row.get("alternate_terms")),
            "missing_coverage": [],
            "sentiment": row.get("sentiment"),
            "tone": row.get("tone"),
            "document_type": row.get("document_type"),
            "reading_level": row.get("reading_level"),
            "author": row.get("author"),
            "enrichment_quality_score": row.get("enrichment_quality_score"),
            "language": None,
            "enrichment_version": row.get("enrichment_version", 0),
            "embedding_model": row.get("embedding_model"),
            "retry_count": row.get("retry_count", 0),
            "has_embedding": _has_emb,
            "embedding_dims": row.get("embedding_dims"),
            "llm_cache": [
                {
                    "step": r["step"],
                    "model": r["model"],
                    "response_chars": r["response_chars"],
                    "prompt_hash": r["prompt_hash"],
                    "created_at": r["created_at"] if isinstance(r["created_at"], str) else (r["created_at"].isoformat() if r["created_at"] else None),
                }
                for r in cache_rows
            ],
        }
    except HTTPException:
        raise
    except Exception:
        elapsed = _time.time() - _start
        log.exception(
            "inspect_document failed",
            request_id=request_id,
            doc_id=str(doc_id),
            elapsed_seconds=elapsed,
        )
        raise


@router.get("/{doc_id}/chunks")
@limiter.limit(rate_limit())
async def get_chunks(doc_id: UUID, request: Request) -> dict:  # type: ignore[misc]
    """
    Return all text chunks for a document (chunk_index + text; embeddings excluded).

    Only documents with chunk_status='chunked' will have rows here.
    Short documents (< 3,000 words) are never chunked; their chunk list will be empty.
    """
    import time as _time

    request_id = _extract_request_id(request)
    log.info(
        "get_chunks started",
        request_id=request_id,
        doc_id=str(doc_id),
    )
    _start = _time.time()
    try:
        pg: PostgresClient = _get_pg(request)
        doc = await pg.get_by_id(doc_id)
        if doc is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
        chunks = await pg.get_chunks(doc_id)
        elapsed = _time.time() - _start
        log.info(
            "get_chunks succeeded",
            request_id=request_id,
            doc_id=str(doc_id),
            chunk_count=len(chunks),
            elapsed_seconds=elapsed,
        )
        return {
            "doc_id": str(doc_id),
            "title": doc.title,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
    except HTTPException:
        raise
    except Exception:
        elapsed = _time.time() - _start
        log.exception(
            "get_chunks failed",
            request_id=request_id,
            doc_id=str(doc_id),
            elapsed_seconds=elapsed,
        )
        raise


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: UUID, request: Request) -> None:
    """
    Permanently delete a document and all associated data (edges, chunks, AQ embeddings).

    Deletes:
      - Postgres row (cascades to graph_edges, document_chunks, aq_embeddings)
      - Flat-file body store entry
      - Redis cache keys for this document (search result cache)

    Requires admin session (``is_admin=True`` on request state).
    """
    import asyncio
    import time as _time

    from sqlalchemy import text

    from dewie.storage.body_store import delete_body

    request_id = _extract_request_id(request)
    log.info(
        "delete_document started",
        request_id=request_id,
        doc_id=str(doc_id),
    )
    _start = _time.time()
    try:
        from dewie.config import settings

        # Admin-gated. When AUTH_ENABLED=false (local dev) all callers are
        # treated as admin, matching _require_admin() used by the admin routes.
        if settings.auth_enabled and not getattr(request.state, "is_admin", False):
            raise HTTPException(status_code=403, detail="Admin session required")

        pg: PostgresClient = _get_pg(request)
        doc = await pg.get_by_id(doc_id)
        if doc is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

        _del_filter = "WHERE id = :id" if getattr(pg, "_is_sqlite", False) else "WHERE id = cast(:id as uuid)"
        async with pg._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM documents {_del_filter}"), {"id": str(doc_id)}
            )

        # Evict from body store (Redis body cache was removed in issue #119)
        await asyncio.to_thread(delete_body, str(doc_id))

        # Evict Redis cache keys for this document so stale results are not served
        cache = getattr(request.app.state, "cache", None)
        if cache is not None:
            try:
                redis = getattr(cache, "_redis", None)
                if redis is not None:
                    await redis.delete(f"doc:{doc_id}")
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to evict Redis cache for doc %s: %s", doc_id, exc)

        elapsed = _time.time() - _start
        log.info(
            "delete_document succeeded",
            request_id=request_id,
            doc_id=str(doc_id),
            elapsed_seconds=elapsed,
        )
        await _audit_doc(request, "doc.delete", "document", str(doc_id))
    except HTTPException:
        raise
    except Exception:
        elapsed = _time.time() - _start
        log.exception(
            "delete_document failed",
            request_id=request_id,
            doc_id=str(doc_id),
            elapsed_seconds=elapsed,
        )
        raise


@router.get("/{doc_id}")
async def get_document(doc_id: UUID, request: Request):
    """Return document metadata (no content body)."""
    import time as _time

    request_id = _extract_request_id(request)
    log.info(
        "get_document started",
        request_id=request_id,
        doc_id=str(doc_id),
    )
    _start = _time.time()
    try:
        pg: PostgresClient = _get_pg(request)
        doc = await pg.get_by_id(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        elapsed = _time.time() - _start
        log.info(
            "get_document succeeded",
            request_id=request_id,
            doc_id=str(doc_id),
            elapsed_seconds=elapsed,
        )
        return {
            "id": str(doc.id),
            "title": doc.title,
            "summary": doc.summary,
            "url": doc.url,
            "source": doc.source,
            "topics": doc.topics,
            "keywords": doc.keywords,
            "entities": doc.entities,
            "answers_questions": doc.answers_questions,
            "sentiment": doc.sentiment,
            "status": doc.status,
            "ingested_at": doc.ingested_at.isoformat() if doc.ingested_at else None,
        }
    except HTTPException:
        raise
    except Exception:
        elapsed = _time.time() - _start
        log.exception(
            "get_document failed",
            request_id=request_id,
            doc_id=str(doc_id),
            elapsed_seconds=elapsed,
        )
        raise
