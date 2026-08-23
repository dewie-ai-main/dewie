# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
/ingest endpoints for submitting content to the pipeline.

Ingestion is fire-and-forget: the endpoint returns immediately with the
document ID(s) and processes enrichment in a background task.

Enrichment is delegated to ``MetadataProcessor`` from the ``enrichment``
package.  The processor is built from config at startup and attached to
``app.state.processor`` — this route does not instantiate its own processor.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import BaseModel

from dewie.api.middleware import limiter, rate_limit
from dewie.config import settings
from dewie.enrichment.processor import MetadataProcessor
from dewie.ingestion.rss import RSSIngester
from dewie.ingestion.web import WebIngester
from dewie.models.content import ContentDocument, IngestRequest
from dewie.storage.postgres import PostgresClient

log = logging.getLogger("dewie.api")

# ── Sensitive field redaction ────────────────────────────────────────────────────

_SENSITIVE_KEYS = frozenset(("api_key", "token", "password", "secret", "authorization"))


def _truncate(value: str, max_len: int = 1000) -> str:
    """Truncate a string to *max_len* characters for safe logging."""
    if len(value) > max_len:
        return value[:max_len] + f"... [truncated to {max_len} chars]"
    return value


def _redact_fields(d: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *d* with sensitive values redacted."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        k_lower = k.lower()
        if any(s in k_lower for s in _SENSITIVE_KEYS):
            out[k] = "***REDACTED***"
        elif isinstance(v, str) and len(v) > 1000:
            out[k] = v[:1000] + "... [truncated]"
        else:
            out[k] = v
    return out


def _extract_request_id(request: Request) -> str:
    """Extract request_id from request state, falling back to 'unknown'."""
    return getattr(request.state, "request_id", "unknown")

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestResponse(BaseModel):
    """Response returned immediately after documents are accepted for processing."""

    accepted: list[str] = []
    message: str = ""


def _get_pg(request: Request) -> PostgresClient:
    return request.app.state.postgres


def _get_processor(request: Request) -> MetadataProcessor:
    return request.app.state.processor


async def _audit_ingest(request: Request, action: str, resource_type: str, resource_id: str,
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


@router.post("", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(rate_limit())
async def ingest(
    request: Request,
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    pg: PostgresClient = Depends(_get_pg),
    processor: MetadataProcessor = Depends(_get_processor),
) -> IngestResponse:
    """
    Submit a URL (article page or RSS feed) for ingestion and enrichment.
    Requires X-Service-Key header matching INTERNAL_SERVICE_KEY env var.

    The response is returned immediately.  Metadata enrichment and
    relationship building happen in a background task via the configured
    enrichment backend (spaCy, local LLM, remote API, or agent).

    Args:
        body: Ingest request containing the URL to process.

    Returns:
        List of accepted document IDs and a status message.

    Raises:
        422: If no content is found at the URL.
        403: If X-Service-Key is missing or invalid.
    """
    request_id = _extract_request_id(request)
    t0 = time.monotonic()

    log.info(
        "ingest started",
        extra={
            "request_id": request_id,
            "url": str(body.url),
            "has_body": bool(body.body),
            "has_corpus_id": body.corpus_id is not None,
        },
    )

    try:
        # Service key auth — ingest workers must pass this header.
        # Authenticated users (session cookie or API key) and same-origin
        # requests (status page) may ingest without the header.
        _service_key = os.environ.get("INTERNAL_SERVICE_KEY", "").strip()
        _provided_key = request.headers.get("X-Service-Key", "").strip()

        if settings.internal_service_key_required and not _service_key:
            raise HTTPException(
                status_code=503,
                detail="Server misconfigured: INTERNAL_SERVICE_KEY is required but not set",
            )

        _has_ingest_scope = "ingest" in getattr(request.state, "key_scopes", [])
        _has_valid_key = _service_key and hmac.compare_digest(_provided_key, _service_key)
        _referer = request.headers.get("Referer", "")
        _same_origin = _referer and _referer.endswith("/ui/status.html")
        # When AUTH_ENABLED=false (local dev) all callers are trusted, matching
        # _require_admin() and delete_document(); the admin panel can ingest
        # without the service key in that mode.
        _auth_disabled = not settings.auth_enabled

        if (
            _service_key
            and not _has_valid_key
            and not _has_ingest_scope
            and not _same_origin
            and not _auth_disabled
        ):
            raise HTTPException(status_code=403, detail="Invalid or missing X-Service-Key")

        # Validate optional provider/model override pair at request boundary.
        if body.enrichment_provider or body.enrichment_model:
            from dewie.model_registry import registry

            valid, reason = await registry.validate_provider_model(
                provider=body.enrichment_provider,
                model=body.enrichment_model,
                include_hidden=False,
            )
            if not valid:
                raise HTTPException(status_code=400, detail=f"Invalid enrichment selection: {reason}")

        url = str(body.url)

        # If caller pre-fetched the body (e.g. ingest_reddit for link posts), use it directly
        # and skip the WebIngester round-trip entirely.
        if body.body:
            from urllib.parse import urlparse as _urlparse

            from dewie.models.content import ContentStatus

            docs: list[ContentDocument] = [
                ContentDocument(
                    url=url,
                    title=body.title or url,
                    body=body.body,
                    source=_urlparse(url).netloc,
                    status=ContentStatus.PENDING,
                    corpus_id=body.corpus_id,
                )
            ]
        else:
            # Detect feed vs. single-page heuristically
            is_feed = any(url.endswith(ext) for ext in (".xml", ".rss", ".atom")) or "feed" in url

            if is_feed:
                ingester = RSSIngester()
                docs = []
                async for doc in await ingester.fetch(url):
                    docs.append(doc)
            else:
                async with WebIngester() as ingester:
                    docs = [doc async for doc in ingester.fetch(url)]

        if not docs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No content found at URL (may be rate-limited, paywalled, or empty).",
            )

        # Propagate corpus_id from request to all docs (no-op when None)
        if body.corpus_id:
            for doc in docs:
                if doc.corpus_id is None:
                    doc.corpus_id = body.corpus_id

        # Propagate gap_fill flag — once set it should never be cleared
        if body.gap_fill:
            for doc in docs:
                doc.gap_fill = True

        # Propagate visibility — private docs are owned by the calling workspace
        if body.visibility:
            for doc in docs:
                doc.visibility = body.visibility

        # ── Separate paywall stubs from enrichable docs (before quality gate) ────
        # Paywall docs have empty/short body by design — they must bypass the gate
        # so we can persist the stub and avoid re-fetching the same paywalled URL.
        paywall_docs = [d for d in docs if getattr(d, "paywall_detected", False)]
        enrichable_docs = [d for d in docs if not getattr(d, "paywall_detected", False)]

        if paywall_docs:
            log.info(
                "ingest: %d paywall doc(s) detected from %s — persisting stubs, skipping enrichment",
                extra={
                    "request_id": request_id,
                    "count": len(paywall_docs),
                    "url": url,
                },
            )

        # ── Content quality gate ─────────────────────────────────────────────────
        # Applied only to non-paywall docs. Uses ContentValidator for comprehensive
        # pre-enrichment checks: length, boilerplate, noise ratio, repetition, title.
        # Falls back to legacy min_body_chars check if validator is disabled.
        from dewie.enrichment.content_validator import ContentValidator

        min_chars = body.min_body_chars
        validator_enabled = getattr(settings, "content_validator_enabled", True)

        if validator_enabled:
            enrichable_docs, rejected = ContentValidator.validate_many(
                enrichable_docs,
                min_body_chars=max(min_chars, 200) if min_chars > 0 else 200,
            )
            for rejected_doc, result in rejected:
                log.info(
                    "content_validator: rejected doc from %s — %s [check=%s]",
                    extra={
                        "request_id": request_id,
                        "url": url,
                        "reason": result.reason,
                        "checks_failed": ", ".join(result.checks_failed),
                    },
                )
        elif min_chars > 0:
            before = len(enrichable_docs)
            enrichable_docs = [
                d for d in enrichable_docs if len(getattr(d, "body", "") or "") >= min_chars
            ]
            dropped = before - len(enrichable_docs)
            if dropped:
                log.info(
                    "quality_gate: dropped docs (body too short)",
                    extra={
                        "request_id": request_id,
                        "dropped": dropped,
                        "total": before,
                        "url": url,
                        "min_chars": min_chars,
                    },
                )

        docs = paywall_docs + enrichable_docs
        if not docs:
            rejected_summary = (
                f"{len(rejected)} doc(s) rejected"
                if validator_enabled
                else f"min_body_chars={min_chars}"
            )
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            log.info(
                "ingest succeeded (all rejected by quality gate)",
                extra={
                    "request_id": request_id,
                    "elapsed_ms": elapsed,
                    # NB: key must not be "message" — reserved on LogRecord, raises at log time.
                    "detail": f"All documents rejected by content quality gate ({rejected_summary}).",
                },
            )
            return IngestResponse(
                accepted=[],
                message=f"All documents rejected by content quality gate ({rejected_summary}).",
            )

        # Persist enrichable documents immediately so IDs are available to callers
        # Note: paywall stubs are NOT persisted — they failed the quality gate
        for doc in enrichable_docs:
            await pg.upsert(doc)

        # Save raw body text to flat files for future corpus rebuilds.
        # Stored at data/bodies/{shard}/{doc_id}.txt — independent of DB.
        # Also write to Postgres body_text column for distributed worker support (Issue #51).
        from dewie.storage.body_store import save_body

        for doc in enrichable_docs:
            if hasattr(doc, "body") and doc.body:
                save_body(doc.id, doc.body)
                try:
                    await pg.write_body_text(doc.id, doc.body)
                except Exception as exc:
                    log.warning(
                        "Failed to write body_text to Postgres for doc %s: %s",
                        extra={"request_id": request_id, "doc_id": str(doc.id)},
                    )

        if enrichable_docs:
            background_tasks.add_task(_enrich_batch, enrichable_docs, pg, processor)

        paywall_note = f"; {len(paywall_docs)} paywall stub(s) skipped" if paywall_docs else ""
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "ingest succeeded",
            extra={
                "request_id": request_id,
                "accepted_count": len(enrichable_docs),
                "elapsed_ms": elapsed,
            },
        )
        for doc in enrichable_docs:
            await _audit_ingest(request, "doc.ingest", "document", str(doc.id))
        return IngestResponse(
            accepted=[str(doc.id) for doc in enrichable_docs],
            message=f"Accepted {len(enrichable_docs)} document(s) for processing{paywall_note}.",
        )
    except HTTPException:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.info(
            "ingest http_error",
            extra={
                "request_id": request_id,
                "status_code": 0,
                "elapsed_ms": elapsed,
            },
        )
        raise
    except Exception:
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        log.exception(
            "ingest failed",
            extra={
                "request_id": request_id,
                "url": url if "url" in locals() else None,
                "elapsed_ms": elapsed,
            },
        )
        raise


# ── Background enrichment ─────────────────────────────────────────────────────


async def _enrich_batch(
    docs: list[ContentDocument],
    pg: PostgresClient,
    processor: MetadataProcessor,
) -> None:
    """Background task: enrich each document in the batch sequentially."""
    for doc in docs:
        await processor.enrich_and_persist(doc, pg)

    # Pass B (pipeline.enrich_docs) intentionally removed — issue #119.
    # Newly ingested docs stay status='pending' and are picked up by the
    # enrichment_flow queue (Pass A), which uses embed_summary correctly.


# ── YouTube ingest ──────────────────────────────────────────────────────────


class YouTubeIngestRequest(BaseModel):
    """Request body for YouTube video or channel ingest."""

    url: str
    limit: int = 50  # max videos when url is a channel/playlist


class YouTubeIngestResponse(BaseModel):
    accepted: int
    skipped: int
    doc_ids: list[str]
    message: str


@router.post("/youtube", response_model=YouTubeIngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_youtube(
    request: Request,
    body: YouTubeIngestRequest,
    background_tasks: BackgroundTasks,
    pg: PostgresClient = Depends(_get_pg),
    processor: MetadataProcessor = Depends(_get_processor),
):
    """
    Ingest one or more YouTube videos as video_transcript documents.

    - Single video URL → ingest that video's transcript
    - Channel URL (/@handle, /channel/UC..., /user/...) or playlist → ingest up to `limit` videos
    """
    import time as _time
    import uuid as _uuid

    from dewie.ingestion.youtube import fetch_video, list_channel_videos  # noqa: PLC0415
    from dewie.models.content import ContentDocument, ContentStatus

    request_id = _extract_request_id(request)
    t0 = _time.monotonic()

    log.info(
        "ingest_youtube started",
        extra={
            "request_id": request_id,
            "url": body.url,
            "limit": body.limit,
        },
    )

    try:
        url = body.url.strip()
        is_channel = any(
            p in url for p in ("/@", "/channel/", "/user/", "/playlist?", "youtube.com/c/")
        )

        if is_channel:
            video_urls = await list_channel_videos(url, limit=body.limit)
        else:
            video_urls = [url]

        if not video_urls:
            elapsed = round((_time.monotonic() - t0) * 1000, 2)
            log.info(
                "ingest_youtube succeeded",
                extra={
                    "request_id": request_id,
                    "accepted": 0,
                    "skipped": 0,
                    "elapsed_ms": elapsed,
                },
            )
            return YouTubeIngestResponse(
                accepted=0, skipped=0, doc_ids=[], message="No videos found at URL"
            )

        accepted_docs: list[ContentDocument] = []
        skipped = 0

        for video_url in video_urls:
            data = await fetch_video(video_url)
            if data is None:
                skipped += 1
                continue

            # Check for duplicate via URL uniqueness (upsert will handle conflict, but skip early)
            if await pg.get_by_url(data["url"]) is not None:
                skipped += 1
                continue

            doc = ContentDocument(
                id=_uuid.uuid4(),
                url=data["url"],
                title=data["title"],
                body=data["body"],
                source=data["source"],
                status=ContentStatus.PENDING,
                document_type=data["doc_type"],
                author=data.get("author"),
                published_at=data.get("published_at"),
            )

            await pg.upsert(doc)

            # Write body to flat-file store for worker compatibility
            try:
                await pg.write_body_text(doc.id, data["body"])
            except Exception:
                pass

            accepted_docs.append(doc)

        if accepted_docs:
            background_tasks.add_task(_enrich_batch, accepted_docs, pg, processor)

        doc_ids = [str(d.id) for d in accepted_docs]
        elapsed = round((_time.monotonic() - t0) * 1000, 2)
        log.info(
            "ingest_youtube succeeded",
            extra={
                "request_id": request_id,
                "accepted": len(accepted_docs),
                "skipped": skipped,
                "elapsed_ms": elapsed,
            },
        )
        for d in accepted_docs:
            await _audit_ingest(request, "doc.ingest", "document", str(d.id))
        return YouTubeIngestResponse(
            accepted=len(accepted_docs),
            skipped=skipped,
            doc_ids=doc_ids,
            message=f"Accepted {len(accepted_docs)} video(s) for enrichment; {skipped} skipped (no transcript or duplicate)",
        )
    except Exception:
        elapsed = round((_time.monotonic() - t0) * 1000, 2)
        log.exception(
            "ingest_youtube failed",
            extra={
                "request_id": request_id,
                "url": body.url,
                "elapsed_ms": elapsed,
            },
        )
        raise
