# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
dewie/api/routes/user.py — Per-user self-service routes.

All routes require an authenticated session (cookie / session token).
API-key auth is intentionally NOT accepted for key-creation endpoints
to prevent privilege escalation.

Routes:
  POST   /user/ingest              — user-initiated ingest
  GET    /user/uploads             — list documents ingested by this user
  GET    /user/usage               — per-day usage (best-effort; empty when unavailable)
  POST   /user/api-keys            — create a personal API key (session auth only)
  GET    /user/api-keys            — list the caller's active API keys
  DELETE /user/api-keys/{key_id}   — revoke a personal API key
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import text

log = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["user"])


# ── Auth helpers ──────────────────────────────────────────────────────────────


_OPEN_MODE_USER_ID = "00000000-0000-0000-0000-000000000002"


def _require_user(request: Request) -> str:
    """Return user_id or raise 401. In open auth mode returns a synthetic local user."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        from dewie.config import settings
        # key_id set means the middleware already authenticated an API key.
        key_id = getattr(request.state, "key_id", None)
        if not settings.auth_enabled or key_id:
            return _OPEN_MODE_USER_ID
        raise HTTPException(status_code=401, detail="Authentication required")
    return str(user_id)


def _require_session_auth(request: Request) -> str:
    """Return user_id only when the request used session (cookie) auth, not an API key."""
    user_id = _require_user(request)
    if getattr(request.state, "key_id", None) is not None:
        raise HTTPException(
            status_code=403, detail="Session authentication required for this operation"
        )
    return user_id


def _pg(request: Request) -> Any:
    return request.app.state.postgres


# ── Ingest ────────────────────────────────────────────────────────────────────


class UserIngestRequest(BaseModel):
    url: str = Field(description="URL to ingest")


class UserIngestResponse(BaseModel):
    doc_id: str
    status: str
    doc_count: int = 1


class UserContextSelectionRequest(BaseModel):
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="User-space provider/model selections. null values remove keys.",
    )


async def _enrich_one(pg: Any, doc: Any) -> None:
    """Kick off background enrichment for a freshly-ingested document."""
    # In production this would enqueue the enrichment flow.
    # Kept as a thin wrapper so tests can patch it easily.
    pass


async def _audit_user(request: Request, action: str, resource_type: str, resource_id: str,
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
        _pg(request),
        settings=settings,
        tenant_id=tenant_id,
        actor_id=actor_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        metadata=metadata,
    )


@router.post("/ingest", response_model=UserIngestResponse, status_code=202)
async def user_ingest(body: UserIngestRequest, request: Request) -> UserIngestResponse:
    """Ingest a URL on behalf of the authenticated user."""
    user_id = _require_user(request)
    pg = _pg(request)

    import asyncio

    from sqlalchemy import text as _text

    from dewie.config import settings
    from dewie.ingestion.source_router import SourceRouter

    log.info("user_ingest started user_id=%s url=%s", user_id, body.url)
    source_router = SourceRouter()
    first_id: str | None = None
    doc_count = 0
    try:
        async for doc in source_router.fetch(body.url):
            await pg.upsert(doc)
            async with pg._engine.begin() as conn:
                row = (
                    await conn.execute(
                        _text("SELECT id FROM documents WHERE url = :url"),
                        {"url": doc.url},
                    )
                ).fetchone()
                actual_id = str(row[0]) if row else str(doc.id)
                await conn.execute(
                    _text("UPDATE documents SET user_id = :uid WHERE id = :doc_id"),
                    {"uid": user_id, "doc_id": actual_id},
                )
            from uuid import UUID as _UUID
            doc.id = _UUID(actual_id)
            if settings.enable_enrichment and request.app.state.processor is not None:
                asyncio.create_task(
                    request.app.state.processor.enrich_and_persist(doc, pg=pg)
                )
            await _audit_user(request, "doc.ingest", "document", actual_id)
            if first_id is None:
                first_id = actual_id
            doc_count += 1

        if first_id is None:
            log.warning("user_ingest: no content yielded url=%s user_id=%s", body.url, user_id)
            raise HTTPException(status_code=422, detail="Could not fetch content from URL")

        log.info("user_ingest succeeded doc_count=%d first_id=%s user_id=%s", doc_count, first_id, user_id)
        return UserIngestResponse(doc_id=first_id, status="pending", doc_count=doc_count)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("user_ingest failed url=%s user_id=%s: %s", body.url, user_id, exc)
        raise HTTPException(status_code=422, detail=f"Could not fetch content: {exc}") from exc
    finally:
        await source_router.close()


# ── Uploads ───────────────────────────────────────────────────────────────────


@router.get("/uploads")
async def list_uploads(request: Request) -> list:
    """List documents ingested by the authenticated user, including error messages for failed docs."""
    user_id = _require_user(request)
    pg = _pg(request)

    async with pg._engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT id, url, title, status, ingested_at "
                    "FROM documents WHERE user_id = :user_id "
                    "ORDER BY ingested_at DESC LIMIT 100"
                ),
                {"user_id": user_id},
            )
        ).fetchall()

    # Build a lookup of latest unresolved error per doc_id for quick lookup
    error_map: dict[str, str] = {}
    doc_ids = [str(r.id) for r in rows if r.status == "failed"]
    if doc_ids:
        try:
            id_list = ", ".join(f"'{did}'" for did in doc_ids)
            error_rows = (
                await conn.execute(
                    text(
                        f"SELECT DISTINCT ON (doc_id) doc_id, message "
                        f"FROM pipeline_errors "
                        f"WHERE doc_id IN ({id_list}) AND resolved = FALSE "
                        f"ORDER BY doc_id, created_at DESC"
                    )
                )
            ).fetchall()
            for er in error_rows:
                error_map[str(er.doc_id)] = er.message
        except Exception:
            pass

    return [
        {
            "id": str(r.id),
            "url": r.url,
            "title": r.title,
            "status": r.status,
            "ingested_at": str(r.ingested_at),
            "error_message": error_map.get(str(r.id)),
        }
        for r in rows
    ]


@router.get("/usage")
async def usage(_request: Request) -> list[dict[str, int | str]]:  # type: ignore[type-arg]
    """Return per-day usage history (empty when unavailable)."""
    return []


@router.get("/model-catalog")
async def user_model_catalog(
    request: Request,
    include_hidden: bool = False,
    purpose: str = "all",
) -> dict[str, Any]:
    """Return user-space catalog of provider/model pairs from filesystem overlays."""
    _require_user(request)
    from dewie.model_registry import registry

    if purpose not in {"all", "chat", "embedding"}:
        raise HTTPException(status_code=400, detail="purpose must be one of: all, chat, embedding")
    return await registry.catalog(context="user", include_hidden=include_hidden, purpose=purpose)


@router.get("/model-selection")
async def get_user_model_selection(request: Request) -> dict[str, Any]:
    """Return current user-space selection state (independent from admin context)."""
    _require_user(request)
    from dewie.model_registry import registry

    values = await registry.get_context_selection("user")
    return {"context": "user", "values": values}


@router.patch("/model-selection")
async def set_user_model_selection(
    body: UserContextSelectionRequest,
    request: Request,
) -> dict[str, Any]:
    """Update user-space selection state without affecting admin/server settings."""
    _require_user(request)
    from dewie.model_registry import registry

    values = await registry.set_context_selection("user", body.values)
    return {"context": "user", "values": values}


# ── API Keys ──────────────────────────────────────────────────────────────────


@router.post("/api-keys", status_code=201)
async def create_user_key(request: Request) -> dict:
    """Create a personal API key. Requires session auth (not API key auth)."""
    user_id = _require_session_auth(request)
    pg = _pg(request)

    import uuid as _uuid

    import dewie.auth as _auth

    raw_key, record = await _auth.create_api_key(
        pg,
        name=f"user-{user_id[:8]}",
        scopes=["read"],
        user_id=_uuid.UUID(user_id),
    )
    await _audit_user(request, "key.create", "api_key", str(record["id"]))
    return {
        "key": raw_key,
        "key_id": str(record["id"]),
        "prefix": record["key_prefix"],
        "created_at": str(record.get("created_at", "")),
    }


@router.get("/api-keys")
async def list_user_keys(request: Request) -> list:
    """List the caller's active API keys (non-sensitive)."""
    user_id = _require_user(request)
    pg = _pg(request)

    async with pg._engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT id, key_prefix, name, scopes, created_at, last_used_at "
                    "FROM api_keys WHERE user_id = :user_id AND revoked_at IS NULL "
                    "ORDER BY created_at DESC"
                ),
                {"user_id": user_id},
            )
        ).fetchall()

    return [
        {
            "id": str(r.id),
            "prefix": r.key_prefix,
            "name": r.name,
            "scopes": list(r.scopes) if r.scopes else [],
            "created_at": str(r.created_at),
            "last_used_at": str(r.last_used_at) if r.last_used_at else None,
        }
        for r in rows
    ]


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_user_key(key_id: uuid.UUID, request: Request) -> None:
    """Revoke one of the caller's API keys."""
    _require_user(request)
    pg = _pg(request)

    import dewie.auth as _auth

    revoked = await _auth.revoke_api_key(pg, key_id=key_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="Key not found or already revoked")

    await _audit_user(request, "key.revoke", "api_key", str(key_id))
