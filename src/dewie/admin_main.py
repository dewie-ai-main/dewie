# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
dewie/admin_main.py — Internal admin FastAPI application.

This app is intended for private network access.

Includes:
  /admin/*      — API key management, invites, user approval
  /pipeline/*   — ingestion pipeline controls
  /auth/me      — session verification for the admin UI
  /ui/          — admin-only static pages

Session cookies issued by the public app (dewie.main) are valid here
because both apps share the same JWT_SECRET.

Run:
    uvicorn dewie.admin_main:admin_app
"""

from __future__ import annotations

import hmac
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from dewie.api.logging_config import get_logger as _get_logger
from dewie.api.middleware import register_api_middleware
from dewie.api.routes.admin import router as admin_router
from dewie.api.routes.auth import router as auth_router
from dewie.api.routes.health import router as health_router
from dewie.api.routes.pipeline import router as pipeline_router
from dewie.api.routes.service_status import router as service_status_router
from dewie.config import settings
from dewie.source_bootstrap import load_public_sources_defaults
from dewie.storage.cache import CacheClient
from dewie.storage.postgres import PostgresClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared resources for the admin app."""
    pg = PostgresClient()
    cache = CacheClient()
    await pg.init_schema()
    if settings.enable_public_sources:
        seed_defaults = load_public_sources_defaults()
        await pg.seed_public_sources(seed_defaults)

    app.state.postgres = pg
    app.state.cache = cache
    # processor is not needed for admin operations; set None so any route that
    # accidentally tries to use it fails loudly rather than silently.
    app.state.processor = None

    # ── Logging config ───────────────────────────────────────────────────
    _admin_logger = _get_logger("dewie.admin")
    _admin_logger.info("Admin API logging configured — request tracking enabled")

    yield

    await pg.close()
    await cache.close()


admin_app = FastAPI(
    title="Dewie Admin",
    description="Internal admin API — not exposed on the public internet.",
    version="0.1.0",
    lifespan=lifespan,
    # No public OpenAPI docs — keep schema off the open web
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# ── Middleware ────────────────────────────────────────────────────────────────
# Auth middleware: accepts X-Admin-Key header OR a valid dewie_session cookie
# (issued by the public app) where is_admin=true.
#
async def _admin_session_middleware(request: Request, call_next: Callable) -> Response:
    """
    Accept requests authenticated via:
      1. X-Admin-Key header (matches ADMIN_KEY env var)
      2. dewie_session JWT cookie (signed with JWT_SECRET) where is_admin=true

    All other requests get 401 (except /health and static assets).
    """
    path = request.url.path

    # Always set defaults
    request.state.workspace_ids = []
    request.state.key_id = None
    request.state.is_admin = False

    # Health check is always exempt
    if path == "/health":
        return await call_next(request)

    # Static assets — let through for CSS/JS/images needed by the login page.
    # HTML entry points (e.g. /ui/admin.html) are NOT exempt: an unauthenticated
    # request to an HTML page is redirected to the login page so the server
    # enforces auth rather than relying on client-side JS.
    if path.startswith("/ui/") or path.startswith("/static/"):
        if not path.endswith(".html"):
            return await call_next(request)
        # HTML pages require a valid session — redirect to login if absent
        session_token = request.cookies.get("dewie_session", "").strip()
        if session_token:
            try:
                from dewie.local_auth import verify_session_token

                payload = verify_session_token(session_token)
                if payload and payload.get("is_admin", False):
                    pg = request.app.state.postgres
                    # Check if the session token has been revoked
                    if pg:
                        try:
                            from dewie.local_auth import is_session_revoked
                            if await is_session_revoked(pg, session_token):
                                return RedirectResponse(url="/ui/login.html", status_code=302)
                        except Exception:
                            pass  # Best-effort: don't block auth if revocation check fails
                    request.state.is_admin = True
                    return await call_next(request)
            except Exception:
                pass

        return RedirectResponse(url="/ui/login.html", status_code=302)

    # Auth endpoints — let through so login/me work without a prior session
    if path.startswith("/auth/"):
        return await call_next(request)

    # ── X-Admin-Key header ────────────────────────────────────────────────────
    provided = request.headers.get("X-Admin-Key", "").strip()
    admin_key = os.environ.get("ADMIN_KEY", "")
    if admin_key and provided and hmac.compare_digest(provided, admin_key):
        request.state.is_admin = True
        return await call_next(request)

    # ── Session cookie (shared JWT_SECRET with public app) ────────────────────
    session_token = request.cookies.get("dewie_session", "").strip()
    if session_token:
        try:
            from dewie.local_auth import verify_session_token

            payload = verify_session_token(session_token)
            if payload and payload.get("is_admin", False):
                pg = request.app.state.postgres
                # Check if the session token has been revoked
                if pg:
                    try:
                        from dewie.local_auth import is_session_revoked
                        if await is_session_revoked(pg, session_token):
                            return Response(
                                content='{"detail": "Session revoked. Please log in again."}',
                                status_code=401,
                                media_type="application/json",
                            )
                    except Exception:
                        pass  # Best-effort: don't block auth if revocation check fails
                request.state.is_admin = True
                return await call_next(request)
        except Exception:
            pass

    return Response(
        content='{"detail": "Admin authentication required"}',
        status_code=401,
        media_type="application/json",
    )


admin_app.middleware("http")(_admin_session_middleware)
register_api_middleware(admin_app)

# ── Routes ────────────────────────────────────────────────────────────────────

admin_app.include_router(auth_router)
admin_app.include_router(admin_router)
admin_app.include_router(pipeline_router)
admin_app.include_router(health_router)
admin_app.include_router(service_status_router)


@admin_app.get("/health", tags=["ops"])
async def health() -> dict:  # type: ignore[type-arg]
    return {"status": "ok"}


@admin_app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/admin.html", status_code=302)


# ── Static files ──────────────────────────────────────────────────────────────

from dewie._static import static_dir

_static_path = static_dir()
if _static_path is not None:
    admin_app.mount("/ui", StaticFiles(directory=str(_static_path), html=True), name="static")


# ── CLI entry point ───────────────────────────────────────────────────────────


def cli() -> None:
    import uvicorn

    host = os.environ.get("ADMIN_HOST", "127.0.0.1")
    port = int(os.environ.get("ADMIN_PORT", "8001"))
    uvicorn.run(
        "dewie.admin_main:admin_app",
        host=host,
        port=port,
        reload=False,
    )
