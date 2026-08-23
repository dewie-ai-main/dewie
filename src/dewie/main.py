# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Dewie — FastAPI application entry point.

Application lifecycle
---------------------
1. On startup:
   a. Initialise PostgreSQL schema.
   b. Build the ``BackendRegistry`` and ``EnrichmentRouter`` from config.
   c. Instantiate the ``MetadataProcessor`` with the router and registry.
   d. Attach all shared clients and services to ``app.state``.
2. On shutdown: gracefully close all storage connections.

Application state
-----------------
The following objects are available on ``request.app.state`` in route handlers:

- ``postgres``           — ``PostgresClient``
- ``cache``              — ``CacheClient``
- ``processor``          — ``MetadataProcessor`` (enrichment pipeline)
- ``network_backend``    — ``NetworkBackend`` (corpus sharing, cloud-overridable)

Run locally
-----------
::

    uvicorn dewie.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import os
import os as _os_main
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path as _Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from dewie.api.logging_config import get_logger as _get_logger
from dewie.api.middleware import register_api_middleware, register_middleware
from dewie.api.routes.admin import router as admin_router
from dewie.api.routes.auth import router as auth_router
from dewie.api.routes.capabilities import router as capabilities_router
from dewie.api.routes.corpus import router as corpus_router
from dewie.api.routes.dashboard import router as dashboard_router
from dewie.api.routes.documents import router as documents_router
from dewie.api.routes.feeds import router as feeds_router

# Load .env.local into os.environ early so that code reading os.environ directly
# (ModelClient, local_auth middleware) sees the same values as pydantic Settings.
_env_local = _Path(".env.local")
if _env_local.exists():
    for _line in _env_local.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            if _k and _k not in _os_main.environ:  # don't overwrite real env vars
                _os_main.environ[_k] = _v.strip()

from dewie.api.routes.graph import router as graph_router
from dewie.api.routes.health import router as health_router
from dewie.api.routes.ingest import router as ingest_router
from dewie.api.routes.investigate_v2 import router as investigate_v2_router
from dewie.api.routes.mcp import router as mcp_router
from dewie.api.routes.me import router as me_router
from dewie.api.routes.pipeline import router as pipeline_router
from dewie.api.routes.query import router as query_router
from dewie.api.routes.research_agent import router as research_agent_router
from dewie.api.routes.search_queue import router as search_queue_router
from dewie.api.routes.service_status import router as service_status_router
from dewie.api.routes.sources import router as sources_router
from dewie.api.routes.traverse import router as traverse_router
from dewie.api.routes.user import router as user_router
from dewie.config import settings
from dewie.enrichment.processor import MetadataProcessor
from dewie.enrichment.registry import BackendRegistry
from dewie.enrichment.router import EnrichmentRouter
from dewie.local_auth import create_local_user, seed_default_admin
from dewie.source_bootstrap import build_local_source, load_public_sources_defaults
from dewie.storage.cache import CacheClient, InProcessCacheClient
from dewie.storage.network import NoopNetworkBackend
from dewie.storage.postgres import PostgresClient
from dewie.workers.chunk_embedder import run_chunk_embedder_loop
from dewie.workers.edge_rebuild import run_edge_rebuild_loop
from dewie.workers.enrichment import run_enrichment_loop
from dewie.workers.ingest_poller import run_ingest_loop


async def _seed_admin_user(pg: PostgresClient) -> None:
    """Create a default admin user if no users exist in the database.

    Credentials come from ADMIN_EMAIL / ADMIN_PASSWORD env vars. If neither is
    set, seeding is skipped — the lifespan handles it via settings.admin_email /
    settings.admin_password, which also require explicit configuration.
    """
    from sqlalchemy import text as _text

    _log = _get_logger(__name__)

    email = os.environ.get("ADMIN_EMAIL", "").strip()
    password = os.environ.get("ADMIN_PASSWORD", "").strip()
    if not email or not password:
        _log.warning(
            "ADMIN_EMAIL / ADMIN_PASSWORD not set — skipping default admin seed. "
            "Set both env vars before first startup to create an admin account."
        )
        return

    async with pg._engine.connect() as _conn:
        count = await _conn.execute(_text("SELECT count(*) FROM users"))
        total = count.scalar()

    if total and total > 0:
        return

    try:
        await create_local_user(pg, email=email, password=password, name="Admin")
        _log.info("Seeded default admin user (email=%s)", email)
    except ValueError:
        # Email already exists (race condition on concurrent startup) — ignore
        pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the lifecycle of all shared resources."""
    from contextlib import AsyncExitStack

    from dewie.api.mcp_streamable import mcp_app

    async with AsyncExitStack() as _mcp_stack:
        # Runs the FastMCP session manager's task group for the life of the
        # app — Mount never propagates ASGI lifespan events into sub-apps, so
        # without this every streamable-HTTP request to /api/mcp-stream hangs.
        # .run() can only be called once per session-manager instance ever —
        # guard against re-entry so tests that invoke lifespan() repeatedly
        # against the same process-wide mcp_app singleton don't blow up.
        if not getattr(mcp_app.session_manager, "_has_started", False):
            await _mcp_stack.enter_async_context(mcp_app.session_manager.run())
        async for _ in _lifespan_body(app):
            yield


async def _lifespan_body(app: FastAPI) -> AsyncIterator[None]:
    # ── Production safety checks ──────────────────────────────────────────────
    import os as _os

    _log = _get_logger(__name__)
    _auth_enabled = _os.environ.get("AUTH_ENABLED", "false").lower() in ("1", "true", "yes")

    _admin_key = _os.environ.get("ADMIN_KEY", "")
    if _admin_key in ("", "dewie-admin-local"):
        _log.warning(
            "ADMIN_KEY is %s — set it to a strong random value before deploying.",
            "empty" if not _admin_key else repr(_admin_key),
        )

    _jwt_secret = _os.environ.get("JWT_SECRET", "")
    if not _jwt_secret:
        _log.warning(
            "JWT_SECRET not set — sessions will not survive restarts. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    elif len(_jwt_secret) < 32:
        _log.warning("JWT_SECRET is only %d chars — recommend 32+ for security.", len(_jwt_secret))

    _internal_key = _os.environ.get("INTERNAL_SERVICE_KEY", "")
    if _auth_enabled and not _internal_key:
        _log.warning(
            "AUTH_ENABLED=true but INTERNAL_SERVICE_KEY is not set — "
            "enrichment workers will not be able to call the ingest API."
        )

    # ── Logging configuration ───────────────────────────────────────────────
    _api_logger = _get_logger("dewie.api")
    _api_logger.info("API logging configured — request tracking enabled")

    # ── Subsystem startup log ─────────────────────────────────────────────────
    _log.info(
        "Subsystems: api=%s enrichment=%s ingestion=%s poller=%s",
        settings.enable_api,
        settings.enable_enrichment,
        settings.enable_ingestion,
        settings.enable_poller,
    )

    # ── Storage clients ───────────────────────────────────────────────────────
    pg = PostgresClient()
    cache = CacheClient() if settings.redis_url else InProcessCacheClient()

    await pg.init_schema()
    await _seed_admin_user(pg)

    local_source = build_local_source()
    if local_source:
        await pg.seed_public_sources([local_source])
        _log.info("Local source seeded: name=%s type=%s", local_source["name"], local_source["type"])

    if settings.enable_public_sources:
        seed_defaults = load_public_sources_defaults()
        seed_result = await pg.seed_public_sources(seed_defaults)
        _log.info(
            "Public source bootstrap completed: seeded=%s",
            seed_result.get("seeded", 0),
        )

    # ── Seed default admin user on first startup ────────────────────────────
    if settings.admin_email and settings.admin_password:
        admin_user = await seed_default_admin(pg, settings.admin_email, settings.admin_password)
        if admin_user:
            _log.info(
                "Default admin user seeded — email=%s (users table was empty)",
                admin_user["email"],
            )
        else:
            _log.info("Default admin seeding skipped — users already exist")
    else:
        _log.info(
            "Default admin seeding skipped — ADMIN_EMAIL/ADMIN_PASSWORD not configured"
        )

    # ── Enrichment pipeline ───────────────────────────────────────────────────
    if settings.enable_enrichment:
        registry = BackendRegistry.from_config(settings)
        router = EnrichmentRouter.from_config(settings, registry)
        processor = MetadataProcessor(
            router=router,
            registry=registry,
            fallback_backend_name=settings.enrichment_fallback_backend,
            max_retries=settings.enrichment_max_retries,
        )
        _log.info(
            "Enrichment pipeline initialised (backend=%s)", settings.enrichment_default_backend
        )
        if settings.enrichment_default_backend == "passthrough":
            _log.warning(
                "enrichment_default_backend is 'passthrough' — documents will be ingested "
                "but NO summary, answers_questions, or keywords will be generated. "
                "Set enrichment_default_backend and enrichment_backends in dewie.yml."
            )
        if not settings.embed_server:
            _log.warning(
                "embed_server is not configured — embeddings will not be stored and "
                "semantic search will return no results. Set embed_server and embed_model "
                "in dewie.yml."
            )
        elif not settings.embed_model:
            _log.warning(
                "embed_model is not configured — embeddings will not be stored. "
                "Set embed_model in dewie.yml."
            )
    else:
        registry = None  # type: ignore[assignment]
        router = None  # type: ignore[assignment]
        processor = None  # type: ignore[assignment]
        _log.info("Enrichment pipeline DISABLED (ENABLE_ENRICHMENT=false)")

    # ── Attach to app state ───────────────────────────────────────────────────
    app.state.postgres = pg
    app.state.cache = cache
    app.state.processor = processor
    app.state.network_backend = NoopNetworkBackend()

    from dewie.api import mcp_shared_state

    mcp_shared_state.configure(pg, processor)

    # Background tasks — neither blocks health-check or requests.
    background_tasks: list[asyncio.Task] = []

    def _spawn(coro):
        task = asyncio.create_task(coro)
        if isinstance(task, asyncio.Task):
            background_tasks.append(task)
        else:
            # Tests may monkeypatch create_task() with a plain Mock.
            # Close the coroutine to avoid "never awaited" warnings.
            coro.close()

    if settings.enable_enrichment and processor is not None:
        n_workers = max(1, getattr(settings, "enrichment_workers", 1))
        if getattr(pg, "_is_sqlite", False):
            n_workers = 1  # SQLite get_pending_docs has no atomic claim
        for _ in range(n_workers):
            _spawn(run_enrichment_loop(pg, processor, settings))
        _spawn(run_chunk_embedder_loop(pg, settings))

    if settings.enable_poller:
        _spawn(run_edge_rebuild_loop(pg, settings))
    else:
        _log.info("Background poller tasks DISABLED (ENABLE_POLLER=false)")

    if settings.enable_ingestion:
        _spawn(run_ingest_loop(pg, processor))

    # Refresh materialized views every 5 minutes
    async def _refresh_mat_views():
        from sqlalchemy import text as _sqlt

        while True:
            try:
                await asyncio.sleep(300)
                async with pg._engine.begin() as _conn:
                    # Try CONCURRENT first (non-blocking); fall back to blocking refresh
                    # if the view has never been populated (ObjectNotInPrerequisiteState)
                    if not pg._is_sqlite:
                        for view in ("corpus_quality_cache", "corpus_sources_cache"):
                            try:
                                await _conn.execute(
                                    _sqlt(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
                                )
                            except Exception:
                                await _conn.execute(_sqlt(f"REFRESH MATERIALIZED VIEW {view}"))
            except Exception as _exc:
                import logging as _log

                _log.getLogger(__name__).warning("mat view refresh failed: %s", _exc)

    _spawn(_refresh_mat_views())

    yield

    # ── Teardown ──────────────────────────────────────────────────────────────
    for task in background_tasks:
        task.cancel()
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)

    await pg.close()
    await cache.close()


app = FastAPI(
    title="Dewie",
    description=(
        "Agent-native retrieval framework. "
        "Ingest your documents, enrich with metadata, then search with full "
        "context so your agents can answer questions your LLM can't."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

register_middleware(app)
register_api_middleware(app)

# No-cache middleware for /ui/ static files — prevents stale HTML in dev
class _NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        if request.url.path.startswith("/ui/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(_NoCacheMiddleware)

# ── Route registration ────────────────────────────────────────────────────────
app.include_router(auth_router, prefix="/api")
app.include_router(capabilities_router, prefix="/api")
app.include_router(corpus_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(feeds_router, prefix="/api")
app.include_router(graph_router, prefix="/api")
app.include_router(health_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")
app.include_router(investigate_v2_router, prefix="/api")
app.include_router(mcp_router, prefix="/api")
app.include_router(me_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(research_agent_router, prefix="/api")
app.include_router(search_queue_router, prefix="/api")
app.include_router(service_status_router, prefix="/api")
app.include_router(sources_router, prefix="/api")
app.include_router(traverse_router, prefix="/api")
app.include_router(user_router, prefix="/api")
app.include_router(admin_router, prefix="/api")

# Serve UI static files
from dewie._static import static_dir

_static = static_dir()
if _static is not None:
    app.mount("/ui", StaticFiles(directory=str(_static)), name="ui")
else:
    _get_logger("dewie").warning("static/ directory not found — UI disabled (API unaffected)")

# In-process MCP Streamable HTTP transport — remote clients connect with just
# a URL + Authorization: Bearer <api_key>, no local Python process needed.
# Endpoint is /api/mcp-stream/mcp (FastMCP's own route lives at "/mcp" inside
# the mounted sub-app) — keeping FastMCP's default route path here avoids a
# 307 redirect that a Mount + Route("/") combination would otherwise require.
# Auth is handled by the global _api_key_middleware, same as every other route.
from dewie.api.mcp_streamable import mcp_app as _mcp_streamable_app  # noqa: E402

app.mount("/api/mcp-stream", _mcp_streamable_app.streamable_http_app())


# ── Root redirects ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root_redirect(request: Request):
    """Redirect unauthenticated users to login, authenticated users to the app."""
    token = request.cookies.get("dewie_session") or request.cookies.get("token") or \
            request.headers.get("Authorization")
    if token:
        return RedirectResponse(url="/ui/app.html", status_code=302)
    return RedirectResponse(url="/ui/login.html", status_code=302)


@app.get("/app", include_in_schema=False)
async def app_redirect():
    """Redirect to the SPA."""
    return RedirectResponse(url="/ui/app.html", status_code=302)


@app.get("/admin", include_in_schema=False)
async def admin_redirect():
    """Redirect to the SPA (admin panel handled client-side)."""
    return RedirectResponse(url="/ui/app.html", status_code=302)


@app.get("/health", include_in_schema=False)
async def health_simple() -> dict:  # type: ignore[type-arg]
    """Simple liveness probe for Docker healthchecks."""
    return {"status": "ok"}


@app.get("/service-status", include_in_schema=False)
async def status_redirect(request: Request):
    """Status endpoint for the /ui/status.html page and monitoring."""
    from dewie.api.routes.dashboard import service_status as _svc_status
    return await _svc_status(request)
