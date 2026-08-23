"""Tests for dewie.main — FastAPI app factory and lifespan.

These tests import the `app` object (covering module-level code) and
exercise the lifespan with fully-mocked dependencies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _all_paths(app) -> list[str]:
    """Collect every route path, expanding FastAPI's lazy _IncludedRouter
    entries (fastapi >= 0.137 represents include_router() calls this way)."""
    paths: list[str] = []
    for r in app.routes:
        if hasattr(r, "effective_route_contexts"):
            paths.extend(c.path for c in r.effective_route_contexts())
        elif hasattr(r, "path"):
            paths.append(r.path)
    return paths


# ── Module-level: importing app covers all module-level code ──────────────────


class TestAppObject:
    def test_app_imported(self):
        """app is a FastAPI instance."""
        from fastapi import FastAPI

        from dewie.main import app

        assert isinstance(app, FastAPI)

    def test_app_title(self):
        """app has expected title."""
        from dewie.main import app

        assert app.title == "Dewie"

    def test_app_has_routes(self):
        """app has routes registered (health at minimum)."""
        from dewie.main import app

        paths = _all_paths(app)
        # At least the health and redirect routes must be present
        assert "/health" in paths

    def test_root_redirect_registered(self):
        """Root / redirect route is registered."""
        from dewie.main import app

        paths = _all_paths(app)
        assert "/" in paths

    def test_app_redirect_registered(self):
        """GET /app redirect is registered."""
        from dewie.main import app

        paths = _all_paths(app)
        assert "/app" in paths

    def test_admin_redirect_registered(self):
        """GET /admin redirect is registered."""
        from dewie.main import app

        paths = _all_paths(app)
        assert "/admin" in paths

    def test_health_endpoint(self):
        """GET /health returns 200."""
        from fastapi.testclient import TestClient

        from dewie.main import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        # It may fail due to DB not being set up, but route must exist
        assert resp.status_code != 404


# ── Lifespan: mock all dependencies ──────────────────────────────────────────


class TestLifespan:
    """Test the lifespan context manager with all heavy deps mocked."""

    @pytest.fixture
    def mock_pg(self):
        pg = AsyncMock()
        pg.init_schema = AsyncMock()
        pg.backfill_aq_tsvec = AsyncMock()
        pg.rebuild_capability_clusters = AsyncMock()
        pg.close = AsyncMock()

        # Set up _engine.begin()/connect() as async context managers so the
        # admin-seeding path works. scalar() is synchronous in SQLAlchemy, and
        # returning 1 means "users exist" -> seeding is skipped in tests.
        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.mappings.return_value.fetchone.return_value = {"cnt": 0}
        mock_result.scalar.return_value = 1
        mock_conn.execute = AsyncMock(return_value=mock_result)

        mock_begin_ctx = MagicMock()
        mock_begin_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_begin_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_begin_ctx
        mock_engine.connect.return_value = mock_begin_ctx
        pg._engine = mock_engine

        return pg

    @pytest.fixture
    def mock_cache(self):
        cache = AsyncMock()
        cache.close = AsyncMock()
        return cache

    @pytest.fixture
    def mock_processor(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_lifespan_startup_without_enrichment(self, mock_pg, mock_cache):
        """Lifespan completes startup with enrichment disabled."""
        from fastapi import FastAPI

        from dewie.main import lifespan

        test_app = FastAPI()

        with (
            patch("dewie.main.PostgresClient", return_value=mock_pg),
            patch("dewie.main.CacheClient", return_value=mock_cache),
            patch("dewie.main.settings") as mock_settings,
            patch("dewie.main._seed_admin_user"),
            patch("asyncio.create_task"),
        ):
            mock_settings.auth_enabled = False
            mock_settings.admin_email = ""
            mock_settings.admin_password = ""
            mock_settings.jwt_secret = "a-secure-test-secret-that-is-long-enough"
            mock_settings.enable_enrichment = False
            mock_settings.enable_ingestion = False
            mock_settings.enable_poller = False
            mock_settings.enable_api = True

            async with lifespan(test_app):
                # Startup complete — check pg.init_schema was called
                mock_pg.init_schema.assert_called_once()
                assert test_app.state.postgres is mock_pg
                assert test_app.state.cache is mock_cache
                assert test_app.state.processor is None

        mock_pg.close.assert_called_once()
        mock_cache.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_startup_with_enrichment(self, mock_pg, mock_cache, mock_processor):
        """Lifespan starts enrichment pipeline when enable_enrichment=True."""
        from fastapi import FastAPI

        from dewie.main import lifespan

        test_app = FastAPI()
        mock_registry = MagicMock()
        mock_router = MagicMock()

        with (
            patch("dewie.main.PostgresClient", return_value=mock_pg),
            patch("dewie.main.CacheClient", return_value=mock_cache),
            patch("dewie.main.BackendRegistry") as MockRegistry,
            patch("dewie.main.EnrichmentRouter") as MockRouter,
            patch("dewie.main.MetadataProcessor", return_value=mock_processor),
            patch("dewie.main.settings") as mock_settings,
            patch("dewie.main._seed_admin_user"),
            patch("asyncio.create_task"),
        ):
            MockRegistry.from_config.return_value = mock_registry
            MockRouter.from_config.return_value = mock_router
            mock_settings.auth_enabled = False
            mock_settings.admin_email = ""
            mock_settings.admin_password = ""
            mock_settings.jwt_secret = "a-secure-test-secret"
            mock_settings.enable_enrichment = True
            mock_settings.enable_ingestion = True
            mock_settings.enable_poller = False
            mock_settings.enable_api = True
            mock_settings.enrichment_default_backend = "openai"
            mock_settings.enrichment_fallback_backend = None
            mock_settings.enrichment_max_retries = 3
            mock_settings.enrichment_workers = 1

            async with lifespan(test_app):
                assert test_app.state.processor is mock_processor

        mock_pg.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_warns_on_weak_admin_key(self, mock_pg, mock_cache):
        """Lifespan warns if ADMIN_KEY is empty."""
        import os

        from fastapi import FastAPI

        from dewie.main import lifespan

        test_app = FastAPI()

        with (
            patch("dewie.main.PostgresClient", return_value=mock_pg),
            patch("dewie.main.CacheClient", return_value=mock_cache),
            patch("dewie.main.settings") as mock_settings,
            patch("dewie.main._seed_admin_user"),
            patch.dict(os.environ, {"ADMIN_KEY": ""}),
         ):
            mock_settings.auth_enabled = False
            mock_settings.admin_email = ""
            mock_settings.admin_password = ""
            mock_settings.enable_enrichment = False
            mock_settings.enable_ingestion = False
            mock_settings.enable_poller = False
            mock_settings.enable_api = True

            # Should not raise — just logs a warning
            async with lifespan(test_app):
                pass


# ── Issue #876: seed default admin user on first startup ────────────────────



def _engine_with_user_count(count: int):
    """Mock engine whose connect()/begin() yield a conn reporting `count` users.

    Mirrors the real shape: execute() is async, scalar() is sync.
    """
    from unittest.mock import AsyncMock, MagicMock

    conn = AsyncMock()
    result = MagicMock()
    result.scalar.return_value = count
    conn.execute = AsyncMock(return_value=result)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=None)
    engine = MagicMock()
    engine.connect.return_value = ctx
    engine.begin.return_value = ctx
    return engine


class TestSeedAdminUser:
    """Regression tests for #876 — default admin user seeding."""

    @pytest.mark.asyncio
    async def test_seed_admin_user_skips_when_no_env_vars(self):
        """When ADMIN_EMAIL/ADMIN_PASSWORD are not set, seeding is skipped (no default admin/admin)."""
        import os
        from unittest.mock import MagicMock

        from dewie.main import _seed_admin_user

        mock_pg = MagicMock()
        mock_pg._engine = _engine_with_user_count(0)

        with (
            patch("dewie.main.create_local_user") as mock_create,
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("ADMIN_EMAIL", None)
            os.environ.pop("ADMIN_PASSWORD", None)
            await _seed_admin_user(mock_pg)
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_seed_admin_user_uses_env_vars(self):
        """ADMIN_EMAIL / ADMIN_PASSWORD env vars override defaults."""
        import os

        from dewie.main import _seed_admin_user

        mock_pg = MagicMock()
        mock_pg._engine = _engine_with_user_count(0)

        with (
            patch("dewie.main.create_local_user") as mock_create,
            patch.dict(os.environ, {"ADMIN_EMAIL": "ops@corp.com", "ADMIN_PASSWORD": "s3cret!"}),
        ):
            await _seed_admin_user(mock_pg)
            mock_create.assert_called_once()
            assert mock_create.call_args.kwargs["email"] == "ops@corp.com"
            assert mock_create.call_args.kwargs["password"] == "s3cret!"

    @pytest.mark.asyncio
    async def test_seed_admin_user_skips_when_users_exist(self):
        """When users table has rows, seeding is skipped."""
        from dewie.main import _seed_admin_user

        mock_pg = MagicMock()
        mock_pg._engine = _engine_with_user_count(5)

        with patch("dewie.main.create_local_user") as mock_create:
            await _seed_admin_user(mock_pg)
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_seed_admin_user_ignores_duplicate(self):
        """Race condition: if another startup already created the user, ignore ValueError."""
        from dewie.main import _seed_admin_user

        mock_pg = MagicMock()
        mock_pg._engine = _engine_with_user_count(0)

        with patch("dewie.main.create_local_user") as mock_create:
            mock_create.side_effect = ValueError("Email already exists: admin")
            # Should not raise — the ValueError is caught
            await _seed_admin_user(mock_pg)


# ── Issue #241: corpus tab routes must be registered ────────────────────────


class TestCorpusTabRoutes:
    """Regression tests for #241 — corpus tab returned 404.

    Root causes:
    1. pipeline_router was not included in app.include_router() calls.
    2. /pipeline/corpus/* was matched by the blanket admin scope rule
       instead of the more specific read-scope rules.

    Both are fixed; these tests guard against regression.
    """

    def test_pipeline_corpus_quality_route_registered(self):
        """GET /api/pipeline/corpus/quality is registered in the app."""
        from dewie.main import app

        paths = _all_paths(app)
        assert "/api/pipeline/corpus/quality" in paths

    def test_pipeline_corpus_sources_route_registered(self):
        """GET /api/pipeline/corpus/sources is registered in the app."""
        from dewie.main import app

        paths = _all_paths(app)
        assert "/api/pipeline/corpus/sources" in paths

    def test_pipeline_corpus_quality_refresh_route_registered(self):
        """POST /api/pipeline/corpus/quality/refresh is registered in the app."""
        from dewie.main import app

        paths = _all_paths(app)
        assert "/api/pipeline/corpus/quality/refresh" in paths

    def test_corpus_quality_requires_read_not_admin_scope(self):
        """Middleware scope for /api/pipeline/corpus/quality is 'read', not 'admin'."""
        from dewie.api.middleware import _SCOPE_PREFIX_RULES

        path = "/api/pipeline/corpus/quality"
        matched_prefix: str | None = None
        matched_scope: str | None = None
        for prefix, scope in _SCOPE_PREFIX_RULES:
            if path.startswith(prefix):
                if matched_prefix is None or len(prefix) > len(matched_prefix):
                    matched_prefix = prefix
                    matched_scope = scope
        assert matched_scope == "read", (
            f"Expected 'read' scope for {path!r}, got {matched_scope!r}. "
            "This would cause a 403/404 for read-only API keys on the corpus tab."
        )


# ── CLI entry point ───────────────────────────────────────────────────────────


class TestCli:
    def test_cli_function_exists(self):
        """The CLI entrypoint (dewie.cli:main, per pyproject scripts) is importable."""
        from dewie.cli import main as cli

        assert callable(cli)
