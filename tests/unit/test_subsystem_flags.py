"""
Unit tests for subsystem flags and SQLite config option (Issue #143).

These tests verify that Settings correctly parses ENABLE_API, ENABLE_ENRICHMENT,
ENABLE_INGESTION, ENABLE_POLLER, and DEWIE_DB from environment variables,
and that defaults are True for all boolean subsystem flags.
"""

from __future__ import annotations

import os
from unittest.mock import patch


def _make_settings(**env_overrides):
    """Construct a fresh Settings instance with env overrides (no .env file)."""
    from dewie.config import Settings

    env = {
        "POSTGRES_DSN": "postgresql+asyncpg://test:test@localhost/test",
        **env_overrides,
    }
    with patch.dict(os.environ, env, clear=False):
        return Settings(
            _env_file=None,  # type: ignore[call-arg]
        )


class TestSubsystemFlagDefaults:
    """All subsystem flags default to True for backward compatibility."""

    def test_enable_api_default_true(self):
        s = _make_settings()
        assert s.enable_api is True

    def test_enable_enrichment_default_true(self):
        s = _make_settings()
        assert s.enable_enrichment is True

    def test_enable_ingestion_default_true(self):
        s = _make_settings()
        assert s.enable_ingestion is True

    def test_enable_poller_default_true(self):
        s = _make_settings()
        assert s.enable_poller is True


class TestSubsystemFlagEnvParsing:
    """Subsystem flags are correctly parsed from environment variables."""

    def test_enable_api_false(self):
        s = _make_settings(ENABLE_API="false")
        assert s.enable_api is False

    def test_enable_api_true_explicit(self):
        s = _make_settings(ENABLE_API="true")
        assert s.enable_api is True

    def test_enable_enrichment_false(self):
        s = _make_settings(ENABLE_ENRICHMENT="false")
        assert s.enable_enrichment is False

    def test_enable_enrichment_true_explicit(self):
        s = _make_settings(ENABLE_ENRICHMENT="true")
        assert s.enable_enrichment is True

    def test_enable_ingestion_false(self):
        s = _make_settings(ENABLE_INGESTION="false")
        assert s.enable_ingestion is False

    def test_enable_ingestion_true_explicit(self):
        s = _make_settings(ENABLE_INGESTION="true")
        assert s.enable_ingestion is True

    def test_enable_poller_false(self):
        s = _make_settings(ENABLE_POLLER="false")
        assert s.enable_poller is False

    def test_enable_poller_true_explicit(self):
        s = _make_settings(ENABLE_POLLER="true")
        assert s.enable_poller is True


class TestSQLiteConfig:
    """DEWIE_DB env var is parsed correctly."""

    def test_dewie_db_default_empty(self):
        s = _make_settings()
        assert s.dewie_db == ""

    def test_dewie_db_sqlite_url(self):
        s = _make_settings(DEWIE_DB="sqlite+aiosqlite:///./dewie.db")
        assert s.dewie_db == "sqlite+aiosqlite:///./dewie.db"

    def test_dewie_db_postgres_override(self):
        dsn = "postgresql+asyncpg://user:pass@host/db"
        s = _make_settings(DEWIE_DB=dsn)
        assert s.dewie_db == dsn
