# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Async PostgreSQL client wrapping SQLAlchemy Core for document storage.

Schema is intentionally kept simple — a single `documents` table with
JSONB columns for metadata arrays, enabling flexible GIN-index queries.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from uuid import UUID

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from dewie.config import settings
from dewie.models.content import ContentDocument, ContentStatus, DocumentType, ReadingLevel
from dewie.models.feed import RSSFeed
from dewie.models.metadata import Relationship

# Weak / default Postgres passwords that should never be used in production.
WEAK_PASSWORDS: frozenset[str] = frozenset(
    {"dewie", "postgres", "password", "admin", "secret", "changeme", ""}
)

_logger = logging.getLogger(__name__)


def check_db_credentials(dsn: str | None = None) -> None:
    """Warn at startup if default or weak database credentials are detected.

    Checks both the POSTGRES_PASSWORD environment variable and the password
    embedded in the DSN (if provided).  Emits a single WARNING-level log
    message when a weak credential is found.

    This is intentionally lightweight — no network calls, no schema checks.
    """
    dsn = dsn or settings.postgres_dsn
    warned = False

    # 1. Check POSTGRES_PASSWORD env var (only if actually set)
    pw_env = os.environ.get("POSTGRES_PASSWORD")
    if pw_env is not None and pw_env.lower() in WEAK_PASSWORDS:
        _logger.warning(
            "SECURITY WARNING: POSTGRES_PASSWORD is set to a weak or default value. "
            "Change it to a strong password (≥16 chars, mixed case, numbers, symbols) "
            "before deploying to production.",
        )
        warned = True

    # 2. Extract password from DSN and check it too
    #    DSN format: postgresql[+asyncpg]://user:password@host:port/db[?params]
    m = re.search(r"postgresql\+?\w*://[^:]+:([^@]+)@", dsn)
    if m:
        pw_dsn = m.group(1)
        if pw_dsn.lower() in WEAK_PASSWORDS:
            if not warned:
                _logger.warning(
                    "SECURITY WARNING: Database password in POSTGRES_DSN is weak or default. "
                    "Change it to a strong password before deploying to production.",
                )
            else:
                # Already warned about env var — still log the DSN detail
                _logger.warning(
                    "SECURITY WARNING: Database password in POSTGRES_DSN is also weak "
                    "(value matches a known default). Update both POSTGRES_PASSWORD and "
                    "POSTGRES_DSN in your .env file.",
                )


_VALID_SOURCE_TYPES = frozenset({"sqlite", "postgres", "mcp"})




async def _expand_query_with_session(query: str, session) -> str:  # type: ignore[type-arg]
    """
    Expand a query with alternate_terms from the corpus.
    E.g. "NBA scores" → "NBA scores basketball National Basketball Association"

    Only adds terms that:
    - appear in at least 2 documents (prevents rare one-off noise)
    - are short (≤4 chars) or acronym-like, OR appear in multiple docs
    - are not already in the query
    Caps expansion at 5 added terms to avoid query bloat.
    """
    words = [w.strip() for w in query.split() if len(w.strip()) > 1]
    if not words:
        return query
    lower_words = [w.lower() for w in words]
    conditions = " OR ".join(
        f"alternate_terms @> CAST(:lw{i} AS jsonb)" for i, _ in enumerate(lower_words)
    )
    params: dict[str, str] = {f"lw{i}": json.dumps([w]) for i, w in enumerate(lower_words)}
    try:
        # Only return terms that appear in ≥2 documents, capped at 10 candidates
        sql = text(f"""
            SELECT term, COUNT(*) AS freq
            FROM (
                SELECT DISTINCT id, jsonb_array_elements_text(alternate_terms) AS term
                FROM documents
                WHERE status = 'ready'
                  AND ({conditions})
            ) sub
            GROUP BY term
            HAVING COUNT(*) >= 2
            ORDER BY freq DESC
            LIMIT 10
        """)
        rows = (await session.execute(sql, params)).all()
        existing = {w.lower() for w in query.split()}
        extra = [r[0] for r in rows if r[0].lower() not in existing]
        if extra:
            return query + " " + " ".join(extra[:5])
    except Exception:
        pass
    return query


def _embed_dimensions_for_model(model: str) -> int:
    """Infer vector dimensions from a known embedding model name.

    The vector column defaults to 1536 dims. Changing EMBED_MODEL to a model
    with different dimensions requires re-embedding all documents and running
    ``ALTER TABLE documents ALTER COLUMN embedding TYPE vector(<dims>)``.
    Override auto-detection via the EMBED_DIMENSIONS env var.
    """
    env_dims = os.environ.get("EMBED_DIMENSIONS")
    if env_dims:
        return int(env_dims)

    requested_dims = os.environ.get("EMBED_OUTPUT_DIMENSIONS")
    if requested_dims:
        return int(requested_dims)

    m = model.lower()
    if "embeddinggemma" in m:
        return 768
    if "qwen3-embedding-8b" in m:
        return 4096
    if "qwen3-embedding-4b" in m:
        return 2560
    if "qwen3-embedding-0.6b" in m:
        return 1024
    if "3-large" in m:
        return 3072
    if "nomic" in m:
        return 768
    if "ada-002" in m or "3-small" in m:
        return 1536
    return 1536  # safe default for most OpenAI-compatible models


async def _get_embedding(query: str) -> list[float] | None:
    """Get query embedding via the configured embedding provider.

    Provider resolution:
        embed_server  — registered server label (see providers/servers.py), or
                        'local' for in-process embeddings. Set via dewie.yml /
                        EMBED_SERVER env var.
        embed_model   — embedding model name (default: text-embedding-3-small)
    """
    try:
        from dewie.providers.factory import get_embedding_provider

        provider = get_embedding_provider()
        vectors = await provider.embed([query])
        if vectors and vectors[0]:
            return vectors[0]
    except Exception:
        pass
    return None


# ── DDL extracted to Alembic migration ────────────────────────────────────────
# See: migrations/versions/000000000000_baseline_schema.py


# ── Well-known seed UUIDs ──────────────────────────────────────────────────────
ROOT_WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000010")
DEFAULT_CORPUS_ID = UUID("00000000-0000-0000-0000-000000000011")


class PostgresClient:
    """Thin async wrapper around SQLAlchemy for document CRUD operations."""

    def __init__(self, dsn: str | None = None) -> None:
        # Use provided DSN, or fallback to settings.postgres_dsn
        dsn = dsn or settings.postgres_dsn
        self._is_sqlite = dsn.startswith("sqlite+") or dsn.startswith("sqlite://")
        # Warn if default/weak credentials are detected (skipped for SQLite)
        if not self._is_sqlite:
            check_db_credentials(dsn)
        if self._is_sqlite:
            self._engine = create_async_engine(dsn)
        else:
            # asyncpg ignores ?ssl=disable in the DSN — pass it explicitly
            _ssl_arg: bool | None = None
            if "ssl=disable" in dsn.lower() or "sslmode=disable" in dsn.lower():
                _ssl_arg = False
                dsn = dsn.split("?")[0]  # strip query params asyncpg doesn't understand
            _connect_args: dict = {"server_settings": {"hnsw.ef_search": "100"}}
            if _ssl_arg is not None:
                _connect_args["ssl"] = _ssl_arg
            self._engine = create_async_engine(
                dsn,
                pool_size=20,
                max_overflow=40,
                pool_timeout=30,
                pool_recycle=1800,
                connect_args=_connect_args,
            )
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
    async def _test_postgres_connection(self, config: dict) -> tuple[bool, str | None]:
        """Try connecting to a postgres source and return (ok, error)."""
        dsn = str(config.get("dsn", "")).strip()

        # If no DSN, build one from host/database/user/password
        if not dsn:
            host = str(config.get("host", "localhost")).strip()
            port = str(config.get("port", "5432")).strip()
            database = str(config.get("database", config.get("dbname", "postgres"))).strip()
            user = str(config.get("user", config.get("username", "postgres"))).strip()
            password = str(config.get("password", "")) or ""
            if password:
                dsn = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}?ssl=disable"
            else:
                dsn = f"postgresql+asyncpg://{user}@{host}:{port}/{database}?ssl=disable"

        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import create_async_engine
            engine = create_async_engine(dsn, pool_size=1, max_overflow=0)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            await engine.dispose()
            return (True, None)
        except Exception as exc:
            return (False, str(exc))

    async def _test_mcp_connection(self, config: dict) -> tuple[bool, str | None]:
        """Validate MCP Dewie source reachability and query API compatibility."""
        endpoint = str(config.get("endpoint", "")).strip().rstrip("/")
        if not endpoint:
            return (False, "Missing endpoint in config")

        api_key = str(config.get("api_key", "")).strip()
        headers: dict[str, str] = {}
        if api_key:
            headers["X-API-Key"] = api_key

        # Prefer query capability checks first, then health probes.
        candidates = [f"{endpoint}/api/query/rankers", f"{endpoint}/query/rankers"]
        if endpoint.endswith("/api"):
            base = endpoint[: -len("/api")]
            candidates = [
                f"{endpoint}/query/rankers",
                f"{base}/api/query/rankers",
                f"{base}/query/rankers",
            ]
        candidates.extend([f"{endpoint}/api/health", f"{endpoint}/health"])

        deduped: list[str] = []
        for url in candidates:
            if url not in deduped:
                deduped.append(url)

        from dewie.ingestion.web import _SSRFSafeTransport

        last_status: int | None = None
        last_error: str | None = None
        try:
            async with httpx.AsyncClient(transport=_SSRFSafeTransport(), timeout=10.0) as client:
                for url in deduped:
                    try:
                        resp = await client.get(url, headers=headers, follow_redirects=True)
                    except Exception as exc:  # noqa: BLE001
                        last_error = f"{type(exc).__name__}: {exc}"
                        continue

                    if resp.status_code == 200:
                        return (True, None)
                    if resp.status_code in {404, 405}:
                        last_status = resp.status_code
                        continue
                    if resp.status_code in {401, 403}:
                        return (False, f"Authentication failed ({resp.status_code})")

                    last_status = resp.status_code
        except Exception as exc:  # noqa: BLE001
            return (False, f"{type(exc).__name__}: {exc}")

        if last_error:
            return (False, f"Connection failed: {last_error}")
        if last_status is not None:
            return (False, f"No compatible API endpoint found (last status {last_status})")
        return (False, "No compatible API endpoint found")


    async def init_schema(self) -> None:
        """Ensure schema is up to date via Alembic migrations.

        For SQLite, falls back to the inline schema builder.
        For PostgreSQL, delegates to ``alembic upgrade head`` which runs
        the baseline migration and any subsequent revision files.
        """
        if getattr(self, "_is_sqlite", False):
            await self._init_sqlite_schema()
            return

        await self._run_alembic_upgrade()
        await self._migrate_corpus_ids()
        await self._ensure_revoked_session_tokens_table()

    @staticmethod
    async def _run_alembic_upgrade() -> None:
        """Run ``alembic upgrade head`` in-process.

        Idempotent — Alembic tracks migration state in ``alembic_version``,
        so running this on every startup is a no-op after the first time.
        """
        from configparser import RawConfigParser
        from pathlib import Path

        from alembic import command
        from alembic.config import Config

        # Resolve paths against the installed package, not the current working
        # directory — a pip-installed dewie is booted from arbitrary cwds where
        # ``alembic.ini`` and ``src/dewie/migrations`` do not exist.
        migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
        repo_ini = Path(__file__).resolve().parent.parent.parent.parent / "alembic.ini"

        alembic_cfg = Config(str(repo_ini) if repo_ini.exists() else None)
        # Replace the file_config with one that has interpolation disabled
        # so %(rev)s and %(slug)s in file_template are not treated as variables
        raw_cp = RawConfigParser(interpolation=None)
        if repo_ini.exists():
            raw_cp.read(str(repo_ini))
        alembic_cfg.file_config = raw_cp
        # Always point script_location at the packaged migrations directory.
        alembic_cfg.set_main_option("script_location", str(migrations_dir))
        # Pass the DSN through so Alembic uses the same connection.
        # Fall back to settings.postgres_dsn so env var isn't required.
        from dewie.config import settings as _settings

        postgres_dsn = (
            os.environ.get("POSTGRES_DSN")
            or os.environ.get("POSTGRES_URL")
            or _settings.postgres_dsn
        )
        if postgres_dsn:
            sync_dsn = postgres_dsn.replace("postgresql+asyncpg://", "postgresql://")
            if "?" in sync_dsn:
                sync_dsn = sync_dsn.split("?")[0]
            alembic_cfg.set_main_option("sqlalchemy.url", sync_dsn)

        def _run() -> None:
            command.upgrade(alembic_cfg, "head")

        # Alembic CLI functions are synchronous — run in executor to avoid blocking
        import asyncio

        await asyncio.get_event_loop().run_in_executor(None, _run)

    async def _init_sqlite_schema(self) -> None:
        """Create a minimal SQLite schema for local OSS mode."""
        sqlite_schema = [
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                embed_summary TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                status TEXT NOT NULL,
                topics TEXT NOT NULL DEFAULT '[]',
                keywords TEXT NOT NULL DEFAULT '[]',
                entities TEXT NOT NULL DEFAULT '[]',
                sentiment REAL,
                crawl_session TEXT,
                search_vec TEXT,
                enrichment_version INTEGER NOT NULL DEFAULT 0,
                embedding_model TEXT,
                enriched_at TEXT,
                author TEXT,
                reading_level TEXT,
                document_type TEXT,
                tone TEXT,
                answers_questions TEXT NOT NULL DEFAULT '[]',
                aq_tsvec TEXT,
                published_at TEXT,
                paywall_detected INTEGER NOT NULL DEFAULT 0,
                paywall_type TEXT NOT NULL DEFAULT 'none',
                alternate_terms TEXT NOT NULL DEFAULT '[]',
                enrichment_quality_score INTEGER,
                body_text TEXT,
                priority INTEGER NOT NULL DEFAULT 0,
                chunk_status TEXT NOT NULL DEFAULT 'none',
                embedding TEXT,
                embedding_full TEXT,
                corpus_id TEXT,
                 sharing_tier TEXT NOT NULL DEFAULT 'private',
                 retain_body INTEGER NOT NULL DEFAULT 0,
                 instance_id TEXT,
                 user_id TEXT
             )
            """,
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)",
            "CREATE INDEX IF NOT EXISTS idx_documents_ingested_at ON documents(ingested_at)",
            "CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url)",
            """
            CREATE TABLE IF NOT EXISTS document_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                rel_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0,
                shared_attrs TEXT,
                PRIMARY KEY (source_id, target_id, rel_type)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                embedding TEXT,
                embedding_model TEXT,
                aq_text TEXT,
                aq_embedding TEXT,
                PRIMARY KEY (doc_id, chunk_index)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                slug TEXT NOT NULL UNIQUE,
                sharing_tier TEXT NOT NULL DEFAULT 'private',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS corpora (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                slug TEXT NOT NULL UNIQUE,
                sharing_tier TEXT NOT NULL DEFAULT 'internal_only',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT DEFAULT CURRENT_TIMESTAMP,
                tenant_id TEXT,
                source TEXT NOT NULL DEFAULT 'api',
                question TEXT NOT NULL,
                model TEXT,
                hops INTEGER DEFAULT 0,
                elapsed_ms INTEGER DEFAULT 0,
                answer TEXT,
                hop_trace TEXT,
                docs_returned TEXT,
                full_results TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                tenant_id TEXT,
                email TEXT NOT NULL UNIQUE,
                name TEXT,
                google_sub TEXT,
                apple_sub TEXT,
                is_admin INTEGER NOT NULL DEFAULT 0,
                plan TEXT NOT NULL DEFAULT 'free',
                activation_status TEXT NOT NULL DEFAULT 'approved',
                password_hash TEXT,
                password_reset_token TEXT,
                password_reset_expires TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_google_sub ON users(google_sub)",
            "CREATE INDEX IF NOT EXISTS idx_users_apple_sub ON users(apple_sub)",
            """
            CREATE TABLE IF NOT EXISTS pipeline_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                step TEXT NOT NULL,
                error_type TEXT NOT NULL,
                message TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                resolved INTEGER NOT NULL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS system_health (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab', abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
                doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                step TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                raw_response TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (doc_id, step, model)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_llm_cache_doc_id ON llm_cache (doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_llm_cache_step ON llm_cache (step)",
            """
            CREATE TABLE IF NOT EXISTS dewie_sources (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL,
                config_json TEXT NOT NULL DEFAULT '{}',
                enabled INTEGER NOT NULL DEFAULT 1,
                created_by TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                tested_at TEXT,
                test_status TEXT,
                test_error TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_dewie_sources_enabled ON dewie_sources(enabled)",
            """
            CREATE TABLE IF NOT EXISTS revoked_session_tokens (
                token_hash TEXT PRIMARY KEY,
                revoked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab', abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
                user_id TEXT,
                workspace_ids TEXT NOT NULL DEFAULT '[]',
                key_hash TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                scopes TEXT NOT NULL DEFAULT '["read"]',
                name TEXT,
                revoked_at TEXT,
                last_used_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix)",
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id     TEXT,
                actor_id      TEXT,
                action        TEXT NOT NULL,
                resource_type TEXT,
                resource_id   TEXT,
                metadata      TEXT NOT NULL DEFAULT '{}',
                ts            TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action)",
            """
            CREATE TABLE IF NOT EXISTS rss_feeds (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                name TEXT,
                corpus_id TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                enabled INTEGER NOT NULL DEFAULT 1,
                poll_interval_minutes INTEGER NOT NULL DEFAULT 60,
                last_polled_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                tenant_id TEXT
            )
            """,
        ]

        async with self._engine.begin() as conn:
            for stmt in sqlite_schema:
                await conn.exec_driver_sql(stmt)

            # Migrations for columns added after initial schema creation
            _sqlite_migrations = [
                "ALTER TABLE documents ADD COLUMN user_id TEXT",
                "ALTER TABLE documents ADD COLUMN embedding_full TEXT",
            ]
            for migration in _sqlite_migrations:
                try:
                    await conn.exec_driver_sql(migration)
                except Exception:
                    pass  # Column already exists

    async def _migrate_corpus_ids(self) -> None:
        """Migrate documents.corpus_id from TEXT to UUID FK pointing at corpora.

        Idempotent — detects whether migration has already run by checking the
        column type.  Skips cleanly when corpus_id is already a UUID column or
        when there are no documents with a corpus_id set.
        """
        async with self._engine.connect() as probe:
            rows = await probe.exec_driver_sql(
                """
                SELECT data_type FROM information_schema.columns
                WHERE table_name = 'documents' AND column_name = 'corpus_id'
                """
            )
            row = rows.fetchone()
            if row is None or row[0].lower() != "text":
                # corpus_id doesn't exist or is already a UUID — nothing to do
                return

        async with self._engine.begin() as conn:
            # Collect distinct corpus_id text values
            rows = await conn.exec_driver_sql(
                "SELECT DISTINCT corpus_id FROM documents WHERE corpus_id IS NOT NULL"
            )
            slugs = [r[0] for r in rows.fetchall()]

            # Ensure a corpora row exists for each slug
            for slug in slugs:
                await conn.exec_driver_sql(
                    """
                    INSERT INTO corpora (workspace_id, name, slug, sharing_tier)
                    VALUES ($1, $2, $3, 'internal_only')
                    ON CONFLICT (slug) DO NOTHING
                    """,
                    (str(ROOT_WORKSPACE_ID), slug, slug),
                )

            # Add the new UUID column
            await conn.exec_driver_sql(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS corpus_id_new UUID"
            )

            # Back-fill new column from slug match
            await conn.exec_driver_sql(
                """
                UPDATE documents d
                SET corpus_id_new = c.id
                FROM corpora c
                WHERE d.corpus_id = c.slug
                """
            )

            # Docs with no corpus_id get assigned the default corpus
            await conn.exec_driver_sql(
                """
                UPDATE documents
                SET corpus_id_new = $1
                WHERE corpus_id_new IS NULL
                """,
                (str(DEFAULT_CORPUS_ID),),
            )

            # Swap columns
            await conn.exec_driver_sql("ALTER TABLE documents DROP COLUMN corpus_id")
            await conn.exec_driver_sql(
                "ALTER TABLE documents RENAME COLUMN corpus_id_new TO corpus_id"
            )
            await conn.exec_driver_sql(
                "ALTER TABLE documents ADD CONSTRAINT documents_corpus_id_fkey "
                "FOREIGN KEY (corpus_id) REFERENCES corpora(id) ON DELETE SET NULL"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_documents_corpus_id ON documents (corpus_id)"
            )

    async def _ensure_revoked_session_tokens_table(self) -> None:
        """Ensure the revoked_session_tokens table exists (PostgreSQL).

        Idempotent — uses CREATE TABLE IF NOT EXISTS so it's safe to run
        on every startup. No ALTER TABLE on startup to avoid prod deadlocks.
        """
        try:
            async with self._engine.begin() as conn:
                await conn.exec_driver_sql("""
                    CREATE TABLE IF NOT EXISTS revoked_session_tokens (
                        token_hash TEXT PRIMARY KEY,
                        revoked_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """)
        except Exception:
            pass  # Best-effort; table may already exist via migration

    async def backfill_aq_tsvec(self) -> None:
        """Backfill aq_tsvec for docs that have answers_questions but no stored tsvector.

        Best-effort: called as a background task after startup so it does not delay
        the health-check. The enrichment pipeline writes aq_tsvec on every re-enrichment,
        so missing rows will be filled in over time even if this backfill is skipped.
        """
        try:
            async with self._engine.begin() as conn:
                await conn.exec_driver_sql("""
                    UPDATE documents
                    SET aq_tsvec = to_tsvector('english', coalesce(answers_questions::text, ''))
                    WHERE aq_tsvec IS NULL
                      AND answers_questions IS NOT NULL
                      AND jsonb_array_length(answers_questions) > 0
                """)
        except Exception:
            pass  # Best-effort; pipeline fills aq_tsvec on every re-enrichment

    async def rebuild_capability_clusters(self, min_docs: int = 5) -> int:
        """
        Precompute the capability cluster index from enrichment metadata.

        Groups documents by their primary topic, identifies the hub document
        (highest edge count) per cluster, extracts its AQ strings as a
        natural-language capability description, and writes to capability_clusters.

        Best-effort: called as a background task. Returns number of clusters written.
        """
        try:
            async with self._engine.begin() as conn:
                # One cluster per distinct primary topic with enough documents
                rows = (
                    (
                        await conn.execute(
                            text("""
                    WITH topic_groups AS (
                        SELECT
                            topic,
                            count(*)                         AS doc_count,
                            min(ingested_at)                 AS earliest_doc,
                            max(ingested_at)                 AS latest_doc
                        FROM (
                            SELECT id, ingested_at,
                                   jsonb_array_elements_text(topics) AS topic
                            FROM documents
                            WHERE status = 'ready'
                              AND topics IS NOT NULL
                              AND jsonb_array_length(topics) > 0
                        ) t
                        GROUP BY topic
                        HAVING count(*) >= :min_docs
                    ),
                    hub_docs AS (
                        SELECT DISTINCT ON (topic)
                            topic,
                            d.id                             AS hub_doc_id,
                            d.answers_questions              AS hub_aqs,
                            d.topics                         AS hub_topics,
                            COALESCE(e.edge_count, 0)        AS edge_count
                        FROM topic_groups tg
                        JOIN LATERAL (
                            SELECT id, answers_questions, topics
                            FROM documents
                            WHERE status = 'ready'
                              AND topics @> jsonb_build_array(tg.topic)
                        ) d ON true
                        LEFT JOIN (
                            SELECT source_id AS doc_id,
                                   count(*) AS edge_count
                            FROM document_edges
                            GROUP BY source_id
                        ) e ON e.doc_id = d.id
                        ORDER BY topic, edge_count DESC
                    )
                    SELECT
                        tg.topic                             AS label,
                        hd.hub_doc_id,
                        hd.hub_aqs,
                        hd.hub_topics,
                        tg.doc_count,
                        tg.earliest_doc,
                        tg.latest_doc,
                        -- coverage_confidence: penalise tiny clusters
                        LEAST(1.0, tg.doc_count::float / 50.0) AS coverage_confidence
                    FROM topic_groups tg
                    JOIN hub_docs hd USING (topic)
                    ORDER BY tg.doc_count DESC
                    LIMIT 200
                """),
                            {"min_docs": min_docs},
                        )
                    )
                    .mappings()
                    .all()
                )

                if not rows:
                    return 0

                # Truncate and rebuild atomically
                await conn.execute(text("TRUNCATE capability_clusters"))
                for row in rows:
                    raw_aqs = row["hub_aqs"] or []
                    if isinstance(raw_aqs, str):
                        import json as _json

                        raw_aqs = _json.loads(raw_aqs)
                    sample_aqs = raw_aqs[:5]  # at most 5 sample questions

                    raw_topics = row["hub_topics"] or []
                    if isinstance(raw_topics, str):
                        import json as _json

                        raw_topics = _json.loads(raw_topics)

                    await conn.execute(
                        text("""
                        INSERT INTO capability_clusters
                            (label, topic_centroid, hub_doc_id, doc_count,
                             coverage_confidence, sample_aqs, earliest_doc, latest_doc)
                        VALUES
                            (:label, :topic_centroid, :hub_doc_id, :doc_count,
                             :coverage_confidence, :sample_aqs, :earliest_doc, :latest_doc)
                    """),
                        {
                            "label": row["label"],
                            "topic_centroid": raw_topics,
                            "hub_doc_id": str(row["hub_doc_id"]) if row["hub_doc_id"] else None,
                            "doc_count": row["doc_count"],
                            "coverage_confidence": float(row["coverage_confidence"]),
                            "sample_aqs": sample_aqs,
                            "earliest_doc": row["earliest_doc"],
                            "latest_doc": row["latest_doc"],
                        },
                    )

                return len(rows)
        except Exception:
            return 0

    async def probe_capabilities(
        self,
        context: str,
        max_clusters: int = 5,
    ) -> list[dict]:  # type: ignore[type-arg]
        """
        Query the capability_clusters index for a given topic context.
        Returns matching clusters ordered by relevance (FTS + doc_count).
        """
        async with self._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        text("""
                SELECT
                    label,
                    hub_doc_id,
                    doc_count,
                    coverage_confidence,
                    sample_aqs,
                    topic_centroid,
                    earliest_doc,
                    latest_doc,
                    ts_rank_cd(
                        to_tsvector('english', label || ' ' || array_to_string(topic_centroid, ' ')),
                        plainto_tsquery('english', :context)
                    ) AS relevance
                FROM capability_clusters
                WHERE to_tsvector('english', label || ' ' || array_to_string(topic_centroid, ' '))
                    @@ plainto_tsquery('english', :context)
                ORDER BY relevance DESC, doc_count DESC
                LIMIT :max_clusters
            """),
                        {"context": context, "max_clusters": max_clusters},
                    )
                )
                .mappings()
                .all()
            )

            return [dict(r) for r in rows]

    async def get_gap_report(
        self,
        topic_filter: list[str] | None = None,
        min_gap_severity: str = "minor",
        as_of: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        """
        Proactive gap analysis of the entire corpus.

        Queries the ``capability_clusters`` table to identify topic areas with
        insufficient coverage (gaps) versus areas with strong coverage.

        Severity levels:
            major  = no docs in cluster
            moderate  = <5 docs or no docs after as_of date
            minor  = thin coverage (coverage_confidence < 0.3)

        Returns structured data suitable for compliance/audit export.
        """
        try:
            from datetime import date, datetime

            # Total ready corpus doc count
            async with self._engine.connect() as conn:
                total_result = await conn.execute(
                    text("SELECT count(*) AS cnt FROM documents WHERE status = 'ready'")
                )
                total_docs = total_result.scalar() or 0

            # Build WHERE clause for optional topic_filter
            where_clauses = ["1 = 1"]
            params: dict = {}

            if topic_filter:
                conditions = []
                for i, topic in enumerate(topic_filter):
                    conditions.append(f"label ILIKE :tf{i}")
                    params[f"tf{i}"] = f"%{topic}%"
                where_clauses.append(f"({' OR '.join(conditions)})")

            where_clause = " AND ".join(where_clauses)

            # Fetch clusters
            async with self._engine.connect() as conn:
                rows = await conn.execute(
                    text(f"""
                        SELECT label, doc_count, coverage_confidence,
                               earliest_doc, latest_doc, sample_aqs
                        FROM capability_clusters
                        WHERE {where_clause}
                        ORDER BY doc_count DESC
                    """),
                    params,
                )
                clusters = [dict(r) for r in rows.fetchall()]

            # Classify each cluster as gap or strong area
            gaps: list[dict] = []
            strong_areas: list[str] = []
            severity_order = {"major": 0, "moderate": 1, "minor": 2}

            as_of_date = None
            if as_of:
                try:
                    as_of_date = date.fromisoformat(as_of)
                except (ValueError, TypeError):
                    pass

            for c in clusters:
                label = c["label"]
                doc_count = c["doc_count"]
                confidence = float(c["coverage_confidence"])
                latest_doc = c.get("latest_doc")

                severity = None
                evidence_parts: list[str] = []

                if doc_count == 0:
                    severity = "major"
                    evidence_parts.append("No documents in this area")
                elif doc_count < 5:
                    severity = "moderate"
                    evidence_parts.append(
                        f"Only {doc_count} documents — thin coverage"
                    )
                elif as_of_date and latest_doc:
                    try:
                        latest_date = (
                            latest_doc.date()
                            if hasattr(latest_doc, "date")
                            else date.fromisoformat(str(latest_doc)[:10])
                        )
                        if latest_date < as_of_date:
                            severity = "moderate"
                            evidence_parts.append(
                                f"No documents dated after {as_of_date} "
                                f"(latest: {latest_date})"
                            )
                    except (ValueError, TypeError):
                        pass

                if confidence < 0.3 and severity is None:
                    severity = "minor"
                    evidence_parts.append(
                        f"Low coverage confidence ({confidence:.2f})"
                    )

                if severity is not None:
                    gaps.append(
                        {
                            "topic": label,
                            "severity": severity,
                            "evidence": "; ".join(evidence_parts),
                            "doc_count_in_area": doc_count,
                            "recommendation": (
                                f"Ingest more content on '{label}' to improve "
                                "corpus coverage"
                            ),
                        }
                    )
                else:
                    strong_areas.append(label)

            # Apply min_gap_severity filter
            min_sev_val = severity_order.get(min_gap_severity, 2)
            gaps = [g for g in gaps if severity_order.get(g["severity"], 3) <= min_sev_val]

            # Determine overall coverage_summary
            if not clusters:
                coverage_summary = "empty"
            elif len(gaps) == 0 and strong_areas or len(gaps) < len(clusters) * 0.2:
                coverage_summary = "deep"
            elif len(gaps) < len(clusters) * 0.5:
                coverage_summary = "moderate"
            elif len(gaps) < len(clusters) * 0.8:
                coverage_summary = "sparse"
            else:
                coverage_summary = "none"

            return {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "corpus_doc_count": total_docs,
                "coverage_summary": coverage_summary,
                "gaps": gaps,
                "strong_areas": strong_areas,
                "gap_count": len(gaps),
            }
        except Exception:
            return {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "corpus_doc_count": 0,
                "coverage_summary": "error",
                "gaps": [],
                "strong_areas": [],
                "gap_count": 0,
            }

    # ── Workspace / corpus management ─────────────────────────────────────────

    async def get_workspaces(self, parent_id: UUID | None = None) -> list[dict]:  # type: ignore[type-arg]
        """Return all workspaces, optionally filtered by parent."""
        async with self._engine.connect() as conn:
            if parent_id is not None:
                rows = await conn.execute(
                    text("SELECT id, name, parent_id, sharing_tier, created_at FROM workspaces WHERE parent_id = :pid ORDER BY name"),
                    {"pid": str(parent_id)},
                )
            else:
                rows = await conn.execute(
                    text("SELECT id, name, parent_id, sharing_tier, created_at FROM workspaces ORDER BY name")
                )
        return [dict(r._mapping) for r in rows.fetchall()]

    async def create_workspace(
        self,
        name: str,
        parent_id: UUID | None = None,
        sharing_tier: str = "internal_only",
    ) -> dict:  # type: ignore[type-arg]
        """Create a new workspace. Returns the created row."""
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    text("""
                        INSERT INTO workspaces (name, parent_id, sharing_tier)
                        VALUES (:name, :parent_id, :sharing_tier)
                        RETURNING id, name, parent_id, sharing_tier, created_at
                    """),
                    {"name": name, "parent_id": str(parent_id) if parent_id else None, "sharing_tier": sharing_tier},
                )
            ).fetchone()
        return dict(row._mapping)

    async def delete_workspace(self, workspace_id: UUID) -> None:
        """Delete a workspace (cascades to corpora)."""
        async with self._engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM workspaces WHERE id = :id"),
                {"id": str(workspace_id)},
            )

    async def get_corpora(self, workspace_id: UUID | None = None) -> list[dict]:  # type: ignore[type-arg]
        """Return all corpora, optionally filtered by workspace."""
        async with self._engine.connect() as conn:
            if workspace_id is not None:
                rows = await conn.execute(
                    text("SELECT id, name, slug, workspace_id, sharing_tier, created_at FROM corpora WHERE workspace_id = :wid ORDER BY name"),
                    {"wid": str(workspace_id)},
                )
            else:
                rows = await conn.execute(
                    text("SELECT id, name, slug, workspace_id, sharing_tier, created_at FROM corpora ORDER BY name")
                )
        return [dict(r._mapping) for r in rows.fetchall()]

    async def create_corpus(
        self,
        name: str,
        slug: str,
        workspace_id: UUID,
        sharing_tier: str = "internal_only",
    ) -> dict:  # type: ignore[type-arg]
        """Create a new corpus. Returns the created row."""
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    text("""
                        INSERT INTO corpora (name, slug, workspace_id, sharing_tier)
                        VALUES (:name, :slug, :workspace_id, :sharing_tier)
                        RETURNING id, name, slug, workspace_id, sharing_tier, created_at
                    """),
                    {"name": name, "slug": slug, "workspace_id": str(workspace_id), "sharing_tier": sharing_tier},
                )
            ).fetchone()
        return dict(row._mapping)

    async def delete_corpus(self, corpus_id: UUID) -> None:
        """Delete a corpus. Documents referencing it have their corpus_id set to NULL."""
        async with self._engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM corpora WHERE id = :id"),
                {"id": str(corpus_id)},
            )

    # ── Multi-source registry (Issue #386) ──────────────────────────────────

    @staticmethod
    def _normalize_source_row(row: dict) -> dict:  # type: ignore[type-arg]
        cfg_raw = row.get("config_json")
        if isinstance(cfg_raw, str):
            try:
                cfg = json.loads(cfg_raw)
            except Exception:
                cfg = {}
        else:
            cfg = cfg_raw or {}

        return {
            "id": row.get("id"),
            "name": row.get("name"),
            "type": row.get("type"),
            "config": cfg,
            "enabled": bool(row.get("enabled", True)),
            "created_by": row.get("created_by"),
            "created_at": row.get("created_at"),
            "tested_at": row.get("tested_at"),
            "test_status": row.get("test_status"),
            "test_error": row.get("test_error"),
            "updated_at": row.get("updated_at"),
        }

    async def list_sources(self) -> list[dict]:  # type: ignore[type-arg]
        """Return all configured sources."""
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        """
                        SELECT id, name, type, config_json, enabled, created_by,
                               created_at, tested_at, test_status, test_error, updated_at
                        FROM dewie_sources
                        ORDER BY created_at DESC
                        """
                    )
                )
            ).mappings().fetchall()
        return [self._normalize_source_row(dict(r)) for r in rows]

    async def get_source(self, source_id: UUID | str) -> dict | None:  # type: ignore[type-arg]
        """Return a single source by id."""
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        """
                        SELECT id, name, type, config_json, enabled, created_by,
                               created_at, tested_at, test_status, test_error, updated_at
                        FROM dewie_sources
                        WHERE id = :id
                        LIMIT 1
                        """
                    ),
                    {"id": str(source_id)},
                )
            ).mappings().fetchone()
        return self._normalize_source_row(dict(row)) if row else None

    async def create_source(
        self,
        *,
        source_id: UUID,
        name: str,
        source_type: str,
        config: dict,
        enabled: bool = True,
        created_by: UUID | None = None,
    ) -> dict:  # type: ignore[type-arg]
        """Create a source and return the created row."""
        # asyncpg JSONB codec requires a JSON string, not a dict
        cfg_json = json.dumps(config)

        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        """
                        INSERT INTO dewie_sources
                            (id, name, type, config_json, enabled, created_by, created_at, updated_at)
                        VALUES
                            (:id, :name, :type, :config_json, :enabled, :created_by, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        RETURNING id, name, type, config_json, enabled, created_by,
                                  created_at, tested_at, test_status, test_error, updated_at
                        """
                    ),
                    {
                        "id": str(source_id),
                        "name": name,
                        "type": source_type,
                        "config_json": cfg_json,
                        "enabled": enabled,
                        "created_by": str(created_by) if created_by else None,
                    },
                )
            ).mappings().fetchone()
        if row is None:
            raise RuntimeError("Failed to create source")
        return self._normalize_source_row(dict(row))

    async def update_source(
        self,
        source_id: UUID | str,
        *,
        name: str | None = None,
        source_type: str | None = None,
        config: dict | None = None,
        enabled: bool | None = None,
    ) -> dict | None:  # type: ignore[type-arg]
        """Update mutable source fields and return updated row; None if not found."""
        updates: dict[str, object] = {}
        if name is not None:
            updates["name"] = name
        if source_type is not None:
            updates["type"] = source_type
        if config is not None:
            updates["config_json"] = json.dumps(config)
        if enabled is not None:
            updates["enabled"] = enabled

        if updates:
            set_clause = ", ".join(f"{k} = :{k}" for k in updates)
            sql = text(
                f"UPDATE dewie_sources SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = :id"
            )
            async with self._engine.begin() as conn:
                result = await conn.execute(sql, {"id": str(source_id), **updates})
                if int(result.rowcount or 0) == 0:
                    return None

        return await self.get_source(source_id)

    async def delete_source(self, source_id: UUID | str) -> bool:
        """Delete a source by id."""
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text("DELETE FROM dewie_sources WHERE id = :id"),
                {"id": str(source_id)},
            )
        return int(result.rowcount or 0) > 0

    async def set_source_test_result(
        self,
        source_id: UUID | str,
        *,
        ok: bool,
        error: str | None,
    ) -> bool:
        """Persist test status for a source."""
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                    UPDATE dewie_sources
                    SET tested_at = CURRENT_TIMESTAMP,
                        test_status = :test_status,
                        test_error = :test_error,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :id
                    """
                ),
                {
                    "id": str(source_id),
                    "test_status": "ok" if ok else "error",
                    "test_error": error,
                },
            )
        return int(result.rowcount or 0) > 0

    async def upsert_source(
        self,
        *,
        name: str,
        source_type: str,
        config: dict,
        enabled: bool = True,
        validate: bool = False,
    ) -> dict:  # type: ignore[type-arg]
        """Idempotently create/update a source by name for startup seeding."""
        source_id = uuid.uuid4()
        if getattr(self, "_is_sqlite", False):
            cfg_value: str | dict = json.dumps(config)
            upsert_sql = text(
                """
                INSERT INTO dewie_sources (id, name, type, config_json, enabled, created_at, updated_at)
                VALUES (:id, :name, :type, :config_json, :enabled, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(name) DO UPDATE SET
                    type = excluded.type,
                    config_json = excluded.config_json,
                    enabled = excluded.enabled,
                    updated_at = CURRENT_TIMESTAMP
                """
            )
        else:
            cfg_value = config
            upsert_sql = text(
                """
                INSERT INTO dewie_sources (id, name, type, config_json, enabled, created_at, updated_at)
                VALUES (:id, :name, :type, CAST(:config_json AS jsonb), :enabled, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(name) DO UPDATE SET
                    type = excluded.type,
                    config_json = excluded.config_json,
                    enabled = excluded.enabled,
                    updated_at = CURRENT_TIMESTAMP
                """
            )

        cfg_json = json.dumps(cfg_value) if isinstance(cfg_value, dict) else cfg_value

        async with self._engine.begin() as conn:
            await conn.execute(
                upsert_sql,
                {
                    "id": str(source_id),
                    "name": name,
                    "type": source_type,
                    "config_json": cfg_json,
                    "enabled": enabled,
                },
            )

            row = (
                await conn.execute(
                    text(
                        """
                        SELECT id, name, type, config_json, enabled, created_by,
                               created_at, tested_at, test_status, test_error, updated_at
                        FROM dewie_sources
                        WHERE name = :name
                        LIMIT 1
                        """
                    ),
                    {"name": name},
                )
            ).mappings().fetchone()

        if row is None:
            raise RuntimeError("Failed to upsert source")
        
        normalized_row = self._normalize_source_row(dict(row))

        if validate:
            ok, error = True, None
            if source_type == "postgres":
                ok, error = await self._test_postgres_connection(config)
            elif source_type == "mcp":
                ok, error = await self._test_mcp_connection(config)
            elif source_type == "sqlite":
                pass
            else:
                validate = False
            
            if validate:
                await self.set_source_test_result(normalized_row["id"], ok=ok, error=error)

        return normalized_row

    async def seed_public_sources(self, defaults: list[dict]) -> dict:  # type: ignore[type-arg]
        """Idempotently seed source defaults at startup."""
        seeded = 0
        for item in defaults:
            name = str(item.get("name", "")).strip()
            source_type = str(item.get("type", "")).strip()
            if not name or not source_type:
                continue
            config = item.get("config") if isinstance(item.get("config"), dict) else {}
            enabled = bool(item.get("enabled", True))
            await self.upsert_source(name=name, source_type=source_type, config=config, enabled=enabled, validate=True)
            seeded += 1
        return {"seeded": seeded}

    # ── Local user management ──────────────────────────────────────────────────

    async def get_local_users(self) -> list[dict]:  # type: ignore[type-arg]
        """List local users for admin management."""
        async with self._engine.connect() as conn:
            rows = await conn.execute(
                text(
                    """
                    SELECT
                        id,
                        email,
                        name,
                        is_admin,
                        activation_status,
                        created_at,
                        last_login_at,
                        CASE
                            WHEN password_hash IS NULL OR password_hash = '' THEN false
                            ELSE true
                        END AS has_password
                    FROM users
                    ORDER BY created_at DESC, email ASC
                    """
                )
            )
        return [dict(r._mapping) for r in rows.fetchall()]

    async def update_local_user(
        self,
        user_id: UUID | str,
        *,
        name: str | None = None,
        is_admin: bool | None = None,
        activation_status: str | None = None,
        password_hash: str | None = None,
    ) -> dict | None:  # type: ignore[type-arg]
        """Update mutable local-user fields and return the updated user."""
        updates: dict[str, object] = {}
        if name is not None:
            updates["name"] = name
        if is_admin is not None:
            updates["is_admin"] = is_admin
        if activation_status is not None:
            updates["activation_status"] = activation_status
        if password_hash is not None:
            updates["password_hash"] = password_hash

        if updates:
            set_clause = ", ".join(f"{key} = :{key}" for key in updates)
            params = {"user_id": str(user_id), **updates}
            async with self._engine.begin() as conn:
                result = await conn.execute(
                    text(f"UPDATE users SET {set_clause} WHERE id = :user_id"),
                    params,
                )
                if int(result.rowcount or 0) == 0:
                    return None

        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        """
                        SELECT
                            id,
                            email,
                            name,
                            is_admin,
                            activation_status,
                            created_at,
                            last_login_at,
                            CASE
                                WHEN password_hash IS NULL OR password_hash = '' THEN false
                                ELSE true
                            END AS has_password
                        FROM users
                        WHERE id = :user_id
                        LIMIT 1
                        """
                    ),
                    {"user_id": str(user_id)},
                )
            ).mappings().fetchone()
        return dict(row) if row else None

    async def delete_local_user(self, user_id: UUID | str) -> bool:
        """Delete a local user by ID."""
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text("DELETE FROM users WHERE id = :user_id"),
                {"user_id": str(user_id)},
            )
        return int(result.rowcount or 0) > 0

    async def get_or_create_default_corpus(self) -> dict:  # type: ignore[type-arg]
        """Return (or lazily create) the default corpus for the root workspace."""
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT id, name, slug, workspace_id, sharing_tier, created_at FROM corpora WHERE id = :id"),
                    {"id": str(DEFAULT_CORPUS_ID)},
                )
            ).fetchone()
        if row:
            return dict(row._mapping)
        # Shouldn't happen after init_schema, but be safe
        return await self.create_corpus(
            name="Default",
            slug="default",
            workspace_id=ROOT_WORKSPACE_ID,
        )

    async def upsert(self, doc: ContentDocument) -> None:
        """Insert or update a document record."""
        search_text = " ".join(
            filter(
                None,
                [
                    doc.title,
                    doc.summary,
                    " ".join(doc.topics),
                    " ".join(doc.keywords),
                    " ".join(doc.entities),
                ],
            )
        )
        if getattr(self, "_is_sqlite", False):
            sql = text("""
                INSERT INTO documents
                    (id, url, title, summary, embed_summary, source, ingested_at,
                     status, topics, keywords, entities, sentiment, crawl_session,
                     search_vec, enrichment_version, embedding_model, enriched_at,
                     author, reading_level, document_type, tone, answers_questions, aq_tsvec,
                     published_at, paywall_detected, paywall_type, alternate_terms,
                     enrichment_quality_score, retain_body, instance_id)
                VALUES
                    (:id, :url, :title, :summary, :embed_summary, :source, :ingested_at,
                     :status, :topics, :keywords, :entities, :sentiment,
                     :crawl_session, :search_text,
                     :enrichment_version, :embedding_model, :enriched_at,
                     :author, :reading_level, :document_type, :tone, :answers_questions,
                     :answers_questions_text,
                     :published_at, :paywall_detected, :paywall_type, :alternate_terms,
                     :enrichment_quality_score, :retain_body, :instance_id)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    summary = CASE
                        WHEN excluded.summary IS NOT NULL AND excluded.summary != '' THEN excluded.summary
                        ELSE documents.summary
                    END,
                    embed_summary = CASE
                        WHEN excluded.embed_summary IS NOT NULL AND excluded.embed_summary != ''
                        THEN excluded.embed_summary
                        ELSE documents.embed_summary
                    END,
                    status = CASE
                        WHEN documents.status = 'ready' THEN documents.status
                        ELSE excluded.status
                    END,
                    topics = excluded.topics,
                    keywords = excluded.keywords,
                    entities = excluded.entities,
                    sentiment = excluded.sentiment,
                    crawl_session = excluded.crawl_session,
                    search_vec = excluded.search_vec,
                    enrichment_version = documents.enrichment_version + 1,
                    embedding_model = excluded.embedding_model,
                    enriched_at = excluded.enriched_at,
                    author = excluded.author,
                    reading_level = excluded.reading_level,
                    document_type = excluded.document_type,
                    tone = excluded.tone,
                    answers_questions = excluded.answers_questions,
                    aq_tsvec = excluded.aq_tsvec,
                    published_at = COALESCE(documents.published_at, excluded.published_at),
                    paywall_detected = excluded.paywall_detected,
                    paywall_type = excluded.paywall_type,
                    alternate_terms = excluded.alternate_terms,
                    enrichment_quality_score = excluded.enrichment_quality_score,
                    retain_body = excluded.retain_body,
                    instance_id = excluded.instance_id
            """)
        else:
            # search_vec: written once here in Pass A upsert; Pass B (pipeline.enrich_docs) was removed in 4b695ee
           sql = text("""
                INSERT INTO documents
                    (id, url, title, summary, embed_summary, source, ingested_at,
                     status, topics, keywords, entities, sentiment, crawl_session,
                     search_vec, enrichment_version, embedding_model, enriched_at,
                     author, reading_level, document_type, tone, answers_questions, aq_tsvec,
                     published_at, paywall_detected, paywall_type, alternate_terms,
                     enrichment_quality_score, retain_body, instance_id)
                VALUES
                    (:id, :url, :title, :summary, :embed_summary, :source, :ingested_at,
                     :status, CAST(:topics AS jsonb), CAST(:keywords AS jsonb), CAST(:entities AS jsonb), :sentiment,
                     :crawl_session, to_tsvector('english', :search_text),
                     :enrichment_version, :embedding_model, :enriched_at,
                     :author, :reading_level, :document_type, :tone, CAST(:answers_questions AS jsonb),
                     to_tsvector('english', coalesce(:answers_questions_text, '')),
                     :published_at, :paywall_detected, :paywall_type, CAST(:alternate_terms AS jsonb),
                     :enrichment_quality_score, :retain_body, :instance_id)
                ON CONFLICT (url) DO UPDATE SET
                    title               = EXCLUDED.title,
                    summary             = CASE WHEN EXCLUDED.summary IS NOT NULL AND EXCLUDED.summary != '' THEN EXCLUDED.summary ELSE documents.summary END,
                    embed_summary       = CASE WHEN EXCLUDED.embed_summary IS NOT NULL AND EXCLUDED.embed_summary != '' THEN EXCLUDED.embed_summary ELSE documents.embed_summary END,
                    status              = CASE
                                               WHEN documents.status = 'ready' THEN documents.status
                                               ELSE EXCLUDED.status
                                           END,
                    topics              = EXCLUDED.topics,
                    keywords            = EXCLUDED.keywords,
                    entities            = EXCLUDED.entities,
                    sentiment           = EXCLUDED.sentiment,
                    crawl_session       = EXCLUDED.crawl_session,
                    search_vec          = EXCLUDED.search_vec,
                    enrichment_version  = documents.enrichment_version + 1,
                    embedding_model     = EXCLUDED.embedding_model,
                    enriched_at         = EXCLUDED.enriched_at,
                    author              = EXCLUDED.author,
                    reading_level       = EXCLUDED.reading_level,
                    document_type       = EXCLUDED.document_type,
                    tone                = EXCLUDED.tone,
                    answers_questions   = EXCLUDED.answers_questions,
                    aq_tsvec           = EXCLUDED.aq_tsvec,
                    published_at        = COALESCE(documents.published_at, EXCLUDED.published_at),
                    paywall_detected    = EXCLUDED.paywall_detected,
                    paywall_type        = EXCLUDED.paywall_type,
                    alternate_terms     = EXCLUDED.alternate_terms,
                    enrichment_quality_score       = EXCLUDED.enrichment_quality_score,
                    retain_body         = EXCLUDED.retain_body,
                    instance_id         = EXCLUDED.instance_id
            """)
        async with self._session_factory() as session:
            await session.execute(
                sql,
                {
                     "id": str(doc.id),
                     "url": doc.url,
                     "title": doc.title,
                     "summary": doc.summary,
                     "embed_summary": doc.embed_summary,
                     "source": doc.source,
                     "ingested_at": doc.ingested_at,
                     "status": doc.status.value,
                     "topics": json.dumps(doc.topics),
                     "keywords": json.dumps(doc.keywords),
                     "entities": json.dumps(doc.entities),
                     "sentiment": doc.sentiment,
                     "crawl_session": str(doc.crawl_session) if doc.crawl_session else None,
                     "search_text": search_text,
                     "enrichment_version": doc.enrichment_version,
                     "embedding_model": doc.embedding_model,
                     "enriched_at": doc.enriched_at,
                     "author": doc.author,
                     "reading_level": doc.reading_level.value if doc.reading_level else None,
                     "document_type": doc.document_type.value if doc.document_type else None,
                     "tone": doc.tone,
                     "answers_questions": json.dumps(doc.answers_questions),
                     "answers_questions_text": " ".join(doc.answers_questions)
                     if doc.answers_questions
                     else "",
                     "published_at": doc.published_at if hasattr(doc, "published_at") else None,
                     "paywall_detected": getattr(doc, "paywall_detected", False),
                     "paywall_type": getattr(doc, "paywall_type", "none"),
                     "alternate_terms": json.dumps(getattr(doc, "alternate_terms", [])),
                     "enrichment_quality_score": getattr(doc, "enrichment_quality_score", None),
                     "retain_body": getattr(doc, "retain_body", False),
                     "instance_id": next((x for x in [doc.instance_id, settings.instance_id] if x and x.strip()), None),
                 },
            )
            await session.commit()

    async def upsert_aq_embeddings(
        self, doc_id: str, aq_pairs: list[tuple[str, list[float]]]
    ) -> None:
        """
        Store per-AQ embeddings for a document.

        aq_pairs: list of (aq_text, embedding_vector) — one per AQ string.

        Deletes existing rows for the doc then inserts fresh ones so that
        re-enrichment always overwrites stale embeddings cleanly.
        """
        if not aq_pairs:
            return
        async with self._engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM document_aq WHERE doc_id = cast(:doc_id as uuid)"),
                {"doc_id": doc_id},
            )
            for aq_text, vec in aq_pairs:
                await conn.execute(
                    text("""
                        INSERT INTO document_aq (doc_id, aq_text, embedding)
                        VALUES (cast(:doc_id as uuid), :aq_text, cast(:vec as vector))
                    """),
                    {"doc_id": doc_id, "aq_text": aq_text, "vec": json.dumps(vec)},
                )

    async def get_by_id(self, doc_id: UUID) -> ContentDocument | None:
        """Retrieve a single document by primary key."""
        sql = text("SELECT * FROM documents WHERE id = :id")
        async with self._session_factory() as session:
            row = (await session.execute(sql, {"id": str(doc_id)})).mappings().first()
        return _row_to_doc(row) if row else None

    async def search(
        self,
        query: str,
        limit: int = 10,
        ranker: str = "rrf",
        min_enrichment_quality_score: int | None = None,
        exclude_reading_levels: list[str] | None = None,
        workspace_ids: list[UUID] | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        source_filter: list[str] | None = None,
    ) -> list[tuple[ContentDocument, float]]:
        """
        Hybrid search using a pluggable ranker strategy.
        Default ranker is 'rrf' (Reciprocal Rank Fusion, k=60).
        See dewie.storage.rankers for available strategies.
        Returns list of (doc, score) tuples.

        Args:
            min_enrichment_quality_score: Exclude docs with enrichment_quality_score < this value.
                               None or -1 = no filter (pass-through).
                               Docs with NULL enrichment_quality_score are always included
                               (not yet enriched with quality signal).
            exclude_reading_levels: List of reading_level values to exclude.
                               E.g. ["quick_read"] drops thin short-form content.
                               None or [] = no filter.
            workspace_ids: Restrict results to docs in corpora belonging to these workspaces.
                           None or empty = all workspaces (no restriction).
            published_after: ISO date string (YYYY-MM-DD). Exclude docs with ingested_at < this date.
            published_before: ISO date string (YYYY-MM-DD). Exclude docs with ingested_at > this date.
        """
        if self._is_sqlite:
            # Per-token scoring via rankers._fts_sqlite — matching the whole query
            # as one LIKE phrase made multi-word queries unfindable.
            from dewie.storage.rankers import _fts_sqlite, _sqlite_in_clause

            async with self._session_factory() as session:
                scored = await _fts_sqlite(session, query, limit)
                if not scored:
                    return []
                placeholders, params = _sqlite_in_clause([doc_id for doc_id, _ in scored])
                rows = (
                    (
                        await session.execute(
                            text(f"SELECT * FROM documents WHERE id IN ({placeholders})"),
                            params,
                        )
                    )
                    .mappings()
                    .all()
                )
            by_id = {str(r["id"]): r for r in rows}
            results = [
                (_row_to_doc(by_id[doc_id]), score)
                for doc_id, score in scored
                if doc_id in by_id
            ]
            if source_filter:
                sf_set = {s.lower() for s in source_filter}
                results = [(doc, score) for doc, score in results if (doc.source or "").lower() in sf_set]
            return results

        from dewie.storage.rankers import RANKER_REGISTRY, run_ranker

        # Fall back to rrf if unknown ranker requested
        if ranker not in RANKER_REGISTRY:
            ranker = "rrf"

        # Expand query with corpus alternate_terms before embedding
        async with self._session_factory() as session:
            expanded_query = await _expand_query_with_session(query, session)

        embedding = await _get_embedding(expanded_query)

        async with self._session_factory() as session:
            scored = await run_ranker(ranker, expanded_query, session, embedding, limit)

        if not scored:
            return []

        top_ids = [doc_id for doc_id, _ in scored[:limit]]
        score_map = {doc_id: score for doc_id, score in scored}

        # Workspace filter — resolve corpus → workspace membership
        ws_ids_str = [str(w) for w in (workspace_ids or [])]

        # Source filter — applied after ranking to avoid modifying every ranker
        if source_filter:
            async with self._session_factory() as session:
                from sqlalchemy import text as _sf_text

                sf_rows = (
                    await session.execute(
                        _sf_text(
                            "SELECT id::text FROM documents WHERE id = ANY(:ids) AND source = ANY(:sources)"
                        ),
                        {"ids": top_ids, "sources": source_filter},
                    )
                ).fetchall()
                allowed = {r[0] for r in sf_rows}
                top_ids = [d for d in top_ids if d in allowed]
                score_map = {k: v for k, v in score_map.items() if k in allowed}

        # Build quality + date + workspace filter clauses for the fetch query
        quality_clauses = []
        quality_params: dict = {"ids": top_ids}

        if min_enrichment_quality_score is not None and min_enrichment_quality_score >= 0:
            # NULL = not yet scored → include (don't penalise unenriched docs)
            quality_clauses.append(
                "(d.enrichment_quality_score IS NULL OR d.enrichment_quality_score >= :min_qs)"
            )
            quality_params["min_qs"] = min_enrichment_quality_score

        if exclude_reading_levels:
            quality_clauses.append("(d.reading_level IS NULL OR d.reading_level != ALL(:excl_rl))")
            quality_params["excl_rl"] = exclude_reading_levels

        if published_after:
            quality_clauses.append("d.ingested_at >= :pub_after::date")
            quality_params["pub_after"] = published_after

        if published_before:
            quality_clauses.append("d.ingested_at <= :pub_before::date")
            quality_params["pub_before"] = published_before

        # Workspace restriction + sharing tier enforcement via corpus JOIN.
        #   public       → visible to all authenticated users
        #   internal_only / private → only within their workspace
        ws_join = ""
        if ws_ids_str:
            ws_join = "JOIN corpora c ON d.corpus_id = c.id"
            quality_clauses.append(
                "(d.sharing_tier = 'public' OR c.workspace_id = ANY(:ws_ids::uuid[]))"
            )
            quality_params["ws_ids"] = ws_ids_str

        quality_filter_sql = ""
        if quality_clauses:
            quality_filter_sql = " AND " + " AND ".join(quality_clauses)

        async with self._session_factory() as session:
            sql_fetch = text(
                f"SELECT d.* FROM documents d {ws_join} WHERE d.id = ANY(:ids) AND d.status = 'ready'{quality_filter_sql}"
            )
            result_rows = (await session.execute(sql_fetch, quality_params)).mappings().all()

        doc_map = {str(r["id"]): r for r in result_rows}
        return [
            (_row_to_doc(doc_map[doc_id]), round(score_map[doc_id], 6))
            for doc_id in top_ids
            if doc_id in doc_map
        ]

    async def find_by_topics(self, topics: list[str], limit: int = 20) -> list[ContentDocument]:
        """Retrieve documents that share at least one of the given topics."""
        if not topics:
            return []
        sql = text("""
            SELECT * FROM documents
            WHERE status = 'ready'
              AND topics ?| ARRAY(
                  SELECT jsonb_array_elements_text(CAST(:topics AS jsonb))
              )
            LIMIT :limit
        """)
        async with self._session_factory() as session:
            rows = (
                (await session.execute(sql, {"topics": json.dumps(topics), "limit": limit}))
                .mappings()
                .all()
            )
        return [_row_to_doc(r) for r in rows]

    async def find_by_entities(self, entities: list[str], limit: int = 20) -> list[ContentDocument]:
        """Retrieve documents sharing any of the given named entities."""
        if not entities:
            return []
        sql = text("""
            SELECT * FROM documents
            WHERE status = 'ready'
              AND entities ?| ARRAY(
                  SELECT jsonb_array_elements_text(CAST(:entities AS jsonb))
              )
            LIMIT :limit
        """)
        async with self._session_factory() as session:
            rows = (
                (await session.execute(sql, {"entities": json.dumps(entities), "limit": limit}))
                .mappings()
                .all()
            )
        return [_row_to_doc(r) for r in rows]

    async def find_by_keywords(self, keywords: list[str], limit: int = 20) -> list[ContentDocument]:
        """Retrieve documents sharing any of the given keywords."""
        if not keywords:
            return []
        sql = text("""
            SELECT * FROM documents
            WHERE status = 'ready'
              AND keywords ?| ARRAY(
                  SELECT jsonb_array_elements_text(CAST(:keywords AS jsonb))
              )
            LIMIT :limit
        """)
        async with self._session_factory() as session:
            rows = (
                (await session.execute(sql, {"keywords": json.dumps(keywords), "limit": limit}))
                .mappings()
                .all()
            )
        return [_row_to_doc(r) for r in rows]

    async def list_recent(self, limit: int = 50, offset: int = 0) -> list[ContentDocument]:
        """Return the most recently ingested documents regardless of status."""
        sql = text("""
            SELECT * FROM documents
            ORDER BY ingested_at DESC
            LIMIT :limit OFFSET :offset
        """)
        async with self._session_factory() as session:
            rows = (await session.execute(sql, {"limit": limit, "offset": offset})).mappings().all()
        return [_row_to_doc(r) for r in rows]

    async def count_by_status(self) -> dict[str, int]:
        """Return a {status: count} breakdown across all documents."""
        sql = text("SELECT status, COUNT(*) AS n FROM documents GROUP BY status")
        async with self._session_factory() as session:
            rows = (await session.execute(sql)).mappings().all()
        return {r["status"]: int(r["n"]) for r in rows}

    async def list_crawl_sessions(self) -> list[dict]:  # type: ignore[type-arg]
        """Summarise crawl activity grouped by session UUID."""
        sql = text("""
            SELECT
                crawl_session,
                COUNT(*)                                        AS total,
                COUNT(*) FILTER (WHERE status = 'ready')       AS ready,
                COUNT(*) FILTER (WHERE status = 'processing')  AS processing,
                COUNT(*) FILTER (WHERE status = 'failed')      AS failed,
                MIN(ingested_at)                               AS started_at,
                MAX(ingested_at)                               AS last_seen_at
            FROM documents
            WHERE crawl_session IS NOT NULL
            GROUP BY crawl_session
            ORDER BY started_at DESC
        """)
        async with self._session_factory() as session:
            rows = (await session.execute(sql)).mappings().all()
        return [dict(r) for r in rows]

    async def mark_status(self, doc_id: UUID, status: ContentStatus) -> None:
        """Update the processing status of a document."""
        sql = text("UPDATE documents SET status = :status WHERE id = :id")
        async with self._session_factory() as session:
            await session.execute(sql, {"status": status.value, "id": str(doc_id)})
            await session.commit()

    async def upsert_relationship(self, rel: Relationship) -> None:
        """Insert or update an edge in document_edges, keeping the highest weight seen."""
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO document_edges
                        (source_id, target_id, rel_type, weight, shared_attrs)
                    VALUES
                        (:src, :tgt, :rel_type, :weight, :shared)
                    ON CONFLICT (source_id, target_id, rel_type)
                    DO UPDATE SET
                        weight       = GREATEST(document_edges.weight, EXCLUDED.weight),
                        shared_attrs = EXCLUDED.shared_attrs
                """),
                {
                    "src": str(rel.source_id),
                    "tgt": str(rel.target_id),
                    "rel_type": rel.relationship_type.value,
                    "weight": rel.weight,
                    "shared": rel.shared_attributes,
                },
            )

    async def get_edge_count(self, doc_id: UUID) -> int:
        """Return the number of edges (in or out) for a document.
        Uses UNION ALL to allow index-only scans on both source and target indexes.
        The OR pattern prevents index use; UNION ALL uses both idx_edges_source_weight
        and idx_edges_target_weight.
        """
        sql = text("""
            SELECT COUNT(*) FROM (
                SELECT source_id FROM document_edges WHERE source_id = cast(:id AS uuid)
                UNION ALL
                SELECT target_id FROM document_edges WHERE target_id = cast(:id AS uuid)
            ) edges
        """)
        async with self._session_factory() as session:
            result = await session.execute(sql, {"id": str(doc_id)})
            return int(result.scalar() or 0)

    async def get_related(self, doc_id: UUID, rel_types: list[str], limit: int = 10) -> list[dict]:  # type: ignore[type-arg]
        """Get related docs from the document_edges table (bidirectional)."""
        sql = text("""
            SELECT neighbor_id AS id,
                   rel_type,
                   weight,
                   shared_terms AS shared,
                   title,
                   summary,
                   topics
            FROM (
                SELECT r.target_id AS neighbor_id, r.rel_type, r.weight, r.shared_terms,
                       d.title, d.summary, d.topics
                FROM relationships r
                JOIN documents d ON d.id = r.target_id
                WHERE r.source_id = :doc_id
                UNION ALL
                SELECT r.source_id AS neighbor_id, r.rel_type, r.weight, r.shared_terms,
                       d.title, d.summary, d.topics
                FROM relationships r
                JOIN documents d ON d.id = r.source_id
                WHERE r.target_id = :doc_id
            ) combined
            ORDER BY weight DESC
            LIMIT :limit
        """)
        async with self._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        sql,
                        {"doc_id": str(doc_id), "limit": limit},
                    )
                )
                .mappings()
                .all()
            )
        return [
            {
                "id": str(r["id"]),
                "rel_type": r["rel_type"],
                "weight": float(r["weight"]),
                "shared": list(r["shared"]) if r["shared"] else [],
                "title": r["title"],
                "summary": r["summary"],
                "topics": r["topics"] if isinstance(r["topics"], list) else json.loads(r["topics"]),
            }
            for r in rows
        ]

    async def set_embedding(self, doc_id: UUID, vector: list[float]) -> None:
        """Store a pgvector embedding for a document."""
        if getattr(self, "_is_sqlite", False):
            async with self._session_factory() as session:
                await session.execute(
                    text("UPDATE documents SET embedding = :vec WHERE id = :id"),
                    {"vec": json.dumps(vector), "id": str(doc_id)},
                )
                await session.commit()
            return

        async with self._session_factory() as session:
            await session.execute(
                text(
                    "UPDATE documents SET embedding = cast(:vec as vector)"
                    " WHERE id = cast(:id as uuid)"
                ),
                {"vec": json.dumps(vector), "id": str(doc_id)},
            )
            await session.commit()

    async def set_embedding_full(self, doc_id: UUID, vector: list[float]) -> None:
        """Store the untruncated embedding (pre embed_dimensions/MRL truncation).

        Only called when embed_store_full_vector is enabled and truncation
        actually happened — used for exact-precision reranking of ANN
        candidates, not for indexed search (no index on this column).
        """
        if getattr(self, "_is_sqlite", False):
            async with self._session_factory() as session:
                await session.execute(
                    text("UPDATE documents SET embedding_full = :vec WHERE id = :id"),
                    {"vec": json.dumps(vector), "id": str(doc_id)},
                )
                await session.commit()
            return

        async with self._session_factory() as session:
            await session.execute(
                text(
                    "UPDATE documents SET embedding_full = cast(:vec as vector)"
                    " WHERE id = cast(:id as uuid)"
                ),
                {"vec": json.dumps(vector), "id": str(doc_id)},
            )
            await session.commit()

    async def get_pending_docs(self, limit: int = 100) -> list[str]:
        """Atomically claim pending docs — marks them processing in one transaction.
        Uses FOR UPDATE SKIP LOCKED so multiple concurrent workers never collide.

        Excludes documents that are paywalled with no body text (terminal stubs),
        since they cannot be meaningfully enriched and clog the queue.
        """
        if getattr(self, "_is_sqlite", False):
            # SQLite: no FOR UPDATE SKIP LOCKED — single-worker mode
            sql = text(
                "SELECT id FROM documents WHERE status = 'pending'"
                " AND (paywall_detected = false"
                "   OR (paywall_detected = true AND body_text IS NOT NULL"
                "       AND length(trim(body_text)) >= 500))"
                " ORDER BY priority DESC, ingested_at ASC LIMIT :limit"
            )
            async with self._session_factory() as session:
                rows = (await session.execute(sql, {"limit": limit})).mappings().all()
            return [str(r["id"]) for r in rows]
        # PostgreSQL: atomically claim + mark processing in one CTE
        async with self._engine.begin() as conn:
            rows = (await conn.execute(text(
                "WITH claimed AS ("
                "  SELECT id FROM documents WHERE status = 'pending'"
                "    AND (paywall_detected = false"
                "      OR (paywall_detected = true AND body_text IS NOT NULL"
                "          AND length(trim(body_text)) >= 500))"
                "  ORDER BY priority DESC, ingested_at ASC LIMIT :limit FOR UPDATE SKIP LOCKED"
                ")"
                " UPDATE documents SET status = 'processing' FROM claimed"
                " WHERE documents.id = claimed.id RETURNING documents.id"
            ), {"limit": limit})).mappings().all()
        return [str(r["id"]) for r in rows]

    async def write_body_text(self, doc_id: UUID | str, body: str) -> None:
        """Best-effort: write body_text to the documents row (Issue #51)."""
        if getattr(self, "_is_sqlite", False):
            sql = text("UPDATE documents SET body_text = :body WHERE id = :id")
        else:
            sql = text("UPDATE documents SET body_text = :body WHERE id = CAST(:id AS UUID)")
        async with self._session_factory() as session:
            await session.execute(sql, {"body": body, "id": str(doc_id)})
            await session.commit()

    async def set_priority(self, doc_id: UUID, priority: int) -> None:
        """Set the queue priority for a document (0 = normal, higher = sooner)."""
        sql = text("UPDATE documents SET priority = :priority WHERE id = :id")
        async with self._session_factory() as session:
            await session.execute(sql, {"priority": priority, "id": str(doc_id)})
            await session.commit()

    async def close(self) -> None:
        """Dispose the connection pool."""
        await self._engine.dispose()

    @property
    def engine(self) -> AsyncEngine:
        """Expose the underlying SQLAlchemy engine for shared use."""
        return self._engine

    # ── Document chunk CRUD ───────────────────────────────────────────────────

    async def get_chunks(self, doc_id: UUID) -> list[dict]:  # type: ignore[type-arg]
        """Return all chunks for a document (text + index only, no embeddings)."""
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                    SELECT chunk_index, text
                    FROM document_chunks
                    WHERE doc_id = cast(:doc_id as uuid)
                    ORDER BY chunk_index
                """),
                        {"doc_id": str(doc_id)},
                    )
                )
                .mappings()
                .all()
            )
        return [{"chunk_index": r["chunk_index"], "text": r["text"]} for r in rows]

    async def insert_chunks(
        self,
        doc_id: UUID,
        chunks: list[tuple[int, str, list[float]]],
        embedding_model: str | None = None,
    ) -> None:
        """
        Insert (chunk_index, text, embedding) rows for a document.
        Deletes existing chunks first so re-runs are idempotent.

        ``embedding_model`` — optional model name (e.g. ``text-embedding-3-small:1536``).
        Written into the ``embedding_model`` column for federated dimension checks.
        """
        if not chunks:
            return
        async with self._engine.begin() as conn:
            if getattr(self, "_is_sqlite", False):
                await conn.execute(
                    text("DELETE FROM document_chunks WHERE doc_id = :doc_id"),
                    {"doc_id": str(doc_id)},
                )
            else:
                await conn.execute(
                    text("DELETE FROM document_chunks WHERE doc_id = cast(:doc_id as uuid)"),
                    {"doc_id": str(doc_id)},
                )
            for chunk_index, chunk_text, vec in chunks:
                if getattr(self, "_is_sqlite", False):
                    await conn.execute(
                        text("""
                            INSERT INTO document_chunks
                                (doc_id, chunk_index, text, token_count, embedding, embedding_model)
                            VALUES (:doc_id, :chunk_index, :text, :token_count, :vec, :embedding_model)
                            ON CONFLICT (doc_id, chunk_index) DO UPDATE
                                SET text = excluded.text,
                                    embedding = excluded.embedding,
                                    embedding_model = excluded.embedding_model
                        """),
                        {
                            "doc_id": str(doc_id),
                            "chunk_index": chunk_index,
                            "text": chunk_text,
                            "token_count": len(chunk_text.split()),
                            "vec": json.dumps(vec),
                            "embedding_model": embedding_model,
                        },
                    )
                else:
                    await conn.execute(
                        text("""
                            INSERT INTO document_chunks
                                (doc_id, chunk_index, text, token_count, embedding, embedding_model)
                            VALUES (
                                cast(:doc_id as uuid),
                                :chunk_index,
                                :text,
                                :token_count,
                                cast(:vec as vector),
                                :embedding_model
                            )
                            ON CONFLICT (doc_id, chunk_index) DO UPDATE
                                SET text = EXCLUDED.text,
                                    embedding = EXCLUDED.embedding,
                                    embedding_model = EXCLUDED.embedding_model
                        """),
                        {
                            "doc_id": str(doc_id),
                            "chunk_index": chunk_index,
                            "text": chunk_text,
                            "token_count": len(chunk_text.split()),
                            "vec": json.dumps(vec),
                            "embedding_model": embedding_model,
                        },
                    )

    async def mark_chunk_status(self, doc_id: UUID, status: str) -> None:
        """Update chunk_status on a document. status: none | chunked | skipped | failed."""
        async with self._engine.begin() as conn:
            if getattr(self, "_is_sqlite", False):
                await conn.execute(
                    text("UPDATE documents SET chunk_status = :status WHERE id = :doc_id"),
                    {"status": status, "doc_id": str(doc_id)},
                )
            else:
                await conn.execute(
                    text(
                        "UPDATE documents SET chunk_status = :status WHERE id = cast(:doc_id as uuid)"
                    ),
                    {"status": status, "doc_id": str(doc_id)},
                )

    async def get_unchunked_docs(self, limit: int = 100) -> list[tuple[str, str, str, str | None]]:
        """
        Return docs eligible for chunking: (id, title, source, body_text).
        Criteria: status=ready, chunk_status=none.
        body_text is included so callers can fall back to the DB column when
        the flat-file body is not present (e.g. distributed workers).
        The caller must check body word count before chunking.
        """
        async with self._engine.connect() as conn:
            if getattr(self, "_is_sqlite", False):
                rows = (
                    await conn.execute(
                        text("""
                        SELECT id, title, source, body_text
                        FROM documents
                        WHERE status = 'ready'
                          AND chunk_status = 'none'
                        ORDER BY ingested_at ASC
                        LIMIT :limit
                    """),
                        {"limit": limit},
                    )
                ).fetchall()
            else:
                rows = (
                    await conn.execute(
                        text("""
                        SELECT id::text, title, source, body_text
                        FROM documents
                        WHERE status = 'ready'
                          AND chunk_status = 'none'
                        ORDER BY ingested_at ASC
                        LIMIT :limit
                    """),
                        {"limit": limit},
                    )
                ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    async def search_chunks_for_docs(
        self,
        query: str,
        doc_ids: list[str],
    ) -> dict[str, dict]:  # type: ignore[type-arg]
        """
        For each doc_id in the list, find the best-matching chunk by cosine similarity.
        Returns {doc_id: {"chunk_index": int, "text": str, "score": float}}.
        Docs with no chunks or no embedding are absent from the result dict.
        """
        if not doc_ids:
            return {}
        vec = await _get_embedding(query)
        if vec is None:
            return {}
        if self._is_sqlite:
            # SQLite: no DISTINCT ON, no <=> vector operator, no pgvector
            # Return empty — chunk search is not available without pgvector
            return {}
        async with self._engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                    SELECT DISTINCT ON (doc_id)
                        doc_id::text,
                        chunk_index,
                        text,
                        1 - (embedding <=> cast(:vec as vector)) AS score
                    FROM document_chunks
                    WHERE doc_id = ANY(cast(:doc_ids as uuid[]))
                      AND embedding IS NOT NULL
                    ORDER BY doc_id, embedding <=> cast(:vec as vector)
                """),
                        {
                            "vec": json.dumps(vec),
                            "doc_ids": [uuid.UUID(str(d)) for d in doc_ids],
                        },
                    )
                )
                .mappings()
                .all()
            )
        return {
            r["doc_id"]: {
                "chunk_index": r["chunk_index"],
                "text": r["text"],
                "score": float(r["score"]),
            }
            for r in rows
        }

    # ── search_queue ───────────────────────────────────────────────────────────

    async def enqueue_search(
        self,
        query: str,
        category: str | None = None,
        priority: int = 5,
    ) -> tuple[bool, str | None]:
        """
        Insert a search_queue item if no pending row exists for this query (case-insensitive).
        Returns (queued: bool, id: str | None).
        """
        async with self._engine.begin() as conn:
            # Dedup: check for an existing pending row
            existing = (
                await conn.execute(
                    text(
                        "SELECT id FROM search_queue WHERE lower(query) = lower(:q) AND status = 'pending'"
                    ),
                    {"q": query},
                )
            ).first()
            if existing:
                return False, None
            row = (
                await conn.execute(
                    text("""
                    INSERT INTO search_queue (query, category, priority)
                    VALUES (:q, :cat, :pri)
                    RETURNING id::text
                """),
                    {"q": query, "cat": category, "pri": priority},
                )
            ).first()
            return True, row[0] if row else None

    async def dequeue_search_batch(self, batch_size: int = 5) -> list[dict]:  # type: ignore[type-arg]
        """
        Claim up to batch_size pending items (highest priority first), mark as processing.
        Returns list of row dicts: id, query, category, priority.
        """
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                    UPDATE search_queue
                    SET status = 'processing'
                    WHERE id IN (
                        SELECT id FROM search_queue
                        WHERE status = 'pending'
                        ORDER BY priority DESC, created_at
                        LIMIT :n
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING id::text, query, category, priority
                """),
                        {"n": batch_size},
                    )
                )
                .mappings()
                .all()
            )
        return [dict(r) for r in rows]

    async def mark_search_queue_status(self, item_id: str, status: str) -> None:
        """Mark a search_queue item as done or failed."""
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    UPDATE search_queue
                    SET status = :status,
                        processed_at = CASE WHEN :status IN ('done', 'failed') THEN now() ELSE processed_at END
                    WHERE id = cast(:id as uuid)
                """),
                {"status": status, "id": item_id},
            )

    async def search_queue_depth(self) -> int:
        """Return count of pending search_queue items."""
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT COUNT(*) FROM search_queue WHERE status = 'pending'")
                )
            ).first()
            return int(row[0]) if row else 0

    async def add_to_review_queue(
        self,
        doc_id,
        reason: str,
        url: str,
        source: str = "",
        notes: str | None = None,
    ) -> None:
        """Add a terminal document to the review_queue. Idempotent."""
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO review_queue (doc_id, reason, url, source, notes)
                    VALUES (cast(:doc_id as uuid), :reason, :url, :source, :notes)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "doc_id": str(doc_id),
                    "reason": reason,
                    "url": url,
                    "source": source,
                    "notes": notes,
                },
            )

    async def review_queue_depth(self) -> int:
        from sqlalchemy import text as _text

        async with self._engine.connect() as conn:
            return (
                await conn.execute(
                    _text("SELECT COUNT(*) FROM review_queue WHERE doc_id NOT IN (SELECT doc_id FROM review_queue WHERE notes IS NOT NULL)")
                )
            ).scalar() or 0

    # ── RSS feed CRUD ────────────────────────────────────────────────────────

    async def create_feed(self, feed: RSSFeed) -> RSSFeed:
        from sqlalchemy import text as _text

        tags_expr = ":tags" if self._is_sqlite else "CAST(:tags AS jsonb)"
        params = {
            "id": str(feed.id),
            "url": feed.url,
            "name": feed.name,
            "corpus_id": feed.corpus_id,
            "tags": json.dumps(feed.tags),
            "enabled": feed.enabled,
            "poll_interval_minutes": feed.poll_interval_minutes,
            "last_polled_at": feed.last_polled_at,
            "created_at": feed.created_at,
            "tenant_id": str(feed.tenant_id),
        }
        async with self._engine.begin() as conn:
            await conn.execute(
                _text(f"""
                    INSERT INTO rss_feeds
                        (id, url, name, corpus_id, tags, enabled,
                         poll_interval_minutes, last_polled_at, created_at, tenant_id)
                    VALUES
                        (:id, :url, :name, :corpus_id, {tags_expr}, :enabled,
                         :poll_interval_minutes, :last_polled_at, :created_at, :tenant_id)
                """),
                params,
            )
        return await self.get_feed(feed.id) or feed

    async def get_feed(self, feed_id: UUID) -> RSSFeed | None:
        from sqlalchemy import text as _text

        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    _text("SELECT * FROM rss_feeds WHERE id = :id"),
                    {"id": str(feed_id)},
                )
            ).fetchone()
        return _row_to_feed(dict(row._mapping)) if row else None

    async def list_feeds(
        self, tenant_id: UUID | None = None, enabled_only: bool = False
    ) -> list[RSSFeed]:
        from sqlalchemy import text as _text

        conditions = []
        params: dict = {}
        if tenant_id is not None:
            conditions.append("tenant_id = :tenant_id")
            params["tenant_id"] = str(tenant_id)
        if enabled_only:
            conditions.append("enabled = TRUE")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    _text(f"SELECT * FROM rss_feeds {where} ORDER BY created_at DESC"),
                    params,
                )
            ).fetchall()
        return [_row_to_feed(dict(r._mapping)) for r in rows]

    async def update_feed(self, feed_id: UUID, **kwargs) -> RSSFeed | None:
        from sqlalchemy import text as _text

        if not kwargs:
            return await self.get_feed(feed_id)
        allowed = {"url", "name", "corpus_id", "tags", "enabled", "poll_interval_minutes"}
        sets = []
        params: dict = {"id": str(feed_id)}
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if k == "tags":
                sets.append(f"{k} = :{k}")
                params[k] = json.dumps(v) if not isinstance(v, str) else v
            else:
                sets.append(f"{k} = :{k}")
                params[k] = v
        if not sets:
            return await self.get_feed(feed_id)
        async with self._engine.begin() as conn:
            await conn.execute(
                _text(f"UPDATE rss_feeds SET {', '.join(sets)} WHERE id = :id"),
                params,
            )
        return await self.get_feed(feed_id)

    async def delete_feed(self, feed_id: UUID) -> bool:
        from sqlalchemy import text as _text

        async with self._engine.begin() as conn:
            result = await conn.execute(
                _text("DELETE FROM rss_feeds WHERE id = :id"),
                {"id": str(feed_id)},
            )
        return result.rowcount > 0

    async def get_feeds_due_for_poll(self) -> list[RSSFeed]:
        from sqlalchemy import text as _text

        if self._is_sqlite:
            sql = _text("""
                SELECT * FROM rss_feeds
                WHERE enabled = 1
                  AND (
                    last_polled_at IS NULL
                    OR last_polled_at < datetime('now', '-' || poll_interval_minutes || ' minutes')
                  )
                ORDER BY COALESCE(last_polled_at, created_at) ASC
            """)
        else:
            sql = _text("""
                SELECT * FROM rss_feeds
                WHERE enabled = TRUE
                  AND (
                    last_polled_at IS NULL
                    OR last_polled_at < NOW() - (poll_interval_minutes * INTERVAL '1 minute')
                  )
                ORDER BY COALESCE(last_polled_at, created_at) ASC
            """)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(sql)).fetchall()
        return [_row_to_feed(dict(r._mapping)) for r in rows]

    async def mark_feed_polled(self, feed_id: UUID) -> None:
        from sqlalchemy import text as _text

        now_expr = "CURRENT_TIMESTAMP" if self._is_sqlite else "NOW()"
        async with self._engine.begin() as conn:
            await conn.execute(
                _text(f"UPDATE rss_feeds SET last_polled_at = {now_expr} WHERE id = :id"),
                {"id": str(feed_id)},
            )
        """Return count of items in review_queue."""
        async with self._engine.connect() as conn:
            row = (await conn.execute(text("SELECT COUNT(*) FROM review_queue"))).first()
            return int(row[0]) if row else 0


# ── Helpers ────────────────────────────────────────────────────────────────────


def _safe_enum(enum_cls, value):
    """Coerce a DB string to an enum, returning None on unknown values instead of raising."""
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        return None


def _row_to_feed(row: dict) -> RSSFeed:
    """Reconstruct an RSSFeed from a database row mapping."""
    from dewie.models.feed import RSSFeed as _RSSFeed

    tags = row.get("tags")
    if isinstance(tags, str):
        tags = json.loads(tags)
    elif tags is None:
        tags = []
    return _RSSFeed(
        id=row["id"],
        url=row["url"],
        name=row.get("name") or "",
        corpus_id=row.get("corpus_id"),
        tags=tags,
        enabled=row.get("enabled", True),
        poll_interval_minutes=row.get("poll_interval_minutes", 60),
        last_polled_at=row.get("last_polled_at"),
        created_at=row["created_at"],
        tenant_id=row["tenant_id"],
    )


def _row_to_doc(row: dict) -> ContentDocument:  # type: ignore[type-arg]
    """Convert a raw SQLAlchemy mapping row into a ContentDocument."""

    def _json(val: object) -> list:
        if isinstance(val, list):
            return val
        if val is None:
            return []
        return json.loads(val)

    try:
        status = ContentStatus(row["status"])
    except ValueError:
        # Unknown status values (e.g. written by a newer/older version) must
        # not make the row unreadable — degrade to pending for re-processing.
        status = ContentStatus.PENDING

    return ContentDocument(
        id=row["id"],
        url=row["url"],
        title=row["title"],
        summary=row["summary"],
        source=row["source"],
        ingested_at=row["ingested_at"],
        status=status,
        topics=_json(row["topics"]),
        keywords=_json(row["keywords"]),
        entities=_json(row["entities"]),
        sentiment=row["sentiment"],
        crawl_session=row["crawl_session"],
        # Enrichment versioning — may be absent on old rows (column added via migration)
        enrichment_version=row.get("enrichment_version") or 0,
        embedding_model=row.get("embedding_model"),
        enriched_at=row.get("enriched_at"),
        # Optional enrichment fields — absent on docs not yet fully enriched
        answers_questions=_json(row["answers_questions"])
        if row.get("answers_questions") is not None
        else [],
        tone=row.get("tone"),
        language=row.get("language") or "en",
        document_type=_safe_enum(DocumentType, row.get("document_type")),
        author=row.get("author"),
        reading_level=_safe_enum(ReadingLevel, row.get("reading_level")),
        embed_summary=row.get("embed_summary") or "",
        published_at=row.get("published_at"),
        paywall_detected=row.get("paywall_detected") or False,
        paywall_type=row.get("paywall_type") or "none",
        alternate_terms=_json(row["alternate_terms"])
        if row.get("alternate_terms") is not None
        else [],
        enrichment_quality_score=row.get("enrichment_quality_score"),
         location=row.get("location"),
         instance_id=str(row["instance_id"]) if row.get("instance_id") is not None else None,
     )


async def init_schema() -> None:
    """Standalone coroutine to initialise the database schema. Suitable for scripting."""
    client = PostgresClient()
    await client.init_schema()
    await client.close()


# Re-export for convenience
__all__ = ["PostgresClient", "init_schema"]
