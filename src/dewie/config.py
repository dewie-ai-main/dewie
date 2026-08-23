# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Central configuration for Dewie.

Settings are read from (in priority order):
  1. Environment variables
  2. .env.local file (in current directory, overrides .env — gitignored, for local overrides)
  3. .env file (in current directory)
  4. dewie.yml (in current directory)
  5. Defaults

Import the singleton ``settings`` object rather than constructing new instances.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _config_file_path() -> Path:
    """Resolve the dewie.yml path the same way admin.py's _config_path() does.

    Must stay in sync with dewie.api.routes.admin._config_path — that's where
    the admin UI writes config edits, and this is where they're read back on
    the next startup. A mismatch here silently discards saved config.
    """
    explicit = os.environ.get("DEWIE_CONFIG_PATH", "").strip()
    if explicit:
        return Path(explicit)
    data_dir = os.environ.get("DEWIE_DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir) / "dewie.yml"
    return Path.cwd() / "dewie.yml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # .env.local is loaded AFTER .env — later entries win in pydantic-settings.
        # This lets local dev override any .env value (e.g. POSTGRES_DSN pointing at
        # a dev/prod DB) without touching the committed .env file.
        # .env.local is gitignored; .env is committed with safe defaults.
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context):
        """Load dewie.yml if it exists for any unset fields.
        
        This runs AFTER pydantic-settings has loaded env vars and .env file.
        Priority: env var/explicit > .env > dewie.yml > defaults
        
        We detect "explicit" by checking if a value differs from its default.
        """
        # If POSTGRES_DSN itself wasn't set but discrete fields were
        # (POSTGRES_HOST/PORT/USER/PASSWORD/DB), assemble the DSN from those.
        # This runs before the yaml block below so an explicit discrete env
        # var still beats dewie.yml, same as postgres_dsn itself would.
        default_dsn = "postgresql+asyncpg://dewie:dewie@localhost:5432/dewie"  # Dev-only default
        if self.postgres_dsn == default_dsn and any(
            (self.postgres_host, self.postgres_port, self.postgres_user, self.postgres_password, self.postgres_db)
        ):
            user = self.postgres_user or "dewie"
            password = self.postgres_password or "dewie"
            host = self.postgres_host or "localhost"
            port = self.postgres_port or "5432"
            dbname = self.postgres_db or "dewie"
            self.postgres_dsn = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{dbname}"

        try:
            import yaml
            yml_path = _config_file_path()
            if not yml_path.exists():
                return

            with open(yml_path) as f:
                yml_dict = yaml.safe_load(f) or {}

            # For postgres_dsn: use yaml only if still default
            is_default_dsn = self.postgres_dsn == default_dsn
            if is_default_dsn:
                if "postgres_dsn" in yml_dict and yml_dict["postgres_dsn"]:
                    self.postgres_dsn = yml_dict["postgres_dsn"]
                elif "dewie_db" in yml_dict and yml_dict["dewie_db"]:
                    self.postgres_dsn = yml_dict["dewie_db"]
            
            # For other fields: load from yaml if they have default values
            # This allows yaml to set api_port, data_dir, etc. without overriding env vars
            for key, value in yml_dict.items():
                if key in ("postgres_dsn", "dewie_db"):
                    # Already handled above
                    continue
                
                if hasattr(self, key) and value is not None:
                    # Get the field default to compare
                    try:
                        field_info = self.model_fields.get(key)
                        if field_info:
                            current_val = getattr(self, key)
                            default_val = field_info.default
                            # Only override if current value is still the default
                            # (covers default=None fields like embed_dimensions —
                            # previously skipped entirely, silently discarding yaml).
                            if current_val == default_val:
                                setattr(self, key, value)
                    except Exception:
                        pass
        except Exception:
            pass

        # Resolve instance_id (may load from persisted file or generate new)
        self.instance_id = self._resolve_instance_id()

    # ── Storage ───────────────────────────────────────────────────────────────
    postgres_dsn: str = Field(
        default="postgresql+asyncpg://dewie:dewie@localhost:5432/dewie",  # Dev-only default — do not use in production
        description=(
            "Async SQLAlchemy DSN for PostgreSQL. Escape hatch for exotic "
            "drivers/options — prefer POSTGRES_HOST/PORT/USER/PASSWORD/DB, "
            "which Dewie assembles into this automatically."
        ),
    )

    postgres_host: str = Field(default="", description="Postgres host. Assembled into postgres_dsn if set.")
    postgres_port: str = Field(default="", description="Postgres port. Assembled into postgres_dsn if set.")
    postgres_user: str = Field(default="", description="Postgres user. Assembled into postgres_dsn if set.")
    postgres_password: str = Field(default="", description="Postgres password. Assembled into postgres_dsn if set.")
    postgres_db: str = Field(default="", description="Postgres database name. Assembled into postgres_dsn if set.")

    dewie_db: str = Field(
        default="",
        description=(
            "Deprecated: use postgres_dsn instead. "
            "If set, overrides postgres_dsn. Kept for backward compatibility."
        ),
    )

    redis_url: str = Field(
        default="",
        description="Redis URL. Leave empty to use in-process cache/queue backends.",
    )

    data_dir: str = Field(
        default="",
        description=(
            "Root directory for persistent data files (flat-file body store, etc.). "
            "Defaults to './data' relative to the working directory when empty. "
            "Set DEWIE_DATA_DIR in the environment or .env to override — "
            "useful in Docker deployments where the mount point differs from cwd."
        ),
    )

    instance_id: str = Field(
        default="",
        description=(
            "Unique identifier for this Dewie instance. Used for deduplication "
            "across federated nodes. Read from DEWIE_INSTANCE_ID env var, or "
            "generated on first startup and persisted to {data_dir}/instance_id.txt."
        ),
    )

    def _resolve_instance_id(self) -> str:
        """Resolve instance_id from env, persisted file, or generate new."""
        if self.instance_id:
            return self.instance_id
        # Try persisted file
        base = Path(self.data_dir) if self.data_dir else Path.cwd() / "data"
        id_file = base / "instance_id.txt"
        try:
            persisted = id_file.read_text().strip()
            if persisted:
                # Persist for next startup
                id_file.write_text(persisted)
                return persisted
        except Exception:
            pass
        # Generate new
        new_id = str(uuid.uuid4())
        try:
            base.mkdir(parents=True, exist_ok=True)
            id_file.write_text(new_id)
        except Exception:
            pass
        return new_id

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=10946)
    api_workers: int = Field(default=4)

    # ── Web search fallback (corpus-first web_search tool) ─────────────────────
    search_provider: str = Field(
        default="",
        description=(
            "Web search backend for the web_search MCP tool: brave | exa | you | stub. "
            "Empty disables web fallback (corpus-only). Requires the matching key env: "
            "BRAVE_API_KEY / EXA_API_KEY / YOU_API_KEY."
        ),
    )

    cors_origins: list[str] = Field(
        default=[
            "https://dewie.ai",
            "https://www.dewie.ai",
            "https://api.dewie.ai",
            "http://localhost:10946",
            "http://localhost:3000",
        ],
        description=(
            "CORS allow-list. Set via CORS_ORIGINS env var as a JSON array: "
            '["https://yourdomain.com","http://localhost:3000"]. '
            "Set CORS_ORIGINS to restrict allowed origins."
        ),
    )

    # ── Recursive query safety limits ─────────────────────────────────────────
    default_max_depth: int = Field(default=3, ge=1, le=10)
    absolute_max_depth: int = Field(default=10, ge=1)
    max_nodes_per_level: int = Field(default=20, ge=1, le=100)
    query_timeout_seconds: int = Field(default=30, ge=1)

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_rpm: int = Field(
        default=60,
        description="Requests per minute per client IP.",
    )

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_ttl_seconds: int = Field(default=300)

    # ── Query logging ─────────────────────────────────────────────────────────
    query_log_save_full_results: bool = Field(
        default=True,
        description=(
            "When True, save the complete ranked SearchResponse (scores, quality, "
            "reading_level, etc.) to query_log.full_results. "
            "Disable when log table grows too large. "
            "Override via QUERY_LOG_SAVE_FULL_RESULTS=false env var or dewie.yml."
        ),
    )

    # ── Query defaults ───────────────────────────────────────────────────────
    query_default_ranker: str = Field(
        default="rrf_chunks",
        description="Default ranker used when request does not explicitly set one.",
    )
    enabled_rankers: list[str] = Field(
        default=[
            "answers_questions_rrf",
            "rrf_chunks",
            "rrf",
            "bm25",
            "vector",
        ],
        description=(
            "Ranked list of ranker IDs exposed to users (API, frontend dropdown, admin UI). "
            "Set via DEWIE_ENABLED_RANKERS env var as a JSON array. "
            "Only rankers whose IDs appear in this list are returned by /query/rankers "
            "and visible in the frontend ranker selector."
        ),
    )

    # ── Crawler ───────────────────────────────────────────────────────────────
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        description="User-Agent header sent for all outbound fetches.",
    )
    crawler_max_depth: int = Field(default=2, ge=0, le=10)
    crawler_max_pages: int = Field(default=100, ge=1)
    crawler_concurrency: int = Field(default=3, ge=1, le=20)
    crawler_politeness_delay: float = Field(default=1.0, ge=0.0)
    crawler_same_domain: bool = Field(default=True)
    crawler_request_timeout: float = Field(default=15.0, ge=1.0)

    # ── Enrichment hook system ────────────────────────────────────────────────

    enrichment_default_backend: str = Field(
        default="passthrough",
        description=(
            "Name of the default enrichment backend used when no routing rule "
            "matches.  Must match a backend 'name' in enrichment_backends."
        ),
    )
    enrichment_backends: str = Field(
        default="[]",
        description=(
            "JSON array of backend descriptor objects.  Each descriptor "
            "must include 'name' and 'type' fields.  See CRAWLER_AND_HOOK_DESIGN.md "
            "§6 for the full descriptor format."
        ),
    )
    enrichment_routing_rules: str = Field(
        default="[]",
        description=(
            "JSON array of routing rule objects evaluated in order.  "
            "Each rule contains one predicate (if_body_shorter_than, "
            "if_body_longer_than, if_document_type, default) and a "
            "'use_backend' value.  See CRAWLER_AND_HOOK_DESIGN.md §2.6."
        ),
    )
    enrichment_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Number of backend attempts before marking a document FAILED.  "
            "The primary backend is tried first; the fallback backend is used "
            "for subsequent attempts."
        ),
    )
    enrichment_fallback_backend: str = Field(
        default="passthrough",
        description=(
            "Name of the backend to use when the primary backend fails.  "
            "Set to 'spacy' only if you have en_core_web_sm installed and want local fallback."
        ),
    )
    max_extraction_chars: int = Field(
        default=80_000,
        ge=1_000,
        description=(
            "Maximum number of body characters sent to the enrichment backend.  "
            "Content beyond this limit is truncated; a note is appended to the "
            "prompt so the backend is aware."
        ),
    )
    max_summary_chars: int = Field(
        default=1_500,
        ge=100,
        description=(
            "Hard cap applied to the summary field after the backend returns it.  "
            "Prevents oversized summaries from being persisted."
        ),
    )
    save_raw_documents: bool = Field(
        default=False,
        description="When true, save raw ingested document bodies to disk under ingested_docs/.",
    )
    enrichment_model: str = Field(
        default="",
        description="LLM model name for enrichment backend. Leave blank to use the first available model from registered providers.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    enrichment_mode: str = Field(
        default="single_pass",
        description=(
            "Enrichment pipeline mode. "
            "single_pass: one combined LLM call (default, cheaper). "
            "dual_pass: separate calls for AQ generation and KE extraction (richer metadata)."
        ),
    )

    # ── Provider selection ────────────────────────────────────────────────────
    # Each step selects a *server label* (see providers/servers.py for the
    # `servers:` dewie.yml schema and built-in labels: openai, anthropic).
    # Per-provider API keys/base URLs live on the server entry,
    # not here — see ServerConfig.api_key_env / endpoint.
    chat_server_aq: str = Field(
        default="",
        description="Server label for AQ generation. Leave blank to use no AQ generation.",
    )
    openai_api_type: str = Field(
        default="chat/completions",
        description="OpenAI API type: chat/completions or v1/responses",
    )
    chat_model_aq: str = Field(
        default="",
        description="Model for AQ generation.",
    )
    chat_server_ke: str = Field(
        default="",
        description="Server label for keyword/entity extraction. Leave blank to use chat_server_aq.",
    )
    chat_model_ke: str = Field(
        default="",
        description="Model for KE extraction. Leave blank to use chat_model_aq.",
    )
    embed_server: str = Field(
        default="local",
        description="Server label for embeddings, or 'local' for in-process embeddings.",
    )
    embed_model: str = Field(
        default="ggml-org/embeddinggemma-300m-qat-q8_0-GGUF",
        description=(
            "Embedding model. Default is EmbeddingGemma-300m as a public GGUF, "
            "run in-process via llama.cpp (zero config, no API key, 768 dims). "
            "A GGUF spec (repo or file) uses llama-cpp-python; any other value "
            "with embed_server=local uses sentence-transformers; otherwise the "
            "model is sent to the configured embed_server."
        ),
    )
    embed_store_full_vector: bool = Field(
        default=False,
        description=(
            "When true, also persist the model's untruncated embedding (before "
            "embed_dimensions/MRL truncation) in embedding_full, for exact-precision "
            "reranking of ANN candidates. Off by default — the truncated vector is "
            "good enough for most use cases."
        ),
    )
    embed_dimensions: int | None = Field(
        default=None,
        description=(
            "Override embedding vector dimensions. "
            "When set, embeddings exceeding this value are truncated. "
            "When unset, auto-detected from model name via _embed_dimensions_for_model(). "
            "Overrides EMBED_DIMENSIONS env var."
        ),
    )
    local_embed_allowed: bool = Field(
        default=True,
        description=(
            "Host-level switch for in-process embedding (embed_server=local). "
            "When False, resolving a local embedding provider fails regardless "
            "of dewie.yml, so a managed host can require the account to be "
            "enabled for local models. Env: LOCAL_EMBED_ALLOWED."
        ),
    )

    # ── Flow control ──────────────────────────────────────────────────────────
    max_enrichment_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            "Maximum number of times a document is re-queued to pending before "
            "being marked FAILED.  Retry count is derived from unresolved "
            "pipeline_errors rows so no schema migration is required."
        ),
    )

    # ── Browse API ────────────────────────────────────────────────────────────

    browse_session_ttl_seconds: int = Field(
        default=14_400,
        ge=60,
        description=(
            "Redis TTL for browse sessions in seconds.  Sessions that have "
            "not been accessed within this window expire automatically.  "
            "Default is 4 hours (14,400 s)."
        ),
    )
    browse_max_neighbors: int = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "Default maximum number of neighbor previews returned per "
            "visit_node or expand call.  Can be overridden per-session "
            "and per-expansion."
        ),
    )

    # ── Auth — API keys (Issue #96) ───────────────────────────────────────────
    auth_enabled: bool = Field(
        default=True,
        description="Enable API key authentication. False = open access.",
    )
    local_auth_enabled: bool = Field(
        default=False,
        description=(
            "Enable local auth mode for development. "
            "When true, requests are treated as an authenticated local user "
            "without requiring X-API-Key."
        ),
    )
    local_auth_user_id: str = Field(
        default="00000000-0000-0000-0000-000000000002",
        description="Synthetic user_id used when LOCAL_AUTH_ENABLED=true.",
    )
    local_auth_email: str = Field(
        default="Dewie Local Catalog",
        description="Username/identity surfaced in /auth/me when LOCAL_AUTH_ENABLED=true. Does not need to be an email address.",
    )
    local_auth_is_admin: bool = Field(
        default=True,
        description="Whether local-auth user has admin privileges.",
    )
    admin_email: str = Field(
        default="admin",
        description=(
            "Email for the default admin user created on first startup when "
            "the users table is empty. Set via ADMIN_EMAIL env var."
        ),
    )
    admin_password: str = Field(
        default="admin",
        description=(
            "Password for the default admin user created on first startup. "
            "Set via ADMIN_PASSWORD env var."
        ),
    )
    internal_service_key_required: bool = Field(
        default=False,
        description=(
            "Require INTERNAL_SERVICE_KEY for /ingest requests. "
            "Recommended true for production deployments."
        ),
    )
    encryption_master_key: str = Field(
        default="",
        description=(
            "Base64 32-byte Fernet key for encrypting server registry API keys at rest. "
            "Empty disables literal-key storage (api_key_env still works). "
            "Set via ENCRYPTION_MASTER_KEY env var."
        ),
    )

    # ── Subsystem flags (Issue #143) ─────────────────────────────────────────
    enable_api: bool = Field(
        default=True,
        description="Enable the public API. Set to false to run worker-only mode.",
    )
    enable_enrichment: bool = Field(
        default=True,
        description="Enable the enrichment pipeline and registry. Set to false to skip enrichment init.",
    )
    enable_ingestion: bool = Field(
        default=True,
        description="Enable RSS/ingest background workers. Set to false to disable polling.",
    )
    enable_poller: bool = Field(
        default=True,
        description="Enable background poller tasks (backfill, cluster rebuild, etc.).",
    )

    # ── Multi-source (Issue #386) ─────────────────────────────────────────────
    enable_public_sources: bool = Field(
        default=False,
        description="Pre-populate dewie_sources with public instances on startup.",
    )
    public_sources_file: str = Field(
        default="config/public_sources.json",
        description="Path to JSON file containing default source definitions for startup seeding.",
    )
    public_sources_json: str = Field(
        default="",
        description="Optional JSON array of source definitions to merge/override file defaults at startup.",
    )

    # ── Worker tuning ─────────────────────────────────────────────────────────
    enrichment_batch_size: int = Field(
        default=2,
        ge=1,
        le=100,
        description=(
            "Docs claimed per enrichment tick. Set to 1 for serial mode "
            "(finish one doc, fetch next) — recommended for local LLMs."
        ),
    )
    enrichment_sleep_secs: int = Field(
        default=30,
        ge=0,
        le=3600,
        description=(
            "Seconds to sleep between enrichment ticks when the queue is empty. "
            "Set to 0 with batch_size=1 for maximum throughput on local models."
        ),
    )
    enrichment_workers: int = Field(
        default=1,
        ge=1,
        le=16,
        description=(
            "Concurrent enrichment loops. get_pending_docs claims docs atomically "
            "(FOR UPDATE SKIP LOCKED), so workers never collide. Match this to the "
            "inference server's parallel slots (llama-server --parallel N). "
            "SQLite installs should keep 1 (no atomic claim there)."
        ),
    )
    chunk_embedder_batch_size: int = Field(default=100, ge=1, le=1000)
    chunk_embedder_sleep_secs: int = Field(default=10, ge=1, le=3600)
    edge_rebuild_sleep_secs: int = Field(default=60, ge=1, le=3600)
    llm_queue_backend: str = Field(
        default="memory",
        description="Queue backend: memory or redis.",
    )

    # ── Compliance / SOC 2 audit hooks (Issue #98) ─────────────────────────────
    # Env binding is case-insensitive: AUDIT_LOG_ENABLED=true still maps here.
    audit_log_enabled: bool = Field(
        default=True,
        description="Enable audit event logging. When False, log_audit_event is a no-op.",
    )
    retention_policy_enabled: bool = Field(
        default=False,
        description="Enable data retention policy enforcement. When False, retention job skips processing.",
    )
    RETENTION_DAYS: int = Field(
        default=365,
        ge=1,
        description="Number of days to retain documents before marking them as expired.",
    )
    PII_SCAN_ENABLED: bool = Field(
        default=False,
        description="Enable PII scanning on document ingestion.",
    )
    ANOMALY_DETECTION_ENABLED: bool = Field(
        default=False,
        description="Enable anomaly detection in query patterns.",
    )

    # ── Plugin startup registration (Issue #819) ─────────────────────────────
    network_backend_class: str = Field(
        default="dewie.storage.network.NoopNetworkBackend",
        description=(
            "Fully-qualified class path for the network backend used by the enrichment pipeline. "
            "Cloud deployments override this via NETWORK_BACKEND_CLASS env var to inject "
            "custom implementations without forking the OSS repo."
        ),
    )
    enrichment_passes: list[str] = Field(
        default=[
            "dewie.enrichment.passes.MetadataPass",
            "dewie.enrichment.passes.ChunkPass",
            "dewie.enrichment.passes.EmbedPass",
        ],
        description=(
            "Ordered list of fully-qualified class paths for enrichment passes. "
            "Each class must implement the EnrichmentPass ABC. Cloud deployments "
            "override this via ENRICHMENT_PASSES env var (as a JSON array) to inject "
            "custom passes without forking the OSS repo."
        ),
    )



    # ── Email ───────────────────────────────────────────────────────────────
    smtp_host: str = Field(default="")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    smtp_from_email: str = Field(default="")
    base_url: str = Field(default="http://localhost:10946")

settings = Settings()
