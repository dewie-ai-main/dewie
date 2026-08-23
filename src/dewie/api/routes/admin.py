# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
dewie/api/routes/admin.py — API key management, workspace/corpus management, query log.

All routes are under /admin prefix.
When AUTH_ENABLED=false (default), these routes are accessible without a key
for local dev convenience. When AUTH_ENABLED=true, admin scope is required.

Routes:
  POST   /admin/keys                    — create a new API key
  GET    /admin/keys                    — list all active keys
  DELETE /admin/keys/{key_id}           — revoke a key

  POST   /admin/workspaces              — create a workspace
  GET    /admin/workspaces              — list workspaces
  DELETE /admin/workspaces/{id}         — delete a workspace

  POST   /admin/corpora                 — create a corpus
  GET    /admin/corpora                 — list corpora (optionally filtered by workspace)
  DELETE /admin/corpora/{id}            — delete a corpus

  GET    /admin/query-log               — list recent query log entries
  GET    /admin/query-log/{id}          — get a single query log entry
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import uuid

log = logging.getLogger("dewie.api")


async def _test_postgres_connection(config: dict) -> tuple[bool, str | None]:
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


async def _test_mcp_connection(config: dict) -> tuple[bool, str | None]:
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

    import httpx

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
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from dewie.config import settings

router = APIRouter(prefix="/admin", tags=["admin"])


_CONFIG_ENV_MAP = {
    "postgres_dsn": "POSTGRES_DSN",
    "redis_url": "REDIS_URL",
    "auth_enabled": "AUTH_ENABLED",
    "local_auth_enabled": "LOCAL_AUTH_ENABLED",
    "local_auth_user_id": "LOCAL_AUTH_USER_ID",
    "local_auth_email": "LOCAL_AUTH_EMAIL",
    "local_auth_is_admin": "LOCAL_AUTH_IS_ADMIN",
    "query_default_ranker": "QUERY_DEFAULT_RANKER",
    "chat_server_aq": "CHAT_SERVER_AQ",
    "chat_model_aq": "CHAT_MODEL_AQ",
    "chat_server_ke": "CHAT_SERVER_KE",
    "chat_model_ke": "CHAT_MODEL_KE",
    "embed_server": "EMBED_SERVER",
    "embed_model": "EMBED_MODEL",
    "enrichment_model": "ENRICHMENT_MODEL",
    "embedding_model": "EMBEDDING_MODEL",
    "enrichment_mode": "ENRICHMENT_MODE",
    "rate_limit_rpm": "RATE_LIMIT_RPM",
    "internal_service_key_required": "INTERNAL_SERVICE_KEY_REQUIRED",
}

_VALID_ENRICHMENT_MODES = frozenset({"single_pass", "dual_pass"})
_VALID_ACTIVATION_STATUSES = frozenset({"pending", "approved", "rejected"})
_VALID_SOURCE_TYPES = frozenset({"sqlite", "postgres", "mcp"})

_CONFIG_KEY_META: dict[str, dict[str, str]] = {
    # ── Query ─────────────────────────────────────────────────────────────────
    "query_default_ranker": {
        "reload_behavior": "hot_reload",
        "description": "Default ranker for /query when ranker is omitted.",
        "section": "query",
    },
    # ── AQ generation ─────────────────────────────────────────────────────────
    "chat_server_aq": {
        "reload_behavior": "restart_required",
        "description": "Registered server label for AQ generation (see /admin/servers).",
        "section": "enrichment",
    },
    "chat_model_aq": {
        "reload_behavior": "restart_required",
        "description": "Model for AQ generation.",
        "section": "enrichment",
    },
    # ── Keyword/entity extraction ─────────────────────────────────────────────
    "chat_server_ke": {
        "reload_behavior": "restart_required",
        "description": "Registered server label for keyword/entity extraction.",
        "section": "enrichment",
    },
    "chat_model_ke": {
        "reload_behavior": "restart_required",
        "description": "Model for keyword/entity extraction.",
        "section": "enrichment",
    },
    # ── Embeddings ────────────────────────────────────────────────────────────
    "embed_server": {
        "reload_behavior": "restart_required",
        "description": "Registered server label for embeddings, or 'local'.",
        "section": "enrichment",
    },
    "embed_model": {
        "reload_behavior": "restart_required",
        "description": "Embedding model name.",
        "section": "enrichment",
    },
    # ── Enrichment pipeline ───────────────────────────────────────────────────
    "enrichment_model": {
        "reload_behavior": "restart_required",
        "description": "Primary enrichment LLM model (legacy field).",
        "section": "enrichment",
    },
    "embedding_model": {
        "reload_behavior": "restart_required",
        "description": "Embedding model (legacy field — prefer embed_model).",
        "section": "enrichment",
    },
    "enrichment_mode": {
        "reload_behavior": "restart_required",
        "description": "Pipeline mode: single_pass (1 LLM call) or dual_pass (separate AQ + KE calls).",
        "section": "enrichment",
    },
    # ── Rate limiting ─────────────────────────────────────────────────────────
    "rate_limit_rpm": {
        "reload_behavior": "hot_reload",
        "description": "Requests per minute per IP/key.",
        "section": "system",
    },
    # ── Local auth ─────────────────────────────────────────────────────────────
    "local_auth_enabled": {
        "reload_behavior": "hot_reload",
        "description": "Enable local auth mode that bypasses API-key checks for a synthetic local user.",
        "section": "security",
    },
    "local_auth_user_id": {
        "reload_behavior": "hot_reload",
        "description": "Synthetic user_id used when local auth is enabled.",
        "section": "security",
    },
    "local_auth_email": {
        "reload_behavior": "hot_reload",
        "description": "Email surfaced by /auth/me for local auth.",
        "section": "security",
    },
    "local_auth_is_admin": {
        "reload_behavior": "hot_reload",
        "description": "Whether the local-auth identity has admin privileges.",
        "section": "security",
    },
    # ── Security ──────────────────────────────────────────────────────────────
    "internal_service_key_required": {
        "reload_behavior": "hot_reload",
        "description": "Require INTERNAL_SERVICE_KEY header on /ingest. Recommended true for production.",
        "section": "security",
    },
}


def _pg(request: Request):  # type: ignore[return]
    return request.app.state.postgres


async def _audit(request: Request, action: str, resource_type: str, resource_id: str,
                 metadata: dict | None = None) -> None:
    """Record an audit event if audit logging is enabled.

    Never raises: audit is telemetry, and a bad tenant_id or storage hiccup
    must not fail the admin action it observes.
    """
    import uuid as _uuid

    try:
        actor_id_raw = getattr(request.state, "actor_id", None)
        try:
            actor_id = str(_uuid.UUID(str(actor_id_raw)))
        except (ValueError, TypeError):
            # No authenticated actor (e.g. auth disabled locally) — actor_id is
            # a UUID column, so a sentinel placeholder, not the literal string
            # "unknown", which asyncpg rejects as an invalid UUID.
            actor_id = "00000000-0000-0000-0000-000000000000"
        tenant_id_str = getattr(request.state, "tenant_id", None)
        try:
            tenant_id = _uuid.UUID(str(tenant_id_str))
        except (ValueError, TypeError):
            tenant_id = _uuid.UUID("00000000-0000-0000-0000-000000000001")

        pg = _pg(request)
        from dewie.compliance import audit_log
        from dewie.config import settings

        await audit_log(
            pg,
            settings=settings,
            tenant_id=tenant_id,
            actor_id=actor_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=metadata,
        )
    except Exception as exc:
        log.warning("audit event %s dropped (non-fatal): %s", action, exc)


def _require_admin(request: Request) -> None:
    """Raise 403 if the request does not have an admin key/session.
    When AUTH_ENABLED=false (local dev), all requests are treated as admin.
    """
    if settings.auth_enabled and not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")


def _require_admin_session(request: Request) -> None:
    """Backward-compatible alias used by legacy tests."""
    return _require_admin(request)


def _config_path() -> Path:
    explicit = os.environ.get("DEWIE_CONFIG_PATH", "").strip()
    if explicit:
        return Path(explicit)
    data_dir = os.environ.get("DEWIE_DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir) / "dewie.yml"
    return Path.cwd() / "dewie.yml"


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml

        if not path.exists():
            return {}
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {exc}") from exc


def _save_config(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {exc}") from exc


def _coerce_value(raw: str, value_type: str) -> Any:
    vt = value_type.lower()
    if vt == "str":
        return raw
    if vt == "int":
        return int(raw)
    if vt == "float":
        return float(raw)
    if vt == "bool":
        norm = raw.strip().lower()
        if norm in {"1", "true", "yes", "on"}:
            return True
        if norm in {"0", "false", "no", "off"}:
            return False
        raise HTTPException(status_code=400, detail=f"Invalid bool value: {raw}")
    if vt == "json":
        return json.loads(raw)
    return raw


def _effective_value(key: str, file_cfg: dict[str, Any]) -> tuple[Any, str]:
    # File (admin-written) takes priority over env vars.
    # Env vars are often injected as a side-effect of imports (e.g. magika calls
    # load_dotenv() which plants .env.local values into os.environ). An explicit
    # admin save to dewie.yml should always win over that.
    if key in file_cfg:
        return file_cfg[key], "file"
    env_key = _CONFIG_ENV_MAP.get(key)
    if env_key:
        env_val = os.environ.get(env_key)
        if env_val is not None and env_val != "":
            return env_val, "env"
    default = getattr(settings, key, None)
    return default, "default"


def _registered_server_labels() -> frozenset[str]:
    """Return registered server labels (servers.py), plus 'local' for embeddings."""
    from dewie.providers.servers import load_servers

    return frozenset(load_servers().keys() | {"local"})


def _validate_config_update(path: str, value: Any) -> None:
    if path not in _CONFIG_KEY_META:
        raise HTTPException(status_code=400, detail=f"Unsupported config key: {path}")
    if path == "query_default_ranker":
        from dewie.storage.rankers import list_rankers as _list_rankers

        valid = {r["id"] for r in _list_rankers()}
        if str(value) not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ranker '{value}'. Valid: {sorted(valid)}",
            )
    if path in {"chat_server_aq", "chat_server_ke", "embed_server"}:
        # Validate against registered server labels (providers/servers.py)
        labels = _registered_server_labels()
        if str(value) not in labels:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid server label '{value}'. Valid: {sorted(labels)}",
            )
    if path in {"chat_model_aq", "chat_model_ke"}:
        # Validate model against registry for the configured server's label, if the
        # model registry happens to know that label (e.g. built-in cloud servers).
        # Read from settings (which reflects .env / dewie.yml) not os.environ,
        # because pydantic-settings does not inject .env values into os.environ.
        settings_key_map = {"chat_model_aq": "chat_server_aq", "chat_model_ke": "chat_server_ke"}
        settings_key = settings_key_map.get(path, "")
        provider_id = getattr(settings, settings_key, "").strip()
        if not provider_id:
            return  # No server configured yet — can't validate model
        from dewie.model_registry import registry

        provider = registry.get_provider(provider_id)
        if provider is None:
            return  # Unknown to the model catalog — allow any model name (custom server)
        # Check if provider has a static model list
        static_models = [m.id for m in provider.models if not m.dynamic]
        if static_models:
            if str(value) not in static_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown model '{value}' for provider '{provider_id}'. Valid: {sorted(static_models)}",
                )
    if path == "enrichment_mode" and str(value) not in _VALID_ENRICHMENT_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid enrichment_mode '{value}'. Valid: {sorted(_VALID_ENRICHMENT_MODES)}",
        )


# ── API Key management ────────────────────────────────────────────────────────


class CreateKeyRequest(BaseModel):
    name: str | None = Field(default=None, description="Human-readable label for the key")
    scopes: list[str] = Field(default=["read"], description="Permission scopes")
    workspace_ids: list[uuid.UUID] = Field(
        default=[], description="Restrict key to these workspaces; empty = all workspaces"
    )
    live: bool = Field(default=True, description="Live key (ck_live_*) vs test key (ck_test_*)")


class KeyResponse(BaseModel):
    id: uuid.UUID
    workspace_ids: list[uuid.UUID] = Field(
        default=[], description="Restrict key to these workspaces; empty = all workspaces"
    )
    key_prefix: str
    scopes: list[str]
    name: str | None
    created_at: str


class CreateKeyResponse(BaseModel):
    key: str = Field(description="Plaintext key — shown once, never stored")
    record: KeyResponse


class ConfigFieldMeta(BaseModel):
    key: str
    reload_behavior: str
    description: str
    section: str = "general"


class ConfigValue(BaseModel):
    key: str
    value: Any
    source: str


class ConfigResponse(BaseModel):
    file_path: str
    values: list[ConfigValue]
    metadata: list[ConfigFieldMeta]


class ConfigSetRequest(BaseModel):
    path: str
    value: str
    value_type: str = Field(default="str", description="str|int|float|bool|json")


class ConfigSetResponse(BaseModel):
    ok: bool
    path: str
    value: Any
    reload_behavior: str
    source: str


class CatalogProviderRequest(BaseModel):
    provider_id: str = Field(description="Provider identifier (e.g. openai, lmstudio-local)")
    base_url: str = Field(default="", description="Provider base URL")
    api_key_env: str | None = Field(default=None, description="Environment variable name for API key")
    probe_url: str | None = Field(default=None, description="Model-discovery endpoint URL")
    probe_timeout: float = Field(default=3.0, ge=0.1, le=60.0)
    dynamic: bool = Field(default=False, description="Whether provider supports runtime discovery")


class CatalogModelAddRequest(BaseModel):
    provider_id: str = Field(description="Provider identifier")
    model_id: str = Field(description="Model identifier")
    display_name: str | None = Field(default=None)
    context_window: int = Field(default=0, ge=0)
    capabilities: list[str] = Field(default_factory=list)
    json_mode: str = Field(default="none")
    prompt_style: str = Field(default="system_user")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=600, ge=1)
    input_per_1m: float = Field(default=0.0, ge=0.0)
    output_per_1m: float = Field(default=0.0, ge=0.0)


class CatalogModelVisibilityRequest(BaseModel):
    provider_id: str
    model_id: str


class ContextSelectionRequest(BaseModel):
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Context-specific provider/model selections. null values remove keys.",
    )


class CatalogOperationResponse(BaseModel):
    ok: bool
    message: str


class CatalogResponse(BaseModel):
    context: str
    purpose: str = "all"
    providers: list[dict[str, Any]]
    models_by_provider: dict[str, list[dict[str, Any]]]
    selections: dict[str, Any]


class EmbeddingStatusResponse(BaseModel):
    provider: str
    model: str
    base_url: str
    probe_url: str | None = None
    provider_available: bool | None = None
    requested_output_dimensions: int | None = None
    storage_dimensions: int | None = None
    is_embedding_model: bool = False
    supports_mrl: bool = False
    min_dimensions: int = 0
    default_dimensions: int = 0
    max_dimensions: int = 0
    dimension_source: str = ""


def _coerce_int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _model_registry():
    from dewie.model_registry import registry

    return registry


@router.get("/model-catalog", response_model=CatalogResponse)
async def get_model_catalog(
    request: Request,
    context: str = "admin",
    include_hidden: bool = True,
    purpose: str = "all",
) -> CatalogResponse:
    """Return effective provider/model catalog from filesystem overlays plus static seed."""
    _require_admin(request)
    if context not in {"admin", "user"}:
        raise HTTPException(status_code=400, detail="context must be one of: admin, user")
    if purpose not in {"all", "chat", "embedding"}:
        raise HTTPException(status_code=400, detail="purpose must be one of: all, chat, embedding")

    reg = _model_registry()
    payload = await reg.catalog(context=context, include_hidden=include_hidden, purpose=purpose)
    return CatalogResponse(**payload)


@router.get("/model-catalog/embedding-status", response_model=EmbeddingStatusResponse)
async def get_embedding_status(request: Request) -> EmbeddingStatusResponse:
    """Return resolved embedding runtime status used by ingest/query paths."""
    _require_admin(request)

    from dewie.providers.factory import _resolve_embed
    from dewie.storage.postgres import _embed_dimensions_for_model

    provider, model, _configured_dims = _resolve_embed()
    reg = _model_registry()

    base_url = ""
    probe_url: str | None = None
    provider_available: bool | None = None

    if provider == "local":
        pass  # in-process, no endpoint
    else:
        from dewie.providers.servers import UnknownServerError, get_server

        try:
            server = get_server(provider)
            base_url = f"{server.endpoint}/v1"
        except UnknownServerError:
            pass
        # Model catalog metadata (availability probe) is keyed the same way for
        # built-in cloud servers; best-effort only.
        provider_info = reg.get_provider(provider)
        if provider_info is not None:
            probe_url = provider_info.probe_url
            provider_available = provider_info.available

    requested_output_dimensions = _coerce_int_or_none(
        os.environ.get("EMBED_OUTPUT_DIMENSIONS") or os.environ.get("EMBED_DIMENSIONS")
    )
    storage_dimensions = _embed_dimensions_for_model(model)

    meta = {
        "is_embedding_model": False,
        "supports_mrl": False,
        "min_dimensions": 0,
        "default_dimensions": 0,
        "max_dimensions": 0,
        "dimension_source": "",
    }
    try:
        payload = await reg.catalog(context="admin", include_hidden=True, purpose="embedding")
        entries = payload.get("models_by_provider", {}).get(provider, [])
        match = next((m for m in entries if str(m.get("id")) == model), None)
        if isinstance(match, dict):
            emb = match.get("embedding") if isinstance(match.get("embedding"), dict) else {}
            meta = {
                "is_embedding_model": bool(match.get("is_embedding_model", False)),
                "supports_mrl": bool(emb.get("supports_mrl", False)),
                "min_dimensions": int(emb.get("min_dimensions", 0) or 0),
                "default_dimensions": int(emb.get("default_dimensions", 0) or 0),
                "max_dimensions": int(emb.get("max_dimensions", 0) or 0),
                "dimension_source": str(emb.get("source", "")),
            }
    except Exception:
        # Keep status endpoint best-effort for operational visibility.
        pass

    return EmbeddingStatusResponse(
        provider=provider,
        model=model,
        base_url=base_url,
        probe_url=probe_url,
        provider_available=provider_available,
        requested_output_dimensions=requested_output_dimensions,
        storage_dimensions=storage_dimensions,
        is_embedding_model=meta["is_embedding_model"],
        supports_mrl=meta["supports_mrl"],
        min_dimensions=meta["min_dimensions"],
        default_dimensions=meta["default_dimensions"],
        max_dimensions=meta["max_dimensions"],
        dimension_source=meta["dimension_source"],
    )


@router.post("/model-catalog/providers", response_model=CatalogOperationResponse)
async def register_model_provider(
    body: CatalogProviderRequest,
    request: Request,
) -> CatalogOperationResponse:
    """Register or update a provider in durable filesystem catalog overlays."""
    _require_admin(request)
    reg = _model_registry()
    try:
        await reg.register_provider(
            body.provider_id,
            base_url=body.base_url,
            api_key_env=body.api_key_env,
            probe_url=body.probe_url,
            probe_timeout=body.probe_timeout,
            dynamic=body.dynamic,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CatalogOperationResponse(ok=True, message="Provider saved")


@router.post("/model-catalog/providers/{provider_id}/refresh", response_model=CatalogOperationResponse)
async def refresh_provider_models(
    provider_id: str,
    request: Request,
) -> CatalogOperationResponse:
    """Fetch model list from provider and persist discovered models."""
    _require_admin(request)
    reg = _model_registry()
    try:
        count = await reg.refresh_provider_models(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CatalogOperationResponse(ok=True, message=f"Discovered {count} models")


@router.delete("/model-catalog/providers/{provider_id}", status_code=204)
async def delete_registry_provider(
    provider_id: str,
    request: Request,
) -> None:
    """Remove a provider from the model registry.

    For user-added providers, the entry is deleted from provider_overrides.yaml.
    For YAML-defined providers (OpenAI, Anthropic, etc.), the provider is marked
    as disabled so it does not re-appear on restart. YAML config is preserved
    so it can be re-enabled later.
    """
    _require_admin(request)
    reg = _model_registry()
    try:
        await reg.remove_provider(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/model-catalog/providers/{provider_id}/reset", response_model=CatalogOperationResponse)
async def reset_provider_models(
    provider_id: str,
    request: Request,
) -> CatalogOperationResponse:
    """Reset discovered entries for a provider and re-fetch fresh models."""
    _require_admin(request)
    reg = _model_registry()
    try:
        count = await reg.reset_provider_models(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CatalogOperationResponse(ok=True, message=f"Reset complete ({count} entries updated)")


@router.post("/model-catalog/models/add", response_model=CatalogOperationResponse)
async def add_catalog_model(
    body: CatalogModelAddRequest,
    request: Request,
) -> CatalogOperationResponse:
    """Add a model to the selectable catalog for a provider."""
    _require_admin(request)
    reg = _model_registry()
    try:
        await reg.add_catalog_model(
            body.provider_id,
            body.model_id,
            {
                "display_name": body.display_name or body.model_id,
                "context_window": body.context_window,
                "capabilities": body.capabilities,
                "enrichment": {
                    "json_mode": body.json_mode,
                    "prompt_style": body.prompt_style,
                    "temperature": body.temperature,
                    "max_tokens": body.max_tokens,
                },
                "cost": {
                    "input_per_1m": body.input_per_1m,
                    "output_per_1m": body.output_per_1m,
                },
                "dynamic": False,
            },
            source="manually_added",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CatalogOperationResponse(ok=True, message="Model added")


@router.post("/model-catalog/models/hide", response_model=CatalogOperationResponse)
async def hide_catalog_model(
    body: CatalogModelVisibilityRequest,
    request: Request,
) -> CatalogOperationResponse:
    """Hide a provider/model pair from selection lists without deleting it."""
    _require_admin(request)
    reg = _model_registry()
    await reg.hide_catalog_model(body.provider_id, body.model_id)
    return CatalogOperationResponse(ok=True, message="Model hidden")


@router.post("/model-catalog/models/unhide", response_model=CatalogOperationResponse)
async def unhide_catalog_model(
    body: CatalogModelVisibilityRequest,
    request: Request,
) -> CatalogOperationResponse:
    """Restore a previously hidden provider/model pair to selection lists."""
    _require_admin(request)
    reg = _model_registry()
    await reg.unhide_catalog_model(body.provider_id, body.model_id)
    return CatalogOperationResponse(ok=True, message="Model restored")


@router.get("/model-catalog/selections/{context}")
async def get_context_selection(context: str, request: Request) -> dict[str, Any]:
    """Get persisted selection state for an independent context (admin or user)."""
    _require_admin(request)
    reg = _model_registry()
    try:
        values = await reg.get_context_selection(context)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"context": context, "values": values}


@router.patch("/model-catalog/selections/{context}")
async def set_context_selection(
    context: str,
    body: ContextSelectionRequest,
    request: Request,
) -> dict[str, Any]:
    """Update persisted selection state for admin/user context without cross-precedence."""
    _require_admin(request)
    reg = _model_registry()
    try:
        values = await reg.set_context_selection(context, body.values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"context": context, "values": values}


# ── Registry endpoints ─────────────────────────────────────────────────────────


class YamlProviderTemplate(BaseModel):
    """Well-known provider definition from models.yaml — offered as a template."""
    id: str
    label: str
    base_url: str
    api_key_env: str | None = None
    model_count: int = 0


class RegistryProviderInfo(BaseModel):
    id: str
    base_url: str
    api_key_env: str | None = None
    probe_url: str | None = None
    probe_timeout: float
    dynamic: bool
    available: bool | None = None


class RegistryModelInfo(BaseModel):
    id: str
    display_name: str
    context_window: int = 0
    capabilities: list[str] = Field(default_factory=list)
    enrichment: dict[str, Any] = Field(default_factory=dict)
    dynamic: bool = False
    provenance: str = "static"


@router.get("/registry/well-known-providers", response_model=list[YamlProviderTemplate])
async def list_well_known_providers(request: Request) -> list[YamlProviderTemplate]:
    """List well-known provider templates from models.yaml.

    These are known services (OpenAI, Anthropic, OpenRouter) whose config
    is defined in models.yaml but are not auto-registered. Users can pick
    one from this list to quickly add a provider with pre-filled config.
    """
    _require_admin(request)
    reg = _model_registry()
    known = reg.well_known_providers()
    return [
        YamlProviderTemplate(
            id=info["id"],
            label=info["label"],
            base_url=info["base_url"],
            api_key_env=info["api_key_env"],
            model_count=info["model_count"],
        )
        for info in sorted(known.values(), key=lambda x: x["label"])
    ]


@router.get("/registry/providers", response_model=list[RegistryProviderInfo])
async def list_registry_providers(request: Request) -> list[RegistryProviderInfo]:
    """List all providers from the model registry."""
    _require_admin(request)
    reg = _model_registry()
    providers = reg.all_providers()
    # Probe once for availability
    await asyncio.gather(*[reg._ensure_probed(p) for p in providers])
    return [
        RegistryProviderInfo(
            id=p.id,
            base_url=p.base_url,
            api_key_env=p.api_key_env,
            probe_url=p.probe_url,
            probe_timeout=p.probe_timeout,
            dynamic=p.dynamic,
            available=p.available,
        )
        for p in sorted(providers, key=lambda x: x.id)
    ]


@router.get("/registry/providers/{provider_id}/models", response_model=list[RegistryModelInfo])
async def list_provider_models(provider_id: str, request: Request) -> list[RegistryModelInfo]:
    """List models for a specific provider from the registry."""
    _require_admin(request)
    reg = _model_registry()
    provider = reg.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")
    # Probe to ensure availability is known
    await reg._ensure_probed(provider)
    # Merge dynamic models
    if provider.dynamic:
        models = reg._merge_dynamic_models(provider)
    else:
        models = list(provider.models)
    return [
        RegistryModelInfo(
            id=m.id,
            display_name=m.display_name,
            context_window=m.context_window,
            capabilities=m.capabilities,
            enrichment={
                "json_mode": m.enrichment.json_mode,
                "prompt_style": m.enrichment.prompt_style,
                "temperature": m.enrichment.temperature,
                "max_tokens": m.enrichment.max_tokens,
            },
            dynamic=m.dynamic,
            provenance=m.provenance,
        )
        for m in sorted(models, key=lambda x: x.id)
    ]


@router.post("/registry/reload", response_model=dict[str, Any])
async def registry_reload(request: Request) -> dict[str, Any]:
    """Reset probe cache and reload the model registry."""
    _require_admin(request)
    reg = _model_registry()
    reg.reset_probe_cache()
    models = await reg.available_models()
    providers = reg.all_providers()
    return {
        "status": "reloaded",
        "model_count": len(models),
        "provider_count": len(providers),
    }


@router.post("/keys", response_model=CreateKeyResponse)
async def create_key(body: CreateKeyRequest, request: Request) -> CreateKeyResponse:
    """Create a new API key."""
    _require_admin(request)
    from dewie.auth import ALL_SCOPES, create_api_key

    invalid = [s for s in body.scopes if s not in ALL_SCOPES]
    if invalid:
        raise HTTPException(
            status_code=400, detail=f"Invalid scopes: {invalid}. Valid: {ALL_SCOPES}"
        )

    pg = _pg(request)

    # The founder key: the middleware flags the fresh-install bootstrap request
    # (no keys exist yet). Force full scopes so it can actually administer the
    # instance — otherwise a default {"name": "..."} POST yields a useless
    # read-only key and the operator is locked out of admin.
    scopes = body.scopes
    if getattr(request.state, "bootstrap_founder", False):
        scopes = list(ALL_SCOPES)

    raw_key, record = await create_api_key(
        pg,
        workspace_ids=body.workspace_ids,
        name=body.name,
        scopes=scopes,
        live=body.live,
    )

    await _audit(request, "key.create", "api_key", str(record["id"]))

    return CreateKeyResponse(
        key=raw_key,
        record=KeyResponse(
            id=record["id"],
            workspace_ids=record["workspace_ids"],
            key_prefix=record["key_prefix"],
            scopes=record["scopes"],
            name=record["name"],
            created_at=str(record["created_at"]),
        ),
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config(request: Request) -> ConfigResponse:
    """List editable runtime config values with source and reload behavior."""
    _require_admin(request)
    path = _config_path()
    cfg = _load_config(path)

    values = [
        ConfigValue(key=k, value=_effective_value(k, cfg)[0], source=_effective_value(k, cfg)[1])
        for k in _CONFIG_KEY_META
    ]
    metadata = [
        ConfigFieldMeta(
            key=k,
            reload_behavior=v["reload_behavior"],
            description=v["description"],
            section=v.get("section", "general"),
        )
        for k, v in _CONFIG_KEY_META.items()
    ]
    return ConfigResponse(file_path=str(path), values=values, metadata=metadata)


@router.patch("/config", response_model=ConfigSetResponse)
async def set_config(body: ConfigSetRequest, request: Request) -> ConfigSetResponse:
    """Set one editable config value in dewie.yml."""
    _require_admin(request)

    value_type = body.value_type.lower()
    if value_type not in {"str", "int", "float", "bool", "json"}:
        raise HTTPException(status_code=400, detail=f"Invalid value_type: {body.value_type}")

    value = _coerce_value(body.value, value_type)
    _validate_config_update(body.path, value)

    path = _config_path()
    cfg = _load_config(path)
    cfg[body.path] = value
    _save_config(path, cfg)

    await _audit(request, "admin.config_update", "config", body.path, {"value_type": value_type})

    meta = _CONFIG_KEY_META[body.path]
    return ConfigSetResponse(
        ok=True,
        path=body.path,
        value=value,
        reload_behavior=meta["reload_behavior"],
        source="file",
    )


# ── Server registry (providers/servers.py) ────────────────────────────────────


class ServerEntry(BaseModel):
    label: str
    api_format: str = Field(description="openai | anthropic")
    endpoint: str
    api_key_env: str | None = None
    api_key_file_env: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict | None = None
    builtin: bool = Field(default=False, description="True for openai/anthropic defaults")
    has_api_key: bool = Field(default=False, description="True if a literal key is stored encrypted")


class ServerListResponse(BaseModel):
    servers: list[ServerEntry]


class ServerUpsertRequest(BaseModel):
    label: str
    api_format: str = Field(default="openai")
    endpoint: str
    api_key_env: str | None = None
    api_key_file_env: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict | None = None
    api_key: str | None = Field(
        default=None,
        description=(
            "Literal API key, stored encrypted (write-only — never echoed back). "
            "Omit to leave an existing stored key untouched."
        ),
    )


_VALID_API_FORMATS = frozenset({"openai", "anthropic"})


@router.get("/servers", response_model=ServerListResponse)
async def list_servers(request: Request) -> ServerListResponse:
    """List all registered servers (built-in + dewie.yml `servers:` entries)."""
    _require_admin(request)
    from dewie.providers.servers import _BUILTIN_SERVERS, load_servers

    servers = load_servers()
    return ServerListResponse(
        servers=[
            ServerEntry(
                label=s.label,
                api_format=s.api_format,
                endpoint=s.endpoint,
                api_key_env=s.api_key_env,
                api_key_file_env=s.api_key_file_env,
                extra_headers=s.extra_headers,
                extra_body=s.extra_body,
                builtin=label in _BUILTIN_SERVERS and s == _BUILTIN_SERVERS.get(label),
                has_api_key=bool(s.api_key_ciphertext),
            )
            for label, s in sorted(servers.items())
        ]
    )


@router.put("/servers/{label}", response_model=ServerEntry)
async def upsert_server(label: str, body: ServerUpsertRequest, request: Request) -> ServerEntry:
    """Create or update a server entry in dewie.yml's `servers:` list."""
    _require_admin(request)
    if body.label != label:
        raise HTTPException(status_code=400, detail="label in body must match URL path")
    if body.api_format not in _VALID_API_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid api_format '{body.api_format}'. Valid: {sorted(_VALID_API_FORMATS)}",
        )
    if not body.endpoint.strip():
        raise HTTPException(status_code=400, detail="endpoint is required")

    path = _config_path()
    cfg = _load_config(path)
    existing = next((s for s in cfg.get("servers", []) if s.get("label") == label), None)
    servers: list[dict] = [s for s in cfg.get("servers", []) if s.get("label") != label]

    if body.api_key:
        if not settings.encryption_master_key:
            raise HTTPException(
                status_code=503,
                detail="ENCRYPTION_MASTER_KEY is not configured — cannot store a literal API key",
            )
        from dewie.crypto import encrypt

        api_key_ciphertext = encrypt(body.api_key)
    else:
        # Omitted -> leave any previously stored key untouched.
        api_key_ciphertext = existing.get("api_key_ciphertext") if existing else None

    servers.append(
        {
            "label": label,
            "api_format": body.api_format,
            "endpoint": body.endpoint.strip(),
            "api_key_env": body.api_key_env,
            "api_key_file_env": body.api_key_file_env,
            "extra_headers": body.extra_headers,
            "extra_body": body.extra_body,
            "api_key_ciphertext": api_key_ciphertext,
        }
    )
    cfg["servers"] = servers
    _save_config(path, cfg)

    await _audit(request, "admin.server_upsert", "server", label)

    from dewie.model_registry import registry as _registry
    from dewie.providers.servers import get_server

    _registry._loaded = False  # re-sync the matching provider entry on next load
    saved = get_server(label)
    return ServerEntry(
        label=saved.label,
        api_format=saved.api_format,
        endpoint=saved.endpoint,
        api_key_env=saved.api_key_env,
        api_key_file_env=saved.api_key_file_env,
        extra_headers=saved.extra_headers,
        extra_body=saved.extra_body,
        builtin=False,
        has_api_key=bool(saved.api_key_ciphertext),
    )


@router.delete("/servers/{label}", status_code=204)
async def delete_server(label: str, request: Request) -> None:
    """Remove a user-defined server entry from dewie.yml. Built-in servers
    (openai, anthropic) cannot be deleted, only shadowed by
    re-registering the same label with different settings."""
    _require_admin(request)
    from dewie.providers.servers import _BUILTIN_SERVERS

    path = _config_path()
    cfg = _load_config(path)
    servers: list[dict] = cfg.get("servers", [])
    remaining = [s for s in servers if s.get("label") != label]
    if len(remaining) == len(servers):
        if label in _BUILTIN_SERVERS:
            raise HTTPException(status_code=400, detail=f"'{label}' is a built-in server and cannot be deleted")
        raise HTTPException(status_code=404, detail=f"No such server: {label}")
    cfg["servers"] = remaining
    _save_config(path, cfg)
    await _audit(request, "admin.server_delete", "server", label)

    from dewie.model_registry import registry as _registry

    await _registry.remove_provider(label)
    _registry._loaded = False

# ── Model Registry management ─────────────────────────────────────────────────


class ProviderListItem(BaseModel):
    id: str
    base_url: str
    dynamic: bool
    available: bool | None


class ModelListItem(BaseModel):
    id: str
    display_name: str
    context_window: int
    capabilities: list[str]
    dynamic: bool
    available: bool


class RegistryReloadResponse(BaseModel):
    status: str
    provider_count: int
    model_count: int


@router.post("/registry/reload", response_model=RegistryReloadResponse)
async def reload_registry(request: Request) -> RegistryReloadResponse:
    """Reload the model registry from disk and re-probe dynamic providers."""
    _require_admin(request)
    from dewie.model_registry import registry as _registry

    _registry._loaded = False
    _registry._load()
    _registry.reset_probe_cache()
    models = await _registry.available_models()
    providers = _registry.all_providers()
    return RegistryReloadResponse(
        status="reloaded",
        provider_count=len(providers),
        model_count=len(models),
    )


@router.get("/keys", response_model=list[KeyResponse])
async def list_keys(request: Request) -> list[KeyResponse]:
    """List all active API keys (non-sensitive fields only)."""
    _require_admin(request)
    from sqlalchemy import text as _text

    pg = _pg(request)
    async with pg._engine.connect() as conn:
        rows = await conn.execute(
            _text(
                "SELECT id, workspace_ids, key_prefix, scopes, name, created_at "
                "FROM api_keys WHERE revoked_at IS NULL "
                "ORDER BY created_at DESC"
            )
        )
        records = rows.mappings().fetchall()

    # SQLite stores the array columns as JSON TEXT; Postgres returns lists.
    is_sqlite = bool(getattr(pg, "_is_sqlite", False))

    def _as_list(v: Any) -> list:
        if is_sqlite and isinstance(v, str):
            import json as _json
            return _json.loads(v) if v else []
        return list(v or [])

    return [
        KeyResponse(
            id=row["id"],
            workspace_ids=[uuid.UUID(str(w)) for w in _as_list(row["workspace_ids"])],
            key_prefix=row["key_prefix"],
            scopes=_as_list(row["scopes"]),
            name=row["name"],
            created_at=str(row["created_at"]),
        )
        for row in records
    ]


@router.delete("/keys/{key_id}", status_code=204)
async def revoke_key(key_id: uuid.UUID, request: Request) -> None:
    """Revoke an API key. Revoked keys cannot be re-activated."""
    _require_admin(request)
    from dewie.auth import revoke_api_key

    pg = _pg(request)
    revoked = await revoke_api_key(pg, key_id=key_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="Key not found or already revoked")

    await _audit(request, "key.revoke", "api_key", str(key_id))


# ── Workspace management ──────────────────────────────────────────────────────


class CreateWorkspaceRequest(BaseModel):
    name: str = Field(description="Human-readable workspace name")
    parent_id: uuid.UUID | None = Field(default=None, description="Parent workspace for nesting")
    sharing_tier: str = Field(default="internal_only", description="Sharing tier")


class WorkspaceResponse(BaseModel):
    id: uuid.UUID
    name: str
    parent_id: uuid.UUID | None
    sharing_tier: str
    created_at: str


@router.post("/workspaces", response_model=WorkspaceResponse)
async def create_workspace(body: CreateWorkspaceRequest, request: Request) -> WorkspaceResponse:
    """Create a new workspace."""
    _require_admin(request)
    pg = _pg(request)
    ws = await pg.create_workspace(
        name=body.name,
        parent_id=body.parent_id,
        sharing_tier=body.sharing_tier,
    )

    await _audit(request, "workspace.create", "workspace", str(ws["id"]), {"name": body.name})

    return WorkspaceResponse(
        id=ws["id"],
        name=ws["name"],
        parent_id=ws.get("parent_id"),
        sharing_tier=ws["sharing_tier"],
        created_at=str(ws["created_at"]),
    )


@router.get("/workspaces", response_model=list[WorkspaceResponse])
async def list_workspaces(request: Request) -> list[WorkspaceResponse]:
    """List all workspaces."""
    _require_admin(request)
    pg = _pg(request)
    workspaces = await pg.get_workspaces()
    return [
        WorkspaceResponse(
            id=ws["id"],
            name=ws["name"],
            parent_id=ws.get("parent_id"),
            sharing_tier=ws["sharing_tier"],
            created_at=str(ws["created_at"]),
        )
        for ws in workspaces
    ]


@router.delete("/workspaces/{workspace_id}", status_code=204)
async def delete_workspace(workspace_id: uuid.UUID, request: Request) -> None:
    """Delete a workspace and all its corpora."""
    _require_admin(request)
    pg = _pg(request)
    await pg.delete_workspace(workspace_id)

    await _audit(request, "workspace.delete", "workspace", str(workspace_id))


# ── Corpus management ─────────────────────────────────────────────────────────


class CreateCorpusRequest(BaseModel):
    name: str = Field(description="Human-readable corpus name")
    slug: str = Field(description="URL-safe identifier (e.g. 'my-corpus')")
    workspace_id: uuid.UUID = Field(description="Parent workspace")
    sharing_tier: str = Field(default="internal_only", description="Sharing tier")


class CorpusResponse(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    workspace_id: uuid.UUID
    sharing_tier: str
    created_at: str


@router.post("/corpora", response_model=CorpusResponse)
async def create_corpus(body: CreateCorpusRequest, request: Request) -> CorpusResponse:
    """Create a new corpus within a workspace."""
    _require_admin(request)
    pg = _pg(request)
    corpus = await pg.create_corpus(
        name=body.name,
        slug=body.slug,
        workspace_id=body.workspace_id,
        sharing_tier=body.sharing_tier,
    )

    await _audit(request, "corpus.create", "corpus", str(corpus["id"]), {"name": body.name})

    return CorpusResponse(
        id=corpus["id"],
        name=corpus["name"],
        slug=corpus["slug"],
        workspace_id=corpus["workspace_id"],
        sharing_tier=corpus["sharing_tier"],
        created_at=str(corpus["created_at"]),
    )


@router.get("/corpora", response_model=list[CorpusResponse])
async def list_corpora(
    request: Request,
    workspace_id: uuid.UUID | None = None,
) -> list[CorpusResponse]:
    """List corpora, optionally filtered by workspace."""
    _require_admin(request)
    pg = _pg(request)
    corpora = await pg.get_corpora(workspace_id=workspace_id)
    return [
        CorpusResponse(
            id=c["id"],
            name=c["name"],
            slug=c["slug"],
            workspace_id=c["workspace_id"],
            sharing_tier=c["sharing_tier"],
            created_at=str(c["created_at"]),
        )
        for c in corpora
    ]


@router.delete("/corpora/{corpus_id}", status_code=204)
async def delete_corpus(corpus_id: uuid.UUID, request: Request) -> None:
    """Delete a corpus. Documents referencing it will have corpus_id set to NULL."""
    _require_admin(request)
    pg = _pg(request)
    await pg.delete_corpus(corpus_id)

    await _audit(request, "corpus.delete", "corpus", str(corpus_id))


# ── Catalog management (Issue #386) ──────────────────────────────────────────

class CreateCatalogRequest(BaseModel):
    name: str = Field(description="Human-readable catalog name")
    type: str = Field(description="Catalog type: sqlite|postgres|mcp")
    config: dict[str, Any] = Field(default_factory=dict, description="Type-specific config")
    enabled: bool = Field(default=True, description="Whether catalog is active")


class UpdateCatalogRequest(BaseModel):
    name: str | None = Field(default=None, description="New catalog name")
    type: str | None = Field(default=None, description="New catalog type")
    config: dict[str, Any] | None = Field(default=None, description="New catalog config")
    enabled: bool | None = Field(default=None, description="Enable/disable catalog")


class CatalogSourceResponse(BaseModel):
    id: uuid.UUID
    name: str
    type: str
    config: dict[str, Any]
    enabled: bool
    created_by: str | None
    created_at: str
    tested_at: str | None
    test_status: str | None
    test_error: str | None
    updated_at: str


class TestCatalogResponse(BaseModel):
    ok: bool
    error: str | None = None


def _to_catalog_response(src: dict[str, Any]) -> CatalogSourceResponse:
    return CatalogSourceResponse(
        id=uuid.UUID(str(src["id"])),
        name=str(src["name"]),
        type=str(src["type"]),
        config=src.get("config") if isinstance(src.get("config"), dict) else {},
        enabled=bool(src.get("enabled", True)),
        created_by=str(src["created_by"]) if src.get("created_by") else None,
        created_at=str(src.get("created_at") or ""),
        tested_at=str(src["tested_at"]) if src.get("tested_at") else None,
        test_status=str(src["test_status"]) if src.get("test_status") else None,
        test_error=str(src["test_error"]) if src.get("test_error") else None,
        updated_at=str(src.get("updated_at") or ""),
    )


@router.post("/catalogs", response_model=CatalogSourceResponse, status_code=201)
async def create_catalog(body: CreateCatalogRequest, request: Request) -> CatalogSourceResponse:
    """Create a new data catalog."""
    _require_admin(request)
    if body.type not in _VALID_SOURCE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid catalog type '{body.type}'. Valid: {sorted(_VALID_SOURCE_TYPES)}",
        )

    pg = _pg(request)
    if not inspect.iscoroutinefunction(getattr(pg, "create_source", None)):
        from sqlalchemy import text as _text

        src_id = uuid.uuid4()
        async with pg._engine.connect() as conn:
            try:
                await conn.execute(
                    _text(
                        """
                        INSERT INTO dewie_sources (id, name, type, config_json, enabled, created_at, updated_at)
                        VALUES (:id, :name, :type, :config_json, :enabled, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """
                    ),
                    {
                        "id": str(src_id),
                        "name": body.name,
                        "type": body.type,
                        "config_json": json.dumps(body.config),
                        "enabled": body.enabled,
                    },
                )
                await conn.commit()
            except Exception as exc:
                await conn.rollback()
                msg = str(exc)
                if "UNIQUE constraint failed" in msg or "unique" in msg.lower():
                    raise HTTPException(
                        status_code=409,
                        detail=f"Catalog '{body.name}' already exists",
                    ) from exc
                raise HTTPException(status_code=500, detail=f"Database error: {msg}") from exc

        return CatalogSourceResponse(
            id=src_id,
            name=body.name,
            type=body.type,
            config=body.config,
            enabled=body.enabled,
            created_by=None,
            created_at="",
            tested_at=None,
            test_status=None,
            test_error=None,
            updated_at="",
        )

    try:
        source = await pg.create_source(
            source_id=uuid.uuid4(),
            name=body.name,
            source_type=body.type,
            config=body.config,
            enabled=body.enabled,
        )
    except Exception as exc:
        msg = str(exc)
        if "unique" in msg.lower() or "duplicate" in msg.lower():
            raise HTTPException(status_code=409, detail=f"Catalog '{body.name}' already exists") from exc
        raise HTTPException(status_code=500, detail=f"Database error: {msg}") from exc

    await _audit(request, "source.create", "source", str(source["id"]), {"name": body.name})
    return _to_catalog_response(source)


@router.get("/catalogs", response_model=list[CatalogSourceResponse])
async def list_catalogs(request: Request) -> list[CatalogSourceResponse]:
    """List configured catalogs."""
    _require_admin(request)
    pg = _pg(request)
    if not inspect.iscoroutinefunction(getattr(pg, "list_sources", None)):
        from sqlalchemy import text as _text

        async with pg._engine.connect() as conn:
            rows = (
                await conn.execute(
                    _text(
                        """
                        SELECT id, name, type, config_json, enabled, created_by,
                               created_at, tested_at, test_status, test_error, updated_at
                        FROM dewie_sources
                        ORDER BY created_at DESC
                        """
                    )
                )
            ).mappings().fetchall()

        parsed = []
        for row in rows:
            cfg_raw = row.get("config_json")
            if isinstance(cfg_raw, str):
                try:
                    cfg = json.loads(cfg_raw)
                except Exception:
                    cfg = {}
            else:
                cfg = cfg_raw or {}
            parsed.append(
                _to_catalog_response(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "type": row["type"],
                        "config": cfg,
                        "enabled": row.get("enabled", True),
                        "created_by": row.get("created_by"),
                        "created_at": row.get("created_at"),
                        "tested_at": row.get("tested_at"),
                        "test_status": row.get("test_status"),
                        "test_error": row.get("test_error"),
                        "updated_at": row.get("updated_at"),
                    }
                )
            )
        return parsed

    rows = await pg.list_sources()
    return [_to_catalog_response(row) for row in rows]


@router.patch("/catalogs/{source_id}", response_model=CatalogSourceResponse)
async def update_catalog(
    source_id: uuid.UUID,
    body: UpdateCatalogRequest,
    request: Request,
) -> CatalogSourceResponse:
    """Update an existing catalog."""
    _require_admin(request)
    if body.type is not None and body.type not in _VALID_SOURCE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid catalog type '{body.type}'. Valid: {sorted(_VALID_SOURCE_TYPES)}",
        )

    pg = _pg(request)
    if not inspect.iscoroutinefunction(getattr(pg, "update_source", None)):
        from sqlalchemy import text as _text

        set_parts: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
        params: dict[str, Any] = {"id": str(source_id)}
        if body.name is not None:
            set_parts.append("name = :name")
            params["name"] = body.name
        if body.type is not None:
            set_parts.append("type = :type")
            params["type"] = body.type
        if body.config is not None:
            set_parts.append("config_json = :config_json")
            params["config_json"] = json.dumps(body.config)
        if body.enabled is not None:
            set_parts.append("enabled = :enabled")
            params["enabled"] = body.enabled

        async with pg._engine.connect() as conn:
            result = await conn.execute(
                _text(f"UPDATE dewie_sources SET {', '.join(set_parts)} WHERE id = :id"),
                params,
            )
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Catalog not found")

            row = (
                await conn.execute(
                    _text(
                        """
                        SELECT id, name, type, config_json, enabled, created_by,
                               created_at, tested_at, test_status, test_error, updated_at
                        FROM dewie_sources
                        WHERE id = :id
                        """
                    ),
                    {"id": str(source_id)},
                )
            ).mappings().fetchone()
            await conn.commit()

        if row is None:
            raise HTTPException(status_code=404, detail="Catalog not found")
        await _audit(request, "source.update", "source", str(source_id), {"name": body.name})
        return _to_catalog_response(
            {
                **dict(row),
                "config": json.loads(row["config_json"]) if isinstance(row.get("config_json"), str) else (row.get("config_json") or {}),
            }
        )

    updated = await pg.update_source(
        source_id,
        name=body.name,
        source_type=body.type,
        config=body.config,
        enabled=body.enabled,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Catalog not found")
    await _audit(request, "source.update", "source", str(source_id), {"name": body.name})
    return _to_catalog_response(updated)


@router.delete("/catalogs/{source_id}", status_code=204)
async def delete_catalog(source_id: uuid.UUID, request: Request) -> None:
    """Delete a catalog by id."""
    _require_admin(request)
    pg = _pg(request)
    if not inspect.iscoroutinefunction(getattr(pg, "delete_source", None)):
        from sqlalchemy import text as _text

        async with pg._engine.connect() as conn:
            result = await conn.execute(
                _text("DELETE FROM dewie_sources WHERE id = :id"),
                {"id": str(source_id)},
            )
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Catalog not found")
            await conn.commit()

        await _audit(request, "source.delete", "source", str(source_id))
        return

    deleted = await pg.delete_source(source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Catalog not found")

    await _audit(request, "source.delete", "source", str(source_id))


@router.post("/catalogs/{source_id}/test", response_model=TestCatalogResponse)
async def test_catalog(source_id: uuid.UUID, request: Request) -> TestCatalogResponse:
    """Run lightweight validation for a catalog and persist test status."""
    _require_admin(request)
    pg = _pg(request)
    if not inspect.iscoroutinefunction(getattr(pg, "get_source", None)):
        from sqlalchemy import text as _text

        async with pg._engine.connect() as conn:
            row = (
                await conn.execute(
                    _text("SELECT type, config_json FROM dewie_sources WHERE id = :id"),
                    {"id": str(source_id)},
                )
            ).mappings().fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Catalog not found")

            src_type = str(row.get("type", ""))
            cfg_raw = row.get("config_json")
            if isinstance(cfg_raw, str):
                try:
                    config = json.loads(cfg_raw)
                except Exception:
                    config = {}
            else:
                config = cfg_raw or {}

            ok = False
            error: str | None = None
            if src_type == "sqlite":
                filepath = str(config.get("filepath", "")).strip()
                if not filepath:
                    error = "Missing filepath in config"
                elif not os.path.exists(filepath):
                    error = f"File not found: {filepath}"
                else:
                    ok = True
            elif src_type == "postgres":
                has_dsn = bool(str(config.get("dsn", "")).strip())
                if not has_dsn:
                    required = ("database", "user")
                    missing = [f for f in required if not config.get(f)]
                    if missing:
                        error = f"Missing required fields: {', '.join(missing)}"
                    else:
                        ok, error = await _test_postgres_connection(config)
                else:
                    ok, error = await _test_postgres_connection(config)
            elif src_type == "mcp":
                ok, error = await _test_mcp_connection(config)
            else:
                error = f"Unknown catalog type: {src_type}"

            await conn.execute(
                _text(
                    """
                    UPDATE dewie_sources
                    SET tested_at = CURRENT_TIMESTAMP,
                        test_status = :status,
                        test_error = :error,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :id
                    """
                ),
                {"status": "ok" if ok else "error", "error": error, "id": str(source_id)},
            )
            await conn.commit()

        return TestCatalogResponse(ok=ok, error=error)

    source = await pg.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Catalog not found")

    src_type = str(source.get("type", ""))
    config = source.get("config") if isinstance(source.get("config"), dict) else {}

    ok = False
    error: str | None = None
    if src_type == "sqlite":
        filepath = str(config.get("filepath", "")).strip()
        if not filepath:
            error = "Missing filepath in config"
        elif not os.path.exists(filepath):
            error = f"File not found: {filepath}"
        else:
            ok = True
    elif src_type == "postgres":
        has_dsn = bool(str(config.get("dsn", "")).strip())
        if not has_dsn:
            required = ("database", "user")
            missing = [f for f in required if not config.get(f)]
            if missing:
                error = f"Missing required fields: {', '.join(missing)}"
            else:
                ok, error = await _test_postgres_connection(config)
        else:
            ok, error = await _test_postgres_connection(config)
    elif src_type == "mcp":
        ok, error = await _test_mcp_connection(config)
    else:
        error = f"Unknown catalog type: {src_type}"

    await pg.set_source_test_result(source_id, ok=ok, error=error)
    return TestCatalogResponse(ok=ok, error=error)


# ── Local user management ──────────────────────────────────────────────────────


class AdminUserResponse(BaseModel):
    id: str
    email: str
    name: str | None
    is_admin: bool
    activation_status: str
    has_password: bool
    created_at: str
    last_login_at: str | None = None


class CreateAdminUserRequest(BaseModel):
    email: str = Field(description="User email address")
    password: str = Field(description="Initial password (min 8 characters)")
    name: str | None = Field(default=None, description="Display name")
    is_admin: bool = Field(default=False, description="Grant admin privileges")
    activation_status: str = Field(default="approved", description="pending|approved|rejected")


class UpdateAdminUserRequest(BaseModel):
    name: str | None = Field(default=None, description="Display name")
    is_admin: bool | None = Field(default=None, description="Admin privileges")
    activation_status: str | None = Field(
        default=None,
        description="pending|approved|rejected",
    )


class SetUserPasswordRequest(BaseModel):
    password: str = Field(description="New password (min 8 characters)")


@router.post("/users", response_model=AdminUserResponse, status_code=201)
async def create_user(body: CreateAdminUserRequest, request: Request) -> AdminUserResponse:
    """Create a new local user."""
    _require_admin(request)
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if body.activation_status not in _VALID_ACTIVATION_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid activation_status '{body.activation_status}'. "
                f"Valid: {sorted(_VALID_ACTIVATION_STATUSES)}"
            ),
        )
    from dewie.local_auth import create_local_user

    pg = _pg(request)
    try:
        user = await create_local_user(pg, email=body.email, password=body.password, name=body.name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # Apply is_admin / activation_status if non-default
    if body.is_admin or body.activation_status != "approved":
        updated = await pg.update_local_user(
            user_id=uuid.UUID(str(user["id"])),
            is_admin=body.is_admin,
            activation_status=body.activation_status,
        )
        if updated:
            user = updated

    await _audit(request, "user.create", "user", str(user["id"]), {"email": body.email})

    return AdminUserResponse(
        id=str(user["id"]),
        email=user["email"],
        name=user.get("name"),
        is_admin=bool(user.get("is_admin", False)),
        activation_status=str(user.get("activation_status") or "approved"),
        has_password=True,
        created_at=str(user["created_at"]),
        last_login_at=None,
    )


@router.get("/users", response_model=list[AdminUserResponse])
async def list_users(request: Request) -> list[AdminUserResponse]:
    """List local users for admin management."""
    _require_admin(request)
    pg = _pg(request)
    users = await pg.get_local_users()
    return [
        AdminUserResponse(
            id=str(u["id"]),
            email=u["email"],
            name=u.get("name"),
            is_admin=bool(u.get("is_admin", False)),
            activation_status=str(u.get("activation_status") or "pending"),
            has_password=bool(u.get("has_password", False)),
            created_at=str(u["created_at"]),
            last_login_at=str(u["last_login_at"]) if u.get("last_login_at") else None,
        )
        for u in users
    ]


@router.patch("/users/{user_id}", response_model=AdminUserResponse)
async def update_user(
    user_id: uuid.UUID,
    body: UpdateAdminUserRequest,
    request: Request,
) -> AdminUserResponse:
    """Update user profile/admin/access fields."""
    _require_admin(request)
    if body.activation_status is not None and body.activation_status not in _VALID_ACTIVATION_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid activation_status "
                f"'{body.activation_status}'. Valid: {sorted(_VALID_ACTIVATION_STATUSES)}"
            ),
        )
    pg = _pg(request)
    updated = await pg.update_local_user(
        user_id=user_id,
        name=body.name,
        is_admin=body.is_admin,
        activation_status=body.activation_status,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    await _audit(request, "user.update", "user", str(user_id))
    return AdminUserResponse(
        id=str(updated["id"]),
        email=updated["email"],
        name=updated.get("name"),
        is_admin=bool(updated.get("is_admin", False)),
        activation_status=str(updated.get("activation_status") or "pending"),
        has_password=bool(updated.get("has_password", False)),
        created_at=str(updated["created_at"]),
        last_login_at=str(updated.get("last_login_at")) if updated.get("last_login_at") else None,
    )


@router.post("/users/{user_id}/password", status_code=204)
async def set_user_password(
    user_id: uuid.UUID,
    body: SetUserPasswordRequest,
    request: Request,
) -> None:
    """Set a local user's password."""
    _require_admin(request)
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    from dewie.local_auth import hash_password

    pg = _pg(request)
    updated = await pg.update_local_user(
        user_id=user_id,
        password_hash=hash_password(body.password),
    )
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    await _audit(request, "user.password_change", "user", str(user_id))


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: uuid.UUID, request: Request) -> None:
    """Delete a local user."""
    _require_admin(request)
    pg = _pg(request)
    deleted = await pg.delete_local_user(user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")

    await _audit(request, "user.delete", "user", str(user_id))


@router.post("/users/{user_id}/suspend", response_model=AdminUserResponse)
async def suspend_user(user_id: uuid.UUID, request: Request) -> AdminUserResponse:
    """Suspend a user by setting activation_status to rejected."""
    _require_admin(request)
    pg = _pg(request)
    updated = await pg.update_local_user(user_id=user_id, activation_status="rejected")
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    await _audit(request, "user.suspend", "user", str(user_id))
    return AdminUserResponse(
        id=str(updated["id"]),
        email=updated["email"],
        name=updated.get("name"),
        is_admin=bool(updated.get("is_admin", False)),
        activation_status=str(updated.get("activation_status") or "pending"),
        has_password=bool(updated.get("has_password", False)),
        created_at=str(updated["created_at"]),
        last_login_at=str(updated["last_login_at"]) if updated.get("last_login_at") else None,
    )


@router.post("/users/{user_id}/unsuspend", response_model=AdminUserResponse)
async def unsuspend_user(user_id: uuid.UUID, request: Request) -> AdminUserResponse:
    """Reactivate a user by setting activation_status to approved."""
    _require_admin(request)
    pg = _pg(request)
    updated = await pg.update_local_user(user_id=user_id, activation_status="approved")
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    await _audit(request, "user.unsuspend", "user", str(user_id))
    return AdminUserResponse(
        id=str(updated["id"]),
        email=updated["email"],
        name=updated.get("name"),
        is_admin=bool(updated.get("is_admin", False)),
        activation_status=str(updated.get("activation_status") or "pending"),
        has_password=bool(updated.get("has_password", False)),
        created_at=str(updated["created_at"]),
        last_login_at=str(updated["last_login_at"]) if updated.get("last_login_at") else None,
    )


class UserDocumentResponse(BaseModel):
    id: str
    title: str
    source_url: str | None = None
    status: str
    corpus_id: str
    owner_user_id: str | None = None
    created_at: str


@router.get("/users/{user_id}/documents", response_model=list[UserDocumentResponse])
async def list_user_documents(user_id: uuid.UUID, request: Request) -> list[UserDocumentResponse]:
    """List documents owned by a user."""
    _require_admin(request)
    pg = _pg(request)
    docs = await pg.get_user_documents(user_id=user_id, limit=100, offset=0)
    return [
        UserDocumentResponse(
            id=str(d["id"]),
            title=str(d["title"]),
            source_url=d.get("source_url"),
            status=str(d["status"]),
            corpus_id=str(d["corpus_id"]),
            owner_user_id=str(d["owner_user_id"]) if d.get("owner_user_id") else None,
            created_at=str(d["created_at"]),
        )
        for d in docs
    ]


@router.delete("/users/{user_id}/documents/{doc_id}", status_code=204)
async def delete_user_document(user_id: uuid.UUID, doc_id: uuid.UUID, request: Request) -> None:
    """Hard-delete a user document."""
    _require_admin(request)
    pg = _pg(request)
    deleted = await pg.delete_user_document(user_id, doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    await _audit(request, "user_doc.delete", "user_document", str(doc_id))


@router.post("/users/{user_id}/documents/{doc_id}/disconnect", status_code=204)
async def disconnect_user_document(user_id: uuid.UUID, doc_id: uuid.UUID, request: Request) -> None:
    """Clear owner_user_id from a user document."""
    _require_admin(request)
    pg = _pg(request)
    disconnected = await pg.disconnect_user_document(user_id, doc_id)
    if not disconnected:
        raise HTTPException(status_code=404, detail="Document not found")

    await _audit(request, "user_doc.disconnect", "user_document", str(doc_id))


# ── Query log viewer ──────────────────────────────────────────────────────────


@router.get("/query-log")
async def list_query_log(request: Request, limit: int = 100) -> list:  # type: ignore[return]
    """List recent query log entries. Requires admin session."""
    _require_admin(request)
    pg = _pg(request)
    import inspect as _inspect

    from sqlalchemy import text as _text
    # Prefer the high-level pg method when it is a *real* coroutine function
    # (not a MagicMock auto-attribute, which is not awaitable).
    _ql = getattr(pg, "get_query_log", None)
    if _ql is not None and _inspect.iscoroutinefunction(_ql):
        return await pg.get_query_log(tenant_id=request.state.tenant_id, limit=limit)

    async with pg._engine.connect() as conn:
        rows = await conn.execute(
            _text(
                """
            SELECT id, ts, source, question, model, hops, elapsed_ms, answer
            FROM query_log
            WHERE tenant_id = :tenant_id
            ORDER BY ts DESC
            LIMIT :limit
        """
            ),
            {"tenant_id": request.state.tenant_id, "limit": limit},
        )
        return [dict(r._mapping) for r in rows.fetchall()]


@router.get("/query-log/{query_id}")
async def get_query_log_entry(query_id: int, request: Request) -> dict:  # type: ignore[return]
    """Get full detail for a single query log entry. Requires admin session."""
    _require_admin(request)
    pg = _pg(request)
    # Tests often provide a convenience method on the pg mock.
    import inspect as _inspect

    from sqlalchemy import text as _text
    _qle = getattr(pg, "get_query_log_entry", None)
    if _qle is not None and _inspect.iscoroutinefunction(_qle):
        r = await pg.get_query_log_entry(query_id=query_id, tenant_id=request.state.tenant_id)
        if isinstance(r, dict):
            return r
    else:
        async with pg._engine.connect() as conn:
            row = await conn.execute(
                _text(
                    """
            SELECT id, ts, source, question, model, hops, elapsed_ms,
                   answer, hop_trace, docs_returned, full_results,
                   correct, input_tokens, output_tokens
            FROM query_log
            WHERE id = :id AND tenant_id = :tenant_id
        """
                ),
                {"id": query_id, "tenant_id": request.state.tenant_id},
            )
            r = row.fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Query not found")
    return dict(r._mapping)


# ── Feeds (all tenants) ────────────────────────────────────────────────────────

@router.get("/feeds")
async def admin_list_feeds(request: Request) -> list[dict]:
    """List all RSS feeds across all tenants. Requires admin session."""
    _require_admin(request)
    pg = _pg(request)
    from sqlalchemy import text as _text

    async with pg._engine.connect() as conn:
        rows = await conn.execute(
            _text("""
                SELECT id, name, url, corpus_id, tags, enabled,
                       poll_interval_minutes, last_polled_at, created_at, tenant_id
                FROM rss_feeds
                ORDER BY created_at DESC
            """)
        )
        return [dict(r._mapping) for r in rows.fetchall()]


@router.post("/feeds", status_code=201)
async def admin_create_feed(request: Request, body: dict) -> dict:
    """Create an RSS feed (admin — can specify any tenant_id). Requires admin session."""
    _require_admin(request)
    import uuid as _uuid

    from dewie.models.feed import RSSFeed

    pg = _pg(request)
    tenant_raw = body.get("tenant_id", "00000000-0000-0000-0000-000000000001")
    feed = RSSFeed(
        id=_uuid.uuid4(),
        url=body["url"],
        name=body["name"],
        corpus_id=body.get("corpus_id"),
        tags=body.get("tags", []),
        enabled=body.get("enabled", True),
        poll_interval_minutes=body.get("poll_interval_minutes", 60),
        tenant_id=_uuid.UUID(str(tenant_raw)),
    )
    created = await pg.create_feed(feed)
    await _audit(request, "feed.create", "feed", str(created.id), {"name": body["name"]})
    return created.model_dump(mode="json")


@router.patch("/feeds/{feed_id}")
async def admin_update_feed(feed_id: str, request: Request, body: dict) -> dict:
    """Update an RSS feed (admin). Requires admin session."""
    _require_admin(request)
    import uuid as _uuid

    pg = _pg(request)
    feed = await pg.update_feed(_uuid.UUID(feed_id), **{k: v for k, v in body.items() if v is not None})
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    await _audit(request, "feed.update", "feed", feed_id)
    return feed.model_dump(mode="json")


@router.delete("/feeds/{feed_id}", status_code=204)
async def admin_delete_feed(feed_id: str, request: Request) -> None:
    """Delete an RSS feed (admin). Requires admin session."""
    _require_admin(request)
    import uuid as _uuid

    pg = _pg(request)
    deleted = await pg.delete_feed(_uuid.UUID(feed_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Feed not found")

    await _audit(request, "feed.delete", "feed", feed_id)


@router.post("/feeds/{feed_id}/poll", status_code=202)
async def admin_poll_feed(feed_id: str, request: Request) -> dict:
    """Manually trigger a poll for any feed (admin). Requires admin session."""
    _require_admin(request)
    import asyncio
    import uuid as _uuid

    from dewie.api.routes.feeds import _poll_feed

    pg = _pg(request)
    processor = getattr(request.app.state, "processor", None)
    feed = await pg.get_feed(_uuid.UUID(feed_id))
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    asyncio.create_task(_poll_feed(feed, pg, processor))
    return {"message": f"Poll triggered for feed '{feed.name}'."}


# ── Model Registry endpoints ───────────────────────────────────────────────────


@router.get("/registry/providers/{provider_id}/models", response_model=list[dict])
async def list_registry_provider_models(
    provider_id: str, request: Request
) -> list[dict]:
    """List models for a specific provider from the model registry."""
    _require_admin(request)
    from dewie.model_registry import registry as _registry

    _registry._load()
    provider = _registry.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")
    models = provider.models
    if not models:
        return []
    return [
        {
            "id": m.id,
            "display_name": m.display_name,
            "context_window": m.context_window,
            "capabilities": m.capabilities,
            "dynamic": m.dynamic,
        }
        for m in models
    ]


@router.post("/restart", status_code=202)
async def restart_server(request: Request) -> dict:
    """Restart the uvicorn server. Kills current process after 2s delay.

    The service will auto-restart. Callers should verify health before proceeding.
    """
    _require_admin(request)
    import asyncio
    import os
    import signal

    async def _delayed_shutdown() -> None:
        await asyncio.sleep(2)
        # Send SIGTERM to the whole process group so uvicorn workers shut down cleanly
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(_delayed_shutdown())
    return {"status": "restarting", "detail": "Server will shut down in 2 seconds"}
