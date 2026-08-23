# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Server registry — single source of truth for "where does a model live".

A server is a label + wire format (api_format) + endpoint. Every pipeline step
(AQ generation, KE extraction, embeddings, enrichment backends, the research
agent) selects a server by label instead of re-deriving endpoint/auth/format
logic itself. This replaces the previously-scattered provider-name vocabularies
in providers/factory.py, enrichment/registry.py, and model_adapter.py.

``api_format`` is an open string key, not a closed enum — "openai" and
"anthropic" ship today because those are the only two wire formats in use, but
adding a third is a matter of teaching the consumers (factory.py, http.py,
model_adapter.py) a new dispatch case, not redesigning this module.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class UnknownServerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServerConfig:
    label: str
    api_format: str
    endpoint: str
    api_key_env: str | None = None
    extra_headers: dict = field(default_factory=dict)
    extra_body: dict | None = None
    # Env var holding a path to a JSON token file, re-read on every request
    # (e.g. a rotating bearer token refreshed by a host cron). Takes precedence
    # over api_key_env when set.
    api_key_file_env: str | None = None
    # Fernet ciphertext of a literal API key, set via the admin UI/API for
    # hosts where the operator can't set process env vars (e.g. a multi-tenant
    # container). Takes precedence over api_key_file_env/api_key_env when
    # ENCRYPTION_MASTER_KEY is configured.
    api_key_ciphertext: str | None = None


def normalize_endpoint(raw: str) -> str:
    """Strip a trailing slash and a single trailing /v1.

    Callers re-append /v1 only where the wire format requires it (e.g. chat/
    completions, embeddings) — this lets users register a server's endpoint
    with or without a trailing /v1 and get identical behavior either way.
    """
    endpoint = raw.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = endpoint[: -len("/v1")]
    return endpoint


_BUILTIN_SERVERS: dict[str, ServerConfig] = {
    "openai": ServerConfig(
        label="openai",
        api_format="openai",
        endpoint="https://api.openai.com",
        api_key_env="OPENAI_API_KEY",
    ),
    "anthropic": ServerConfig(
        label="anthropic",
        api_format="anthropic",
        endpoint="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    # OpenRouter is OpenAI-wire-compatible; endpoint gets /v1 appended by the
    # factory -> https://openrouter.ai/api/v1/chat/completions. Chat only
    # (no embeddings endpoint) — use openai/ollama/local for embeddings.
    "openrouter": ServerConfig(
        label="openrouter",
        api_format="openai",
        endpoint="https://openrouter.ai/api",
        api_key_env="OPENROUTER_API_KEY",
    ),
}


def _get_yml_servers() -> list[dict]:
    try:
        import yaml  # type: ignore[import-untyped]

        from dewie.config import _config_file_path

        yml_path = _config_file_path()
        if not yml_path.exists():
            return []
        with yml_path.open() as f:
            data = yaml.safe_load(f) or {}
        servers = data.get("servers", [])
        return servers if isinstance(servers, list) else []
    except Exception as e:
        log.debug(f"Could not load servers from dewie.yml: {e}")
        return []


def _get_env_servers() -> list[dict]:
    """SERVERS_JSON env var — same descriptor shape as `servers:` in dewie.yml.
    Lets env-only deployments (e.g. docker-compose with no dewie.yml) register
    servers without a config file."""
    raw = os.environ.get("SERVERS_JSON", "").strip()
    if not raw:
        return []
    try:
        import json

        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception as e:
        log.warning("Could not parse SERVERS_JSON: %s", e)
        return []


def _to_server_config(descriptor: dict) -> ServerConfig | None:
    label = descriptor.get("label")
    if not label:
        log.warning("Skipping server descriptor with no label: %r", descriptor)
        return None
    return ServerConfig(
        label=label,
        api_format=descriptor.get("api_format", "openai"),
        endpoint=normalize_endpoint(descriptor.get("endpoint", "")),
        api_key_env=descriptor.get("api_key_env"),
        extra_headers=descriptor.get("extra_headers", {}),
        api_key_file_env=descriptor.get("api_key_file_env"),
        api_key_ciphertext=descriptor.get("api_key_ciphertext"),
        extra_body=descriptor.get("extra_body"),
    )


def load_servers() -> dict[str, ServerConfig]:
    """Built-in servers (openai, anthropic), overridable/extendable
    by a ``servers:`` list in dewie.yml and/or a ``SERVERS_JSON`` env var
    (same descriptor shape) — env wins over yml, both win over built-ins."""
    servers = dict(_BUILTIN_SERVERS)
    for descriptor in (*_get_yml_servers(), *_get_env_servers()):
        server = _to_server_config(descriptor)
        if server is not None:
            servers[server.label] = server
    return servers


def get_server(label: str) -> ServerConfig:
    servers = load_servers()
    server = servers.get(label)
    if server is None:
        raise UnknownServerError(
            f"Unknown server label: {label!r}. Registered servers: "
            f"{sorted(servers.keys()) or '(none)'}. "
            "Add one under `servers:` in dewie.yml or via the admin UI."
        )
    return server


def resolve_api_key(server: ServerConfig) -> str:
    """Resolve the current API key/token for a server.

    api_key_ciphertext (a literal key stored encrypted, for hosts where the
    operator can't set process env vars) takes precedence when a master key
    is configured. Otherwise api_key_file_env (a fresh-read-per-call token
    file, e.g. a rotating bearer token) takes precedence over the static
    api_key_env when both are set.
    """
    if server.api_key_ciphertext:
        from dewie.config import settings as _settings

        if _settings.encryption_master_key:
            from dewie.crypto import decrypt

            try:
                return decrypt(server.api_key_ciphertext)
            except Exception as e:
                log.warning(
                    "Could not decrypt api_key_ciphertext for server %r: %s",
                    server.label, e,
                )
    if server.api_key_file_env:
        token_file = os.environ.get(server.api_key_file_env, "")
        if token_file:
            try:
                import json

                with open(token_file) as f:
                    data = json.load(f)
                token = data.get("token") or data.get("access_token") or ""
                if token:
                    return token
            except Exception as e:
                log.warning(
                    "Could not read token file %r for server %r: %s",
                    token_file, server.label, e,
                )
    if not server.api_key_env:
        return ""
    return os.environ.get(server.api_key_env, "")
