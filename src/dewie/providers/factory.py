# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Provider factory — resolves the correct ChatProvider or EmbeddingProvider
for a given pipeline step.

Resolution: each step picks a *server label* (chat_server_aq, chat_server_ke,
embed_server settings fields — set via dewie.yml or env var) plus a model
name. The label is looked up in the server registry (providers/servers.py),
which is the single source of truth for endpoint/auth/wire-format — see that
module for the `servers:` dewie.yml schema.
"""

from __future__ import annotations

import logging
import os

from .base import ChatProvider, EmbeddingProvider
from .servers import ServerConfig, get_server, resolve_api_key

log = logging.getLogger(__name__)


def _get_settings():
    """Lazy import to avoid circular imports at module load time."""
    from dewie.config import settings

    return settings


def _resolve_step(step: str) -> tuple[str, str]:
    """Resolve (server_label, model) for a given step.

    Steps understood: "aq_generation", "keyword_extraction", "default".
    """
    settings = _get_settings()

    if step == "aq_generation":
        return settings.chat_server_aq, settings.chat_model_aq
    if step == "keyword_extraction":
        return (
            settings.chat_server_ke or settings.chat_server_aq,
            settings.chat_model_ke or settings.chat_model_aq,
        )
    # "default" or any unrecognised step
    return settings.chat_server_aq, settings.chat_model_aq


def _resolve_embed() -> tuple[str, str, int | None]:
    """Resolve (server_label, model, dimensions) for embeddings."""
    settings = _get_settings()

    dimensions = settings.embed_dimensions
    if dimensions is None:
        env_dims = os.environ.get("EMBED_DIMENSIONS")
        if env_dims:
            dimensions = int(env_dims)

    return settings.embed_server, settings.embed_model, dimensions


def _build_chat(server: ServerConfig, model: str) -> ChatProvider:
    settings = _get_settings()
    api_key = resolve_api_key(server) or None

    if server.api_format == "anthropic":
        from .anthropic_provider import AnthropicChatProvider

        return AnthropicChatProvider(
            model=model, api_key=api_key, base_url=f"{server.endpoint}/v1"
        )

    if server.api_format == "openai":
        from .openai_provider import OpenAIChatProvider

        provider = OpenAIChatProvider(
            model=model,
            api_key=api_key,
            base_url=f"{server.endpoint}/v1",
            api_type=settings.openai_api_type,
            extra_body=server.extra_body or None,
        )
        if server.extra_headers:
            provider._extra_headers = server.extra_headers  # type: ignore[attr-defined]
        return provider

    raise RuntimeError(
        f"Unknown api_format {server.api_format!r} for server {server.label!r}. "
        "Valid values: openai, anthropic."
    )


def _build_embed(server: ServerConfig, model: str, dimensions: int | None) -> EmbeddingProvider:
    if server.api_format == "anthropic":
        raise RuntimeError(
            f"Server {server.label!r} uses api_format=anthropic, which has no "
            "embeddings API. Configure embed_server to point at an openai-format server."
        )

    if server.api_format == "openai":
        from .openai_provider import OpenAIEmbeddingProvider

        api_key = resolve_api_key(server) or None
        provider = OpenAIEmbeddingProvider(
            model=model,
            api_key=api_key,
            base_url=f"{server.endpoint}/v1",
            dimensions=dimensions,
        )
        if server.extra_headers:
            provider._extra_headers = server.extra_headers
        return provider

    raise RuntimeError(
        f"Unknown api_format {server.api_format!r} for server {server.label!r}. "
        "Valid values: openai, anthropic."
    )


def get_chat_provider(step: str = "default") -> ChatProvider:
    """Return the configured chat provider for a given pipeline step.

    Steps: "aq_generation", "keyword_extraction", "default"
    """
    label, model = _resolve_step(step)
    if not label:
        raise RuntimeError(
            f"No server configured for step {step!r}. "
            "Set chat_server_aq (and chat_server_ke, if different) in dewie.yml."
        )
    server = get_server(label)
    log.debug(f"get_chat_provider(step={step!r}) -> server={label!r} model={model!r}")
    return _build_chat(server, model)


def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider."""
    label, model, dimensions = _resolve_embed()
    if label == "local":
        if not _get_settings().local_embed_allowed:
            raise RuntimeError(
                "In-process embedding (embed_server=local) is disabled on this "
                "host (LOCAL_EMBED_ALLOWED=false). Ask the operator to enable "
                "local embedding for this account, or register your own "
                "embedding server and point embed_server at it."
            )
        # A GGUF model spec runs in-process via llama.cpp (the zero-config
        # default, EmbeddingGemma-300m); anything else is a sentence-transformers
        # model. See gguf_embed._looks_like_gguf.
        from .gguf_embed import _looks_like_gguf

        if _looks_like_gguf(model):
            from .gguf_embed import GgufEmbeddingProvider

            return GgufEmbeddingProvider(model_spec=model, dimensions=dimensions)
        from .local_embed import LocalEmbeddingProvider

        return LocalEmbeddingProvider(model_name=model, dimensions=dimensions)
    if not label:
        raise RuntimeError(
            "No embedding server configured. Set embed_server in dewie.yml "
            "to a registered server label, or to 'local' for in-process embeddings."
        )
    server = get_server(label)
    log.debug(
        f"get_embedding_provider() -> server={label!r} model={model!r} dims={dimensions!r}"
    )
    return _build_embed(server, model, dimensions)
