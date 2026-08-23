# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
BackendRegistry and PassRegistry — named lookup for enrichment backends and passes.

The registries are the single source of truth for which backends and passes
are available at runtime.  They are built at application startup from config
and attached to ``app.state`` so all request handlers share the same instance.

Responsibilities
----------------
- **BackendRegistry**: Store named ``EnrichmentBackend`` instances and provide
  ``get(name)`` for direct lookup by routing rules.
- **PassRegistry**: Store named ``EnrichmentPass`` instances and provide
  an ordered list for pipeline execution.
- Build themselves from the ``ENRICHMENT_BACKENDS`` and ``ENRICHMENT_PASSES``
  config settings.

Thread safety
-------------
The registry is populated once at startup and read-only thereafter.
Concurrent reads from async handlers are safe without locking.

Config-driven construction
--------------------------
``BackendRegistry.from_config(settings)`` parses the ``enrichment_backends``
setting (a JSON array of backend descriptor dicts) and instantiates the
appropriate backend class for each entry.

Backend descriptor format::

    {
        "name":    "ollama_3b",     # required; must be unique
        "type":    "http",          # "http", "spacy", or "passthrough"

        # http-specific fields:
        "mode":        "ollama",    # "ollama" or "openai"
        "base_url":    "http://localhost:11434",
        "model":       "llama3.2:3b",
        "api_key_env": "ANTHROPIC_API_KEY",  # optional
        "timeout":     30,
        "max_tokens":  0,           # 0 = no limit (omits param from API call)
        "extra_headers": { "anthropic-version": "2023-06-01" },

        # passthrough-specific fields:
        "response": "{ ... }"       # static JSON string
    }
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from dewie.enrichment.base import EnrichmentBackend, EnrichmentPass

if TYPE_CHECKING:
    from dewie.config import Settings
    from dewie.enrichment.router import EnrichmentRouter
    from dewie.ingestion.models import ContentDocument
    from dewie.storage.postgres import PostgresClient

logger = logging.getLogger(__name__)


class BackendRegistry:
    """
    Named registry for ``EnrichmentBackend`` instances.

    Backends must be registered before the application begins serving requests.
    Typically this is done in the FastAPI lifespan handler via
    ``BackendRegistry.from_config(settings)``.

    Example::

        registry = BackendRegistry()
        registry.register(HttpBackend(name="openai_gpt4o", ...))
        registry.set_default("openai_gpt4o")

        backend = registry.default()
    """

    def __init__(self) -> None:
        self._backends: dict[str, EnrichmentBackend] = {}
        self._default_name: str | None = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, backend: EnrichmentBackend) -> None:
        """
        Add a backend to the registry.

        Args:
            backend: Any object satisfying the ``EnrichmentBackend`` Protocol.

        Raises:
            ValueError: If a backend with the same name is already registered.
        """
        if backend.name in self._backends:
            raise ValueError(
                f"A backend named '{backend.name}' is already registered.  "
                "Use a unique name for each backend."
            )
        self._backends[backend.name] = backend
        logger.info("Registered enrichment backend: %s", backend.name)

    def set_default(self, name: str) -> None:
        """
        Designate a registered backend as the default.

        The default is used when no routing rule matches a document.

        Args:
            name: The ``name`` property of a previously registered backend.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in self._backends:
            raise KeyError(
                f"Cannot set default: backend '{name}' is not registered.  "
                f"Registered backends: {self.names()}"
            )
        self._default_name = name
        logger.info("Default enrichment backend set to: %s", name)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> EnrichmentBackend:
        """
        Retrieve a backend by name.

        Args:
            name: Backend identifier as registered via ``register()``.

        Returns:
            The registered ``EnrichmentBackend`` instance.

        Raises:
            KeyError: If no backend with ``name`` is registered.
        """
        if name not in self._backends:
            raise KeyError(
                f"Backend '{name}' is not registered.  Registered backends: {self.names()}"
            )
        return self._backends[name]

    def default(self) -> EnrichmentBackend:
        """
        Return the default backend.

        Returns:
            The backend designated via ``set_default()``, or the first
            registered backend if no default has been set.

        Raises:
            RuntimeError: If no backends are registered at all.
        """
        if not self._backends:
            raise RuntimeError(
                "BackendRegistry is empty.  Register at least one backend before use."
            )
        if self._default_name:
            return self._backends[self._default_name]
        # Fallback: first registered
        first = next(iter(self._backends))
        logger.warning("No default backend configured; using first registered: %s", first)
        return self._backends[first]

    def names(self) -> list[str]:
        """Return the names of all registered backends in registration order."""
        return list(self._backends.keys())

    def list_backends(self) -> list[str]:
        """Return names of all registered backends."""
        return list(self._backends.keys())

    def backend_info(self) -> list[dict]:
        """Return [{name, type, description}] for all registered backends."""
        result = []
        for name, backend in self._backends.items():
            cls = type(backend)
            # Derive a short type slug from the class name (e.g. "HttpBackend" → "http")
            type_slug = cls.__name__.lower().replace("backend", "")
            # Use first non-empty line of the class docstring as description
            doc = cls.__doc__ or ""
            description = next((line.strip() for line in doc.splitlines() if line.strip()), "")
            result.append({"name": name, "type": type_slug, "description": description})
        return result

    # ── Config-driven construction ────────────────────────────────────────────

    @classmethod
    def from_config(cls, settings: Settings) -> BackendRegistry:
        """
        Build a ``BackendRegistry`` from application settings.

        Parses ``settings.enrichment_backends`` (a JSON array string) and
        instantiates the appropriate backend class for each descriptor.

        Always registers a ``PassthroughBackend`` as a safe no-op fallback.
        SpaCy has been removed — LLM failures fail hard with no silent fallback.

        Args:
            settings: Application settings singleton.

        Returns:
            A fully populated ``BackendRegistry`` with the default set to
            ``settings.enrichment_default_backend``.
        """
        # Import here to avoid circular imports at module level
        from dewie.enrichment.backends.http import HttpBackend
        from dewie.enrichment.backends.passthrough import PassthroughBackend

        registry = cls()

        # Always ensure passthrough is available as a safe no-op fallback
        registry.register(PassthroughBackend())

        descriptors: list[dict] = []  # type: ignore[type-arg]
        raw = settings.enrichment_backends
        if raw and raw.strip() not in ("", "[]"):
            try:
                descriptors = json.loads(raw)
            except json.JSONDecodeError:
                logger.error(
                    "ENRICHMENT_BACKENDS is not valid JSON; no additional backends loaded."
                )

        for descriptor in descriptors:
            backend_type = descriptor.get("type", "")
            backend_name = descriptor.get("name", "")

            if not backend_name:
                logger.warning("Backend descriptor missing 'name'; skipping: %s", descriptor)
                continue

            if backend_type == "spacy":
                logger.warning(
                    "SpaCy backend has been removed. Ignoring descriptor: %s", backend_name
                )
                continue

            try:
                if backend_type == "http":
                    from dewie.providers.servers import UnknownServerError, get_server

                    server_label = descriptor.get("server", "")
                    if not server_label:
                        logger.error(
                            "HttpBackend '%s': descriptor has no 'server' label "
                            "(must reference an entry under `servers:` in dewie.yml); skipping.",
                            backend_name,
                        )
                        continue
                    try:
                        server = get_server(server_label)
                    except UnknownServerError as exc:
                        logger.error("HttpBackend '%s': %s; skipping.", backend_name, exc)
                        continue

                    # model can be omitted → resolved from settings
                    model = (
                        descriptor.get("model")
                        or settings.chat_model_aq
                        or settings.enrichment_model
                        or ""
                    )
                    if not model:
                        logger.warning(
                            "HttpBackend '%s': no model configured in descriptor or settings "
                            "(chat_model_aq / enrichment_model). Requests will likely fail.",
                            backend_name,
                        )

                    extra_headers = {**server.extra_headers, **(descriptor.get("extra_headers") or {})}
                    registry.register(
                        HttpBackend(
                            name=backend_name,
                            base_url=f"{server.endpoint}/v1",
                            model=model,
                            mode=server.api_format,  # "openai" | "anthropic"
                            api_key_env=descriptor.get("api_key_env") or server.api_key_env,
                            timeout=float(descriptor.get("timeout", 30)),
                            extra_headers=extra_headers or None,
                            max_tokens=int(descriptor.get("max_tokens", 0)),
                            extra_body={**server.extra_body, **(descriptor.get("extra_body") or {})} if server.extra_body or descriptor.get("extra_body") else None,
                        )
                    )

                elif backend_type == "agent":
                    from dewie.enrichment.backends.agent import AgentBackend

                    registry.register(
                        AgentBackend(
                            name=backend_name,
                            endpoint=descriptor["endpoint"],
                            model=descriptor["model"],
                            provider=descriptor.get("provider", "custom"),
                            auth_token_env=descriptor.get("auth_token_env"),
                            auth_token=descriptor.get("auth_token"),
                            timeout=float(descriptor.get("timeout", 60)),
                            extra_headers=descriptor.get("extra_headers"),
                        )
                    )

                elif backend_type == "passthrough":
                    registry.register(
                        PassthroughBackend(
                            name=backend_name,
                            response_json=descriptor.get("response", ""),
                        )
                    )

                else:
                    logger.warning(
                        "Unknown backend type '%s' for '%s'; skipping.",
                        backend_type,
                        backend_name,
                    )

            except (KeyError, ValueError) as exc:
                logger.error("Failed to register backend '%s': %s", backend_name, exc)

        # Set default
        default_name = settings.enrichment_default_backend
        if default_name in registry.names():
            registry.set_default(default_name)
        elif registry.names():
            registry.set_default(registry.names()[0])

        return registry


# ── PassRegistry ────────────────────────────────────────────────────────────────


class PassRegistry:
    """
    Named registry for ``EnrichmentPass`` instances.

    Passes are registered at application startup and executed in
    registration order during the enrichment pipeline.  External
    packages (cloud layer, community plugins) can register additional
    passes by calling ``register()`` at startup.

    Configuration-driven construction
    ---------------------------------
    ``PassRegistry.from_config(settings, router, backend_registry)`` parses
    ``settings.enrichment_passes`` (a JSON array of fully-qualified class
    path strings) and imports each pass class dynamically.

    The built-in passes (``MetadataPass``, ``EmbedPass``, ``ChunkPass``) are
    always registered first, in that order.  Config-passes are appended
    after the built-ins.

    Pass descriptor format (config)::

        [
            "dewie.enrichment.passes.MetadataPass",
            "dewie.enrichment.passes.EmbedPass",
            "myapp.enrichment.AQPass",      # external plugin
            "myapp.enrichment.ClassifyPass", # external plugin
        ]

    Thread safety
    -------------
    The registry is populated once at startup and read-only thereafter.
    Concurrent reads from async handlers are safe without locking.
    """

    def __init__(self) -> None:
        self._passes: dict[str, EnrichmentPass] = {}
        self._order: list[str] = []

    def register(self, pass_instance: EnrichmentPass) -> None:
        """
        Add a pass to the registry.

        Args:
            pass_instance: An ``EnrichmentPass`` instance.

        Raises:
            ValueError: If a pass with the same name is already registered.
        """
        if pass_instance.name in self._passes:
            raise ValueError(
                f"A pass named '{pass_instance.name}' is already registered.  "
                "Use a unique name for each pass."
            )
        self._passes[pass_instance.name] = pass_instance
        self._order.append(pass_instance.name)
        logger.info("Registered enrichment pass: %s", pass_instance.name)

    def get(self, name: str) -> EnrichmentPass:
        """
        Retrieve a pass by name.

        Args:
            name: Pass identifier as registered via ``register()``.

        Returns:
            The registered ``EnrichmentPass`` instance.

        Raises:
            KeyError: If no pass with ``name`` is registered.
        """
        if name not in self._passes:
            raise KeyError(
                f"Pass '{name}' is not registered.  Registered passes: {self.names()}"
            )
        return self._passes[name]

    def names(self) -> list[str]:
        """Return the names of all registered passes in registration order."""
        return list(self._order)

    def list_passes(self) -> list[str]:
        """Return names of all registered passes."""
        return list(self._order)

    def get_ordered(self) -> list[EnrichmentPass]:
        """Return all registered passes in execution order."""
        return [self._passes[name] for name in self._order]

    def pass_info(self) -> list[dict]:
        """Return [{name, description}] for all registered passes."""
        result = []
        for name, pass_instance in self._passes.items():
            cls = type(pass_instance)
            doc = cls.__doc__ or ""
            description = next((line.strip() for line in doc.splitlines() if line.strip()), "")
            result.append({"name": name, "description": description})
        return result

    @classmethod
    def from_config(
        cls,
        settings,
        router: EnrichmentRouter | None = None,
        backend_registry: BackendRegistry | None = None,
    ) -> PassRegistry:
        """
        Build a ``PassRegistry`` from application settings.

        Parses ``settings.enrichment_passes`` (a JSON array of fully-qualified
        class path strings) and imports each pass class dynamically.

        Always registers the built-in passes (``MetadataPass``, ``EmbedPass``,
        ``ChunkPass``) in that order before the config passes.

        Args:
            settings:          Application settings singleton.
            router:            ``EnrichmentRouter`` for ``MetadataPass``.
            backend_registry:  ``BackendRegistry`` for ``MetadataPass``.

        Returns:
            A fully populated ``PassRegistry``.
        """
        from dewie.enrichment.base import EnrichmentPass

        registry = cls()

        # Register built-in passes in order
        from dewie.enrichment.passes import ChunkPass, EmbedPass, MetadataPass

        registry.register(MetadataPass(router, backend_registry) if router and backend_registry else _NullPass())
        registry.register(EmbedPass())
        registry.register(ChunkPass())

        # Parse config
        descriptors: list[str] = []  # type: ignore[type-arg]
        raw = settings.enrichment_passes
        if raw and raw.strip() not in ("", "[]"):
            try:
                descriptors = json.loads(raw)
            except json.JSONDecodeError:
                logger.error(
                    "ENRICHMENT_PASSES is not valid JSON; no additional passes loaded."
                )

        for descriptor in descriptors:
            if isinstance(descriptor, dict):
                # Descriptor with path and optional params
                class_path = descriptor.get("path", "")
                params = descriptor.get("params", {})
            else:
                # Simple string path
                class_path = str(descriptor)
                params = {}

            if not class_path:
                logger.warning("Pass descriptor missing 'path'; skipping: %s", descriptor)
                continue

            try:
                pass_instance = _load_pass_class(class_path, params, router, backend_registry)
                if isinstance(pass_instance, EnrichmentPass):
                    registry.register(pass_instance)
                else:
                    logger.warning(
                        "Pass at '%s' does not implement EnrichmentPass; skipping.",
                        class_path,
                    )
            except Exception as exc:
                logger.error("Failed to register pass '%s': %s", class_path, exc)

        return registry


# ── Private helpers ─────────────────────────────────────────────────────────────


def _load_pass_class(
    class_path: str,
    params: dict,
    router: EnrichmentRouter | None = None,
    backend_registry: BackendRegistry | None = None,
) -> EnrichmentPass:
    """
    Dynamically import and instantiate a pass class from its fully-qualified path.

    Special-cases ``MetadataPass`` to inject the router and backend_registry
    if not provided via params.

    Args:
        class_path: Fully-qualified class path (e.g. "dewie.enrichment.passes.MetadataPass").
        params:     Keyword arguments to pass to the constructor.
        router:     EnrichmentRouter to inject for MetadataPass.
        backend_registry: BackendRegistry to inject for MetadataPass.

    Returns:
        An instance of the pass class.
    """
    import importlib

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    pass_class = getattr(module, class_name)

    # Inject router/backend_registry for MetadataPass if not provided
    if class_name == "MetadataPass" and "router" not in params and router:
        params["router"] = router
        params["registry"] = backend_registry

    return pass_class(**params)


class _NullPass(EnrichmentPass):
    """
    A no-op pass used as placeholder when router/registry are not available.

    This allows the built-in pass list to always have a MetadataPass entry
    even when the registry is constructed without the dependencies needed
    for actual enrichment (e.g. during unit tests).
    """

    name = "_null"

    async def run(self, doc: ContentDocument, pg: PostgresClient) -> None:  # type: ignore[override]
        """No-op — does nothing."""
        pass
