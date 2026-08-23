# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Model Registry — loads config/models.yaml, probes dynamic providers,
and exposes a unified view of available models + their capabilities.

Usage:
    from dewie.model_registry import registry

    # All available models (probes run async on first access)
    models = await registry.available_models()

    # Get enrichment config for a specific model
    cfg = registry.enrichment_config("gpt-4.1", provider="openai")

    # Get all models for a provider
    models = await registry.models_for_provider("openai")
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

log = logging.getLogger(__name__)

# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class EnrichmentConfig:
    """Per-model enrichment settings."""

    json_mode: str = "none"  # "json_schema" | "json_object" | "none"
    prompt_style: str = "system_user"
    temperature: float = 0.1
    max_tokens: int = 600
    notes: str = ""


@dataclass
class ModelInfo:
    """Single model entry from registry."""

    id: str
    provider: str
    display_name: str
    context_window: int = 0
    capabilities: list[str] = field(default_factory=list)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    available: bool = True  # False = provider offline
    dynamic: bool = False  # True = discovered at runtime
    provenance: str = "static"  # static|discovered|manually_added
    cost_input_per_1m: float = 0.0
    cost_output_per_1m: float = 0.0
    supports_mrl: bool = False
    min_dimensions: int = 0
    default_dimensions: int = 0
    max_dimensions: int = 0
    dimension_source: str = ""

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "provider": self.provider,
            "source": self.provider,  # legacy compat for model-list consumers
            "display_name": self.display_name,
            "context_window": self.context_window,
            "capabilities": self.capabilities,
            "available": self.available,
            "dynamic": self.dynamic,
            "provenance": self.provenance,
            "enrichment": {
                "json_mode": self.enrichment.json_mode,
                "temperature": self.enrichment.temperature,
                "max_tokens": self.enrichment.max_tokens,
            },
            "cost": {
                "input_per_1m": self.cost_input_per_1m,
                "output_per_1m": self.cost_output_per_1m,
            },
            "embedding": {
                "supports_mrl": self.supports_mrl,
                "min_dimensions": self.min_dimensions,
                "default_dimensions": self.default_dimensions,
                "max_dimensions": self.max_dimensions,
                "source": self.dimension_source,
            },
        }


@dataclass
class ProviderInfo:
    """Provider config entry."""

    id: str
    base_url: str
    api_key_env: str | None = None
    probe_url: str | None = None
    probe_timeout: float = 3.0
    dynamic: bool = False
    models: list[ModelInfo] = field(default_factory=list)
    available: bool | None = None  # None = not yet probed

    @property
    def api_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


# ── Registry ───────────────────────────────────────────────────────────────────


class ModelRegistry:
    """
    Loads models.yaml and provides a unified view of all providers/models.
    Probes dynamic providers (LM Studio) once per process, then caches.
    """

    _CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "models.yaml"
    _OVERLAY_PROVIDER_FILE = "provider_overrides.yaml"
    _OVERLAY_MODEL_FILE = "model_overrides.yaml"
    _OVERLAY_HIDDEN_FILE = "model_hidden.yaml"
    _OVERLAY_SELECTION_FILE = "context_selections.yaml"
    _EMBED_HINTS = ("embed", "embedding", "nomic", "e5", "bge", "gte", "minilm")
    _NON_EMBED_HINTS = ("rerank", "ranker", "chat", "instruct")

    @classmethod
    def _looks_like_embedding_id(cls, model_id: str) -> bool:
        mid = model_id.lower()
        if any(h in mid for h in cls._NON_EMBED_HINTS):
            return False
        return any(h in mid for h in cls._EMBED_HINTS)

    @classmethod
    def _embedding_defaults_for_model(
        cls,
        model_id: str,
        capabilities: list[str],
    ) -> tuple[bool, int, int, int, str]:
        caps = {str(c).lower() for c in capabilities}
        is_embed = "embedding" in caps or "embed" in caps or cls._looks_like_embedding_id(model_id)
        if not is_embed:
            return False, 0, 0, 0, ""

        mid = model_id.lower()
        if "qwen3-embedding-8b" in mid:
            return True, 32, 4096, 4096, "qwen_model_card"
        if "qwen3-embedding-4b" in mid:
            return True, 32, 2560, 2560, "qwen_model_card"
        if "qwen3-embedding-0.6b" in mid:
            return True, 32, 1024, 1024, "qwen_model_card"
        if "text-embedding-3-large" in mid:
            return True, 1, 3072, 3072, "openai_api"
        if "text-embedding-3-small" in mid:
            return True, 1, 1536, 1536, "openai_api"
        if "text-embedding-ada-002" in mid:
            return False, 1536, 1536, 1536, "openai_api"
        if "nomic" in mid:
            return False, 768, 768, 768, "model_name_heuristic"
        return False, 1536, 1536, 1536, "default"

    @classmethod
    def _is_embedding_model(cls, model: ModelInfo) -> bool:
        caps = {str(c).lower() for c in model.capabilities}
        if "embedding" in caps or "embed" in caps:
            return True
        return cls._looks_like_embedding_id(model.id)

    def __init__(self) -> None:
        self._providers: dict[str, ProviderInfo] = {}
        self._probe_cache: dict[str, list[str]] = {}  # provider_id → live model ids
        self._probe_lock = asyncio.Lock()
        self._overlay_lock = asyncio.Lock()
        self._loaded = False
        self._provider_overrides: dict[str, dict[str, Any]] = {}
        self._model_overrides: dict[str, dict[str, dict[str, Any]]] = {}
        self._hidden_models: set[tuple[str, str]] = set()
        self._context_selections: dict[str, dict[str, Any]] = {"user": {}, "admin": {}}
        # Raw YAML configs for well-known providers (kept for template UI)
        self._yaml_provider_configs: dict[str, dict[str, Any]] = {}

    def _overlay_dir(self) -> Path:
        from dewie.config import settings

        base = Path(settings.data_dir) if settings.data_dir else Path.cwd() / "data"
        return base / "config"

    @staticmethod
    def _read_yaml_file(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return default
        try:
            with open(path, encoding="utf-8") as f:
                parsed = yaml.safe_load(f) or {}
            return parsed if isinstance(parsed, dict) else default
        except Exception as exc:
            log.warning("Failed to read overlay file %s: %s", path, exc)
            return default

    @staticmethod
    def _write_yaml_file(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _load_overlay_files(self) -> None:
        base = self._overlay_dir()

        providers = self._read_yaml_file(base / self._OVERLAY_PROVIDER_FILE, {"providers": {}})
        raw_providers = providers.get("providers")
        self._provider_overrides = raw_providers if isinstance(raw_providers, dict) else {}

        models = self._read_yaml_file(base / self._OVERLAY_MODEL_FILE, {"models": {}})
        raw_models = models.get("models")
        self._model_overrides = raw_models if isinstance(raw_models, dict) else {}

        hidden = self._read_yaml_file(base / self._OVERLAY_HIDDEN_FILE, {"hidden": []})
        self._hidden_models = set()
        for item in hidden.get("hidden") or []:
            if not isinstance(item, dict):
                continue
            provider = str(item.get("provider", "")).strip()
            model = str(item.get("model", "")).strip()
            if provider and model:
                self._hidden_models.add((provider, model))

        selections = self._read_yaml_file(
            base / self._OVERLAY_SELECTION_FILE,
            {"contexts": {"user": {}, "admin": {}}},
        )
        contexts = selections.get("contexts") if isinstance(selections, dict) else None
        if isinstance(contexts, dict):
            self._context_selections = {
                "user": dict(contexts.get("user") or {}),
                "admin": dict(contexts.get("admin") or {}),
            }
        else:
            self._context_selections = {"user": {}, "admin": {}}

    def _persist_provider_overrides(self) -> None:
        self._write_yaml_file(
            self._overlay_dir() / self._OVERLAY_PROVIDER_FILE,
            {"providers": self._provider_overrides},
        )

    def _persist_model_overrides(self) -> None:
        self._write_yaml_file(
            self._overlay_dir() / self._OVERLAY_MODEL_FILE,
            {"models": self._model_overrides},
        )

    def _persist_hidden_models(self) -> None:
        hidden = [
            {"provider": provider, "model": model}
            for provider, model in sorted(self._hidden_models)
        ]
        self._write_yaml_file(
            self._overlay_dir() / self._OVERLAY_HIDDEN_FILE,
            {"hidden": hidden},
        )

    def _persist_context_selections(self) -> None:
        self._write_yaml_file(
            self._overlay_dir() / self._OVERLAY_SELECTION_FILE,
            {"contexts": self._context_selections},
        )

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _model_from_config(self, provider_id: str, model_id: str, data: dict[str, Any]) -> ModelInfo:
        enrichment = data.get("enrichment") if isinstance(data.get("enrichment"), dict) else {}
        cost = data.get("cost") if isinstance(data.get("cost"), dict) else {}
        capabilities = data.get("capabilities")
        capability_list = capabilities if isinstance(capabilities, list) else []
        embedding = data.get("embedding") if isinstance(data.get("embedding"), dict) else {}
        supports_mrl, min_dims, default_dims, max_dims, dim_source = self._embedding_defaults_for_model(
            model_id,
            [str(c) for c in capability_list],
        )

        return ModelInfo(
            id=model_id,
            provider=provider_id,
            display_name=str(data.get("display_name") or model_id),
            context_window=self._to_int(data.get("context_window"), 0),
            capabilities=[str(c) for c in capability_list],
            enrichment=EnrichmentConfig(
                json_mode=str(enrichment.get("json_mode", "none")),
                prompt_style=str(enrichment.get("prompt_style", "system_user")),
                temperature=self._to_float(enrichment.get("temperature"), 0.1),
                max_tokens=self._to_int(enrichment.get("max_tokens"), 600),
                notes=str(enrichment.get("notes", "")),
            ),
            dynamic=bool(data.get("dynamic", False)),
            provenance=str(data.get("source") or data.get("provenance") or "manually_added"),
            cost_input_per_1m=self._to_float(cost.get("input_per_1m"), 0.0),
            cost_output_per_1m=self._to_float(cost.get("output_per_1m"), 0.0),
            supports_mrl=bool(embedding.get("supports_mrl", supports_mrl)),
            min_dimensions=self._to_int(embedding.get("min_dimensions", min_dims), min_dims),
            default_dimensions=self._to_int(
                embedding.get("default_dimensions", default_dims),
                default_dims,
            ),
            max_dimensions=self._to_int(embedding.get("max_dimensions", max_dims), max_dims),
            dimension_source=str(embedding.get("source") or dim_source),
        )

    def _load(self) -> None:
        """Load and parse models.yaml synchronously."""
        if self._loaded:
            return
        path = self._CONFIG_PATH
        if not path.exists():
            log.warning("models.yaml not found at %s — registry empty", path)
            self._loaded = True
            return

        with open(path) as f:
            raw = yaml.safe_load(f)

        self._load_overlay_files()

        # Store all YAML-defined provider configs for the well-known provider
        # template dropdown (regardless of whether they get auto-registered).
        raw_providers = raw.get("providers") or {}
        self._yaml_provider_configs = dict(raw_providers)

        for provider_id, pcfg in raw_providers.items():
            api_key_env = pcfg.get("api_key_env")
            dynamic = bool(pcfg.get("dynamic", False))

            # Dynamic providers (no fixed endpoint, e.g. openai-compatible) are
            # templates for the "Add Provider" form — never auto-register them.
            if dynamic and not api_key_env:
                continue

            # Static providers only register when their API key is present.
            if api_key_env and not os.environ.get(api_key_env):
                continue

            provider = ProviderInfo(
                id=provider_id,
                base_url=pcfg.get("base_url", ""),
                api_key_env=api_key_env,
                probe_url=pcfg.get("probe_url"),
                probe_timeout=float(pcfg.get("probe_timeout", 3)),
                dynamic=bool(pcfg.get("dynamic", False)),
            )
            for mcfg in pcfg.get("models") or []:
                ecfg = mcfg.get("enrichment") or {}
                cost = mcfg.get("cost") or {}
                model = ModelInfo(
                    id=mcfg["id"],
                    provider=provider_id,
                    display_name=mcfg.get("display_name", mcfg["id"]),
                    context_window=mcfg.get("context_window", 0),
                    capabilities=mcfg.get("capabilities") or [],
                    enrichment=EnrichmentConfig(
                        json_mode=ecfg.get("json_mode", "none"),
                        prompt_style=ecfg.get("prompt_style", "system_user"),
                        temperature=float(ecfg.get("temperature", 0.1)),
                        max_tokens=int(ecfg.get("max_tokens", 600)),
                        notes=ecfg.get("notes", ""),
                    ),
                    provenance="static",
                    cost_input_per_1m=float(cost.get("input_per_1m", 0)),
                    cost_output_per_1m=float(cost.get("output_per_1m", 0)),
                    supports_mrl=bool((mcfg.get("embedding") or {}).get("supports_mrl", False)),
                    min_dimensions=int((mcfg.get("embedding") or {}).get("min_dimensions", 0) or 0),
                    default_dimensions=int(
                        (mcfg.get("embedding") or {}).get("default_dimensions", 0) or 0
                    ),
                    max_dimensions=int((mcfg.get("embedding") or {}).get("max_dimensions", 0) or 0),
                    dimension_source=str((mcfg.get("embedding") or {}).get("source", "")),
                )
                if not model.dimension_source and self._is_embedding_model(model):
                    (
                        model.supports_mrl,
                        model.min_dimensions,
                        model.default_dimensions,
                        model.max_dimensions,
                        model.dimension_source,
                    ) = self._embedding_defaults_for_model(model.id, model.capabilities)
                provider.models.append(model)
            self._providers[provider_id] = provider

        self._load_overlay_files()
        for provider_id, override in self._provider_overrides.items():
            if not isinstance(override, dict):
                continue
            existing = self._providers.get(provider_id)
            if existing is None:
                self._providers[provider_id] = ProviderInfo(
                    id=provider_id,
                    base_url=str(override.get("base_url") or ""),
                    api_key_env=(str(override.get("api_key_env")) if override.get("api_key_env") else None),
                    probe_url=(str(override.get("probe_url")) if override.get("probe_url") else None),
                    probe_timeout=self._to_float(override.get("probe_timeout"), 3.0),
                    dynamic=bool(override.get("dynamic", False)),
                )
                continue

            if "base_url" in override:
                existing.base_url = str(override.get("base_url") or "")
            if "api_key_env" in override:
                existing.api_key_env = (
                    str(override.get("api_key_env")) if override.get("api_key_env") else None
                )
            if "probe_url" in override:
                existing.probe_url = str(override.get("probe_url")) if override.get("probe_url") else None
            if "probe_timeout" in override:
                existing.probe_timeout = self._to_float(override.get("probe_timeout"), 3.0)
            if "dynamic" in override:
                existing.dynamic = bool(override.get("dynamic", False))

        for provider_id, models in self._model_overrides.items():
            if not isinstance(models, dict):
                continue
            provider = self._providers.get(provider_id)
            if provider is None:
                provider = ProviderInfo(id=provider_id, base_url="")
                self._providers[provider_id] = provider
            model_map = {m.id: m for m in provider.models}
            for model_id, model_data in models.items():
                if not isinstance(model_data, dict):
                    continue
                model = self._model_from_config(provider_id, model_id, model_data)
                model_map[model_id] = model
            provider.models = list(model_map.values())

        # Every registered server (providers/servers.py) gets a matching
        # provider entry so it shows up in the model catalog without a
        # separate "add provider" step — the server label is the single
        # identity for "where do models live" across config-key selection
        # (chat_server_aq/embed_server) and model curation (hide/unhide).
        # Built-in servers (openai/anthropic) already have a
        # matching models.yaml provider entry and are skipped here.
        from dewie.providers.servers import load_servers

        for label, server in load_servers().items():
            if label in self._providers:
                continue
            dynamic = server.api_format == "openai"
            base_url = server.endpoint
            probe_url = None
            if dynamic:
                base_url = base_url.rstrip("/") + "/v1"
                probe_url = base_url + "/models"
            self._providers[label] = ProviderInfo(
                id=label,
                base_url=base_url,
                api_key_env=server.api_key_env,
                probe_url=probe_url,
                dynamic=dynamic,
            )

        self._loaded = True

    async def _probe_provider(self, provider: ProviderInfo) -> bool:
        """Probe a provider's liveness. Returns True if reachable."""
        probe_url = provider.probe_url
        if not probe_url and provider.dynamic and provider.base_url:
            probe_url = provider.base_url.rstrip("/") + "/models"
        if not probe_url:
            return True  # nothing to probe → assume up
        try:
            async with httpx.AsyncClient(timeout=provider.probe_timeout) as client:
                r = await client.get(probe_url)
                r.raise_for_status()
                if provider.dynamic:
                    data = r.json()
                    # OpenAI-compatible /v1/models response
                    ids = [m["id"] for m in data.get("data", [])]
                    self._probe_cache[provider.id] = ids
                return True
        except Exception as exc:
            log.debug("Provider %s unreachable: %s", provider.id, exc)
            return False

    async def _ensure_probed(self, provider: ProviderInfo) -> None:
        """Probe once, cache result."""
        if provider.available is not None:
            return
        async with self._probe_lock:
            if provider.available is not None:
                return
            provider.available = await self._probe_provider(provider)

    def _merge_dynamic_models(self, provider: ProviderInfo) -> list[ModelInfo]:
        """
        Merge static model entries with dynamically discovered ones.
        Static entries take precedence (they carry capability/enrichment config).
        Dynamic-only models get default settings.
        """
        static_ids = {m.id for m in provider.models}
        dynamic_ids = self._probe_cache.get(provider.id, [])

        result = list(provider.models)
        for mid in dynamic_ids:
            if mid not in static_ids:
                caps = ["embedding"] if self._looks_like_embedding_id(mid) else ["json_schema"]
                supports_mrl, min_dims, default_dims, max_dims, dim_source = self._embedding_defaults_for_model(
                    mid,
                    caps,
                )
                result.append(
                    ModelInfo(
                        id=mid,
                        provider=provider.id,
                        display_name=mid,
                        dynamic=True,
                        provenance="discovered",
                        capabilities=caps,
                        enrichment=EnrichmentConfig(json_mode="json_schema"),
                        supports_mrl=supports_mrl,
                        min_dimensions=min_dims,
                        default_dimensions=default_dims,
                        max_dimensions=max_dims,
                        dimension_source=dim_source,
                    )
                )
        return result

    async def available_models(
        self,
        provider_filter: str | None = None,
        include_hidden: bool = False,
    ) -> list[ModelInfo]:
        """
        Return all models that are currently available.
        Probes dynamic providers. Results are cached per process lifetime.
        """
        self._load()
        providers = list(self._providers.values())
        if provider_filter:
            providers = [p for p in providers if p.id == provider_filter]

        # Probe all in parallel
        await asyncio.gather(*[self._ensure_probed(p) for p in providers])

        result: list[ModelInfo] = []
        for provider in providers:
            available = provider.available is not False
            if provider.dynamic:
                models = self._merge_dynamic_models(provider)
            else:
                models = list(provider.models)
            for m in models:
                m.available = available
            if include_hidden:
                result.extend(models)
            else:
                result.extend(
                    [m for m in models if (m.provider, m.id) not in self._hidden_models]
                )

        return result

    def get(self, model_id: str) -> ModelInfo | None:
        """Return ModelInfo for the first provider that has this model_id (static + cached dynamic)."""
        self._load()
        for provider in self._providers.values():
            # Check static models
            for m in provider.models:
                if m.id == model_id:
                    if (provider.id, m.id) in self._hidden_models:
                        return None
                    return m
            # Check dynamically discovered models (populated by available_models probe)
            if provider.dynamic:
                for mid in self._probe_cache.get(provider.id, []):
                    if mid == model_id:
                        if (provider.id, mid) in self._hidden_models:
                            return None
                        return ModelInfo(
                            id=mid,
                            provider=provider.id,
                            display_name=mid,
                            dynamic=True,
                            provenance="discovered",
                        )
        return None

    def get_provider(self, provider_id: str) -> ProviderInfo | None:
        self._load()
        return self._providers.get(provider_id)

    def enrichment_config(self, model_id: str, provider_id: str | None = None) -> EnrichmentConfig:
        """
        Get enrichment config for a model. Searches all providers if provider_id not given.
        Returns defaults if not found.
        """
        self._load()
        providers = (
            [self._providers[provider_id]]
            if provider_id and provider_id in self._providers
            else list(self._providers.values())
        )
        for provider in providers:
            for m in provider.models:
                if m.id == model_id:
                    return m.enrichment
        return EnrichmentConfig()  # defaults

    def all_providers(self) -> list[ProviderInfo]:
        self._load()
        return list(self._providers.values())

    def reset_probe_cache(self) -> None:
        """Force re-probe on next available_models() call."""
        for p in self._providers.values():
            p.available = None
        self._probe_cache.clear()

    def well_known_providers(self) -> dict[str, dict[str, Any]]:
        """Return YAML-defined providers as templates for the UI to offer.

        Only offers cloud services (OpenAI, Anthropic) and generic
        OpenAI-compatible endpoints — not internal/local hosting tools
        like LM Studio, llama.cpp, etc.
        """
        _ALLOWED_TEMPLATE_IDS = frozenset(("openai", "anthropic", "openai-compatible"))
        _LABELS = {
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "openai-compatible": "OpenAI Compatible",
        }
        if not self._loaded:
            self._load()

        result: dict[str, dict[str, Any]] = {}

        # Add YAML-defined providers that are in the allowed set
        for pid, info in self._yaml_provider_configs.items():
            if pid not in _ALLOWED_TEMPLATE_IDS:
                continue
            result[pid] = {
                "id": pid,
                "label": _LABELS[pid],
                "base_url": info["base_url"],
                "api_key_env": info.get("api_key_env"),
                "model_count": len(info.get("models") or []),
                "dynamic": pid == "openai-compatible",
            }

        # Always include the openai-compatible template even if not in YAML
        if "openai-compatible" not in result:
            result["openai-compatible"] = {
                "id": "openai-compatible",
                "label": "OpenAI Compatible",
                "base_url": "",
                "api_key_env": None,
                "model_count": 0,
                "dynamic": True,
            }

        return result

    async def remove_provider(self, provider_id: str) -> None:
        """Remove a provider from the registry, along with any manually-added/
        discovered models and hidden-state recorded under it — otherwise a
        ghost provider with no base_url reappears on next load (recreated by
        the model_overrides reconciliation in `_load()`).
        """
        provider_id = provider_id.strip()
        if not provider_id:
            raise ValueError("provider_id is required")

        async with self._overlay_lock:
            if not self._loaded:
                self._load()

            # Remove in-memory
            self._providers.pop(provider_id, None)

            # Remove from overrides if user-added
            self._provider_overrides.pop(provider_id, None)
            self._model_overrides.pop(provider_id, None)
            self._hidden_models = {
                (p, m) for p, m in self._hidden_models if p != provider_id
            }

            self._persist_provider_overrides()
            self._persist_model_overrides()
            self._persist_hidden_models()

    async def register_provider(
        self,
        provider_id: str,
        *,
        base_url: str,
        api_key_env: str | None = None,
        probe_url: str | None = None,
        probe_timeout: float = 3.0,
        dynamic: bool = False,
    ) -> None:
        provider_id = provider_id.strip()
        if not provider_id:
            raise ValueError("provider_id is required")

        async with self._overlay_lock:
            if not self._loaded:
                self._load()

            self._provider_overrides[provider_id] = {
                "base_url": base_url,
                "api_key_env": api_key_env,
                "probe_url": probe_url,
                "probe_timeout": probe_timeout,
                "dynamic": dynamic,
            }
            self._persist_provider_overrides()
            self._loaded = False

    async def add_catalog_model(
        self,
        provider_id: str,
        model_id: str,
        data: dict[str, Any],
        *,
        source: str = "manually_added",
    ) -> None:
        provider_id = provider_id.strip()
        model_id = model_id.strip()
        if not provider_id or not model_id:
            raise ValueError("provider_id and model_id are required")

        async with self._overlay_lock:
            if not self._loaded:
                self._load()

            provider_models = self._model_overrides.setdefault(provider_id, {})
            entry = dict(data)
            entry["source"] = source
            provider_models[model_id] = entry
            self._persist_model_overrides()
            self._loaded = False

    async def hide_catalog_model(self, provider_id: str, model_id: str) -> None:
        async with self._overlay_lock:
            if not self._loaded:
                self._load()
            self._hidden_models.add((provider_id, model_id))
            self._persist_hidden_models()

    async def unhide_catalog_model(self, provider_id: str, model_id: str) -> None:
        async with self._overlay_lock:
            if not self._loaded:
                self._load()
            self._hidden_models.discard((provider_id, model_id))
            self._persist_hidden_models()

    async def refresh_provider_models(self, provider_id: str) -> int:
        self._load()
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"Unknown provider: {provider_id}")
        if not provider.probe_url:
            raise ValueError(f"Provider '{provider_id}' has no probe_url configured")

        ok = await self._probe_provider(provider)
        if not ok:
            raise ValueError(f"Provider '{provider_id}' is unreachable")

        discovered_ids = self._probe_cache.get(provider_id, [])
        if not discovered_ids:
            return 0

        async with self._overlay_lock:
            models = self._model_overrides.setdefault(provider_id, {})
            for model_id in discovered_ids:
                existing = models.get(model_id, {})
                if not isinstance(existing, dict):
                    existing = {}
                existing.setdefault("display_name", model_id)
                existing.setdefault("capabilities", ["json_schema"])
                existing.setdefault("enrichment", {"json_mode": "json_schema"})
                existing["dynamic"] = True
                existing["source"] = "discovered"
                models[model_id] = existing
            self._persist_model_overrides()
            self._loaded = False

        return len(discovered_ids)

    async def reset_provider_models(self, provider_id: str) -> int:
        async with self._overlay_lock:
            if not self._loaded:
                self._load()

            removed = 0
            provider_models = self._model_overrides.get(provider_id, {})
            if isinstance(provider_models, dict):
                to_delete = [
                    model_id
                    for model_id, data in provider_models.items()
                    if isinstance(data, dict) and str(data.get("source", "")) == "discovered"
                ]
                for model_id in to_delete:
                    provider_models.pop(model_id, None)
                    removed += 1
                if not provider_models:
                    self._model_overrides.pop(provider_id, None)
            self._persist_model_overrides()
            self._loaded = False

        discovered = await self.refresh_provider_models(provider_id)
        return removed + discovered

    async def catalog(
        self,
        *,
        context: str = "admin",
        include_hidden: bool = True,
        purpose: str = "all",
    ) -> dict[str, Any]:
        models = await self.available_models(include_hidden=True)
        by_provider: dict[str, list[dict[str, Any]]] = {}
        if purpose not in {"all", "chat", "embedding"}:
            raise ValueError("purpose must be one of: all, chat, embedding")

        for model in models:
            is_embedding = self._is_embedding_model(model)
            if purpose == "chat" and is_embedding:
                continue
            if purpose == "embedding" and not is_embedding:
                continue
            entry = model.to_dict()
            hidden = (model.provider, model.id) in self._hidden_models
            entry["hidden"] = hidden
            entry["selectable"] = not hidden
            entry["is_embedding_model"] = is_embedding
            by_provider.setdefault(model.provider, []).append(entry)

        providers = [
            {
                "id": provider.id,
                "base_url": provider.base_url,
                "api_key_env": provider.api_key_env,
                "probe_url": provider.probe_url,
                "probe_timeout": provider.probe_timeout,
                "dynamic": provider.dynamic,
                "available": provider.available,
            }
            for provider in sorted(self.all_providers(), key=lambda p: p.id)
        ]

        if not include_hidden:
            by_provider = {
                provider_id: [m for m in items if not m.get("hidden")]
                for provider_id, items in by_provider.items()
            }

        return {
            "context": context,
            "purpose": purpose,
            "providers": providers,
            "models_by_provider": by_provider,
            "selections": dict(self._context_selections.get(context, {})),
        }

    async def get_context_selection(self, context: str) -> dict[str, Any]:
        if context not in {"admin", "user"}:
            raise ValueError("context must be 'admin' or 'user'")
        self._load()
        return dict(self._context_selections.get(context, {}))

    async def set_context_selection(self, context: str, values: dict[str, Any]) -> dict[str, Any]:
        if context not in {"admin", "user"}:
            raise ValueError("context must be 'admin' or 'user'")

        async with self._overlay_lock:
            if not self._loaded:
                self._load()
            current = dict(self._context_selections.get(context, {}))
            for key, value in values.items():
                if value is None:
                    current.pop(key, None)
                else:
                    current[key] = value
            self._context_selections[context] = current
            self._persist_context_selections()
        return dict(current)

    async def validate_provider_model(
        self,
        *,
        provider: str | None,
        model: str | None,
        include_hidden: bool = False,
    ) -> tuple[bool, str | None]:
        """
        Validate provider/model selection pair.

        Rules:
        - both provider and model omitted -> valid
        - one provided without the other -> invalid
        - both provided -> must exist in effective catalog
        """
        provider_val = (provider or "").strip()
        model_val = (model or "").strip()

        if not provider_val and not model_val:
            return True, None
        if (provider_val and not model_val) or (model_val and not provider_val):
            return False, "provider and model must be provided together"

        models = await self.available_models(provider_filter=provider_val, include_hidden=include_hidden)
        if any(m.id == model_val and m.provider == provider_val for m in models):
            return True, None
        return False, f"Unknown provider/model pair: {provider_val}/{model_val}"


# ── Singleton ─────────────────────────────────────────────────────────────────

registry = ModelRegistry()
