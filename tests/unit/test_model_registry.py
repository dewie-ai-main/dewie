"""Tests for dewie.model_registry — ModelRegistry, ModelInfo, EnrichmentConfig."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dewie.config import settings
from dewie.model_registry import (
    EnrichmentConfig,
    ModelInfo,
    ModelRegistry,
    ProviderInfo,
)

# ── EnrichmentConfig ──────────────────────────────────────────────────────────


def test_enrichment_config_defaults():
    cfg = EnrichmentConfig()
    assert cfg.json_mode == "none"
    assert cfg.prompt_style == "system_user"
    assert cfg.temperature == 0.1
    assert cfg.max_tokens == 600


def test_enrichment_config_custom():
    cfg = EnrichmentConfig(json_mode="json_schema", temperature=0.5, max_tokens=1000)
    assert cfg.json_mode == "json_schema"
    assert cfg.temperature == 0.5
    assert cfg.max_tokens == 1000


# ── ModelInfo ─────────────────────────────────────────────────────────────────


def test_model_info_has_capability():
    m = ModelInfo(
        id="gpt-4", provider="openai", display_name="GPT-4", capabilities=["tools", "vision"]
    )
    assert m.has_capability("tools") is True
    assert m.has_capability("audio") is False


def test_model_info_to_dict():
    m = ModelInfo(
        id="gpt-4",
        provider="openai",
        display_name="GPT-4",
        context_window=128000,
        capabilities=["tools"],
        cost_input_per_1m=2.5,
        cost_output_per_1m=10.0,
    )
    d = m.to_dict()
    assert d["id"] == "gpt-4"
    assert d["provider"] == "openai"
    assert d["source"] == "openai"  # legacy compat
    assert d["display_name"] == "GPT-4"
    assert d["context_window"] == 128000
    assert d["capabilities"] == ["tools"]
    assert d["cost"]["input_per_1m"] == 2.5
    assert "enrichment" in d


# ── ProviderInfo ──────────────────────────────────────────────────────────────


def test_provider_info_api_key_from_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-key")
    p = ProviderInfo(id="test", base_url="http://localhost", api_key_env="MY_KEY")
    assert p.api_key == "secret-key"


def test_provider_info_api_key_none_when_no_env():
    p = ProviderInfo(id="test", base_url="http://localhost")
    assert p.api_key is None


# ── ModelRegistry._load ───────────────────────────────────────────────────────

SAMPLE_YAML = """
providers:
  openai:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    models:
      - id: gpt-4o
        display_name: GPT-4o
        context_window: 128000
        capabilities: [tools, vision]
        enrichment:
          json_mode: json_schema
          temperature: 0.1
          max_tokens: 800
        cost:
          input_per_1m: 2.5
          output_per_1m: 10.0
      - id: gpt-4o-mini
        display_name: GPT-4o Mini
        context_window: 128000
        capabilities: [tools]
  openai-compatible:
    base_url: http://localhost:1234/v1
    probe_url: http://localhost:1234/v1/models
    dynamic: true
    models: []
"""


def _make_registry_with_yaml(yaml_content: str) -> ModelRegistry:
    import os
    import re
    import tempfile

    from dewie.config import settings as _settings

    reg = ModelRegistry()

    # Use settings.data_dir if it's already been monkeypatched to a temp path
    # (so sibling registries created in the same test share the same overlay dir).
    # Otherwise point to a fresh temp dir so the real data/config/
    # doesn't filter out test providers like openai/openai-compatible.
    if _settings.data_dir and not _settings.data_dir.startswith(str(Path(__file__).parents[2])):
        overlay_dir = Path(_settings.data_dir) / "config"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = overlay_dir
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=tmp_dir) as f:
            f.write(yaml_content)
            tmp_path = Path(f.name)
    else:
        tmp_dir = Path(tempfile.mkdtemp())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=tmp_dir) as f:
            f.write(yaml_content)
            tmp_path = Path(f.name)
        reg._overlay_dir = lambda: tmp_dir  # type: ignore[method-assign]

    reg._CONFIG_PATH = tmp_path

    # Auto-set any api_key_env variables referenced in the YAML so those
    # providers get registered (the real code skips them when the env var
    # is not set, but tests expect them to be available). Overwrite empty
    # values too: magika's import-time load_dotenv() can plant KEY="" from a
    # developer .env, and setdefault would leave that in place.
    for match in re.finditer(r'api_key_env:\s*(\w+)', yaml_content):
        var = match.group(1)
        if not os.environ.get(var):
            os.environ[var] = "test-key-for-" + var.lower()
    reg._load()
    return reg


def test_load_parses_providers():
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    providers = reg.all_providers()
    provider_ids = [p.id for p in providers]
    assert "openai" in provider_ids
    # openai-compatible is dynamic with no api_key_env — it's a template, not auto-registered


def test_load_parses_models():
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    provider = reg.get_provider("openai")
    assert provider is not None
    model_ids = [m.id for m in provider.models]
    assert "gpt-4o" in model_ids
    assert "gpt-4o-mini" in model_ids


def test_load_parses_enrichment_config():
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    cfg = reg.enrichment_config("gpt-4o", "openai")
    assert cfg.json_mode == "json_schema"
    assert cfg.temperature == 0.1
    assert cfg.max_tokens == 800


def test_load_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    count_before = len(reg.all_providers())
    reg._load()  # second call should be no-op
    assert len(reg.all_providers()) == count_before


def test_load_missing_yaml():
    reg = ModelRegistry()
    reg._CONFIG_PATH = Path("/nonexistent/models.yaml")
    reg._load()
    assert reg._loaded is True
    assert reg.all_providers() == []


# ── get ───────────────────────────────────────────────────────────────────────


def test_get_returns_model():
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    m = reg.get("gpt-4o")
    assert m is not None
    assert m.id == "gpt-4o"


def test_get_returns_none_for_unknown():
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    assert reg.get("unknown-model-xyz") is None


# ── get_provider ──────────────────────────────────────────────────────────────


def test_get_provider_returns_provider():
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    p = reg.get_provider("openai")
    assert p is not None
    assert p.id == "openai"


def test_get_provider_unknown_returns_none():
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    assert reg.get_provider("missing") is None


# ── enrichment_config ─────────────────────────────────────────────────────────


def test_enrichment_config_unknown_model_returns_defaults():
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    cfg = reg.enrichment_config("does-not-exist")
    assert isinstance(cfg, EnrichmentConfig)
    assert cfg.json_mode == "none"


def test_enrichment_config_with_provider_filter():
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    cfg = reg.enrichment_config("gpt-4o", provider_id="openai")
    assert cfg.json_mode == "json_schema"


# ── _probe_provider ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_provider_no_probe_url_returns_true():
    reg = ModelRegistry()
    provider = ProviderInfo(id="local", base_url="http://localhost", probe_url=None)
    result = await reg._probe_provider(provider)
    assert result is True


@pytest.mark.asyncio
async def test_probe_provider_reachable():
    reg = ModelRegistry()
    provider = ProviderInfo(
        id="test", base_url="http://localhost", probe_url="http://localhost/health"
    )

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("dewie.model_registry.httpx.AsyncClient", return_value=mock_cm):
        result = await reg._probe_provider(provider)

    assert result is True


@pytest.mark.asyncio
async def test_probe_provider_unreachable():
    reg = ModelRegistry()
    provider = ProviderInfo(
        id="test", base_url="http://localhost", probe_url="http://localhost/health"
    )

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("dewie.model_registry.httpx.AsyncClient", return_value=mock_cm):
        result = await reg._probe_provider(provider)

    assert result is False


@pytest.mark.asyncio
async def test_probe_dynamic_provider_populates_cache():
    reg = ModelRegistry()
    provider = ProviderInfo(
        id="openai-compatible",
        base_url="http://localhost",
        probe_url="http://localhost/v1/models",
        dynamic=True,
    )

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value={"data": [{"id": "llama3.2:3b"}, {"id": "embed-model"}]}
    )

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("dewie.model_registry.httpx.AsyncClient", return_value=mock_cm):
        await reg._probe_provider(provider)

    # discovered models include embeddings; filtering happens by catalog purpose
    assert "llama3.2:3b" in reg._probe_cache.get("openai-compatible", [])
    assert "embed-model" in reg._probe_cache.get("openai-compatible", [])


# ── _merge_dynamic_models ─────────────────────────────────────────────────────


def test_merge_dynamic_models_adds_discovered():
    reg = ModelRegistry()
    provider = ProviderInfo(id="openai-compatible", base_url="http://localhost", dynamic=True)
    reg._probe_cache["openai-compatible"] = ["new-model", "another-model"]

    merged = reg._merge_dynamic_models(provider)
    assert any(m.id == "new-model" for m in merged)
    assert any(m.id == "another-model" for m in merged)


def test_merge_dynamic_models_static_takes_precedence():
    reg = ModelRegistry()
    static_model = ModelInfo(
        id="gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        enrichment=EnrichmentConfig(json_mode="json_schema"),
    )
    provider = ProviderInfo(id="openai", base_url="http://api", dynamic=True, models=[static_model])
    reg._probe_cache["openai"] = ["gpt-4o", "new-model"]

    merged = reg._merge_dynamic_models(provider)
    gpt4o_entries = [m for m in merged if m.id == "gpt-4o"]
    assert len(gpt4o_entries) == 1
    assert gpt4o_entries[0].enrichment.json_mode == "json_schema"


# ── reset_probe_cache ─────────────────────────────────────────────────────────


def test_reset_probe_cache():
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    provider = reg.get_provider("openai")
    provider.available = True
    reg._probe_cache["openai"] = ["some-model"]

    reg.reset_probe_cache()

    assert provider.available is None
    assert reg._probe_cache == {}


# ── available_models ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_available_models_returns_static_models():
    reg = _make_registry_with_yaml(SAMPLE_YAML)

    # Mark all providers as available so no real probing
    for p in reg.all_providers():
        p.available = True

    models = await reg.available_models()
    model_ids = [m.id for m in models]
    assert "gpt-4o" in model_ids
    assert "gpt-4o-mini" in model_ids


@pytest.mark.asyncio
async def test_available_models_provider_filter():
    reg = _make_registry_with_yaml(SAMPLE_YAML)
    for p in reg.all_providers():
        p.available = True

    models = await reg.available_models(provider_filter="openai")
    assert all(m.provider == "openai" for m in models)


@pytest.mark.asyncio
async def test_filesystem_overlay_persists_provider_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))

    reg = _make_registry_with_yaml("providers: {}")
    await reg.register_provider(
        "custom-provider",
        base_url="http://localhost:9999/v1",
        dynamic=False,
    )
    await reg.add_catalog_model(
        "custom-provider",
        "my-model",
        {
            "display_name": "My Model",
            "capabilities": ["json_schema"],
            "enrichment": {"json_mode": "json_schema", "max_tokens": 777},
        },
        source="manually_added",
    )

    first = await reg.available_models(provider_filter="custom-provider")
    assert any(m.id == "my-model" for m in first)

    reg2 = ModelRegistry()
    reg2._CONFIG_PATH = reg._CONFIG_PATH
    second = await reg2.available_models(provider_filter="custom-provider")
    model = next((m for m in second if m.id == "my-model"), None)
    assert model is not None
    assert model.provenance == "manually_added"


@pytest.mark.asyncio
async def test_hidden_models_filtered_from_default_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))

    reg = _make_registry_with_yaml(SAMPLE_YAML)
    for p in reg.all_providers():
        p.available = True

    await reg.hide_catalog_model("openai", "gpt-4o")

    visible = await reg.available_models(provider_filter="openai")
    visible_ids = {m.id for m in visible}
    assert "gpt-4o" not in visible_ids
    assert "gpt-4o-mini" in visible_ids

    all_models = await reg.available_models(provider_filter="openai", include_hidden=True)
    all_ids = {m.id for m in all_models}
    assert "gpt-4o" in all_ids


@pytest.mark.asyncio
async def test_context_selections_are_independent_and_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))

    reg = _make_registry_with_yaml(SAMPLE_YAML)
    await reg.set_context_selection("admin", {"chat_provider_aq": "openai", "chat_model_aq": "gpt-4o"})
    await reg.set_context_selection(
        "user",
        {"chat_provider_aq": "anthropic", "chat_model_aq": "claude-3-5-sonnet"},
    )

    admin_sel = await reg.get_context_selection("admin")
    user_sel = await reg.get_context_selection("user")
    assert admin_sel["chat_provider_aq"] == "openai"
    assert user_sel["chat_provider_aq"] == "anthropic"

    reg2 = ModelRegistry()
    reg2._CONFIG_PATH = reg._CONFIG_PATH
    admin_sel2 = await reg2.get_context_selection("admin")
    user_sel2 = await reg2.get_context_selection("user")
    assert admin_sel2["chat_model_aq"] == "gpt-4o"
    assert user_sel2["chat_model_aq"] == "claude-3-5-sonnet"


@pytest.mark.asyncio
async def test_catalog_purpose_embedding_filters_chat_models(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))

    reg = _make_registry_with_yaml(
        """
providers:
  mixed:
    base_url: http://localhost:1234/v1
    dynamic: false
    models:
      - id: gpt-4o-mini
        display_name: GPT-4o Mini
        capabilities: [json_schema]
      - id: "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M"
        display_name: Qwen Embedding
        capabilities: [embedding]
"""
    )
    for p in reg.all_providers():
        p.available = True

    payload = await reg.catalog(context="admin", include_hidden=False, purpose="embedding")
    models = payload["models_by_provider"]["mixed"]
    ids = {m["id"] for m in models}

    assert "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M" in ids
    assert "gpt-4o-mini" not in ids


@pytest.mark.asyncio
async def test_catalog_purpose_chat_filters_embedding_models(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))

    reg = _make_registry_with_yaml(
        """
providers:
  mixed:
    base_url: http://localhost:1234/v1
    dynamic: false
    models:
      - id: gpt-4o-mini
        display_name: GPT-4o Mini
        capabilities: [json_schema]
      - id: nomic-embed-text
        display_name: Nomic Embed
        capabilities: [embedding]
"""
    )
    for p in reg.all_providers():
        p.available = True

    payload = await reg.catalog(context="admin", include_hidden=False, purpose="chat")
    models = payload["models_by_provider"]["mixed"]
    ids = {m["id"] for m in models}

    assert "gpt-4o-mini" in ids
    assert "nomic-embed-text" not in ids
