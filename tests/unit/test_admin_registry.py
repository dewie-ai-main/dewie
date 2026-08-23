"""Tests for registry-based model/provider selection in admin routes.

Covers:
- _registered_server_labels() includes built-in and dewie.yml-registered servers
- _validate_config_update rejects unknown models for known providers
- _validate_config_update allows models for providers with no static list
- GET /admin/registry/providers lists providers
- GET /admin/registry/providers/{id}/models lists models
- POST /admin/registry/reload triggers reload
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# ── Helper: build a minimal ModelRegistry from inline YAML ───────────────────


MINI_YAML = """
providers:
  openai:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    models:
      - id: gpt-4o-mini
        display_name: GPT-4o Mini
        context_window: 128000
        capabilities: [json_object]
  openai-compatible:
    base_url: http://localhost:1234/v1
    probe_url: http://localhost:1234/v1/models
    dynamic: true
    models: []
"""


def _make_mini_registry():
    import os
    import re
    import tempfile

    from dewie.model_registry import ModelRegistry

    # Auto-set api_key_env vars referenced in the YAML
    for match in re.finditer(r'api_key_env:\s*(\w+)', MINI_YAML):
        os.environ.setdefault(match.group(1), "test-key-for-" + match.group(1).lower())

    tmp_dir = Path(tempfile.mkdtemp())
    reg = ModelRegistry()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=tmp_dir) as f:
        f.write(MINI_YAML)
        tmp = Path(f.name)
    reg._CONFIG_PATH = tmp
    # Isolated overlay dir so real data/config/ doesn't bleed into test registry
    reg._overlay_dir = lambda: tmp_dir  # type: ignore[method-assign]
    reg._load()
    return reg


# ── _registered_server_labels ─────────────────────────────────────────────────


def test_registered_server_labels_includes_builtins_and_yaml(tmp_path, monkeypatch):
    from dewie.api.routes.admin import _registered_server_labels

    yml = tmp_path / "dewie.yml"
    yml.write_text("servers:\n  - label: my-server\n    api_format: openai\n    endpoint: http://x\n")
    monkeypatch.setenv("DEWIE_CONFIG_PATH", str(yml))

    labels = _registered_server_labels()

    # Must include built-in servers
    assert "openai" in labels
    assert "anthropic" in labels
    # Must include user-registered servers from dewie.yml
    assert "my-server" in labels
    # 'local' is always allowed for embed_server
    assert "local" in labels


# ── _validate_config_update — model validation ────────────────────────────────


def test_validate_config_model_accepted_when_in_registry(monkeypatch):
    import dewie.config as _cfg
    from dewie.api.routes.admin import _validate_config_update

    reg = _make_mini_registry()
    monkeypatch.setattr(_cfg.settings, "chat_server_aq", "openai")

    with patch("dewie.model_registry.registry", reg):
        # Should not raise
        _validate_config_update("chat_model_aq", "gpt-4o-mini")


def test_validate_config_model_rejected_when_not_in_registry(monkeypatch):
    from fastapi import HTTPException

    import dewie.config as _cfg
    from dewie.api.routes.admin import _validate_config_update

    reg = _make_mini_registry()
    monkeypatch.setattr(_cfg.settings, "chat_server_aq", "openai")

    with patch("dewie.model_registry.registry", reg):
        with pytest.raises(HTTPException) as exc_info:
            _validate_config_update("chat_model_aq", "gpt-99-ultra-fake")

    assert exc_info.value.status_code == 400
    assert "gpt-99-ultra-fake" in exc_info.value.detail
    assert "gpt-4o-mini" in exc_info.value.detail


def test_validate_config_model_allowed_for_dynamic_provider(monkeypatch):
    """Dynamic providers (no static model list) should allow any model name."""
    import dewie.config as _cfg
    from dewie.api.routes.admin import _validate_config_update

    reg = _make_mini_registry()
    monkeypatch.setattr(_cfg.settings, "chat_server_aq", "openai-compatible")

    with patch("dewie.model_registry.registry", reg):
        # openai-compatible has no static models — should not raise
        _validate_config_update("chat_model_aq", "any-random-model-name")


def test_validate_config_model_allowed_for_unknown_provider(monkeypatch):
    """Provider not in registry should not block model selection (manual/custom case)."""
    import dewie.config as _cfg
    from dewie.api.routes.admin import _validate_config_update

    reg = _make_mini_registry()
    monkeypatch.setattr(_cfg.settings, "chat_server_aq", "my-custom-provider-not-in-registry")

    with patch("dewie.model_registry.registry", reg):
        # Should not raise — unknown provider means we can't validate
        _validate_config_update("chat_model_aq", "any-model")


def test_validate_config_ke_model_validated(monkeypatch):
    """chat_model_ke is also validated."""
    from fastapi import HTTPException

    import dewie.config as _cfg
    from dewie.api.routes.admin import _validate_config_update

    reg = _make_mini_registry()
    monkeypatch.setattr(_cfg.settings, "chat_server_ke", "openai")

    with patch("dewie.model_registry.registry", reg):
        with pytest.raises(HTTPException) as exc_info:
            _validate_config_update("chat_model_ke", "not-a-real-model")

    assert exc_info.value.status_code == 400


def test_validate_config_provider_validated_against_registry():
    """Server labels are validated against the registered server list."""
    from fastapi import HTTPException

    from dewie.api.routes.admin import _validate_config_update

    with pytest.raises(HTTPException):
        _validate_config_update("chat_server_aq", "totally-unknown-server-xyz")


# ── API endpoints (ASGI) ──────────────────────────────────────────────────────


def _make_admin_app_with_registry(reg):
    """Build a minimal admin FastAPI app for endpoint testing."""
    from fastapi import FastAPI

    from dewie.admin_main import _admin_session_middleware
    from dewie.api.routes.admin import router as admin_router

    app = FastAPI()
    app.middleware("http")(_admin_session_middleware)
    app.include_router(admin_router)

    app.state.postgres = AsyncMock()
    app.state.cache = AsyncMock()
    app.state.processor = None
    return app


@pytest.mark.asyncio
async def test_list_providers_endpoint():
    import os

    from httpx import ASGITransport, AsyncClient

    reg = _make_mini_registry()
    app = _make_admin_app_with_registry(reg)

    with (
        patch("dewie.model_registry.registry", reg),
        patch.dict(os.environ, {"ADMIN_KEY": "test-key"}),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/admin/registry/providers", headers={"X-Admin-Key": "test-key"}
            )

    assert resp.status_code == 200
    body = resp.json()
    provider_ids = [p["id"] for p in body]
    assert "openai" in provider_ids
    # openai-compatible is a template (dynamic, no api_key_env) — not auto-registered


@pytest.mark.asyncio
async def test_list_provider_models_endpoint():
    import os

    from httpx import ASGITransport, AsyncClient

    reg = _make_mini_registry()
    app = _make_admin_app_with_registry(reg)

    with (
        patch("dewie.model_registry.registry", reg),
        patch.dict(os.environ, {"ADMIN_KEY": "test-key"}),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/admin/registry/providers/openai/models",
                headers={"X-Admin-Key": "test-key"},
            )

    assert resp.status_code == 200
    body = resp.json()
    model_ids = [m["id"] for m in body]
    assert "gpt-4o-mini" in model_ids


@pytest.mark.asyncio
async def test_list_provider_models_404_for_unknown():
    import os

    from httpx import ASGITransport, AsyncClient

    reg = _make_mini_registry()
    app = _make_admin_app_with_registry(reg)

    with (
        patch("dewie.model_registry.registry", reg),
        patch.dict(os.environ, {"ADMIN_KEY": "test-key"}),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/admin/registry/providers/no-such-provider/models",
                headers={"X-Admin-Key": "test-key"},
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_registry_reload_endpoint():
    import os

    from httpx import ASGITransport, AsyncClient

    reg = _make_mini_registry()
    # Pre-mark providers so no real probing happens
    for p in reg.all_providers():
        p.available = True

    app = _make_admin_app_with_registry(reg)

    with (
        patch("dewie.model_registry.registry", reg),
        patch.dict(os.environ, {"ADMIN_KEY": "test-key"}),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/admin/registry/reload", headers={"X-Admin-Key": "test-key"}
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "reloaded"
    assert "model_count" in body
    assert "provider_count" in body
