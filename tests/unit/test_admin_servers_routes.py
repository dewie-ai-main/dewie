"""Tests for /admin/servers routes — literal encrypted API key storage."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet


def _request(tmp_path, monkeypatch):
    from dewie.config import settings as _settings

    monkeypatch.setattr(_settings, "auth_enabled", False)
    monkeypatch.setenv("DEWIE_CONFIG_PATH", str(tmp_path / "dewie.yml"))
    req = MagicMock()
    req.state.is_admin = True
    req.state.actor_id = None
    req.state.tenant_id = None
    return req


def test_upsert_server_with_api_key_is_encrypted_and_never_echoed(tmp_path, monkeypatch):
    from dewie.api.routes.admin import ServerUpsertRequest, upsert_server
    from dewie.config import settings as _settings

    monkeypatch.setattr(_settings, "encryption_master_key", Fernet.generate_key().decode())
    req = _request(tmp_path, monkeypatch)

    body = ServerUpsertRequest(
        label="my-llama", api_format="openai", endpoint="http://x:8080", api_key="sk-literal"
    )
    entry = asyncio.run(upsert_server("my-llama", body, req))

    assert entry.has_api_key is True
    assert "sk-literal" not in entry.model_dump_json()

    raw = (tmp_path / "dewie.yml").read_text()
    assert "sk-literal" not in raw
    assert "api_key_ciphertext" in raw


def test_upsert_server_with_api_key_but_no_master_key_returns_clean_503(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from dewie.api.routes.admin import ServerUpsertRequest, upsert_server
    from dewie.config import settings as _settings

    monkeypatch.setattr(_settings, "encryption_master_key", "")
    req = _request(tmp_path, monkeypatch)

    body = ServerUpsertRequest(
        label="my-llama", api_format="openai", endpoint="http://x:8080", api_key="sk-literal"
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(upsert_server("my-llama", body, req))
    assert exc_info.value.status_code == 503


def test_upsert_server_omitting_api_key_preserves_existing(tmp_path, monkeypatch):
    from dewie.api.routes.admin import ServerUpsertRequest, upsert_server
    from dewie.config import settings as _settings
    from dewie.providers.servers import get_server, resolve_api_key

    monkeypatch.setattr(_settings, "encryption_master_key", Fernet.generate_key().decode())
    req = _request(tmp_path, monkeypatch)

    body = ServerUpsertRequest(
        label="my-llama", api_format="openai", endpoint="http://x:8080", api_key="sk-literal"
    )
    asyncio.run(upsert_server("my-llama", body, req))

    # Update endpoint only, omit api_key entirely.
    body2 = ServerUpsertRequest(label="my-llama", api_format="openai", endpoint="http://y:9090")
    entry = asyncio.run(upsert_server("my-llama", body2, req))

    assert entry.has_api_key is True
    assert resolve_api_key(get_server("my-llama")) == "sk-literal"


def test_list_servers_reports_has_api_key(tmp_path, monkeypatch):
    from dewie.api.routes.admin import ServerUpsertRequest, list_servers, upsert_server
    from dewie.config import settings as _settings

    monkeypatch.setattr(_settings, "encryption_master_key", Fernet.generate_key().decode())
    req = _request(tmp_path, monkeypatch)

    asyncio.run(
        upsert_server(
            "with-key",
            ServerUpsertRequest(label="with-key", api_format="openai", endpoint="http://a", api_key="sk-1"),
            req,
        )
    )
    asyncio.run(
        upsert_server(
            "without-key",
            ServerUpsertRequest(label="without-key", api_format="openai", endpoint="http://b"),
            req,
        )
    )

    resp = asyncio.run(list_servers(req))
    by_label = {s.label: s for s in resp.servers}
    assert by_label["with-key"].has_api_key is True
    assert by_label["without-key"].has_api_key is False
    assert all("ciphertext" not in s.model_dump_json() for s in resp.servers)
