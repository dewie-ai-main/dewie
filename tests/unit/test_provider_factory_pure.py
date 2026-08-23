"""Tests for dewie.providers.servers — pure helpers."""

from __future__ import annotations

import pytest

# ── normalize_endpoint ────────────────────────────────────────────────────────


def test_normalize_endpoint_strips_trailing_slash():
    from dewie.providers.servers import normalize_endpoint

    assert normalize_endpoint("http://x:8080/") == "http://x:8080"


def test_normalize_endpoint_strips_trailing_v1():
    from dewie.providers.servers import normalize_endpoint

    assert normalize_endpoint("http://x:8080/v1") == "http://x:8080"
    assert normalize_endpoint("http://x:8080/v1/") == "http://x:8080"


def test_normalize_endpoint_no_v1_unchanged():
    from dewie.providers.servers import normalize_endpoint

    assert normalize_endpoint("http://x:8080") == "http://x:8080"


# ── load_servers / get_server ─────────────────────────────────────────────────


def test_load_servers_includes_builtins():
    from dewie.providers.servers import load_servers

    servers = load_servers()
    assert "openai" in servers
    assert servers["openai"].api_format == "openai"
    assert "anthropic" in servers
    assert servers["anthropic"].api_format == "anthropic"
    # OpenRouter is OpenAI-wire-compatible (chat only, no embeddings)
    assert "openrouter" in servers
    assert servers["openrouter"].api_format == "openai"
    assert servers["openrouter"].endpoint == "https://openrouter.ai/api"
    assert servers["openrouter"].api_key_env == "OPENROUTER_API_KEY"


def test_load_servers_yml_override(tmp_path, monkeypatch):
    from dewie.providers.servers import load_servers

    yml = tmp_path / "dewie.yml"
    yml.write_text(
        "servers:\n  - label: my-server\n    api_format: openai\n"
        "    endpoint: http://localhost:8080/v1\n"
    )
    monkeypatch.setenv("DEWIE_CONFIG_PATH", str(yml))

    servers = load_servers()
    assert "my-server" in servers
    assert servers["my-server"].endpoint == "http://localhost:8080"  # /v1 normalized off


def test_load_servers_env_json(monkeypatch):
    from dewie.providers.servers import load_servers

    monkeypatch.setenv(
        "SERVERS_JSON",
        '[{"label": "env-server", "api_format": "openai", "endpoint": "http://y:9000"}]',
    )
    servers = load_servers()
    assert "env-server" in servers
    assert servers["env-server"].endpoint == "http://y:9000"


def test_get_server_unknown_raises():
    from dewie.providers.servers import UnknownServerError, get_server

    with pytest.raises(UnknownServerError, match="Unknown server label"):
        get_server("totally-unknown-server-xyz")


def test_get_server_known_returns_config():
    from dewie.providers.servers import get_server

    server = get_server("openai")
    assert server.label == "openai"
    assert server.api_key_env == "OPENAI_API_KEY"


# ── resolve_api_key ────────────────────────────────────────────────────────────


def test_resolve_api_key_env_var(monkeypatch):
    from dewie.providers.servers import ServerConfig, resolve_api_key

    monkeypatch.setenv("MY_KEY", "secret123")
    server = ServerConfig(label="x", api_format="openai", endpoint="http://x", api_key_env="MY_KEY")
    assert resolve_api_key(server) == "secret123"


def test_resolve_api_key_no_key_configured():
    from dewie.providers.servers import ServerConfig, resolve_api_key

    server = ServerConfig(label="x", api_format="openai", endpoint="http://x")
    assert resolve_api_key(server) == ""


def test_resolve_api_key_file(tmp_path, monkeypatch):
    import json

    from dewie.providers.servers import ServerConfig, resolve_api_key

    token_file = tmp_path / "token.json"
    token_file.write_text(json.dumps({"token": "rotating-token"}))
    monkeypatch.setenv("MY_TOKEN_FILE", str(token_file))

    server = ServerConfig(
        label="x", api_format="openai", endpoint="http://x", api_key_file_env="MY_TOKEN_FILE"
    )
    assert resolve_api_key(server) == "rotating-token"


def test_resolve_api_key_ciphertext_wins_over_env_var(monkeypatch):
    from cryptography.fernet import Fernet

    from dewie.config import settings as _settings
    from dewie.providers.servers import ServerConfig, resolve_api_key

    master_key = Fernet.generate_key().decode()
    monkeypatch.setattr(_settings, "encryption_master_key", master_key)
    monkeypatch.setenv("MY_KEY", "env-secret")

    from dewie.crypto import encrypt

    ciphertext = encrypt("literal-secret")
    server = ServerConfig(
        label="x",
        api_format="openai",
        endpoint="http://x",
        api_key_env="MY_KEY",
        api_key_ciphertext=ciphertext,
    )
    assert resolve_api_key(server) == "literal-secret"


def test_resolve_api_key_ciphertext_without_master_key_falls_back(monkeypatch):
    from dewie.config import settings as _settings
    from dewie.providers.servers import ServerConfig, resolve_api_key

    monkeypatch.setattr(_settings, "encryption_master_key", "")
    monkeypatch.setenv("MY_KEY", "env-secret")

    server = ServerConfig(
        label="x",
        api_format="openai",
        endpoint="http://x",
        api_key_env="MY_KEY",
        api_key_ciphertext="not-a-real-ciphertext",
    )
    assert resolve_api_key(server) == "env-secret"
