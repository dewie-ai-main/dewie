"""Tests for AgentBackend — OpenClaw gateway enrichment backend."""

from __future__ import annotations

from unittest.mock import (  # patch used in test_registry_builds_agent_backend
    AsyncMock,
    MagicMock,
)

import httpx
import pytest

from dewie.enrichment.backends.agent import AgentBackend
from dewie.enrichment.base import BackendError


def make_backend(**kwargs):
    defaults = dict(
        name="test_agent",
        endpoint="http://localhost:18789",
        model="gpt-4o",
        provider="custom",
        auth_token="test-gateway-token",
        timeout=30.0,
    )
    defaults.update(kwargs)
    return AgentBackend(**defaults)


def _mock_response(content: str, status: int = 200):
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = {"choices": [{"message": {"content": content}}]}
    mock.text = content
    return mock


@pytest.mark.asyncio
async def test_complete_success():
    backend = make_backend()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response('{"title": "Test"}'))
    backend._http_client = mock_client

    result = await backend.complete("extract this doc")
    assert result == '{"title": "Test"}'
    call_kwargs = mock_client.post.call_args
    assert "localhost:18789/v1/chat/completions" in call_kwargs[0][0]
    headers = call_kwargs[1]["headers"]
    assert headers["Authorization"] == "Bearer test-gateway-token"


@pytest.mark.asyncio
async def test_complete_uses_auth_token_env(monkeypatch):
    monkeypatch.setenv("MY_GW_TOKEN", "env-token-xyz")
    backend = make_backend(auth_token=None, auth_token_env="MY_GW_TOKEN")
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response('{"ok": true}'))
    backend._http_client = mock_client

    await backend.complete("prompt")
    headers = mock_client.post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer env-token-xyz"


@pytest.mark.asyncio
async def test_complete_no_auth_token():
    backend = make_backend(auth_token=None, auth_token_env=None)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response('{"ok": true}'))
    backend._http_client = mock_client

    await backend.complete("prompt")
    headers = mock_client.post.call_args[1]["headers"]
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_complete_401_raises():
    backend = make_backend()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response("Unauthorized", status=401))
    backend._http_client = mock_client

    with pytest.raises(BackendError, match="auth failed"):
        await backend.complete("prompt")


@pytest.mark.asyncio
async def test_complete_429_raises():
    backend = make_backend()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response("Rate limited", status=429))
    backend._http_client = mock_client

    with pytest.raises(BackendError, match="rate limited"):
        await backend.complete("prompt")


@pytest.mark.asyncio
async def test_complete_connection_refused():
    backend = make_backend()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    backend._http_client = mock_client

    with pytest.raises(BackendError, match="connection refused"):
        await backend.complete("prompt")


@pytest.mark.asyncio
async def test_complete_timeout():
    backend = make_backend()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
    backend._http_client = mock_client

    with pytest.raises(BackendError, match="timed out"):
        await backend.complete("prompt")


@pytest.mark.asyncio
async def test_complete_empty_response():
    backend = make_backend()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_mock_response("   "))
    backend._http_client = mock_client

    with pytest.raises(BackendError, match="empty response"):
        await backend.complete("prompt")


def test_name():
    backend = make_backend(name="remote_agent")
    assert backend.name == "remote_agent"


def test_endpoint_normalised():
    backend = make_backend(endpoint="http://localhost:18789/")
    assert backend._endpoint == "http://localhost:18789/v1/chat/completions"


def test_registry_builds_agent_backend():
    """Registry correctly instantiates AgentBackend for type=agent."""
    import json
    from unittest.mock import MagicMock

    from dewie.enrichment.registry import BackendRegistry

    descriptor = json.dumps(
        [
            {
                "name": "remote_agent",
                "type": "agent",
                "endpoint": "http://localhost:18789",
                "model": "gpt-4o",
                "provider": "custom",
                "auth_token_env": "MOMUS_GATEWAY_TOKEN",
                "timeout": 60,
            }
        ]
    )

    mock_settings = MagicMock()
    mock_settings.enrichment_backends = descriptor
    mock_settings.enrichment_default_backend = "remote_agent"

    registry = BackendRegistry.from_config(mock_settings)
    backend = registry.get("remote_agent")
    assert backend is not None
    assert backend.name == "remote_agent"
    assert isinstance(backend, AgentBackend)
    assert backend._provider == "custom"
