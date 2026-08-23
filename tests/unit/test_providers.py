"""Tests for dewie providers — OpenAI, Anthropic, Ollama, OpenRouter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_http_cm(response):
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.get = AsyncMock(return_value=response)
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    return mock_cm, mock_client


def _mock_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(return_value=json_data or {})
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return resp


# ── OpenAIChatProvider ────────────────────────────────────────────────────────


def test_openai_chat_provider_name():
    from dewie.providers.openai_provider import OpenAIChatProvider

    p = OpenAIChatProvider(api_key="sk-test")
    assert p.name == "openai"


def test_openai_chat_provider_custom_base_url():
    from dewie.providers.openai_provider import OpenAIChatProvider

    p = OpenAIChatProvider(api_key="k", base_url="http://custom-llm:8000/v1")
    assert "custom-llm" in p._chat_url


@pytest.mark.asyncio
async def test_openai_chat_returns_empty_without_api_key():
    from dewie.providers.openai_provider import OpenAIChatProvider

    p = OpenAIChatProvider(model="gpt-4o-mini", api_key=None)
    p._api_key = ""  # ensure empty
    result = await p.complete([{"role": "user", "content": "Hello"}])
    assert result == ""


@pytest.mark.asyncio
async def test_openai_chat_complete_success():
    from dewie.providers.openai_provider import OpenAIChatProvider

    p = OpenAIChatProvider(api_key="sk-test")

    resp = _mock_response(200, {"choices": [{"message": {"content": "  Hello!  "}}]})
    cm, _ = _mock_http_cm(resp)
    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.complete([{"role": "user", "content": "hi"}])
    assert result == "Hello!"


@pytest.mark.asyncio
async def test_openai_chat_complete_exception_returns_empty():
    from dewie.providers.openai_provider import OpenAIChatProvider

    p = OpenAIChatProvider(api_key="sk-test")

    cm, client = _mock_http_cm(None)
    client.post = AsyncMock(side_effect=Exception("connection error"))
    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.complete([{"role": "user", "content": "hi"}])
    assert result == ""


# ── OpenAIEmbeddingProvider ───────────────────────────────────────────────────


def test_openai_embed_provider_name():
    from dewie.providers.openai_provider import OpenAIEmbeddingProvider

    p = OpenAIEmbeddingProvider(api_key="sk-test")
    assert p.name == "openai"


@pytest.mark.asyncio
async def test_openai_embed_returns_none_without_key():
    from dewie.providers.openai_provider import OpenAIEmbeddingProvider

    p = OpenAIEmbeddingProvider(api_key=None)
    p._api_key = ""
    result = await p.embed(["text"])
    assert result is None


@pytest.mark.asyncio
async def test_openai_embed_success():
    from dewie.providers.openai_provider import OpenAIEmbeddingProvider

    p = OpenAIEmbeddingProvider(api_key="sk-test")

    resp = _mock_response(
        200,
        {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]},
            ]
        },
    )
    cm, _ = _mock_http_cm(resp)
    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.embed(["text1", "text2"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_openai_embed_exception_returns_none():
    from dewie.providers.openai_provider import OpenAIEmbeddingProvider

    p = OpenAIEmbeddingProvider(api_key="sk-test")

    cm, client = _mock_http_cm(None)
    client.post = AsyncMock(side_effect=Exception("timeout"))
    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.embed(["text"])
    assert result is None


# ── AnthropicChatProvider ─────────────────────────────────────────────────────


def test_anthropic_provider_name():
    from dewie.providers.anthropic_provider import AnthropicChatProvider

    p = AnthropicChatProvider(api_key="ant-test")
    assert p.name == "anthropic"


@pytest.mark.asyncio
async def test_anthropic_returns_empty_without_key():
    from dewie.providers.anthropic_provider import AnthropicChatProvider

    p = AnthropicChatProvider(api_key=None)
    p._api_key = ""
    result = await p.complete([{"role": "user", "content": "hi"}])
    assert result == ""


@pytest.mark.asyncio
async def test_anthropic_complete_success():
    from dewie.providers.anthropic_provider import AnthropicChatProvider

    p = AnthropicChatProvider(api_key="ant-test", model="claude-3-haiku-20240307")

    resp = _mock_response(200, {"content": [{"type": "text", "text": "Hello from Claude"}]})
    cm, _ = _mock_http_cm(resp)
    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.complete([{"role": "user", "content": "hi"}])
    assert result == "Hello from Claude"


@pytest.mark.asyncio
async def test_anthropic_omits_temperature_on_current_models():
    """Current top-tier models 400 on temperature — it must not be sent."""
    from dewie.providers.anthropic_provider import AnthropicChatProvider

    p = AnthropicChatProvider(api_key="ant-test", model="claude-opus-4-8")
    resp = _mock_response(200, {"content": [{"type": "text", "text": "ok"}]})
    cm, client = _mock_http_cm(resp)
    with patch("httpx.AsyncClient", return_value=cm):
        await p.complete([{"role": "user", "content": "hi"}])
    payload = client.post.call_args.kwargs["json"]
    assert "temperature" not in payload
    assert payload["model"] == "claude-opus-4-8"
    assert payload["max_tokens"] > 0


@pytest.mark.asyncio
async def test_anthropic_sends_temperature_on_older_models():
    """Models that still accept sampling params should receive temperature."""
    from dewie.providers.anthropic_provider import AnthropicChatProvider

    p = AnthropicChatProvider(api_key="ant-test", model="claude-haiku-4-5-20251001")
    resp = _mock_response(200, {"content": [{"type": "text", "text": "ok"}]})
    cm, client = _mock_http_cm(resp)
    with patch("httpx.AsyncClient", return_value=cm):
        await p.complete([{"role": "user", "content": "hi"}], temperature=0.2)
    payload = client.post.call_args.kwargs["json"]
    assert payload["temperature"] == 0.2


@pytest.mark.asyncio
async def test_anthropic_complete_exception():
    from dewie.providers.anthropic_provider import AnthropicChatProvider

    p = AnthropicChatProvider(api_key="ant-test")

    cm, client = _mock_http_cm(None)
    client.post = AsyncMock(side_effect=Exception("error"))
    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.complete([{"role": "user", "content": "hi"}])
    assert result == ""


# ── OllamaChatProvider ────────────────────────────────────────────────────────


def test_ollama_chat_provider_name():
    from dewie.providers.ollama import OllamaChatProvider

    p = OllamaChatProvider(model="llama3.2:3b")
    assert p.name == "ollama"


@pytest.mark.asyncio
async def test_ollama_chat_complete_success():
    from dewie.providers.ollama import OllamaChatProvider

    p = OllamaChatProvider(model="llama3.2:3b")

    resp = _mock_response(200, {"choices": [{"message": {"content": "Ollama response"}}]})
    cm, _ = _mock_http_cm(resp)
    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.complete([{"role": "user", "content": "hi"}])
    assert result == "Ollama response"


@pytest.mark.asyncio
async def test_ollama_chat_exception_returns_empty():
    from dewie.providers.ollama import OllamaChatProvider

    p = OllamaChatProvider(model="llama3.2:3b")

    cm, client = _mock_http_cm(None)
    client.post = AsyncMock(side_effect=Exception("ollama down"))
    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.complete([{"role": "user", "content": "hi"}])
    assert result == ""


def test_ollama_embed_provider_name():
    from dewie.providers.ollama import OllamaEmbeddingProvider

    p = OllamaEmbeddingProvider(model="nomic-embed-text")
    assert p.name == "ollama"


@pytest.mark.asyncio
async def test_ollama_embed_success():
    from dewie.providers.ollama import OllamaEmbeddingProvider

    p = OllamaEmbeddingProvider(model="nomic-embed-text")

    # Ollama /api/embeddings returns one embedding per call
    resp1 = _mock_response(200, {"embedding": [0.1, 0.2]})
    resp2 = _mock_response(200, {"embedding": [0.3, 0.4]})

    call_count = [0]

    async def mock_post(url, **kwargs):
        call_count[0] += 1
        return resp1 if call_count[0] == 1 else resp2

    mock_client = AsyncMock()
    mock_client.post = mock_post
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.embed(["text1", "text2"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_ollama_embed_exception():
    from dewie.providers.ollama import OllamaEmbeddingProvider

    p = OllamaEmbeddingProvider(model="nomic-embed-text")

    cm, client = _mock_http_cm(None)
    client.post = AsyncMock(side_effect=Exception("error"))
    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.embed(["text"])
    assert result is None


# ── OpenRouterChatProvider ────────────────────────────────────────────────────


def test_openrouter_provider_name():
    from dewie.providers.openrouter import OpenRouterChatProvider

    p = OpenRouterChatProvider(api_key="or-test")
    assert p.name == "openrouter"


@pytest.mark.asyncio
async def test_openrouter_returns_empty_without_key():
    from dewie.providers.openrouter import OpenRouterChatProvider

    p = OpenRouterChatProvider(api_key=None)
    p._api_key = ""
    result = await p.complete([{"role": "user", "content": "hi"}])
    assert result == ""


@pytest.mark.asyncio
async def test_openrouter_complete_success():
    from dewie.providers.openrouter import OpenRouterChatProvider

    p = OpenRouterChatProvider(api_key="or-test", model="anthropic/claude-3.5-sonnet")

    resp = _mock_response(200, {"choices": [{"message": {"content": "OpenRouter response"}}]})
    cm, _ = _mock_http_cm(resp)
    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.complete([{"role": "user", "content": "hi"}])
    assert result == "OpenRouter response"


@pytest.mark.asyncio
async def test_openrouter_complete_exception():
    from dewie.providers.openrouter import OpenRouterChatProvider

    p = OpenRouterChatProvider(api_key="or-test")

    cm, client = _mock_http_cm(None)
    client.post = AsyncMock(side_effect=Exception("error"))
    with patch("httpx.AsyncClient", return_value=cm):
        result = await p.complete([{"role": "user", "content": "hi"}])
    assert result == ""
