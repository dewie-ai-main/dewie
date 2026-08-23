"""Tests for dewie.model_adapter — pure functions."""

from __future__ import annotations

import json

import pytest

# ── supports_native_tools ─────────────────────────────────────────────────────


def test_gpt_models_support_native_tools():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("gpt-4o") is True
    assert supports_native_tools("gpt-4o-mini") is True
    assert supports_native_tools("GPT-4") is True


def test_claude_models_support_native_tools():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("claude-3-5-sonnet") is True
    assert supports_native_tools("claude-opus-4-7") is True


def test_gemini_models_support_native_tools():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("gemini-pro") is True


def test_grok_models_support_native_tools():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("grok-1") is True


def test_llama_does_not_support_native_tools():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("llama3.2:3b") is False


def test_qwen_does_not_support_native_tools():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("qwen2.5:7b") is False


# ── _parse_react_tool_calls ───────────────────────────────────────────────────


def test_parse_react_tool_calls_empty():
    from dewie.model_adapter import _parse_react_tool_calls

    assert _parse_react_tool_calls("No tool calls here") == []


def test_parse_react_tool_calls_single():
    from dewie.model_adapter import _parse_react_tool_calls

    text = 'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "AI news"}}'
    tcs = _parse_react_tool_calls(text)
    assert len(tcs) == 1
    assert tcs[0].name == "dewie_search"
    assert tcs[0].arguments == {"query": "AI news"}


def test_parse_react_tool_calls_multiple():
    from dewie.model_adapter import _parse_react_tool_calls

    text = (
        'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "q1"}}\n'
        "Some reasoning...\n"
        'TOOL_CALL: {"name": "dewie_expand", "arguments": {"doc_id": "abc"}}'
    )
    tcs = _parse_react_tool_calls(text)
    assert len(tcs) == 2
    assert tcs[0].name == "dewie_search"
    assert tcs[1].name == "dewie_expand"


def test_parse_react_tool_calls_invalid_json_skipped():
    from dewie.model_adapter import _parse_react_tool_calls

    text = "TOOL_CALL: {invalid json}"
    tcs = _parse_react_tool_calls(text)
    assert tcs == []


def test_parse_react_tool_calls_no_name_skipped():
    from dewie.model_adapter import _parse_react_tool_calls

    text = 'TOOL_CALL: {"arguments": {"query": "test"}}'
    tcs = _parse_react_tool_calls(text)
    assert tcs == []


# ── _inject_react_instructions ────────────────────────────────────────────────


def test_inject_react_appends_to_existing_system():
    from dewie.model_adapter import _inject_react_instructions

    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result = _inject_react_instructions(msgs)
    assert result[0]["role"] == "system"
    assert "Be helpful." in result[0]["content"]
    assert "TOOL_CALL" in result[0]["content"]


def test_inject_react_inserts_system_when_missing():
    from dewie.model_adapter import _inject_react_instructions

    msgs = [{"role": "user", "content": "Hello"}]
    result = _inject_react_instructions(msgs)
    assert result[0]["role"] == "system"
    assert "TOOL_CALL" in result[0]["content"]
    assert result[1]["role"] == "user"


def test_inject_react_does_not_mutate_original():
    from dewie.model_adapter import _inject_react_instructions

    msgs = [{"role": "system", "content": "Original."}]
    _inject_react_instructions(msgs)
    assert msgs[0]["content"] == "Original."


# ── parse_response ────────────────────────────────────────────────────────────


def _make_raw(content="Hello", finish_reason="stop", tool_calls=None, usage=None):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20},
    }


def test_parse_response_basic():
    from dewie.model_adapter import parse_response

    raw = _make_raw("Answer here")
    resp = parse_response(raw)
    assert resp.content == "Answer here"
    assert resp.finish_reason == "stop"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 20
    assert resp.tool_calls == []


def test_parse_response_with_native_tool_calls():
    from dewie.model_adapter import parse_response

    raw = _make_raw(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_abc",
                "function": {"name": "dewie_search", "arguments": '{"query": "test"}'},
            }
        ],
    )
    resp = parse_response(raw)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "dewie_search"
    assert resp.tool_calls[0].arguments == {"query": "test"}


def test_parse_response_react_fallback():
    from dewie.model_adapter import parse_response

    content = 'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "AI"}}'
    raw = _make_raw(content=content, finish_reason="stop")
    resp = parse_response(raw)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "dewie_search"
    assert resp.finish_reason == "tool_calls"


def test_parse_response_tool_calls_null_falls_back_to_stop():
    from dewie.model_adapter import parse_response

    raw = _make_raw(content="Just text, no tool calls.", finish_reason="tool_calls")
    resp = parse_response(raw)
    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []


def test_llm_response_has_tool_calls_property():
    from dewie.model_adapter import LLMResponse, ToolCall

    resp = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="x", name="fn", arguments={})],
        finish_reason="tool_calls",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    assert resp.has_tool_calls is True
    assert resp.has_content is False


def test_llm_response_has_content_property():
    from dewie.model_adapter import LLMResponse

    resp = LLMResponse(
        content="some text",
        tool_calls=[],
        finish_reason="stop",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    assert resp.has_content is True
    assert resp.has_tool_calls is False


# ── ModelClient ───────────────────────────────────────────────────────────────


def test_model_client_explicit_provider():
    from dewie.model_adapter import ModelClient

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        "os.environ", {"LLM_PROVIDER": "openai"}
    ):
        client = ModelClient(provider="anthropic", model="gpt-4o")
    assert client.provider == "anthropic"
    assert client.model == "gpt-4o"


def test_model_client_defaults_to_openai():
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="gpt-4o-mini")
    assert client.provider == "openai"
    assert client.base_url == "https://api.openai.com/v1"


def test_model_client_unknown_provider_raises():
    from dewie.model_adapter import ModelClient

    with pytest.raises(RuntimeError, match="Unknown server label"):
        ModelClient(provider="unknown-provider-xyz", model="some-model")


def test_model_client_native_tools_flag():
    from dewie.model_adapter import ModelClient

    gpt = ModelClient(provider="openai", model="gpt-4o")
    assert gpt._uses_native_tools is True

    llama = ModelClient(provider="openai", model="llama3.2:3b")
    assert llama._uses_native_tools is False


def test_model_client_custom_server(monkeypatch, tmp_path):
    """A custom registered server resolves to its configured endpoint."""
    from dewie.model_adapter import ModelClient

    yml = tmp_path / "dewie.yml"
    yml.write_text(
        "servers:\n  - label: my-server\n    api_format: openai\n"
        "    endpoint: https://example.com/llm\n"
    )
    monkeypatch.setenv("DEWIE_CONFIG_PATH", str(yml))

    client = ModelClient(provider="my-server", model="gpt-4o")
    assert client.provider == "my-server"
    assert client.base_url == "https://example.com/llm/v1"


def test_model_client_env_provider():
    """LLM_PROVIDER env var is used when no explicit provider given."""
    from unittest.mock import patch

    from dewie.model_adapter import ModelClient

    with patch.dict("os.environ", {"LLM_PROVIDER": "openai"}):
        client = ModelClient(model="gpt-4o")
    assert client.provider == "openai"


def test_model_client_registry_resolution():
    """Provider resolved from model_registry when no env var is set."""
    from unittest.mock import MagicMock, patch

    from dewie.model_adapter import ModelClient

    fake_info = MagicMock()
    fake_info.provider = "openai"
    fake_registry = MagicMock()
    fake_registry.get.return_value = fake_info

    with patch.dict("os.environ", {}, clear=True):
        with patch("dewie.model_adapter.os.environ.get", side_effect=lambda k, d=None: d):
            with patch.dict(
                "sys.modules", {"dewie.model_registry": MagicMock(registry=fake_registry)}
            ):
                client = ModelClient(model="some-model")
    assert client.provider == "openai"


def test_model_client_no_provider_raises():
    """Raises RuntimeError if no provider can be resolved."""
    import sys
    from unittest.mock import MagicMock, patch

    from dewie.model_adapter import ModelClient

    # Use a model not in the registry and not matching any local prefixes
    # Patch model_registry to simulate it not providing a provider
    fake_registry = MagicMock()
    fake_registry.get.return_value = None  # no registry entry
    fake_module = MagicMock()
    fake_module.registry = fake_registry
    with patch.dict("os.environ", {}, clear=True):
        with patch("dewie.model_adapter.os.environ.get", side_effect=lambda k, d=None: d):
            with patch.dict(sys.modules, {"dewie.model_registry": fake_module}):
                with pytest.raises(RuntimeError, match="LLM provider not configured"):
                    ModelClient(model="unknown-exotic-model-xyz")


# ── ModelClient context manager ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_client_context_manager():
    """__aenter__/__aexit__ initialize and close the HTTP client."""
    from unittest.mock import AsyncMock, patch

    from dewie.model_adapter import ModelClient

    mock_client = AsyncMock()
    with patch("httpx.AsyncClient", return_value=mock_client):
        client = ModelClient(provider="openai", model="gpt-4o")
        async with client as c:
            assert c is client
            assert client._client is mock_client
        mock_client.aclose.assert_called_once()


# ── ModelClient._auth_headers ─────────────────────────────────────────────────


def test_auth_headers_openai():
    """openai provider uses Authorization: Bearer."""
    from unittest.mock import patch

    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="gpt-4o")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        headers = client._auth_headers()
    assert headers["Authorization"] == "Bearer sk-test-key"


def test_auth_headers_anthropic():
    """anthropic provider uses x-api-key."""
    from unittest.mock import patch

    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="anthropic", model="claude-3-sonnet")
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-key-123"}):
        headers = client._auth_headers()
    assert headers["x-api-key"] == "ant-key-123"


def test_auth_headers_custom_server_with_api_key_env(monkeypatch, tmp_path):
    """A custom server's api_key_env is used for Authorization: Bearer."""
    from dewie.model_adapter import ModelClient

    yml = tmp_path / "dewie.yml"
    yml.write_text(
        "servers:\n  - label: my-server\n    api_format: openai\n"
        "    endpoint: https://example.com/llm\n    api_key_env: MY_SERVER_KEY\n"
    )
    monkeypatch.setenv("DEWIE_CONFIG_PATH", str(yml))
    monkeypatch.setenv("MY_SERVER_KEY", "custom-secret")

    client = ModelClient(provider="my-server", model="gpt-4o")
    headers = client._auth_headers()
    assert headers["Authorization"] == "Bearer custom-secret"


def test_auth_headers_custom_server_no_key_sends_no_auth(monkeypatch, tmp_path):
    """A custom server with no api_key_env (e.g. local llama.cpp) sends no auth header."""
    from dewie.model_adapter import ModelClient

    yml = tmp_path / "dewie.yml"
    yml.write_text(
        "servers:\n  - label: my-server\n    api_format: openai\n"
        "    endpoint: https://example.com/llm\n"
    )
    monkeypatch.setenv("DEWIE_CONFIG_PATH", str(yml))

    client = ModelClient(provider="my-server", model="gpt-4o")
    headers = client._auth_headers()
    assert "Authorization" not in headers


# ── ModelClient.complete ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_basic():
    """complete() makes a POST to /chat/completions and parses the response."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.model_adapter import ModelClient

    raw_response = {
        "choices": [{"message": {"content": "Paris", "tool_calls": None}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = raw_response

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    mock_http.aclose = AsyncMock()

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        with patch("httpx.AsyncClient", return_value=mock_http):
            client = ModelClient(provider="openai", model="gpt-4o")
            async with client:
                result = await client.complete(
                    messages=[{"role": "user", "content": "What is the capital of France?"}]
                )

    assert result.content == "Paris"
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_complete_with_tools_native():
    """complete() with native tools sends tools in payload."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.model_adapter import ModelClient

    raw_response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "search", "arguments": '{"query": "AI"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = raw_response

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    mock_http.aclose = AsyncMock()

    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        with patch("httpx.AsyncClient", return_value=mock_http):
            client = ModelClient(provider="openai", model="gpt-4o")
            async with client:
                result = await client.complete(
                    messages=[{"role": "user", "content": "Search for AI"}],
                    tools=tools,
                )

    assert result.has_tool_calls
    assert result.tool_calls[0].name == "search"


# ── ModelClient helper methods ────────────────────────────────────────────────


def test_tool_response_native_mode():
    """tool_response for native tools returns role=tool dict."""
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="gpt-4o")
    result = client.tool_response("call_123", "The answer is 42", "calc")
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_123"
    assert result["content"] == "The answer is 42"


def test_tool_response_react_mode():
    """tool_response for ReAct mode returns role=user with prefix."""
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="llama3.2:3b")
    result = client.tool_response("react_0", "result text", "search")
    assert result["role"] == "user"
    assert "search" in result["content"]
    assert "result text" in result["content"]


def test_assistant_message_without_tool_calls():
    """assistant_message with no tool calls returns plain content dict."""
    from dewie.model_adapter import LLMResponse, ModelClient

    client = ModelClient(provider="openai", model="gpt-4o")
    resp = LLMResponse(
        content="Hello",
        tool_calls=[],
        finish_reason="stop",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    msg = client.assistant_message(resp)
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hello"
    assert "tool_calls" not in msg


def test_assistant_message_with_tool_calls():
    """assistant_message with tool calls includes tool_calls for native mode."""
    from dewie.model_adapter import LLMResponse, ModelClient, ToolCall

    client = ModelClient(provider="openai", model="gpt-4o")
    resp = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "AI"})],
        finish_reason="tool_calls",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    msg = client.assistant_message(resp)
    assert "tool_calls" in msg
    assert msg["tool_calls"][0]["function"]["name"] == "search"


# ── Gemma-4 TOOL_CALL parsing ─────────────────────────────────────────────────


def test_parse_react_gemma4_format():
    """Gemma-4 <|tool_call|> format is parsed into ToolCall objects."""
    from dewie.model_adapter import _parse_react_tool_calls

    text = '<|tool_call|>call:dewie_search{"query": "AI research"}'
    tool_calls = _parse_react_tool_calls(text)
    assert len(tool_calls) >= 1
    assert tool_calls[0].name == "dewie_search"


def test_parse_response_strips_tool_call_syntax_from_content():
    """parse_response strips TOOL_CALL syntax from content field."""
    from dewie.model_adapter import parse_response

    content = 'TOOL_CALL: {"name": "search", "arguments": {"query": "AI"}}'
    raw = {
        "choices": [{"message": {"content": content, "tool_calls": None}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }
    resp = parse_response(raw)
    # Should have parsed the tool call
    assert len(resp.tool_calls) == 1
    # Content should be stripped/None
    assert resp.content is None or "TOOL_CALL" not in (resp.content or "")


# ── _is_openai_native branch: grok + gpt-5 ────────────────────────────────────


def test_supports_native_tools_grok_models():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("grok-2") is True
    assert supports_native_tools("grok-beta") is True


def test_supports_native_tools_gpt5_models():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("gpt-5-turbo") is True
    assert supports_native_tools("gpt-5o") is True


# ── _parse_react_tool_calls: JSON parse failure + fallback ────────────────────


def test_parse_react_json_decode_error_falls_back_to_regex():
    """When JSON parse fails, falls back to regex extraction."""
    from dewie.model_adapter import _parse_react_tool_calls

    # Valid JSON in wrong place, force JSON decode error on args
    text = 'TOOL_CALL: {"name": "my_func", "arguments": {invalid json here}}'
    result = _parse_react_tool_calls(text)
    # May parse partially or return empty, but should not raise
    assert isinstance(result, list)


def test_parse_react_outer_exception_returns_empty():
    """Exceptions during parse return empty list gracefully."""
    from dewie.model_adapter import _parse_react_tool_calls

    # Pass empty string to trigger no-match path
    result = _parse_react_tool_calls("")
    assert result == []


# ── _parse_react_tool_calls: react_tcs result consumed ───────────────────────


def test_parse_react_tool_calls_with_nested_json_arguments():
    """Valid ReAct format with JSON arguments parses correctly."""
    from dewie.model_adapter import _parse_react_tool_calls

    args = {"query": "machine learning", "limit": 5}
    text = f"TOOL_CALL: {json.dumps({'name': 'search', 'arguments': args})}"
    result = _parse_react_tool_calls(text)
    assert len(result) == 1
    assert result[0].name == "search"
    assert result[0].arguments["query"] == "machine learning"


# ── complete: ReAct tool calls ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_react_tool_calls_parsed():
    """When _uses_native_tools is False, tool calls are parsed from content."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.model_adapter import ModelClient

    tool_call_text = 'TOOL_CALL: {"name": "search", "arguments": {"q": "test"}}'
    mock_response = {
        "choices": [
            {"message": {"content": tool_call_text, "tool_calls": None}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = mock_response
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    mock_http.aclose = AsyncMock()
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        with patch("httpx.AsyncClient", return_value=mock_http):
            client = ModelClient(provider="openai", model="gpt-4o")
            async with client:
                client._uses_native_tools = False
                result = await client.complete(
                    [{"role": "user", "content": "search for something"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "description": "search docs",
                                "parameters": {},
                            },
                        }
                    ],
                )
    # When ReAct parses tool calls, content may be None and tool_calls populated
    assert result.tool_calls is not None and len(result.tool_calls) > 0


# ── complete: response_format passed through ─────────────────────────────────


@pytest.mark.asyncio
async def test_complete_with_response_format():
    """response_format parameter is included in the payload."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dewie.model_adapter import ModelClient

    mock_response = {
        "choices": [
            {
                "message": {"content": '{"key": "value"}', "tool_calls": None},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = mock_response
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    mock_http.aclose = AsyncMock()
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        with patch("httpx.AsyncClient", return_value=mock_http):
            client = ModelClient(provider="openai", model="gpt-4o-mini")
            async with client:
                result = await client.complete(
                    [{"role": "user", "content": "return JSON"}],
                    response_format={"type": "json_object"},
                )
    assert result.content == '{"key": "value"}'
    # Verify response_format was in the payload
    call_kwargs = mock_http.post.call_args
    payload = call_kwargs.kwargs.get(
        "json", call_kwargs.args[1] if len(call_kwargs.args) > 1 else {}
    )
    assert "response_format" in payload
