"""Tests for qwen model tool calls via openclaw provider.

Covers: supports_native_tools for qwen variants, ModelClient._uses_native_tools flag,
ReAct instruction injection, _parse_react_tool_calls parsing, full agent flow,
and AgentBackend payload construction.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── supports_native_tools for qwen ───────────────────────────────────────────


def test_qwen_does_not_support_native_tools():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("qwen2.5:7b") is False
    assert supports_native_tools("qwen3:32b") is False
    assert supports_native_tools("Qwen3-235B-A22B") is False
    assert supports_native_tools("qwen2.5-vl:72b") is False
    assert supports_native_tools("qwen") is False
    assert supports_native_tools("QWEN-72B") is False


def test_qwen_model_lowercase_edge_cases():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("Qwen/Qwen2.5-7B") is False
    assert supports_native_tools("Qwen/Qwen3-32B") is False
    assert supports_native_tools("qwen-3-235b") is False


def test_qwen_not_confused_with_native_models():
    from dewie.model_adapter import supports_native_tools

    # Make sure qwen doesn't accidentally match gpt/claude/gemini/grok prefixes
    assert supports_native_tools("qwen2.5-coder:1.5b") is False
    assert supports_native_tools("qwen2.5-math:7b") is False


# ── ModelClient._uses_native_tools for qwen ──────────────────────────────────


def test_model_client_qwen_native_tools_flag_false():
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="qwen2.5:7b")
    assert client._uses_native_tools is False

    client2 = ModelClient(provider="openai", model="qwen3:32b")
    assert client2._uses_native_tools is False


def test_model_client_qwen_camelcase_flag():
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="Qwen3-235B-A22B")
    assert client._uses_native_tools is False


def test_model_client_qwen_vs_gpt_flag():
    from dewie.model_adapter import ModelClient

    gpt_client = ModelClient(provider="openai", model="gpt-4o")
    qwen_client = ModelClient(provider="openai", model="qwen2.5:7b")

    assert gpt_client._uses_native_tools is True
    assert qwen_client._uses_native_tools is False


def test_model_client_qwen_lmstudio_provider():
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="qwen2.5:7b")
    assert client._uses_native_tools is False
    assert client.provider == "openai"


def test_model_client_qwen_openrouter_provider(monkeypatch):
    from dewie.model_adapter import ModelClient
    from dewie.providers import servers

    # openrouter is no longer a built-in server (issue #27) — register a stub
    monkeypatch.setitem(
        servers._BUILTIN_SERVERS,
        "openrouter",
        servers.ServerConfig(
            label="openrouter",
            api_format="openai",
            endpoint="https://openrouter.ai/api",
            api_key_env="OPENROUTER_API_KEY",
        ),
    )
    client = ModelClient(provider="openrouter", model="qwen2.5:7b")
    assert client._uses_native_tools is False
    assert client.provider == "openrouter"


# ── _inject_react_instructions for qwen ──────────────────────────────────────


def test_inject_react_qwen_messages():
    from dewie.model_adapter import _inject_react_instructions

    msgs = [{"role": "user", "content": "What is Dewie?"}]
    result = _inject_react_instructions(msgs)
    assert result[0]["role"] == "system"
    assert "TOOL_CALL:" in result[0]["content"]
    assert "dewie_search" in result[0]["content"]
    assert "dewie_expand" in result[0]["content"]
    assert "dewie_read" in result[0]["content"]


def test_inject_react_qwen_with_existing_system():
    from dewie.model_adapter import _inject_react_instructions

    msgs = [
        {"role": "system", "content": "You are a helpful assistant for Dewie."},
        {"role": "user", "content": "Search for AI"},
    ]
    result = _inject_react_instructions(msgs)
    assert "You are a helpful assistant for Dewie." in result[0]["content"]
    assert "TOOL_CALL:" in result[0]["content"]


def test_react_instructions_format():
    from dewie.model_adapter import _REACT_INSTRUCTIONS

    assert "TOOL_CALL:" in _REACT_INSTRUCTIONS
    assert '{"name": "<tool_name>", "arguments": {<args>}' in _REACT_INSTRUCTIONS
    assert "dewie_search" in _REACT_INSTRUCTIONS
    assert "dewie_expand" in _REACT_INSTRUCTIONS
    assert "dewie_read" in _REACT_INSTRUCTIONS


# ── _parse_react_tool_calls for qwen-style responses ─────────────────────────


def test_parse_react_qwen_single_tool_call():
    from dewie.model_adapter import _parse_react_tool_calls

    text = 'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "AI research"}}'
    result = _parse_react_tool_calls(text)
    assert len(result) == 1
    assert result[0].name == "dewie_search"
    assert result[0].arguments["query"] == "AI research"
    assert result[0].id.startswith("react_")


def test_parse_react_qwen_multiple_tool_calls():
    from dewie.model_adapter import _parse_react_tool_calls

    text = (
        "I need to search first, then expand the result.\n"
        'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "Dewie"}}\n'
        "Found it, now expanding.\n"
        'TOOL_CALL: {"name": "dewie_expand", "arguments": {"doc_id": "doc-123", "limit": 10}}'
    )
    result = _parse_react_tool_calls(text)
    assert len(result) == 2
    assert result[0].name == "dewie_search"
    assert result[1].name == "dewie_expand"


def test_parse_react_qwen_nested_arguments():
    from dewie.model_adapter import _parse_react_tool_calls

    text = 'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "machine learning", "filters": {"date": "2024", "source": "arxiv"}}}'
    result = _parse_react_tool_calls(text)
    assert len(result) == 1
    assert result[0].arguments["filters"]["date"] == "2024"
    assert result[0].arguments["filters"]["source"] == "arxiv"


def test_parse_react_qwen_dewie_read():
    from dewie.model_adapter import _parse_react_tool_calls

    text = 'TOOL_CALL: {"name": "dewie_read", "arguments": {"doc_id": "doc-abc"}}'
    result = _parse_react_tool_calls(text)
    assert len(result) == 1
    assert result[0].name == "dewie_read"
    assert result[0].arguments["doc_id"] == "doc-abc"


def test_parse_react_qwen_dewie_expand():
    from dewie.model_adapter import _parse_react_tool_calls

    text = 'TOOL_CALL: {"name": "dewie_expand", "arguments": {"doc_id": "doc-xyz", "limit": 5}}'
    result = _parse_react_tool_calls(text)
    assert len(result) == 1
    assert result[0].name == "dewie_expand"
    assert result[0].arguments["doc_id"] == "doc-xyz"
    assert result[0].arguments["limit"] == 5


def test_parse_react_qwen_invalid_json():
    from dewie.model_adapter import _parse_react_tool_calls

    text = "TOOL_CALL: {invalid json here}"
    result = _parse_react_tool_calls(text)
    assert result == []


def test_parse_react_qwen_missing_name():
    from dewie.model_adapter import _parse_react_tool_calls

    text = 'TOOL_CALL: {"arguments": {"query": "test"}}'
    result = _parse_react_tool_calls(text)
    assert result == []


def test_parse_react_qwen_no_tool_calls():
    from dewie.model_adapter import _parse_react_tool_calls

    text = "The answer is 42, no tools needed."
    result = _parse_react_tool_calls(text)
    assert result == []


# ── parse_response for qwen-style responses ─────────────────────────────────


def test_parse_response_qwen_tool_calls_stop():
    from dewie.model_adapter import parse_response

    content = 'Let me search for that.\nTOOL_CALL: {"name": "dewie_search", "arguments": {"query": "AI news"}}'
    raw = {
        "choices": [
            {"message": {"content": content, "tool_calls": None}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10},
    }
    resp = parse_response(raw)
    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "dewie_search"
    assert resp.tool_calls[0].arguments["query"] == "AI news"
    assert "TOOL_CALL" not in (resp.content or "")


def test_parse_response_qwen_tool_calls_tool_calls_reason():
    from dewie.model_adapter import parse_response

    content = 'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "test"}}'
    raw = {
        "choices": [
            {"message": {"content": content, "tool_calls": None}, "finish_reason": "tool_calls"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8},
    }
    resp = parse_response(raw)
    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "dewie_search"


def test_parse_response_qwen_no_react_lines():
    from dewie.model_adapter import parse_response

    raw = {
        "choices": [
            {
                "message": {"content": "I don't know the answer.", "tool_calls": None},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6},
    }
    resp = parse_response(raw)
    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []
    assert resp.content == "I don't know the answer."


def test_parse_response_qwen_tool_calls_null_no_react():
    from dewie.model_adapter import parse_response

    raw = {
        "choices": [
            {
                "message": {"content": "Just text response.", "tool_calls": None},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    resp = parse_response(raw)
    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []


def test_parse_response_qwen_multiple_tool_calls_stripped():
    from dewie.model_adapter import parse_response

    content = (
        'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "AI"}}\n'
        'TOOL_CALL: {"name": "dewie_expand", "arguments": {"doc_id": "doc-1"}}'
    )
    raw = {
        "choices": [
            {"message": {"content": content, "tool_calls": None}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15},
    }
    resp = parse_response(raw)
    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0].name == "dewie_search"
    assert resp.tool_calls[1].name == "dewie_expand"
    assert resp.content is None or "TOOL_CALL" not in resp.content


# ── ModelClient complete with qwen ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_qwen_react_injects_instructions():
    """complete() with qwen model injects ReAct instructions instead of sending tools."""
    from dewie.model_adapter import ModelClient

    raw_response = {
        "choices": [
            {
                "message": {
                    "content": 'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "AI"}}',
                    "tool_calls": None,
                },
                "finish_reason": "stop",
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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "dewie_search",
                "description": "Search the corpus",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        with patch("httpx.AsyncClient", return_value=mock_http):
            client = ModelClient(provider="openai", model="qwen2.5:7b")
            async with client:
                await client.complete(
                    messages=[{"role": "user", "content": "Search for AI"}],
                    tools=tools,
                )

    call_kwargs = mock_http.post.call_args
    payload = call_kwargs[1]["json"]
    # ReAct: tools NOT in payload
    assert "tools" not in payload
    # ReAct: messages have instructions injected
    assert len(payload["messages"]) == 2
    assert "TOOL_CALL:" in payload["messages"][0]["content"]
    assert "dewie_search" in payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_complete_qwen_parses_react_tool_calls():
    """complete() with qwen model parses TOOL_CALL lines from response."""
    from dewie.model_adapter import ModelClient

    raw_response = {
        "choices": [
            {
                "message": {
                    "content": "Let me search.\nTOOL_CALL: {\"name\": \"dewie_search\", \"arguments\": {\"query\": \"Dewie\"}}",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
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

    tools = [{"type": "function", "function": {"name": "dewie_search", "parameters": {}}}]

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        with patch("httpx.AsyncClient", return_value=mock_http):
            client = ModelClient(provider="openai", model="qwen2.5:7b")
            async with client:
                result = await client.complete(
                    messages=[{"role": "user", "content": "Search for Dewie"}],
                    tools=tools,
                )

    assert result.has_tool_calls is True
    assert result.tool_calls[0].name == "dewie_search"
    assert result.tool_calls[0].arguments["query"] == "Dewie"
    assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_complete_qwen_no_tool_calls_returns_content():
    """complete() with qwen model returns content when no TOOL_CALL lines present."""
    from dewie.model_adapter import ModelClient

    raw_response = {
        "choices": [
            {
                "message": {
                    "content": "Dewie is a retrieval framework.",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6},
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = raw_response

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    mock_http.aclose = AsyncMock()

    tools = [{"type": "function", "function": {"name": "dewie_search", "parameters": {}}}]

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        with patch("httpx.AsyncClient", return_value=mock_http):
            client = ModelClient(provider="openai", model="qwen2.5:7b")
            async with client:
                result = await client.complete(
                    messages=[{"role": "user", "content": "What is Dewie?"}],
                    tools=tools,
                )

    assert result.has_tool_calls is False
    assert result.has_content is True
    assert result.content == "Dewie is a retrieval framework."
    assert result.finish_reason == "stop"


# ── ModelClient helper methods with qwen ─────────────────────────────────────


def test_tool_response_qwen_react_mode():
    """tool_response() for qwen returns role=user with prefix."""
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="qwen2.5:7b")
    result = client.tool_response("react_0", "search results", "dewie_search")
    assert result["role"] == "user"
    assert "Tool result (dewie_search):" in result["content"]
    assert "search results" in result["content"]


def test_tool_response_qwen_react_mode_no_tool_name():
    """tool_response() for qwen without tool name uses generic prefix."""
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="qwen3:32b")
    result = client.tool_response("react_1", "expanded docs")
    assert result["role"] == "user"
    assert "Tool result:" in result["content"]
    assert "expanded docs" in result["content"]


def test_assistant_message_qwen_no_tool_calls_in_msg():
    """assistant_message() for qwen doesn't include tool_calls field."""
    from dewie.model_adapter import LLMResponse, ModelClient, ToolCall

    client = ModelClient(provider="openai", model="qwen2.5:7b")
    resp = LLMResponse(
        content="Let me search for that.",
        tool_calls=[ToolCall(id="react_0", name="dewie_search", arguments={"query": "AI"})],
        finish_reason="tool_calls",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    msg = client.assistant_message(resp)
    assert msg["role"] == "assistant"
    assert msg["content"] == "Let me search for that."
    assert "tool_calls" not in msg


def test_assistant_message_qwen_with_content():
    """assistant_message() for qwen includes content text."""
    from dewie.model_adapter import LLMResponse, ModelClient

    client = ModelClient(provider="openai", model="qwen3:32b")
    resp = LLMResponse(
        content="Here is what I found.",
        tool_calls=[],
        finish_reason="stop",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    msg = client.assistant_message(resp)
    assert msg["role"] == "assistant"
    assert msg["content"] == "Here is what I found."


# ── Full agent flow with qwen ───────────────────────────────────────────────


def test_full_agent_qwen_react_flow():
    """Full agent flow: user prompt -> qwen returns TOOL_CALL -> process result."""
    from dewie.model_adapter import (
        ModelClient,
        parse_response,
    )

    # Step 1: Qwen model returns text with TOOL_CALL lines
    content = (
        "I'll search the corpus for information about Dewie.\n"
        'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "Dewie retrieval framework"}}'
    )
    raw = {
        "choices": [
            {"message": {"content": content, "tool_calls": None}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 14},
    }

    # Step 2: Parse the raw response
    response = parse_response(raw)
    assert response.finish_reason == "tool_calls"
    assert response.has_tool_calls is True
    # Reasoning text is kept, TOOL_CALL syntax is stripped
    assert response.content == "I'll search the corpus for information about Dewie."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "dewie_search"
    assert response.tool_calls[0].arguments["query"] == "Dewie retrieval framework"

    # Step 3: Build tool result message (ReAct mode for qwen)
    client = ModelClient(provider="openai", model="qwen2.5:7b")
    tool_result = client.tool_response(
        tool_call_id="react_0",
        content='[{"title": "Dewie Docs", "url": "https://dewie.ai/docs", "summary": "Retrieval framework..."}]',
        tool_name="dewie_search",
    )
    assert tool_result["role"] == "user"
    assert "Tool result (dewie_search):" in tool_result["content"]

    # Step 4: Reconstruct assistant message
    assistant_msg = client.assistant_message(response)
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "I'll search the corpus for information about Dewie."
    assert "tool_calls" not in assistant_msg


def test_full_agent_qwen_multi_step_flow():
    """Qwen agent makes multiple tool calls in sequence."""
    from dewie.model_adapter import parse_response

    # First call: qwen returns dewie_search tool call
    content1 = 'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "AI trends 2024"}}'
    raw1 = {
        "choices": [
            {"message": {"content": content1, "tool_calls": None}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10},
    }
    resp1 = parse_response(raw1)
    assert resp1.has_tool_calls is True
    assert resp1.tool_calls[0].name == "dewie_search"

    # Simulate second call after tool result feedback
    content2 = (
        "Now let me expand the most relevant document.\n"
        'TOOL_CALL: {"name": "dewie_expand", "arguments": {"doc_id": "doc-abc-123", "limit": 10}}'
    )
    raw2 = {
        "choices": [
            {
                "message": {"content": content2, "tool_calls": None},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 25, "completion_tokens": 12},
    }
    resp2 = parse_response(raw2)
    assert resp2.has_tool_calls is True
    assert resp2.tool_calls[0].name == "dewie_expand"
    assert resp2.tool_calls[0].arguments["doc_id"] == "doc-abc-123"


def test_full_agent_qwen_final_answer():
    """Qwen agent returns final answer after tool calls."""
    from dewie.model_adapter import parse_response

    # Final response: qwen gives answer without tool calls
    content = (
        "Based on my search, Dewie is a small-model-friendly retrieval framework. "
        "It achieves strong factual recall with Gemma-4-26b (+0.69 delta)."
    )
    raw = {
        "choices": [
            {"message": {"content": content, "tool_calls": None}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 25},
    }
    resp = parse_response(raw)
    assert resp.finish_reason == "stop"
    assert resp.has_tool_calls is False
    assert resp.has_content is True
    assert "Dewie" in resp.content
    assert "Gemma-4-26b" in resp.content


# ── AgentBackend with qwen model ────────────────────────────────────────────


def test_agent_backend_qwen_model():
    """AgentBackend can be configured with qwen model via openclaw provider."""
    from dewie.enrichment.backends.agent import AgentBackend

    backend = AgentBackend(
        name="qwen_enrichment",
        endpoint="http://gateway.example.com:18789",
        model="qwen2.5:7b",
        provider="openclaw",
        auth_token="test-token",
    )
    assert backend._provider == "openclaw"
    assert backend._model == "qwen2.5:7b"
    assert "gateway.example.com" in backend._endpoint


def test_agent_backend_qwen_builds_payload():
    """AgentBackend.complete constructs correct payload for qwen model."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.enrichment.backends.agent import AgentBackend

    backend = AgentBackend(
        name="qwen_test",
        endpoint="http://localhost:18789",
        model="qwen3:32b",
        provider="openclaw",
        auth_token="gateway-token",
    )
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Extracted summary"}}]
    }
    mock_client.post = AsyncMock(return_value=mock_resp)
    backend._http_client = mock_client

    import asyncio

    asyncio.run(backend.complete("Summarize this document"))

    call_kwargs = mock_client.post.call_args
    assert "localhost:18789/v1/chat/completions" in call_kwargs[0][0]
    payload = call_kwargs[1]["json"]
    assert payload["model"] == "qwen3:32b"
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Summarize this document"


def test_agent_backend_qwen_error_handling():
    """AgentBackend raises BackendError on empty qwen response."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.enrichment.backends.agent import AgentBackend
    from dewie.enrichment.base import BackendError

    backend = AgentBackend(
        name="qwen_test",
        endpoint="http://gateway.local:18789",
        model="qwen2.5:7b",
        provider="openclaw",
        auth_token="t",
    )
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
    mock_client.post = AsyncMock(return_value=mock_resp)
    backend._http_client = mock_client

    import asyncio

    with pytest.raises(BackendError, match="empty response"):
        asyncio.run(backend.complete("test"))


# ── ReAct instructions mention correct Dewie tool names ─────────────────────


def test_react_instructions_include_all_dewie_tools():
    """ReAct instructions list all three Dewie tools."""
    from dewie.model_adapter import _REACT_INSTRUCTIONS

    assert "dewie_search(query, limit=5)" in _REACT_INSTRUCTIONS
    assert "dewie_expand(doc_id, limit=10)" in _REACT_INSTRUCTIONS
    assert "dewie_read(doc_id)" in _REACT_INSTRUCTIONS


def test_react_instructions_format_example():
    """ReAct instructions show the exact TOOL_CALL format."""
    from dewie.model_adapter import _REACT_INSTRUCTIONS

    assert '{"name": "<tool_name>", "arguments": {<args>}' in _REACT_INSTRUCTIONS


# ── Qwen model variants edge cases ──────────────────────────────────────────


def test_qwen_variants_coding_model():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("qwen2.5-coder:1.5b") is False
    assert supports_native_tools("qwen2.5-coder:7b") is False
    assert supports_native_tools("Qwen2.5-Coder-32B") is False


def test_qwen_variants_math_model():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("qwen2.5-math:1.5b") is False
    assert supports_native_tools("qwen2.5-math:7b") is False


def test_qwen_variants_vl_model():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("qwen2.5-vl:7b") is False
    assert supports_native_tools("qwen2.5-vl:72b") is False


def test_qwen_variants_base_model():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("qwen:7b") is False
    assert supports_native_tools("qwen:14b") is False
    assert supports_native_tools("qwen:72b") is False


# ── ModelClient local prefix heuristic for qwen ─────────────────────────────


