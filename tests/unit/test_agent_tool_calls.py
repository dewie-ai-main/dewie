"""Tests for agents making tool calls to Dewey.

Covers ToolCall/LLMResponse dataclasses, native tool calling,
ReAct format parsing, ModelClient tool preparation, MCP endpoints,
and AgentBackend payload construction.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── ToolCall / LLMResponse dataclasses ────────────────────────────────────────


def test_toolcall_dataclass_creation():
    from dewie.model_adapter import ToolCall

    tc = ToolCall(id="call_abc", name="dewie_search", arguments={"query": "AI"})
    assert tc.id == "call_abc"
    assert tc.name == "dewie_search"
    assert tc.arguments == {"query": "AI"}


def test_llm_response_basic():
    from dewie.model_adapter import LLMResponse

    resp = LLMResponse(
        content="Hello world",
        tool_calls=[],
        finish_reason="stop",
        input_tokens=10,
        output_tokens=5,
        raw={"id": "resp-1"},
    )
    assert resp.content == "Hello world"
    assert resp.finish_reason == "stop"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5
    assert resp.raw == {"id": "resp-1"}


def test_llm_response_has_tool_calls_true():
    from dewie.model_adapter import LLMResponse, ToolCall

    resp = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="c1", name="fn", arguments={})],
        finish_reason="tool_calls",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    assert resp.has_tool_calls is True
    assert resp.has_content is False


def test_llm_response_has_tool_calls_false():
    from dewie.model_adapter import LLMResponse

    resp = LLMResponse(
        content=None,
        tool_calls=[],
        finish_reason="stop",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    assert resp.has_tool_calls is False


def test_llm_response_has_content_true():
    from dewie.model_adapter import LLMResponse

    resp = LLMResponse(
        content="  some text  ",
        tool_calls=[],
        finish_reason="stop",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    assert resp.has_content is True


def test_llm_response_has_content_empty():
    from dewie.model_adapter import LLMResponse

    resp = LLMResponse(
        content="",
        tool_calls=[],
        finish_reason="stop",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    assert resp.has_content is False

    resp2 = LLMResponse(
        content="   ",
        tool_calls=[],
        finish_reason="stop",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    assert resp2.has_content is False

    resp3 = LLMResponse(
        content=None,
        tool_calls=[],
        finish_reason="stop",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    assert resp3.has_content is False


# ── Native tool call parsing ─────────────────────────────────────────────────


def test_parse_response_native_tool_call():
    from dewie.model_adapter import LLMResponse, parse_response

    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_search1",
                            "type": "function",
                            "function": {
                                "name": "dewie_search",
                                "arguments": json.dumps({"query": "machine learning", "limit": 5}),
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 8},
    }
    resp = parse_response(raw)
    assert isinstance(resp, LLMResponse)
    assert resp.has_tool_calls is True
    assert resp.has_content is False
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "call_search1"
    assert resp.tool_calls[0].name == "dewie_search"
    assert resp.tool_calls[0].arguments == {"query": "machine learning", "limit": 5}
    assert resp.finish_reason == "tool_calls"
    assert resp.input_tokens == 15
    assert resp.output_tokens == 8


def test_parse_response_multiple_native_tool_calls():
    from dewie.model_adapter import parse_response

    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "dewie_search",
                                "arguments": json.dumps({"query": "AI"}),
                            },
                        },
                        {
                            "id": "call_2",
                            "function": {
                                "name": "dewie_expand",
                                "arguments": json.dumps({"doc_id": "doc-abc-123"}),
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 12},
    }
    resp = parse_response(raw)
    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0].name == "dewie_search"
    assert resp.tool_calls[1].name == "dewie_expand"


def test_parse_response_native_tool_call_invalid_args():
    from dewie.model_adapter import parse_response

    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "function": {"name": "dewie_search", "arguments": "not-json"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    resp = parse_response(raw)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "dewie_search"
    assert resp.tool_calls[0].arguments == {}


def test_parse_response_content_with_tool_calls():
    """Some models return both content and tool_calls."""
    from dewie.model_adapter import parse_response

    raw = {
        "choices": [
            {
                "message": {
                    "content": "Searching for you...",
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "function": {
                                "name": "dewie_search",
                                "arguments": json.dumps({"query": "test"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }
    resp = parse_response(raw)
    assert resp.content == "Searching for you..."
    assert len(resp.tool_calls) == 1


# ── ReAct format parsing ─────────────────────────────────────────────────────


def test_parse_react_single_tool_call():
    from dewie.model_adapter import _parse_react_tool_calls

    text = 'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "Dewie search engine"}}'
    result = _parse_react_tool_calls(text)
    assert len(result) == 1
    assert result[0].name == "dewie_search"
    assert result[0].arguments == {"query": "Dewie search engine"}
    assert result[0].id.startswith("react_")


def test_parse_react_multiple_tool_calls():
    from dewie.model_adapter import _parse_react_tool_calls

    text = (
        'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "LLM"}}\n'
        "Let me also expand the most relevant document.\n"
        'TOOL_CALL: {"name": "dewie_expand", "arguments": {"doc_id": "doc-001", "limit": 10}}'
    )
    result = _parse_react_tool_calls(text)
    assert len(result) == 2
    assert result[0].name == "dewie_search"
    assert result[1].name == "dewie_expand"


def test_parse_react_mixed_text_and_tool_calls():
    from dewie.model_adapter import _parse_react_tool_calls

    text = (
        "I'll search for information about the project first.\n"
        'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "project history"}}\n'
        "Then I'll read the document.\n"
        'TOOL_CALL: {"name": "dewie_read", "arguments": {"doc_id": "proj-doc"}}'
    )
    result = _parse_react_tool_calls(text)
    assert len(result) == 2
    assert result[0].name == "dewie_search"
    assert result[1].name == "dewie_read"


def test_parse_react_no_tool_calls():
    from dewie.model_adapter import _parse_react_tool_calls

    text = "The answer is 42. No tools needed."
    assert _parse_react_tool_calls(text) == []


def test_parse_react_invalid_json_skipped():
    from dewie.model_adapter import _parse_react_tool_calls

    text = "TOOL_CALL: {broken json}"
    assert _parse_react_tool_calls(text) == []


def test_parse_react_empty_name_skipped():
    from dewie.model_adapter import _parse_react_tool_calls

    text = 'TOOL_CALL: {"arguments": {}}'
    assert _parse_react_tool_calls(text) == []


def test_parse_react_nested_json_arguments():
    from dewie.model_adapter import _parse_react_tool_calls

    text = 'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "AI", "filters": {"date": "2024"}}}'
    result = _parse_react_tool_calls(text)
    assert len(result) == 1
    assert result[0].arguments["filters"]["date"] == "2024"


# ── Gemma-4 native tool call format ──────────────────────────────────────────


def test_parse_react_gemma_brace_format():
    from dewie.model_adapter import _parse_react_tool_calls

    text = '<|tool_call|>call:dewie_search{"query": "test"}'
    result = _parse_react_tool_calls(text)
    assert len(result) >= 1
    assert result[0].name == "dewie_search"
    assert result[0].id.startswith("gemma_")


def test_parse_react_gemma_paren_format():
    from dewie.model_adapter import _parse_react_tool_calls

    text = '<|tool_call|>call:dewie_search(query="test", limit=5)'
    result = _parse_react_tool_calls(text)
    assert len(result) >= 1
    assert result[0].name == "dewie_search"
    assert result[0].arguments["query"] == "test"


def test_parse_react_gemma_mixed():
    from dewie.model_adapter import _parse_react_tool_calls

    text = (
        "Thinking...\n"
        '<|tool_call|>call:dewie_search{"query": "AI"}\n'
        "Now let me expand the result.\n"
        '<|tool_call|>call:dewie_expand(doc_id="doc-1", limit=10)'
    )
    result = _parse_react_tool_calls(text)
    assert len(result) >= 2


# ── parse_response with ReAct fallback ───────────────────────────────────────


def test_parse_response_stop_with_tool_call_lines():
    from dewie.model_adapter import parse_response

    content = (
        'Let me search for that.\n'
        'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "AI news"}}\n'
        "Then I'll read the top result."
    )
    raw = {
        "choices": [
            {"message": {"content": content, "tool_calls": None}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    resp = parse_response(raw)
    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "dewie_search"
    # TOOL_CALL syntax should be stripped from content
    assert "TOOL_CALL" not in (resp.content or "")


def test_parse_response_tool_calls_finish_reason_with_content_react():
    from dewie.model_adapter import parse_response

    content = 'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "test"}}'
    raw = {
        "choices": [
            {"message": {"content": content, "tool_calls": None}, "finish_reason": "tool_calls"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }
    resp = parse_response(raw)
    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 1


def test_parse_response_tool_calls_finish_reason_no_react_fallback():
    from dewie.model_adapter import parse_response

    raw = {
        "choices": [
            {"message": {"content": "No tools here", "tool_calls": None}, "finish_reason": "tool_calls"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }
    resp = parse_response(raw)
    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []


# ── supports_native_tools ────────────────────────────────────────────────────


def test_supports_native_tools_gpt():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("gpt-4o") is True
    assert supports_native_tools("gpt-4o-mini") is True
    assert supports_native_tools("gpt-4") is True


def test_supports_native_tools_claude():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("claude-3-5-sonnet") is True
    assert supports_native_tools("claude-opus-4") is True


def test_supports_native_tools_gemini():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("gemini-pro") is True
    assert supports_native_tools("gemini-2.0-flash") is True


def test_supports_native_tools_grok():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("grok-2") is True


def test_supports_native_tools_gpt5():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("gpt-5-turbo") is True


def test_supports_native_tools_does_not_support():
    from dewie.model_adapter import supports_native_tools

    assert supports_native_tools("llama3.2:3b") is False
    assert supports_native_tools("qwen2.5:7b") is False
    assert supports_native_tools("glm-4.7-flash") is False
    assert supports_native_tools("gemma-2-2b") is False


# ── ModelClient tool preparation ─────────────────────────────────────────────


def test_model_client_native_tools_flag_gpt():
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="gpt-4o")
    assert client._uses_native_tools is True


def test_model_client_native_tools_flag_claude(monkeypatch):
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
    client = ModelClient(provider="openrouter", model="claude-3-sonnet")
    assert client._uses_native_tools is True


def test_model_client_native_tools_flag_local_llm():
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="llama3.2:3b")
    assert client._uses_native_tools is False


def test_model_client_native_tools_flag_qwen():
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="qwen2.5:7b")
    assert client._uses_native_tools is False


@pytest.mark.asyncio
async def test_complete_native_tools_sends_tools_in_payload():
    """Native tool models include tools in the API payload."""
    from dewie.model_adapter import ModelClient

    raw_response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "function": {
                                "name": "dewie_search",
                                "arguments": json.dumps({"query": "AI"}),
                            },
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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "dewie_search",
                "description": "Search the Dewie corpus",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        with patch("httpx.AsyncClient", return_value=mock_http):
            client = ModelClient(provider="openai", model="gpt-4o")
            async with client:
                await client.complete(
                    messages=[{"role": "user", "content": "Search for AI"}],
                    tools=tools,
                )

    call_kwargs = mock_http.post.call_args
    payload = call_kwargs[1]["json"]
    assert "tools" in payload
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_complete_react_tools_injects_instructions():
    """ReAct models do NOT send tools; instead ReAct instructions are injected."""
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
                "description": "Search the Dewie corpus",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        with patch("httpx.AsyncClient", return_value=mock_http):
            client = ModelClient(provider="openai", model="llama3.2:3b")
            async with client:
                await client.complete(
                    messages=[{"role": "user", "content": "Search for AI"}],
                    tools=tools,
                )

    call_kwargs = mock_http.post.call_args
    payload = call_kwargs[1]["json"]
    # Tools should NOT be in payload for ReAct
    assert "tools" not in payload
    # But ReAct instructions should be injected into messages
    assert len(payload["messages"]) == 2
    assert "TOOL_CALL" in payload["messages"][0]["content"]
    assert "dewie_search" in payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_complete_react_mode_parses_tool_calls_from_response():
    """ReAct mode parses TOOL_CALL lines from response content."""
    from dewie.model_adapter import ModelClient

    raw_response = {
        "choices": [
            {
                "message": {
                    "content": "Let me search for that.\nTOOL_CALL: {\"name\": \"dewie_search\", \"arguments\": {\"query\": \"Dewie\"}}",
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
            "function": {"name": "dewie_search", "parameters": {}},
        }
    ]

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        with patch("httpx.AsyncClient", return_value=mock_http):
            client = ModelClient(provider="openai", model="llama3.2:3b")
            async with client:
                result = await client.complete(
                    messages=[{"role": "user", "content": "Search for Dewie"}],
                    tools=tools,
                )

    assert result.has_tool_calls is True
    assert result.tool_calls[0].name == "dewie_search"
    assert result.tool_calls[0].arguments == {"query": "Dewie"}
    assert result.finish_reason == "tool_calls"


# ── ModelClient helper methods ───────────────────────────────────────────────


def test_tool_response_native_mode():
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="gpt-4o")
    result = client.tool_response("call_123", "search results here", "dewie_search")
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_123"
    assert result["content"] == "search results here"


def test_tool_response_react_mode():
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="llama3.2:3b")
    result = client.tool_response("react_0", "search results here", "dewie_search")
    assert result["role"] == "user"
    assert "Tool result (dewie_search):" in result["content"]
    assert "search results here" in result["content"]


def test_tool_response_react_mode_no_tool_name():
    from dewie.model_adapter import ModelClient

    client = ModelClient(provider="openai", model="qwen2.5:7b")
    result = client.tool_response("react_0", "result")
    assert result["role"] == "user"
    assert "Tool result:" in result["content"]
    assert "result" in result["content"]


def test_assistant_message_without_tools():
    from dewie.model_adapter import LLMResponse, ModelClient

    client = ModelClient(provider="openai", model="gpt-4o")
    resp = LLMResponse(
        content="The answer is 42",
        tool_calls=[],
        finish_reason="stop",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    msg = client.assistant_message(resp)
    assert msg["role"] == "assistant"
    assert msg["content"] == "The answer is 42"
    assert "tool_calls" not in msg


def test_assistant_message_with_native_tool_calls():
    from dewie.model_adapter import LLMResponse, ModelClient, ToolCall

    client = ModelClient(provider="openai", model="gpt-4o")
    resp = LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id="call_1", name="dewie_search", arguments={"query": "AI news"})
        ],
        finish_reason="tool_calls",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    msg = client.assistant_message(resp)
    assert msg["role"] == "assistant"
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["id"] == "call_1"
    assert msg["tool_calls"][0]["type"] == "function"
    assert msg["tool_calls"][0]["function"]["name"] == "dewie_search"
    assert msg["tool_calls"][0]["function"]["arguments"] == json.dumps({"query": "AI news"})


def test_assistant_message_with_react_no_tool_calls_in_msg():
    """ReAct models don't include tool_calls in assistant message dict."""
    from dewie.model_adapter import LLMResponse, ModelClient, ToolCall

    client = ModelClient(provider="openai", model="llama3.2:3b")
    resp = LLMResponse(
        content="Let me search...",
        tool_calls=[ToolCall(id="react_0", name="dewie_search", arguments={"query": "AI"})],
        finish_reason="tool_calls",
        input_tokens=0,
        output_tokens=0,
        raw={},
    )
    msg = client.assistant_message(resp)
    assert msg["role"] == "assistant"
    assert msg["content"] == "Let me search..."
    # ReAct models don't add tool_calls field to assistant message
    assert "tool_calls" not in msg


# ── ReAct instructions injection ─────────────────────────────────────────────


def test_inject_react_appends_to_system():
    from dewie.model_adapter import _inject_react_instructions

    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    result = _inject_react_instructions(msgs)
    assert result[0]["role"] == "system"
    assert "You are a helpful assistant." in result[0]["content"]
    assert "TOOL_CALL:" in result[0]["content"]
    assert "dewie_search" in result[0]["content"]


def test_inject_react_adds_system_when_missing():
    from dewie.model_adapter import _inject_react_instructions

    msgs = [{"role": "user", "content": "Hello"}]
    result = _inject_react_instructions(msgs)
    assert result[0]["role"] == "system"
    assert "TOOL_CALL:" in result[0]["content"]
    assert result[1]["role"] == "user"


def test_inject_react_does_not_mutate_original():
    from dewie.model_adapter import _inject_react_instructions

    msgs = [{"role": "system", "content": "Original content"}]
    _inject_react_instructions(msgs)
    assert msgs[0]["content"] == "Original content"


def test_react_instructions_mention_dewie_tools():
    from dewie.model_adapter import _REACT_INSTRUCTIONS

    assert "dewie_search" in _REACT_INSTRUCTIONS
    assert "dewie_expand" in _REACT_INSTRUCTIONS
    assert "dewie_read" in _REACT_INSTRUCTIONS


# ── MCP tool manifest ────────────────────────────────────────────────────────


def test_mcp_tool_manifest_structure():
    from dewie.api.routes.mcp import _TOOL_MANIFEST

    assert "schema_version" in _TOOL_MANIFEST
    assert _TOOL_MANIFEST["schema_version"] == "1.0"
    assert "tools" in _TOOL_MANIFEST

    tool_names = {t["name"] for t in _TOOL_MANIFEST["tools"]}
    # Core tools that must always be advertised; the manifest may grow.
    assert {
        "search_corpus",
        "ingest_url",
        "expand",
        "read",
        "intersect",
        "bridge",
        "browse",
        "research",
        "web_search",
    } <= tool_names


def test_mcp_tool_manifest_search_corpus_schema():
    from dewie.api.routes.mcp import _TOOL_MANIFEST

    search_tool = next(t for t in _TOOL_MANIFEST["tools"] if t["name"] == "search_corpus")
    assert search_tool["description"] is not None
    props = search_tool["input_schema"]["properties"]
    assert "query" in props
    assert props["query"]["type"] == "string"
    assert "corpus_id" in props
    assert "limit" in props
    assert props["limit"]["default"] == 10
    assert "query" in _TOOL_MANIFEST["tools"][0]["input_schema"].get("required", [])


def test_mcp_tool_manifest_ingest_url_schema():
    from dewie.api.routes.mcp import _TOOL_MANIFEST

    ingest_tool = next(t for t in _TOOL_MANIFEST["tools"] if t["name"] == "ingest_url")
    props = ingest_tool["input_schema"]["properties"]
    assert "url" in props
    assert props["url"]["type"] == "string"
    assert "url" in _TOOL_MANIFEST["tools"][1]["input_schema"].get("required", [])


@pytest.mark.asyncio
async def test_mcp_manifest_endpoint():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from dewie.api.middleware import limiter
    from dewie.api.routes.mcp import router

    app = FastAPI()
    app.state.limiter = limiter
    app.state.postgres = AsyncMock()
    app.state.processor = None
    app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/mcp")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    tool_names = {t["name"] for t in data["tools"]}
    assert {"search_corpus", "web_search"} <= tool_names


@pytest.mark.asyncio
async def test_mcp_call_search_corpus():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from dewie.api.middleware import limiter
    from dewie.api.routes.mcp import router

    app = FastAPI()
    app.state.limiter = limiter

    async def _no_auth(request, call_next):
        request.state.user_id = None
        request.state.workspace_ids = []
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_no_auth)
    app.include_router(router)
    app.state.postgres = AsyncMock()
    app.state.processor = None

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/mcp", json={"tool": "search_corpus", "input": {"query": "test query"}}
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_mcp_call_unknown_tool():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from dewie.api.middleware import limiter
    from dewie.api.routes.mcp import router

    app = FastAPI()
    app.state.limiter = limiter

    async def _no_auth(request, call_next):
        request.state.user_id = None
        request.state.workspace_ids = []
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_no_auth)
    app.include_router(router)
    app.state.postgres = AsyncMock()
    app.state.processor = None

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/mcp", json={"tool": "nonexistent_tool", "input": {}}
        )
    assert resp.status_code == 401


# ── MCPToolCall / MCPToolResult models ───────────────────────────────────────


def test_mcp_tool_call_model():
    from dewie.api.routes.mcp import MCPToolCall

    call = MCPToolCall(tool="search_corpus", input={"query": "AI", "limit": 5})
    assert call.tool == "search_corpus"
    assert call.input["query"] == "AI"
    assert call.input["limit"] == 5


def test_mcp_tool_call_model_required_query():
    from dewie.api.routes.mcp import MCPToolCall

    # MCPToolCall doesn't enforce "query" at the pydantic model level,
    # that's handled in the route handler
    call = MCPToolCall(tool="search_corpus", input={})
    assert call.tool == "search_corpus"


def test_mcp_tool_result_model():
    from dewie.api.routes.mcp import MCPToolResult

    result = MCPToolResult(
        tool="search_corpus",
        content={"results": [], "count": 0},
    )
    assert result.type == "tool_result"
    assert result.tool == "search_corpus"
    assert result.content == {"results": [], "count": 0}


# ── AgentBackend payload construction ────────────────────────────────────────


def test_agent_backend_builds_payload_for_prompt():
    """AgentBackend.complete constructs correct payload structure."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.enrichment.backends.agent import AgentBackend

    backend = AgentBackend(
        name="test",
        endpoint="http://localhost:18789",
        model="gpt-4o",
        provider="openclaw",
        auth_token="test-token",
    )
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Extracted title"}}]
    }
    mock_client.post = AsyncMock(return_value=mock_resp)
    backend._http_client = mock_client

    import asyncio

    asyncio.run(backend.complete("Extract the title"))

    call_kwargs = mock_client.post.call_args
    assert "localhost:18789/v1/chat/completions" in call_kwargs[0][0]
    payload = call_kwargs[1]["json"]
    assert payload["model"] == "gpt-4o"
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Extract the title"
    assert payload["max_tokens"] == 4000


def test_agent_backend_headers_include_content_type():
    """AgentBackend includes Content-Type header."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.enrichment.backends.agent import AgentBackend

    backend = AgentBackend(
        name="test",
        endpoint="http://localhost:18789",
        model="gpt-4o",
        auth_token="my-token",
    )
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_client.post = AsyncMock(return_value=mock_resp)
    backend._http_client = mock_client

    import asyncio

    asyncio.run(backend.complete("test"))

    headers = mock_client.post.call_args[1]["headers"]
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer my-token"


def test_agent_backend_includes_extra_headers():
    """AgentBackend merges extra_headers into request."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.enrichment.backends.agent import AgentBackend

    backend = AgentBackend(
        name="test",
        endpoint="http://localhost:18789",
        model="gpt-4o",
        auth_token="t",
        extra_headers={"X-Custom": "value", "Authorization": "Bearer extra"},
    )
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_client.post = AsyncMock(return_value=mock_resp)
    backend._http_client = mock_client

    import asyncio

    asyncio.run(backend.complete("test"))

    headers = mock_client.post.call_args[1]["headers"]
    assert headers["X-Custom"] == "value"


def test_agent_backend_error_on_unexpected_response_shape():
    """AgentBackend raises BackendError on malformed response."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.enrichment.backends.agent import AgentBackend
    from dewie.enrichment.base import BackendError

    backend = AgentBackend(
        name="test",
        endpoint="http://localhost:18789",
        model="gpt-4o",
        auth_token="t",
    )
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"error": "something went wrong"}
    mock_resp.text = '{"error": "bad response"}'
    mock_client.post = AsyncMock(return_value=mock_resp)
    backend._http_client = mock_client

    import asyncio

    with pytest.raises(BackendError, match="unexpected response shape"):
        asyncio.run(backend.complete("test"))


def test_agent_backend_error_on_empty_content():
    """AgentBackend raises BackendError when content is empty."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.enrichment.backends.agent import AgentBackend
    from dewie.enrichment.base import BackendError

    backend = AgentBackend(
        name="test",
        endpoint="http://localhost:18789",
        model="gpt-4o",
        auth_token="t",
    )
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
    mock_resp.text = '{"choices": [{"message": {"content": ""}}]}'
    mock_client.post = AsyncMock(return_value=mock_resp)
    backend._http_client = mock_client

    import asyncio

    with pytest.raises(BackendError, match="empty response"):
        asyncio.run(backend.complete("test"))


def test_agent_backend_error_on_whitespace_only_content():
    """AgentBackend raises BackendError when content is whitespace only."""
    from unittest.mock import AsyncMock, MagicMock

    from dewie.enrichment.backends.agent import AgentBackend
    from dewie.enrichment.base import BackendError

    backend = AgentBackend(
        name="test",
        endpoint="http://localhost:18789",
        model="gpt-4o",
        auth_token="t",
    )
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "   \n  "}}]}
    mock_resp.text = "   "
    mock_client.post = AsyncMock(return_value=mock_resp)
    backend._http_client = mock_client

    import asyncio

    with pytest.raises(BackendError, match="empty response"):
        asyncio.run(backend.complete("test"))


# ── End-to-end: agent flow from prompt to tool call result ──────────────────


def test_full_agent_tool_call_flow():
    """Simulate: user prompt -> model returns tool call -> agent processes result."""
    from dewie.model_adapter import (
        ModelClient,
        parse_response,
    )

    # Step 1: Model returns a tool call (native format)
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "function": {
                                "name": "dewie_search",
                                "arguments": json.dumps({"query": "AI research trends 2024"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 25, "completion_tokens": 8},
    }

    # Step 2: Parse the raw response
    response = parse_response(raw)
    assert response.has_tool_calls is True
    assert response.has_content is False
    assert response.tool_calls[0].name == "dewie_search"
    assert response.tool_calls[0].arguments["query"] == "AI research trends 2024"

    # Step 3: Build the tool result message (native mode)
    client = ModelClient(provider="openai", model="gpt-4o")
    tool_result = client.tool_response(
        tool_call_id="call_abc",
        content='[{"title": "AI Trends", "url": "https://example.com", "summary": "Recent advances..."}]',
        tool_name="dewie_search",
    )
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == "call_abc"
    assert "AI Trends" in tool_result["content"]

    # Step 4: Reconstruct assistant message for history
    assistant_msg = client.assistant_message(response)
    assert assistant_msg["role"] == "assistant"
    assert "tool_calls" in assistant_msg


def test_full_agent_react_flow():
    """Simulate: ReAct model -> parse TOOL_CALL lines -> process tool result."""
    from dewie.model_adapter import (
        ModelClient,
        parse_response,
    )

    # Step 1: Model returns text with TOOL_CALL lines (ReAct format)
    content = (
        "I'll search for information about AI research trends.\n"
        'TOOL_CALL: {"name": "dewie_search", "arguments": {"query": "AI research trends 2024"}}'
    )
    raw = {
        "choices": [
            {"message": {"content": content, "tool_calls": None}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 12},
    }

    # Step 2: Parse the raw response (should detect TOOL_CALL lines)
    response = parse_response(raw)
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "dewie_search"

    # Step 3: Build tool result message (ReAct mode)
    client = ModelClient(provider="openai", model="llama3.2:3b")
    tool_result = client.tool_response(
        tool_call_id="react_0",
        content='[{"title": "AI Trends", "summary": "Recent advances..."}]',
        tool_name="dewie_search",
    )
    assert tool_result["role"] == "user"
    assert "Tool result (dewie_search):" in tool_result["content"]

    # Step 4: Reconstruct assistant message
    assistant_msg = client.assistant_message(response)
    assert assistant_msg["role"] == "assistant"
    # ReAct: content may be stripped, no tool_calls field in msg
    assert "tool_calls" not in assistant_msg


def test_agent_backends_search_corpus_mcp():
    """AgentBackend can be configured to call Dewey MCP tools via agent gateway."""
    from dewie.enrichment.backends.agent import AgentBackend

    backend = AgentBackend(
        name="dewey_search_agent",
        endpoint="http://example-dewie-instance.local:18789",
        model="gpt-4o",
        provider="openclaw",
        auth_token_env="GATEWAY_TOKEN",
    )
    assert backend.name == "dewey_search_agent"
    assert backend._provider == "openclaw"
    assert "example-dewie-instance.local" in backend._endpoint
    assert backend._endpoint.endswith("/v1/chat/completions")


# ── MCP query logging ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_log_mcp_call_helper_creates_entry():
    from unittest.mock import patch

    from dewie.api.mcp_dispatch import _log_mcp_call

    with patch("dewie.storage.query_logger.log_query") as mock_log:
        with patch("asyncio.create_task") as mock_task:
            _log_mcp_call(
                "search_corpus",
                {"query": "test query", "limit": 5},
                "test-user-123",
                None,
                42.5,
                result_summary={"doc_count": 3},
            )

    mock_task.assert_called_once()
    coro = mock_task.call_args[0][0]
    await coro
    mock_log.assert_called_once()
    entry = mock_log.call_args[0][0]
    assert entry.source == "mcp"
    assert entry.question.startswith("mcp:search_corpus")
    assert entry.user_id == "test-user-123"
    assert entry.elapsed_ms == 42  # banker's rounding


@pytest.mark.asyncio
async def test_mcp_log_mcp_call_with_model_header():
    from unittest.mock import patch

    from dewie.api.mcp_dispatch import _log_mcp_call

    with patch("dewie.storage.query_logger.log_query") as mock_log:
        with patch("asyncio.create_task") as mock_task:
            _log_mcp_call(
                "search_corpus", {"query": "test"}, None, "gpt-4o", 10.0, result_summary={"doc_count": 0}
            )

    coro = mock_task.call_args[0][0]
    await coro
    entry = mock_log.call_args[0][0]
    assert entry.model == "gpt-4o"


@pytest.mark.asyncio
async def test_mcp_log_mcp_call_logs_elapsed_time():
    from unittest.mock import patch

    from dewie.api.mcp_dispatch import _log_mcp_call

    with patch("dewie.storage.query_logger.log_query") as mock_log:
        with patch("asyncio.create_task") as mock_task:
            _log_mcp_call(
                "search_corpus", {"query": "test"}, None, None, 150.7, result_summary={"doc_count": 1}
            )

    coro = mock_task.call_args[0][0]
    await coro
    entry = mock_log.call_args[0][0]
    assert entry.elapsed_ms == 151  # rounded


@pytest.mark.asyncio
async def test_mcp_log_mcp_call_failure_logs_error_info():
    from unittest.mock import patch

    from dewie.api.mcp_dispatch import _log_mcp_call

    with patch("dewie.storage.query_logger.log_query") as mock_log:
        with patch("asyncio.create_task") as mock_task:
            _log_mcp_call(
                "search_corpus",
                {"query": "test"},
                None,
                None,
                50.0,
                result_summary={"success": False, "error": "search_failed"},
            )

    coro = mock_task.call_args[0][0]
    await coro
    entry = mock_log.call_args[0][0]
    assert entry.docs_returned == [{"success": False, "error": "search_failed"}]


@pytest.mark.asyncio
async def test_mcp_call_search_corpus_logs_to_query_log():
    from unittest.mock import AsyncMock, patch

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from dewie.api.middleware import limiter
    from dewie.api.routes.mcp import router

    app = FastAPI()
    app.state.limiter = limiter
    app.state.postgres = AsyncMock()

    # Setup mock search results
    mock_doc = AsyncMock()
    mock_doc.title = "Test Document"
    mock_doc.url = "https://example.com"
    mock_doc.summary = "Test summary"
    mock_doc.document_type = AsyncMock()
    mock_doc.document_type.value = "article"
    mock_doc.source = "web"

    app.state.postgres.search = AsyncMock(return_value=[(mock_doc, 0.95)])

    async def _set_user(request, call_next):
        request.state.user_id = "00000000-0000-0000-0000-000000000001"
        request.state.workspace_ids = []
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_set_user)
    app.include_router(router)

    with patch("dewie.storage.query_logger.log_query") as mock_log:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/mcp", json={"tool": "search_corpus", "input": {"query": "test query"}}
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "content" in data
    assert "results" in data["content"]

    # Verify query_log was called with source=mcp
    mock_log.assert_called_once()
    entry = mock_log.call_args[0][0]
    assert entry.source == "mcp"
    assert entry.elapsed_ms >= 0
    assert entry.user_id == "00000000-0000-0000-0000-000000000001"


@pytest.mark.asyncio
async def test_mcp_call_failure_logs_error():
    from unittest.mock import AsyncMock, patch

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from dewie.api.middleware import limiter
    from dewie.api.routes.mcp import router

    app = FastAPI()
    app.state.limiter = limiter
    app.state.postgres = AsyncMock()

    # Setup mock search to raise an error
    app.state.postgres.search = AsyncMock(side_effect=Exception("database error"))

    async def _set_user(request, call_next):
        request.state.user_id = "00000000-0000-0000-0000-000000000001"
        request.state.workspace_ids = []
        request.state.key_id = None
        return await call_next(request)

    app.middleware("http")(_set_user)
    app.include_router(router)

    with patch("dewie.storage.query_logger.log_query") as mock_log:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/mcp", json={"tool": "search_corpus", "input": {"query": "test query"}}
            )

    assert resp.status_code == 500

    # Verify query_log was called with error info
    mock_log.assert_called_once()
    entry = mock_log.call_args[0][0]
    assert entry.source == "mcp"
    assert entry.elapsed_ms >= 0
    # docs_returned should contain the error summary
    assert len(entry.docs_returned) >= 1
    assert "success" in entry.docs_returned[0]
    assert entry.docs_returned[0]["success"] is False
