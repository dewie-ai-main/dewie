# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
model_adapter.py — Normalize LLM responses across providers.

Different providers return subtly different shapes from the same OpenAI-
compatible API spec. This module wraps those differences so callers only
deal with clean, predictable objects.

Tool-call strategies:
  - GPT models (gpt-*): native function calling — tools sent in payload,
    tool_calls parsed from response.
  - Everything else (Claude, Gemini, …): ReAct mode — tools NOT sent to
    the API; instead, ReAct instructions are injected into the system
    message and TOOL_CALL: {...} lines are parsed from the text response.

Provider resolution: ``provider`` is a server label looked up in the shared
server registry (``providers/servers.py`` — built-in labels are ``openai``,
``anthropic``, ``openrouter``; custom servers are registered under
``servers:`` in dewie.yml or via the admin UI). Endpoint, API key, and wire
format all come from that one place — set LLM_PROVIDER (or pass ``provider=``)
to a registered label, and LLM_MODEL to a model name.

Note: the Anthropic wire format (native Messages API) is NOT translated here —
this client always speaks OpenAI-style ``/chat/completions``, so an
``api_format=anthropic`` server will not work correctly through ModelClient
today. This is a pre-existing limitation, not introduced by the server
registry change.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response from any LLM call."""

    content: str | None  # text answer, if any
    tool_calls: list[ToolCall]  # structured tool calls, if any
    finish_reason: str  # "stop", "tool_calls", "length", etc.
    input_tokens: int
    output_tokens: int
    raw: dict  # original response for debugging

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def has_content(self) -> bool:
        return bool(self.content and self.content.strip())


_REACT_INSTRUCTIONS = (
    "You have access to tools. Call them using this exact format on its own line:\n"
    'TOOL_CALL: {"name": "<tool_name>", "arguments": {<args>}}\n'
    "Available tools: dewie_search(query, limit=5), "
    "dewie_expand(doc_id, limit=10), dewie_read(doc_id)"
)


def supports_native_tools(model: str) -> bool:
    """Returns True for models that support native OpenAI-compatible function calling."""
    m = model.lower()
    # GPT models
    if m.startswith("gpt-"):
        return True
    # Claude (supports tool_use natively in OpenAI-compatible format)
    if m.startswith("claude-"):
        return True
    # Gemini
    if m.startswith("gemini-"):
        return True
    # Grok
    if m.startswith("grok-"):
        return True
    # GPT-5 family
    if m.startswith("gpt-5"):
        return True
    return False


def _parse_react_tool_calls(text: str) -> list[ToolCall]:
    """Parse tool calls from model text output.

    Handles two formats:
    1. ReAct: TOOL_CALL: {"name": "fn", "arguments": {...}}
    2. Gemma-4 native: <|tool_call>call:fn{key: "val"} or call:fn(key="val")
    """
    tool_calls: list[ToolCall] = []

    # Format 1: TOOL_CALL: {...}
    for match in re.finditer(r"TOOL_CALL:\s*(\{.*\})", text):
        try:
            parsed = json.loads(match.group(1))
            name = parsed.get("name", "")
            args = parsed.get("arguments", {})
            if name:
                tool_calls.append(
                    ToolCall(
                        id=f"react_{len(tool_calls)}",
                        name=name,
                        arguments=args if isinstance(args, dict) else {},
                    )
                )
        except (json.JSONDecodeError, KeyError):
            pass

    # Format 2: Gemma-4 native <|tool_call>call:name{...} or call:name(...)
    for match in re.finditer(r"<\|tool_call\|?>call:(\w+)([\{\(].*?)(?:[\}\)]|$)", text, re.DOTALL):
        try:
            name = match.group(1)
            body = match.group(2).strip()
            args: dict = {}
            # Try JSON-like {key: "val", ...}
            if body.startswith("{"):
                # Normalize JS-style keys to JSON
                normalized = re.sub(r"(\w+)\s*:", r'"\1":', body.rstrip("}") + "}")
                try:
                    args = json.loads(normalized)
                except json.JSONDecodeError:
                    # Fallback: extract key="value" pairs
                    args = {
                        k: v for k, v in re.findall(r'(\w+):\s*["\']?([^,"\'}\)]+)["\']?', body)
                    }
            elif body.startswith("("):
                # kwargs style: (key="value", ...)
                args = {k: v for k, v in re.findall(r'(\w+)=["\']?([^,"\')\s]+)["\']?', body)}
            if name:
                tool_calls.append(
                    ToolCall(
                        id=f"gemma_{len(tool_calls)}",
                        name=name,
                        arguments=args,
                    )
                )
        except Exception:
            pass

    return tool_calls


def _inject_react_instructions(messages: list[dict]) -> list[dict]:
    """Append ReAct tool-calling instructions to the system message."""
    result = list(messages)
    for i, msg in enumerate(result):
        if msg.get("role") == "system":
            result[i] = {**msg, "content": msg["content"] + "\n\n" + _REACT_INSTRUCTIONS}
            return result
    result.insert(0, {"role": "system", "content": _REACT_INSTRUCTIONS})
    return result


def parse_response(raw: dict) -> LLMResponse:
    """
    Parse a raw OpenAI-compatible chat completion response into an LLMResponse.

    Quirk: some providers return finish_reason="tool_calls" with tool_calls=null.
    In that case we try to parse ReAct TOOL_CALL lines from content; if none
    found we fall back to finish_reason="stop".
    """
    choice = raw["choices"][0]
    message = choice["message"]
    finish_reason = choice.get("finish_reason", "stop")
    usage = raw.get("usage", {})

    content = message.get("content") or None
    raw_tool_calls = message.get("tool_calls") or []

    tool_calls: list[ToolCall] = []
    for tc in raw_tool_calls:
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}
        tool_calls.append(
            ToolCall(
                id=tc.get("id", ""),
                name=tc["function"]["name"],
                arguments=args,
            )
        )

    # finish_reason="tool_calls" but tool_calls=null: try ReAct parse first.
    if finish_reason == "tool_calls" and not tool_calls and content:
        react_tcs = _parse_react_tool_calls(content)
        if react_tcs:
            tool_calls = react_tcs
        else:
            finish_reason = "stop"

    # Models like glm-4.7-flash emit TOOL_CALL: {...} lines with finish_reason="stop".
    # Detect and parse them so they're handled as proper tool calls, not leaked as answers.
    if (
        finish_reason == "stop"
        and not tool_calls
        and content
        and ("TOOL_CALL:" in content or "<|tool_call" in content)
    ):
        react_tcs = _parse_react_tool_calls(content)
        if react_tcs:
            tool_calls = react_tcs
            finish_reason = "tool_calls"
            # Strip tool-call syntax from content so it doesn't bleed into the answer
            content = re.sub(r"TOOL_CALL:\s*\{.*\}", "", content, flags=re.DOTALL).strip()
            content = re.sub(
                r"<\|tool_call\|?>call:\w+[\(\{].*?[\)\}]", "", content, flags=re.DOTALL
            ).strip()
            content = content or None

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        raw=raw,
    )


class ModelClient:
    """
    Thin async wrapper around an OpenAI-compatible endpoint.

    Transparently handles native tool calling (GPT) vs. ReAct text-based
    tool calling (Claude, Gemini, etc.). Callers always receive an
    LLMResponse with tool_calls populated regardless of which path was used.

    Configuration via env vars:
        LLM_PROVIDER  — registered server label (see providers/servers.py)
        LLM_MODEL     — model id (uses first available registered model if unset)
    """

    def __init__(self, provider: str | None = None, model: str | None = None):
        if model is None:
            model = os.environ.get("LLM_MODEL", "")

        # Resolve provider label: explicit arg → env var → model registry lookup
        # — NO SILENT FALLBACK.
        if provider is None:
            provider = os.environ.get("LLM_PROVIDER")

        if provider is None:
            try:
                from dewie.model_registry import registry

                info = registry.get(model)
                provider = info.provider if info else None
            except Exception:
                provider = None

        if provider is None:
            raise RuntimeError(
                "LLM provider not configured. Set LLM_PROVIDER to a registered server "
                "label (see providers/servers.py / GET /admin/servers). "
                "No silent fallback to OpenAI."
            )

        from dewie.providers.servers import UnknownServerError, get_server

        try:
            server = get_server(provider)
        except UnknownServerError as exc:
            raise RuntimeError(str(exc)) from exc

        self.provider = provider
        self._server = server
        self.model = model
        self.base_url = f"{server.endpoint}/v1"

        self._uses_native_tools = supports_native_tools(model)
        self._client = None  # lazy init

    async def __aenter__(self):
        import httpx

        self._client = httpx.AsyncClient(timeout=120)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def _auth_headers(self) -> dict[str, str]:
        """Return auth headers for the current provider (read fresh on every call)."""
        from dewie.providers.servers import resolve_api_key

        key = resolve_api_key(self._server)
        headers: dict[str, str] = {}
        if key:
            if self._server.api_format == "anthropic":
                headers["x-api-key"] = key
            else:
                headers["Authorization"] = f"Bearer {key}"
        headers.update(self._server.extra_headers)
        return headers

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request and return a normalized response.

        If tools are provided:
          - GPT models: tools are sent natively in the payload.
          - All other models: tools are NOT sent; ReAct instructions are
            injected into the system message instead, and TOOL_CALL lines
            are parsed from the response content.
        """
        send_messages = messages
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": send_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Per-server request params (e.g. thinking suppression) configured on
        # ServerConfig.extra_body — same contract as the enrichment providers.
        if getattr(self._server, "extra_body", None):
            payload.update(self._server.extra_body)

        if response_format:
            payload["response_format"] = response_format

        if tools:
            if self._uses_native_tools:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice
            else:
                payload["messages"] = _inject_react_instructions(messages)

        url = f"{self.base_url}/chat/completions"

        resp = await self._client.post(url, json=payload, headers=self._auth_headers())
        resp.raise_for_status()
        response = parse_response(resp.json())

        # For ReAct mode, also catch finish_reason="stop" responses that
        # contain TOOL_CALL lines (e.g. Gemini via proxy).
        if tools and not self._uses_native_tools and not response.tool_calls:
            react_tcs = _parse_react_tool_calls(response.content or "")
            if react_tcs:
                response.tool_calls = react_tcs
                response.finish_reason = "tool_calls"

        return response

    def tool_response(self, tool_call_id: str, content: str, tool_name: str = "") -> dict:
        """Build a tool result message appropriate for the current mode."""
        if self._uses_native_tools:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        else:
            prefix = f"Tool result ({tool_name}): " if tool_name else "Tool result: "
            return {"role": "user", "content": prefix + content}

    def assistant_message(self, response: LLMResponse) -> dict:
        """Reconstruct the assistant message from a parsed response (for history)."""
        msg: dict[str, Any] = {"role": "assistant", "content": response.content}
        if self._uses_native_tools and response.has_tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
        return msg
