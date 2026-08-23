"""Tests for dewie.enrichment.backends — passthrough, http."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── PassthroughBackend ────────────────────────────────────────────────────────


def test_passthrough_name():
    from dewie.enrichment.backends.passthrough import PassthroughBackend

    b = PassthroughBackend(name="test_stub")
    assert b.name == "test_stub"


def test_passthrough_default_name():
    from dewie.enrichment.backends.passthrough import PassthroughBackend

    b = PassthroughBackend()
    assert b.name == "passthrough"


@pytest.mark.asyncio
async def test_passthrough_complete_returns_custom_json():
    from dewie.enrichment.backends.passthrough import PassthroughBackend

    custom = '{"summary": "test", "keywords": ["a"]}'
    b = PassthroughBackend(response_json=custom)
    result = await b.complete("ignored prompt")
    assert result == custom


@pytest.mark.asyncio
async def test_passthrough_complete_returns_default_on_empty():
    from dewie.enrichment.backends.passthrough import PassthroughBackend

    b = PassthroughBackend(response_json="")
    result = await b.complete("anything")
    assert "document_type" in result


@pytest.mark.asyncio
async def test_passthrough_ignores_prompt():
    from dewie.enrichment.backends.passthrough import PassthroughBackend

    b = PassthroughBackend(response_json='{"x": 1}')
    r1 = await b.complete("prompt A")
    r2 = await b.complete("prompt B")
    assert r1 == r2


# ── HttpBackend ───────────────────────────────────────────────────────────────


def test_http_backend_name():
    from dewie.enrichment.backends.http import HttpBackend

    b = HttpBackend(name="my-backend", base_url="http://localhost:11434", model="llama3.2:3b")
    assert b.name == "my-backend"


def _make_http_backend(**kwargs):
    from dewie.enrichment.backends.http import HttpBackend

    defaults = dict(name="test", base_url="http://localhost:11434", model="llama3.2:3b")
    defaults.update(kwargs)
    return HttpBackend(**defaults)


def _mock_response(status=200, json_data=None, headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(return_value=json_data or {})
    resp.headers = headers or {}
    resp.text = str(json_data or "")
    return resp


def _mock_http_cm(response):
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_http_backend_ollama_mode_success():
    from dewie.enrichment.backends.http import HttpBackend

    b = HttpBackend(name="b", base_url="http://localhost:11434", model="llama3.2", mode="ollama")

    resp = _mock_response(200, {"response": '{"summary": "test"}'})
    cm = _mock_http_cm(resp)

    with patch("httpx.AsyncClient", return_value=cm):
        result = await b.complete("test prompt")
    assert result == '{"summary": "test"}'


@pytest.mark.asyncio
async def test_http_backend_openai_mode_success():
    from dewie.enrichment.backends.http import HttpBackend

    b = HttpBackend(
        name="b",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        mode="openai",
        api_key_env="OPENAI_API_KEY",
    )

    resp = _mock_response(200, {"choices": [{"message": {"content": '{"summary": "openai"}'}}]})
    cm = _mock_http_cm(resp)

    with (
        patch("httpx.AsyncClient", return_value=cm),
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
    ):
        result = await b.complete("prompt")
    assert result == '{"summary": "openai"}'


@pytest.mark.asyncio
async def test_http_backend_raises_backend_error_on_non_200():
    from dewie.enrichment.backends.http import HttpBackend
    from dewie.enrichment.base import BackendError

    b = HttpBackend(name="b", base_url="http://localhost:11434", model="llama3.2", mode="ollama")

    resp = _mock_response(500, {})
    cm = _mock_http_cm(resp)

    with patch("httpx.AsyncClient", return_value=cm), pytest.raises(BackendError):
        await b.complete("prompt")


@pytest.mark.asyncio
async def test_http_backend_raises_on_timeout():
    import httpx

    from dewie.enrichment.backends.http import HttpBackend
    from dewie.enrichment.base import BackendError

    b = HttpBackend(name="b", base_url="http://localhost:11434", model="m", mode="ollama")

    client = AsyncMock()
    client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=cm):
        with pytest.raises(BackendError, match="timed out"):
            await b.complete("prompt")


@pytest.mark.asyncio
async def test_http_backend_raises_on_connect_error():
    import httpx

    from dewie.enrichment.backends.http import HttpBackend
    from dewie.enrichment.base import BackendError

    b = HttpBackend(name="b", base_url="http://localhost:9999", model="m", mode="ollama")

    client = AsyncMock()
    client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=cm):
        with pytest.raises(BackendError, match="connection refused"):
            await b.complete("prompt")


@pytest.mark.asyncio
async def test_http_backend_extra_headers():
    from dewie.enrichment.backends.http import HttpBackend

    b = HttpBackend(
        name="b",
        base_url="http://localhost",
        model="m",
        mode="openai",
        extra_headers={"anthropic-version": "2023-06-01"},
    )
    resp = _mock_response(200, {"choices": [{"message": {"content": "hi"}}]})
    cm = _mock_http_cm(resp)
    client = cm.__aenter__.return_value

    with patch("httpx.AsyncClient", return_value=cm):
        await b.complete("prompt")

    call_headers = (
        client.post.call_args[1].get("headers") or client.post.call_args[0][1]
        if len(client.post.call_args[0]) > 1
        else {}
    )
    # Just verify backend didn't crash with extra headers
    client.post.assert_called_once()


# ── Thinking model: reasoning_content fallback ────────────────────────────────


@pytest.mark.asyncio
async def test_http_backend_reasoning_content_fallback():
    """When content is empty but reasoning_content has JSON, return extracted JSON."""
    from dewie.enrichment.backends.http import HttpBackend

    b = HttpBackend(name="glm", base_url="http://localhost:11434", model="glm-4-flash", mode="openai")

    json_answer = '{"summary": "test doc", "keywords": ["a", "b"]}'
    resp = _mock_response(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": f"<think>Let me analyse this.</think>\n{json_answer}",
                    }
                }
            ]
        },
    )
    cm = _mock_http_cm(resp)

    with patch("httpx.AsyncClient", return_value=cm):
        result = await b.complete("prompt")
    assert result == json_answer


@pytest.mark.asyncio
async def test_http_backend_reasoning_content_fallback_none_content():
    """When content is None and reasoning_content has JSON, return extracted JSON."""
    from dewie.enrichment.backends.http import HttpBackend

    b = HttpBackend(name="glm", base_url="http://localhost:11434", model="glm-4-flash", mode="openai")

    json_answer = '{"document_type": "article", "language": "en"}'
    resp = _mock_response(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": json_answer,
                    }
                }
            ]
        },
    )
    cm = _mock_http_cm(resp)

    with patch("httpx.AsyncClient", return_value=cm):
        result = await b.complete("prompt")
    assert result == json_answer


@pytest.mark.asyncio
async def test_http_backend_normal_content_not_overridden_by_reasoning():
    """When content is non-empty, reasoning_content is ignored."""
    from dewie.enrichment.backends.http import HttpBackend

    b = HttpBackend(name="glm", base_url="http://localhost:11434", model="glm-4-flash", mode="openai")

    real_answer = '{"summary": "real answer"}'
    resp = _mock_response(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": real_answer,
                        "reasoning_content": '{"summary": "should be ignored"}',
                    }
                }
            ]
        },
    )
    cm = _mock_http_cm(resp)

    with patch("httpx.AsyncClient", return_value=cm):
        result = await b.complete("prompt")
    assert result == real_answer


def test_extract_json_from_reasoning_object():
    from dewie.enrichment.backends.http import HttpBackend

    text = '<think>thinking...</think>\n{"foo": "bar", "n": 42}'
    result = HttpBackend._extract_json_from_reasoning(text)
    assert result == '{"foo": "bar", "n": 42}'


def test_extract_json_from_reasoning_array():
    from dewie.enrichment.backends.http import HttpBackend

    text = 'Here is the answer: [1, 2, 3]'
    result = HttpBackend._extract_json_from_reasoning(text)
    assert result == '[1, 2, 3]'


def test_extract_json_from_reasoning_no_json():
    from dewie.enrichment.backends.http import HttpBackend

    result = HttpBackend._extract_json_from_reasoning("just plain text with no JSON here")
    assert result == ""


def test_extract_json_from_reasoning_markdown_fences():
    from dewie.enrichment.backends.http import HttpBackend

    text = '```json\n{"summary": "clean"}\n```'
    result = HttpBackend._extract_json_from_reasoning(text)
    assert result == '{"summary": "clean"}'


# ── Issue #52: reasoning_content overflow scenario (GLM-4.7-Flash thinking model) ──

@pytest.mark.asyncio
async def test_http_backend_reasoning_content_overflow_scenario():
    """
    Regression test for issue #52.

    GLM-4.7-Flash (thinking model) exhausts its token budget writing
    reasoning_content before producing structured JSON in content.  When
    max_tokens is too small for the full reasoning + answer, the API returns
    an empty `content` field but a rich `reasoning_content`.

    The backend must fall back to extracting JSON from reasoning_content.
    """
    from dewie.enrichment.backends.http import HttpBackend

    # Simulate a model that spent all its tokens on reasoning; content is empty.
    b = HttpBackend(
        name="glm-4.7-flash",
        base_url="http://localhost:11434",
        model="GLM-4.7-Flash",
        mode="openai",
        max_tokens=512,  # small budget — triggers the overflow scenario
    )

    # Realistic reasoning block: long think section, JSON answer at the end
    long_think = "I need to extract metadata from this document. " * 50  # ~250 tokens of thinking
    json_answer = '{"summary": "overflow scenario", "keywords": ["glm", "thinking", "overflow"], "language": "en"}'
    reasoning_block = f"<think>{long_think}</think>\n{json_answer}"

    resp = _mock_response(
        200,
        {
            "choices": [
                {
                    "message": {
                        # Empty content — token budget exhausted before answer
                        "content": "",
                        "reasoning_content": reasoning_block,
                    },
                    "finish_reason": "length",  # confirms token budget hit
                }
            ],
            "model": "GLM-4.7-Flash",
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": 512,  # capped at max_tokens
                "reasoning_tokens": 500,
            },
        },
    )
    cm = _mock_http_cm(resp)

    with patch("httpx.AsyncClient", return_value=cm):
        result = await b.complete("Extract metadata from this document.")

    # Must recover the JSON from reasoning_content
    assert result == json_answer
    parsed = json.loads(result)
    assert parsed["summary"] == "overflow scenario"
    assert "glm" in parsed["keywords"]


@pytest.mark.asyncio
async def test_http_backend_reasoning_content_no_json_returns_raw():
    """
    When reasoning_content is non-empty but contains no extractable JSON,
    return the raw reasoning_content so the caller gets a useful error
    rather than a silent empty string.
    """
    from dewie.enrichment.backends.http import HttpBackend

    b = HttpBackend(name="glm", base_url="http://localhost:11434", model="GLM-4.7-Flash", mode="openai")

    resp = _mock_response(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "I was thinking a lot but produced no JSON at all.",
                    }
                }
            ]
        },
    )
    cm = _mock_http_cm(resp)

    with patch("httpx.AsyncClient", return_value=cm):
        result = await b.complete("prompt")

    # Falls back to raw reasoning_content when no JSON found
    assert result == "I was thinking a lot but produced no JSON at all."
