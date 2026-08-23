"""Unit tests for OpenAIEmbeddingProvider dimension handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_embed_includes_dimensions_when_requested(monkeypatch):
    from dewie.providers.openai_provider import OpenAIEmbeddingProvider

    monkeypatch.setenv("EMBED_OUTPUT_DIMENSIONS", "1024")

    provider = OpenAIEmbeddingProvider(
        model="Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [{"index": 0, "embedding": [0.1] * 1024}],
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("dewie.providers.openai_provider.httpx.AsyncClient", return_value=mock_cm):
        vectors = await provider.embed(["hello"])

    assert vectors is not None
    assert len(vectors[0]) == 1024
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["dimensions"] == 1024


@pytest.mark.asyncio
async def test_embed_retries_without_dimensions_and_downsamples_for_mrl(monkeypatch):
    from dewie.providers.openai_provider import OpenAIEmbeddingProvider

    monkeypatch.setenv("EMBED_OUTPUT_DIMENSIONS", "1024")

    provider = OpenAIEmbeddingProvider(
        model="Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
    )

    bad_response = MagicMock()
    bad_response.status_code = 400

    good_response = MagicMock()
    good_response.status_code = 200
    good_response.raise_for_status = MagicMock()
    good_response.json.return_value = {
        "data": [{"index": 0, "embedding": [0.2] * 4096}],
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[bad_response, good_response])

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("dewie.providers.openai_provider.httpx.AsyncClient", return_value=mock_cm):
        vectors = await provider.embed(["hello"])

    assert vectors is not None
    assert len(vectors[0]) == 1024
    first_payload = mock_client.post.call_args_list[0].kwargs["json"]
    second_payload = mock_client.post.call_args_list[1].kwargs["json"]
    assert first_payload["dimensions"] == 1024
    assert "dimensions" not in second_payload
