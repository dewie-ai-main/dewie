"""Tests for dewie.providers.factory — provider resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dewie.providers.servers import ServerConfig

# ── _resolve_step ─────────────────────────────────────────────────────────────


def test_resolve_step_aq_generation():
    from dewie.providers.factory import _resolve_step

    mock_settings = MagicMock()
    mock_settings.chat_server_aq = "openai"
    mock_settings.chat_model_aq = "gpt-4o-mini"

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        label, model = _resolve_step("aq_generation")

    assert label == "openai"
    assert model == "gpt-4o-mini"


def test_resolve_step_keyword_extraction():
    from dewie.providers.factory import _resolve_step

    mock_settings = MagicMock()
    mock_settings.chat_server_ke = "anthropic"
    mock_settings.chat_model_ke = "claude-3-haiku-20240307"
    mock_settings.chat_server_aq = ""
    mock_settings.chat_model_aq = ""

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        label, model = _resolve_step("keyword_extraction")

    assert label == "anthropic"
    assert model == "claude-3-haiku-20240307"


def test_resolve_step_ke_falls_back_to_aq():
    from dewie.providers.factory import _resolve_step

    mock_settings = MagicMock()
    mock_settings.chat_server_ke = ""
    mock_settings.chat_model_ke = ""
    mock_settings.chat_server_aq = "openai"
    mock_settings.chat_model_aq = "gpt-4o-mini"

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        label, model = _resolve_step("keyword_extraction")

    assert label == "openai"
    assert model == "gpt-4o-mini"


def test_resolve_step_default_falls_back():
    from dewie.providers.factory import _resolve_step

    mock_settings = MagicMock()
    mock_settings.chat_server_aq = ""
    mock_settings.chat_model_aq = ""

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        label, model = _resolve_step("some_unknown_step")

    assert isinstance(label, str)
    assert isinstance(model, str)


# ── _resolve_embed ────────────────────────────────────────────────────────────


def test_resolve_embed():
    from dewie.providers.factory import _resolve_embed

    mock_settings = MagicMock()
    mock_settings.embed_server = "openai"
    mock_settings.embed_model = "text-embedding-3-small"
    mock_settings.embed_dimensions = None

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        label, model, dimensions = _resolve_embed()

    assert label == "openai"
    assert model == "text-embedding-3-small"
    assert dimensions is None


# ── _build_chat ───────────────────────────────────────────────────────────────


def test_build_chat_openai():
    from dewie.providers.factory import _build_chat

    server = ServerConfig(label="openai", api_format="openai", endpoint="https://api.openai.com")
    mock_settings = MagicMock()
    mock_settings.openai_api_type = "chat/completions"

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        provider = _build_chat(server, "gpt-4o-mini")

    assert provider is not None
    assert provider.name == "openai"


def test_build_chat_anthropic():
    from dewie.providers.factory import _build_chat

    server = ServerConfig(label="anthropic", api_format="anthropic", endpoint="https://api.anthropic.com")
    mock_settings = MagicMock()

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        provider = _build_chat(server, "claude-3-haiku-20240307")

    assert provider is not None
    assert provider.name == "anthropic"


def test_build_chat_custom_server():
    """A custom server with api_format=openai builds an OpenAI-format provider
    pointed at the custom endpoint."""
    from dewie.providers.factory import _build_chat

    server = ServerConfig(label="my-llama", api_format="openai", endpoint="http://custom-llm:8000")
    mock_settings = MagicMock()
    mock_settings.openai_api_type = "chat/completions"

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        provider = _build_chat(server, "my-model")

    assert provider is not None
    assert "custom-llm" in provider._chat_url
    assert provider._chat_url.endswith("/v1/chat/completions")


def test_build_chat_unknown_api_format_raises():
    from dewie.providers.factory import _build_chat

    server = ServerConfig(label="weird", api_format="weird-format", endpoint="http://x")
    mock_settings = MagicMock()

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        with pytest.raises(RuntimeError, match="Unknown api_format"):
            _build_chat(server, "some-model")


# ── _build_embed ──────────────────────────────────────────────────────────────


def test_build_embed_openai():
    from dewie.providers.factory import _build_embed

    server = ServerConfig(label="openai", api_format="openai", endpoint="https://api.openai.com")
    provider = _build_embed(server, "text-embedding-3-small", None)

    assert provider is not None
    assert provider.name == "openai"


def test_build_embed_anthropic_raises():
    from dewie.providers.factory import _build_embed

    server = ServerConfig(label="anthropic", api_format="anthropic", endpoint="https://api.anthropic.com")

    with pytest.raises(RuntimeError, match="no embeddings API"):
        _build_embed(server, "some-model", None)


def test_build_embed_unknown_api_format_raises():
    from dewie.providers.factory import _build_embed

    server = ServerConfig(label="weird", api_format="weird-format", endpoint="http://x")

    with pytest.raises(RuntimeError, match="Unknown api_format"):
        _build_embed(server, "some-model", None)


# ── get_chat_provider / get_embedding_provider ────────────────────────────────


def test_get_chat_provider(tmp_path, monkeypatch):
    from dewie.providers.factory import get_chat_provider

    monkeypatch.setenv("DEWIE_CONFIG_PATH", str(tmp_path / "dewie.yml"))
    mock_settings = MagicMock()
    mock_settings.chat_server_aq = "openai"
    mock_settings.chat_model_aq = "gpt-4o-mini"
    mock_settings.openai_api_type = "chat/completions"

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        provider = get_chat_provider("aq_generation")

    assert provider is not None


def test_get_chat_provider_no_server_raises():
    from dewie.providers.factory import get_chat_provider

    mock_settings = MagicMock()
    mock_settings.chat_server_aq = ""
    mock_settings.chat_model_aq = ""

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        with pytest.raises(RuntimeError, match="No server configured"):
            get_chat_provider("aq_generation")


def test_get_embedding_provider(tmp_path, monkeypatch):
    from dewie.providers.factory import get_embedding_provider

    monkeypatch.setenv("DEWIE_CONFIG_PATH", str(tmp_path / "dewie.yml"))
    mock_settings = MagicMock()
    mock_settings.embed_server = "openai"
    mock_settings.embed_model = "text-embedding-3-small"
    mock_settings.embed_dimensions = None

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        provider = get_embedding_provider()

    assert provider is not None


def test_get_embedding_provider_local():
    from dewie.providers.factory import get_embedding_provider

    mock_settings = MagicMock()
    mock_settings.embed_server = "local"
    mock_settings.embed_model = "all-MiniLM-L6-v2"
    mock_settings.embed_dimensions = None

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        provider = get_embedding_provider()

    assert provider is not None
    assert provider.name == "local"


def test_get_embedding_provider_local_disallowed_raises():
    """LOCAL_EMBED_ALLOWED=false blocks embed_server=local even if configured."""
    from dewie.providers.factory import get_embedding_provider

    mock_settings = MagicMock()
    mock_settings.embed_server = "local"
    mock_settings.embed_model = "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF"
    mock_settings.embed_dimensions = None
    mock_settings.local_embed_allowed = False

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        with pytest.raises(RuntimeError, match="disabled on this host"):
            get_embedding_provider()


def test_get_embedding_provider_local_gguf():
    """embed_server=local + a GGUF model -> in-process llama.cpp provider."""
    from dewie.providers.factory import get_embedding_provider

    mock_settings = MagicMock()
    mock_settings.embed_server = "local"
    mock_settings.embed_model = "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF"
    mock_settings.embed_dimensions = None

    with patch("dewie.providers.factory._get_settings", return_value=mock_settings):
        provider = get_embedding_provider()

    assert type(provider).__name__ == "GgufEmbeddingProvider"
    assert provider.name == "gguf-local"


def test_gguf_spec_parsing():
    from dewie.providers.gguf_embed import _looks_like_gguf, _parse_spec

    assert _looks_like_gguf("ggml-org/embeddinggemma-300m-qat-q8_0-GGUF")
    assert not _looks_like_gguf("all-MiniLM-L6-v2")

    # repo only -> auto-pick a .gguf file
    assert _parse_spec("ggml-org/embeddinggemma-300m-qat-q8_0-GGUF") == (
        "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF", "*.gguf", None,
    )
    # repo + explicit file
    assert _parse_spec("ggml-org/repo-GGUF/model-Q8_0.gguf") == (
        "ggml-org/repo-GGUF", "model-Q8_0.gguf", None,
    )
    # hf: prefix is stripped
    assert _parse_spec("hf:ggml-org/repo-GGUF/model.gguf") == (
        "ggml-org/repo-GGUF", "model.gguf", None,
    )
    # absolute local path
    assert _parse_spec("/models/embeddinggemma.gguf") == (None, None, "/models/embeddinggemma.gguf")
