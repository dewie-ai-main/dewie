"""Tests for the in-process GGUF embedding provider (dewie.providers.gguf_embed)."""

from __future__ import annotations

import pytest

from dewie.providers.base import EmbeddingDimensionMismatchError
from dewie.providers.gguf_embed import (
    GgufEmbeddingProvider,
    _looks_like_gguf,
    _normalize,
    _parse_spec,
)


class TestLooksLikeGguf:
    def test_positive(self):
        assert _looks_like_gguf("ggml-org/embeddinggemma-300m-qat-q8_0-GGUF")
        assert _looks_like_gguf("/models/model.gguf")

    def test_negative(self):
        assert not _looks_like_gguf("all-MiniLM-L6-v2")
        assert not _looks_like_gguf("text-embedding-3-small")


class TestParseSpec:
    def test_repo_only_auto_picks_file(self):
        assert _parse_spec("ggml-org/embeddinggemma-300m-qat-q8_0-GGUF") == (
            "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF", "*.gguf", None,
        )

    def test_repo_plus_file(self):
        assert _parse_spec("ggml-org/repo-GGUF/model-Q8_0.gguf") == (
            "ggml-org/repo-GGUF", "model-Q8_0.gguf", None,
        )

    def test_repo_plus_nested_file(self):
        assert _parse_spec("owner/repo/sub/model.gguf") == (
            "owner/repo", "sub/model.gguf", None,
        )

    def test_hf_prefix_stripped(self):
        assert _parse_spec("hf:ggml-org/repo-GGUF/model.gguf") == (
            "ggml-org/repo-GGUF", "model.gguf", None,
        )

    def test_absolute_local_path(self):
        assert _parse_spec("/models/embeddinggemma.gguf") == (
            None, None, "/models/embeddinggemma.gguf",
        )

    def test_home_local_path(self):
        assert _parse_spec("~/models/x.gguf") == (None, None, "~/models/x.gguf")


class TestNormalize:
    def test_single_flat_vector(self):
        assert _normalize([0.1, 0.2, 0.3], n_texts=1) == [[0.1, 0.2, 0.3]]

    def test_flat_vector_with_multiple_texts_is_ambiguous(self):
        # A flat list of floats can't be N>1 vectors.
        assert _normalize([0.1, 0.2], n_texts=2) is None

    def test_batch_of_vectors(self):
        raw = [[1.0, 2.0], [3.0, 4.0]]
        assert _normalize(raw, n_texts=2) == [[1.0, 2.0], [3.0, 4.0]]

    def test_token_level_mean_pooled(self):
        # One text -> list of token vectors -> mean-pooled to one vector.
        raw = [[[2.0, 4.0], [4.0, 8.0]]]
        assert _normalize(raw, n_texts=1) == [[3.0, 6.0]]

    def test_empty(self):
        assert _normalize([], n_texts=1) is None
        assert _normalize("not a list", n_texts=1) is None


class _FakeLlama:
    def __init__(self, out):
        self._out = out

    def embed(self, texts):
        return self._out


@pytest.mark.asyncio
async def test_embed_returns_vectors(monkeypatch):
    prov = GgufEmbeddingProvider("ggml-org/x-GGUF")
    monkeypatch.setattr(prov, "_get_model", lambda: _FakeLlama([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    out = await prov.embed(["a", "b"])
    assert out == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


@pytest.mark.asyncio
async def test_embed_truncates_to_dimensions(monkeypatch):
    prov = GgufEmbeddingProvider("ggml-org/x-GGUF", dimensions=2)
    monkeypatch.setattr(prov, "_get_model", lambda: _FakeLlama([[1.0, 2.0, 3.0, 4.0]]))
    out = await prov.embed(["a"])
    assert out == [[1.0, 2.0]]


@pytest.mark.asyncio
async def test_embed_raises_when_model_dim_too_small(monkeypatch):
    prov = GgufEmbeddingProvider("ggml-org/x-GGUF", dimensions=8)
    monkeypatch.setattr(prov, "_get_model", lambda: _FakeLlama([[1.0, 2.0]]))
    with pytest.raises(EmbeddingDimensionMismatchError):
        await prov.embed(["a"])


@pytest.mark.asyncio
async def test_embed_degrades_to_none_when_llama_cpp_missing(monkeypatch):
    """Missing llama-cpp-python must NOT fail the document — embed returns None
    so the doc stays full-text searchable."""
    def _raise_import(*_a, **_k):
        raise ImportError("llama-cpp-python is not installed")

    prov = GgufEmbeddingProvider("ggml-org/x-GGUF")
    monkeypatch.setattr(prov, "_get_model", _raise_import)
    assert await prov.embed(["a"]) is None


@pytest.mark.asyncio
async def test_embed_returns_none_on_model_error(monkeypatch):
    class _Boom:
        def embed(self, texts):
            raise RuntimeError("llama boom")

    prov = GgufEmbeddingProvider("ggml-org/x-GGUF")
    monkeypatch.setattr(prov, "_get_model", lambda: _Boom())
    assert await prov.embed(["a"]) is None


def test_provider_exposes_model_name_for_dim_inference():
    prov = GgufEmbeddingProvider("ggml-org/embeddinggemma-300m-qat-q8_0-GGUF")
    # chunk_embedder resolves dims via getattr(provider, "model", ...)
    assert getattr(prov, "model", None) == "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF"
    assert prov.name == "gguf-local"
