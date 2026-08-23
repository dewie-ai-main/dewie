"""Unit tests for dewie.providers.local_embed."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── ImportError when sentence_transformers is absent ─────────────────────────


def test_import_error_without_sentence_transformers():
    """Raises ImportError with a helpful pip install message when package is missing."""
    # Ensure any cached import of the module doesn't hide the error
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        from dewie.providers.local_embed import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider(model_name="some/model")
        # Reset cached model to force re-import
        provider._model = None

        with pytest.raises(ImportError, match="pip install dewie\\[local\\]"):
            provider._get_model()


# ── embed() calls model.encode and returns list[list[float]] ─────────────────


@pytest.mark.asyncio
async def test_embed_returns_list_of_float_lists():
    """embed() calls model.encode and converts numpy arrays to list[list[float]]."""
    from dewie.providers.local_embed import LocalEmbeddingProvider

    fake_vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    mock_model = MagicMock()
    mock_model.encode.return_value = fake_vectors

    provider = LocalEmbeddingProvider(model_name="some/model")
    provider._model = mock_model  # inject pre-loaded model

    result = await provider.embed(["hello", "world"])

    mock_model.encode.assert_called_once()
    call_args = mock_model.encode.call_args
    assert call_args[0][0] == ["hello", "world"]
    assert call_args[1].get("convert_to_numpy") is True

    assert isinstance(result, list)
    assert len(result) == 2
    for vec in result:
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)


@pytest.mark.asyncio
async def test_embed_returns_none_on_encode_error():
    """embed() returns None (not raise) when encode() throws a non-ImportError."""
    from dewie.providers.local_embed import LocalEmbeddingProvider

    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("GPU OOM")

    provider = LocalEmbeddingProvider(model_name="some/model")
    provider._model = mock_model

    result = await provider.embed(["text"])
    assert result is None


# ── Device selection ──────────────────────────────────────────────────────────


def test_device_selection_prefers_mps():
    """_select_device returns 'mps' when MPS is available."""
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.cuda.is_available.return_value = True

    with patch.dict(sys.modules, {"torch": mock_torch}):
        from importlib import reload

        import dewie.providers.local_embed as mod

        reload(mod)
        assert mod._select_device() == "mps"


def test_device_selection_falls_back_to_cuda():
    """_select_device returns 'cuda' when MPS is unavailable but CUDA is."""
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.cuda.is_available.return_value = True

    with patch.dict(sys.modules, {"torch": mock_torch}):
        from importlib import reload

        import dewie.providers.local_embed as mod

        reload(mod)
        assert mod._select_device() == "cuda"


def test_device_selection_falls_back_to_cpu():
    """_select_device returns 'cpu' when neither MPS nor CUDA is available."""
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"torch": mock_torch}):
        from importlib import reload

        import dewie.providers.local_embed as mod

        reload(mod)
        assert mod._select_device() == "cpu"


def test_device_selection_falls_back_to_cpu_when_torch_missing():
    """_select_device returns 'cpu' when torch is not installed at all."""
    with patch.dict(sys.modules, {"torch": None}):
        from importlib import reload

        import dewie.providers.local_embed as mod

        reload(mod)
        assert mod._select_device() == "cpu"


# ── Provider metadata ─────────────────────────────────────────────────────────


def test_provider_name():
    from dewie.providers.local_embed import LocalEmbeddingProvider

    provider = LocalEmbeddingProvider(model_name="any/model")
    assert provider.name == "local"
