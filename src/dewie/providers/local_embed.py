# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Local (in-process) embedding provider backed by sentence-transformers.

Lazy-loads the model on first call — startup stays fast.
Runs on CPU by default; uses GPU/MPS if available.
"""

from __future__ import annotations

import asyncio
import logging

from .base import EmbeddingDimensionMismatchError, EmbeddingProvider

log = logging.getLogger(__name__)


def _select_device() -> str:
    """Return the best available device: mps > cuda > cpu."""
    try:
        import torch  # type: ignore[import-untyped]

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    In-process embedding via sentence-transformers.

    Lazy-loads the model on first call — startup stays fast.
    Runs on CPU by default; uses GPU/MPS if available.
    """

    def __init__(self, model_name: str, dimensions: int | None = None) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = None

    @property
    def name(self) -> str:
        return "local"

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install dewie[local]"
            ) from exc

        device = _select_device()
        log.info("Loading local embedding model %r on device %r", self.model_name, device)
        self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        loop = asyncio.get_running_loop()
        try:
            model = self._get_model()
            vectors = await loop.run_in_executor(
                None,
                lambda: model.encode(texts, convert_to_numpy=True),
            )
            embeddings = [v.tolist() for v in vectors]
        except ImportError:
            # Missing optional dep: degrade to no-embedding rather than failing
            # the document (stays full-text searchable). See gguf_embed for the
            # same behavior on the GGUF path.
            log.warning(
                "sentence-transformers is not installed — skipping local embeddings. "
                "Install it (pip install 'dewie[local]') to enable local semantic "
                "search. Documents remain full-text searchable."
            )
            return None
        except Exception as exc:
            log.error("Local embedding failed: %s", exc)
            return None

        if self.dimensions is not None and embeddings:
            first_dim = len(embeddings[0])
            if first_dim != self.dimensions:
                if first_dim > self.dimensions:
                    log.warning(
                        "LocalEmbeddingProvider: truncating %d-dim embedding to %d dims "
                        "(model=%r). Set EMBED_DIMENSIONS=%d to match.",
                        first_dim, self.dimensions, self.model_name, self.dimensions,
                    )
                    embeddings = [v[: self.dimensions] for v in embeddings]
                else:
                    raise EmbeddingDimensionMismatchError(
                        expected=self.dimensions, actual=first_dim, model=self.model_name
                    )

        return embeddings
