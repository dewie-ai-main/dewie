# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Local (in-process) GGUF embedding provider backed by llama-cpp-python.

This is the zero-config default: it runs a small GGUF embedding model
(EmbeddingGemma-300m by default) entirely in-process via llama.cpp — no
external server, no API key, and the default model is a public re-host so
there is no HuggingFace license gate to accept. Same approach many local-first tools use
locally (there via node-llama-cpp; here via the Python binding).

Model spec formats accepted (``embed_model`` setting):
  - ``"ggml-org/embeddinggemma-300m-qat-q8_0-GGUF"``      (repo — file auto-picked)
  - ``"ggml-org/…-GGUF/embeddinggemma-300m-qat-Q8_0.gguf"`` (repo + explicit file)
  - ``"hf:<repo>/<file.gguf>"``                            (hf: prefix)
  - ``"/abs/path/to/model.gguf"``                          (local file)

The model is lazy-loaded on first embed (startup stays fast) and cached for
the process's life. llama.cpp's context is not safe for concurrent calls, so
embeds are serialized behind an async lock.
"""

from __future__ import annotations

import asyncio
import logging

from .base import EmbeddingDimensionMismatchError, EmbeddingProvider

log = logging.getLogger(__name__)

_DEFAULT_FILENAME_GLOB = "*.gguf"


def _looks_like_gguf(model: str) -> bool:
    """True if a model spec refers to a GGUF model (repo or file)."""
    return "gguf" in model.lower()


def _parse_spec(spec: str) -> tuple[str | None, str | None, str | None]:
    """Split a model spec into (repo_id, filename, local_path).

    Exactly one of (repo_id, local_path) is returned non-None.
    """
    s = spec[3:] if spec.lower().startswith("hf:") else spec

    if s.endswith(".gguf") and "/" in s and not s.lower().startswith("http"):
        # Could be a local path or "repo/.../file.gguf". A repo id is
        # "owner/name"; anything with a leading slash or >2 path parts before
        # the file that isn't an owner/name pair we treat as a local path.
        if s.startswith("/") or s.startswith("~") or s.startswith("."):
            return None, None, s
        parts = s.split("/")
        # owner/name/<...>/file.gguf  -> repo="owner/name", filename=rest
        if len(parts) >= 3:
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
            return repo_id, filename, None
        # bare "something.gguf" with a single slash is ambiguous; treat as path
        return None, None, s

    # No explicit file: treat the whole thing as a repo id, auto-pick a .gguf.
    return s, _DEFAULT_FILENAME_GLOB, None


class GgufEmbeddingProvider(EmbeddingProvider):
    """In-process embedding via a GGUF model loaded by llama-cpp-python."""

    # llama.cpp Llama is not safe for concurrent .embed() calls; serialize.
    _lock = asyncio.Lock()

    def __init__(
        self,
        model_spec: str,
        dimensions: int | None = None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
    ) -> None:
        self.model_spec = model_spec
        # Exposed as ``model``/``model_name`` so callers that infer dimensions
        # from the model string (e.g. chunk_embedder) resolve the right dims.
        self.model = model_spec
        self.model_name = model_spec
        self.dimensions = dimensions
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._llama = None

    @property
    def name(self) -> str:
        return "gguf-local"

    def _get_model(self):  # pragma: no cover - needs the real GGUF weights + llama.cpp
        if self._llama is not None:
            return self._llama
        try:
            from llama_cpp import Llama  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. Run: pip install dewie[local]"
            ) from exc

        repo_id, filename, local_path = _parse_spec(self.model_spec)
        common = dict(
            embedding=True,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            verbose=False,
        )
        if local_path:
            log.info("Loading GGUF embedding model from %r", local_path)
            self._llama = Llama(model_path=local_path, **common)
        else:
            log.info(
                "Loading GGUF embedding model repo=%r file=%r (downloads on first use)",
                repo_id,
                filename,
            )
            self._llama = Llama.from_pretrained(
                repo_id=repo_id, filename=filename, **common
            )
        return self._llama

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        try:
            async with self._lock:
                model = self._get_model()
                loop = asyncio.get_running_loop()
                raw = await loop.run_in_executor(None, model.embed, texts)
        except ImportError:
            # Missing optional dep: degrade to no-embedding rather than failing
            # the document. It stays searchable via full-text search; semantic
            # search lights up once the dep is installed.
            log.warning(
                "llama-cpp-python is not installed — skipping local embeddings. "
                "Install it (pip install 'dewie[local]', or add llama-cpp-python) "
                "to enable local semantic search. Documents remain full-text searchable."
            )
            return None
        except Exception as exc:
            log.error("GGUF embedding failed: %s", exc)
            return None

        embeddings = _normalize(raw, len(texts))
        if embeddings is None:
            log.error("GGUF embedding returned an unexpected shape")
            return None

        if self.dimensions is not None and embeddings:
            first_dim = len(embeddings[0])
            if first_dim != self.dimensions:
                if first_dim > self.dimensions:
                    log.warning(
                        "GgufEmbeddingProvider: truncating %d-dim embedding to %d dims "
                        "(model=%r). Set EMBED_DIMENSIONS=%d to match.",
                        first_dim, self.dimensions, self.model_spec, self.dimensions,
                    )
                    embeddings = [v[: self.dimensions] for v in embeddings]
                else:
                    raise EmbeddingDimensionMismatchError(
                        expected=self.dimensions, actual=first_dim, model=self.model_spec
                    )

        return embeddings


def _normalize(raw, n_texts: int) -> list[list[float]] | None:
    """Coerce llama_cpp .embed() output into list[list[float]].

    llama_cpp returns a flat list[float] for a single input and a
    list[list[float]] for a batch; some builds return token-level embeddings
    (list[list[list[float]]]) when pooling is off — mean-pool those.
    """
    if not isinstance(raw, list) or not raw:
        return None

    # Single vector (flat list of floats) for a single text.
    if isinstance(raw[0], (int, float)):
        return [ [float(x) for x in raw] ] if n_texts == 1 else None

    out: list[list[float]] = []
    for item in raw:
        if not isinstance(item, list) or not item:
            return None
        if isinstance(item[0], (int, float)):
            out.append([float(x) for x in item])
        elif isinstance(item[0], list):
            # token-level -> mean pool across tokens
            dim = len(item[0])
            sums = [0.0] * dim
            for tok in item:
                for j, v in enumerate(tok):
                    sums[j] += v
            out.append([s / len(item) for s in sums])
        else:
            return None
    return out
