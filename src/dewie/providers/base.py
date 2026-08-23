# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Base abstract classes for chat and embedding providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingDimensionMismatchError(Exception):
    """Raised when an embedding vector's dimension count doesn't match the expected value."""

    def __init__(self, expected: int, actual: int, model: str = "") -> None:
        self.expected = expected
        self.actual = actual
        self.model = model
        super().__init__(
            f"Embedding dimension mismatch: expected {expected}, got {actual} "
            f"(model={model!r}). "
            f"Truncate the vector or set EMBED_DIMENSIONS to {actual}."
        )


class ChatProvider(ABC):
    """Abstract base for all chat/completion providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> str:
        """Return the assistant message text, or empty string on failure."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""


class EmbeddingProvider(ABC):
    """Abstract base for all embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Return embeddings for a batch of texts, or None on failure."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
