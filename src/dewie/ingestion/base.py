# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Abstract base class for all content ingesters.

Every ingester must yield ContentDocument objects in PENDING status.
The metadata enrichment pipeline handles tagging after ingestion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from dewie.models.content import ContentDocument


class BaseIngester(ABC):
    """Protocol that all ingestion adapters must satisfy."""

    @abstractmethod
    async def fetch(self, url: str) -> AsyncIterator[ContentDocument]:
        """
        Yield one or more ContentDocuments from the given URL.

        A URL may resolve to a single article (WebIngester) or a feed
        containing many items (RSSIngester).
        """
        ...

    async def __aenter__(self) -> BaseIngester:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Release any held resources (HTTP sessions, file handles, etc.)."""
