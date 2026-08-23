# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class SourceRecord(BaseModel):
    """Describes a remote Dewie instance or catalog connection."""

    endpoint: str = Field(description="Base URL of the remote Dewie instance")
    name: str = Field(description="Human-readable name for this source")
    api_key: str | None = Field(default=None, description="Optional API key for authentication")
    status: str = Field(default="active", description="Registration status: active/disabled")
    corpus_filter: dict[str, Any] | None = Field(
        default=None,
        description="Optional filter criteria for corpus discovery",
    )
    registered_at: str | None = Field(
        default=None,
        description="ISO-8601 timestamp when this source was registered",
    )


class NetworkBackend(ABC):
    """Abstract base class for corpus sharing network backends.

    Cloud deployments override this with a real implementation.
    OSS ships NoopNetworkBackend as the default.
    """

    @abstractmethod
    async def register_node(self, endpoint: str, api_key: str | None = None) -> str:
        """Register this node with the corpus sharing network.

        Returns the registered node identifier.
        """

    @abstractmethod
    async def discover_peers(self, corpus_filter: dict | None = None) -> list[SourceRecord]:
        """Discover peer Dewie instances matching the given corpus filter.

        Returns an empty list when no peers match or are available.
        """

    @abstractmethod
    async def federated_search(
        self,
        query_vector: list[float],
        k: int,
        sources: list[SourceRecord] | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a federated search across peer sources.

        Returns a merged, reranked list of search results.
        """


class NoopNetworkBackend(NetworkBackend):
    """Default OSS implementation. Corpus sharing requires Dewie Cloud.

    All methods are safe no-ops — they return empty results or raise
    for mutating operations so OSS nodes never accidentally talk to
    a network they don't understand.
    """

    async def register_node(self, endpoint: str, api_key: str | None = None) -> str:
        """OSS nodes cannot register — corpus sharing requires Dewie Cloud."""
        raise NotImplementedError(
            "Network registration requires Dewie Cloud. "
            "OSS nodes can still connect manually via dewie_sources."
        )

    async def discover_peers(self, corpus_filter: dict | None = None) -> list[SourceRecord]:
        """No-op: returns empty list. Peer discovery requires Dewie Cloud."""
        return []

    async def federated_search(
        self,
        query_vector: list[float],
        k: int,
        sources: list[SourceRecord] | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """No-op: returns empty list. Federated search requires Dewie Cloud."""
        return []
