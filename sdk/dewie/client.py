# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Sync and async clients for the Dewie API.
"""

from __future__ import annotations

from typing import Optional

import httpx

from .models import BridgePath, SearchResponse, TraverseResponse


class DewieClient:
    """Synchronous Dewie API client.

    Usage::

        client = DewieClient(api_url="https://api.dewie.ai", api_key="ck_live_...")

        response = client.query("How does CRISPR work", limit=10)
        if response.has_gap:
            print(response.gap_message)

        traverse = client.traverse(["CRISPR", "gene editing"], max_documents=20)
        print(traverse.next_clusters)

        path = client.bridge(source_id="uuid-a", target_id="uuid-b")
        print(f"Connected in {path.hops} hops")
    """

    def __init__(
        self,
        api_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = api_url.rstrip("/")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["X-API-Key"] = api_key
        self._timeout = timeout
        self._http = httpx.Client(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
        )

    def query(
        self,
        query: str,
        limit: int = 5,
        ranker: str = "rrf_aq",
        **kwargs,
    ) -> SearchResponse:
        """Search the corpus. Check ``response.has_gap`` before trusting results."""
        payload = {"query": query, "limit": limit, "ranker": ranker, **kwargs}
        resp = self._http.post("/query", json=payload)
        resp.raise_for_status()
        return SearchResponse.model_validate(resp.json())

    def traverse(
        self,
        seeds: list[str],
        max_documents: int = 20,
        max_depth: int = 3,
        **kwargs,
    ) -> TraverseResponse:
        """Graph traversal from seed topics or doc IDs."""
        payload = {
            "seed_keywords": seeds,
            "max_documents": max_documents,
            **kwargs,
        }
        resp = self._http.post("/traverse", json=payload)
        resp.raise_for_status()
        return TraverseResponse.model_validate(resp.json())

    def bridge(
        self,
        source_id: str,
        target_id: str,
        max_hops: Optional[int] = None,
        **kwargs,
    ) -> BridgePath:
        """Find the shortest bridge path between two documents."""
        payload = {"source_id": source_id, "target_id": target_id, **kwargs}
        if max_hops is not None:
            payload["max_hops"] = max_hops
        resp = self._http.post("/graph/bridge", json=payload)
        resp.raise_for_status()
        return BridgePath.model_validate(resp.json())

    def expand(self, doc_id: str, limit: int = 10) -> list[dict]:
        """
        Return the graph neighbors of a document, sorted by edge weight.

        Use this after a search hit to follow the knowledge graph and surface
        related documents that may not rank highly on keyword or vector search.
        """
        resp = self._http.get(f"/graph/neighbors/{doc_id}", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()

    def ingest(self, url: str, title: Optional[str] = None, body: Optional[str] = None) -> dict:
        """
        Submit a URL for ingestion and enrichment.

        Returns immediately with the accepted document ID(s). Enrichment
        (metadata extraction, embedding, edge building) happens in the background.

        Args:
            url:   The URL to fetch and index.
            title: Optional title override (skips fetching if body is also provided).
            body:  Optional pre-fetched body text (skips HTTP fetch entirely).
        """
        payload: dict = {"url": url}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        resp = self._http.post("/ingest", json=payload)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> DewieClient:
        return self

    def __exit__(self, *_) -> None:
        self.close()


class AsyncDewieClient:
    """Async Dewie API client.

    Usage::

        async with AsyncDewieClient(api_url="...", api_key="...") as client:
            response = await client.query("How does CRISPR work")
            if response.has_gap:
                print(response.gap_message)
    """

    def __init__(
        self,
        api_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = api_url.rstrip("/")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["X-API-Key"] = api_key
        self._timeout = timeout
        self._http: httpx.AsyncClient | None = None

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout,
            )
        return self._http

    async def query(
        self,
        query: str,
        limit: int = 5,
        ranker: str = "rrf_aq",
        **kwargs,
    ) -> SearchResponse:
        """Search the corpus. Check ``response.has_gap`` before trusting results."""
        payload = {"query": query, "limit": limit, "ranker": ranker, **kwargs}
        resp = await self._get_http().post("/query", json=payload)
        resp.raise_for_status()
        return SearchResponse.model_validate(resp.json())

    async def traverse(
        self,
        seeds: list[str],
        max_documents: int = 20,
        max_depth: int = 3,
        **kwargs,
    ) -> TraverseResponse:
        """Graph traversal from seed topics or doc IDs."""
        payload = {
            "seed_keywords": seeds,
            "max_documents": max_documents,
            **kwargs,
        }
        resp = await self._get_http().post("/traverse", json=payload)
        resp.raise_for_status()
        return TraverseResponse.model_validate(resp.json())

    async def bridge(
        self,
        source_id: str,
        target_id: str,
        max_hops: Optional[int] = None,
        **kwargs,
    ) -> BridgePath:
        """Find the shortest bridge path between two documents."""
        payload = {"source_id": source_id, "target_id": target_id, **kwargs}
        if max_hops is not None:
            payload["max_hops"] = max_hops
        resp = await self._get_http().post("/graph/bridge", json=payload)
        resp.raise_for_status()
        return BridgePath.model_validate(resp.json())

    async def expand(self, doc_id: str, limit: int = 10) -> list[dict]:
        """Return graph neighbors for a document, sorted by edge weight."""
        resp = await self._get_http().get(f"/graph/neighbors/{doc_id}", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()

    async def ingest(self, url: str, title: Optional[str] = None, body: Optional[str] = None) -> dict:
        """Submit a URL for ingestion and enrichment. Returns immediately; enrichment is async."""
        payload: dict = {"url": url}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        resp = await self._get_http().post("/ingest", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> AsyncDewieClient:
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
