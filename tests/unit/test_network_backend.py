"""Tests for NetworkBackend ABC and NoopNetworkBackend."""

from __future__ import annotations

import pytest

from dewie.storage.network import NetworkBackend, NoopNetworkBackend, SourceRecord


class TestSourceRecord:
    def test_defaults(self):
        record = SourceRecord(endpoint="https://dewie.example.com", name="test")
        assert record.endpoint == "https://dewie.example.com"
        assert record.name == "test"
        assert record.api_key is None
        assert record.status == "active"
        assert record.corpus_filter is None
        assert record.registered_at is None

    def test_full(self):
        record = SourceRecord(
            endpoint="https://dewie.example.com",
            name="test",
            api_key="ck_live_abc",
            status="disabled",
            corpus_filter={"tags": ["research"]},
            registered_at="2025-01-01T00:00:00Z",
        )
        assert record.api_key == "ck_live_abc"
        assert record.status == "disabled"
        assert record.corpus_filter == {"tags": ["research"]}
        assert record.registered_at == "2025-01-01T00:00:00Z"

    def test_to_json(self):
        record = SourceRecord(endpoint="https://dewie.example.com", name="test")
        data = record.model_dump()
        assert data["endpoint"] == "https://dewie.example.com"
        assert data["api_key"] is None


class TestNoopNetworkBackend:
    @pytest.mark.asyncio
    async def test_register_node_raises(self):
        backend = NoopNetworkBackend()
        with pytest.raises(NotImplementedError, match="Dewie Cloud"):
            await backend.register_node("https://dewie.example.com", "ck_live_abc")

    @pytest.mark.asyncio
    async def test_discover_peers_returns_empty(self):
        backend = NoopNetworkBackend()
        peers = await backend.discover_peers()
        assert peers == []

    @pytest.mark.asyncio
    async def test_discover_peers_with_filter_returns_empty(self):
        backend = NoopNetworkBackend()
        peers = await backend.discover_peers({"tags": ["research"]})
        assert peers == []

    @pytest.mark.asyncio
    async def test_federated_search_returns_empty(self):
        backend = NoopNetworkBackend()
        results = await backend.federated_search([0.1, 0.2, 0.3], k=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_federated_search_with_sources_returns_empty(self):
        backend = NoopNetworkBackend()
        sources = [SourceRecord(endpoint="https://peer1.example.com", name="peer1")]
        results = await backend.federated_search([0.1, 0.2], k=5, sources=sources)
        assert results == []


class TestNetworkBackendABC:
    def test_is_abc(self):
        assert (
            not hasattr(NetworkBackend, "__abstractmethods__")
            or len(NetworkBackend.__abstractmethods__) > 0
        )

    def test_cannot_instantiate_without_implementing(self):
        with pytest.raises(TypeError):
            NetworkBackend()  # type: ignore[abstract]
