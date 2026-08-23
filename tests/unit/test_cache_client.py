"""Tests for dewie.storage.cache — CacheClient and _make_key."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

# ── _make_key ─────────────────────────────────────────────────────────────────


def test_make_key_deterministic():
    from dewie.storage.cache import _make_key

    k1 = _make_key("query", "public", "NBA scores", "topics")
    k2 = _make_key("query", "public", "NBA scores", "topics")
    assert k1 == k2


def test_make_key_differs_by_namespace():
    from dewie.storage.cache import _make_key

    k1 = _make_key("query", "public", "q")
    k2 = _make_key("related", "public", "q")
    assert k1 != k2


def test_make_key_differs_by_parts():
    from dewie.storage.cache import _make_key

    k1 = _make_key("query", "t1", "q")
    k2 = _make_key("query", "t2", "q")
    assert k1 != k2


def test_make_key_format():
    from dewie.storage.cache import _make_key

    k = _make_key("query", "public", "test")
    assert k.startswith("dewie:query:")
    assert len(k) > 20


# ── CacheClient ───────────────────────────────────────────────────────────────


def _make_cache_client():
    from dewie.storage.cache import CacheClient

    client = object.__new__(CacheClient)
    client._ttl = 300
    client._redis = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_get_query_result_miss():
    c = _make_cache_client()
    c._redis.get = AsyncMock(return_value=None)
    result = await c.get_query_result("test query", "topics")
    assert result is None


@pytest.mark.asyncio
async def test_get_query_result_hit():
    c = _make_cache_client()
    data = {"results": [{"id": "abc"}]}
    c._redis.get = AsyncMock(return_value=json.dumps(data))
    result = await c.get_query_result("test query", "topics")
    assert result == data


@pytest.mark.asyncio
async def test_set_query_result():
    c = _make_cache_client()
    c._redis.setex = AsyncMock()
    await c.set_query_result("test query", "topics", {"results": []})
    c._redis.setex.assert_called_once()
    args = c._redis.setex.call_args[0]
    assert args[1] == 300  # TTL


@pytest.mark.asyncio
async def test_get_related_result_miss():
    c = _make_cache_client()
    c._redis.get = AsyncMock(return_value=None)
    result = await c.get_related_result("doc-1", 2, "topics")
    assert result is None


@pytest.mark.asyncio
async def test_get_related_result_hit():
    c = _make_cache_client()
    data = {"nodes": []}
    c._redis.get = AsyncMock(return_value=json.dumps(data))
    result = await c.get_related_result("doc-1", 2, "topics")
    assert result == data


@pytest.mark.asyncio
async def test_get_tenant_plan_returns_value():
    c = _make_cache_client()
    c._redis.get = AsyncMock(return_value="pro")
    plan = await c.get_tenant_plan("tenant-123")
    assert plan == "pro"


@pytest.mark.asyncio
async def test_set_tenant_plan():
    c = _make_cache_client()
    c._redis.setex = AsyncMock()
    await c.set_tenant_plan("tenant-123", "pro")
    c._redis.setex.assert_called_once()
    args = c._redis.setex.call_args[0]
    assert args[1] == 300
    assert args[2] == "pro"


@pytest.mark.asyncio
async def test_incr_quota_first_increment_sets_expiry():
    c = _make_cache_client()
    c._redis.incrby = AsyncMock(return_value=1)
    c._redis.expire = AsyncMock()
    count = await c.incr_quota("tenant-123", "2026-04-27")
    assert count == 1
    c._redis.expire.assert_called_once()


@pytest.mark.asyncio
async def test_incr_quota_subsequent_does_not_set_expiry():
    c = _make_cache_client()
    c._redis.incrby = AsyncMock(return_value=5)
    c._redis.expire = AsyncMock()
    count = await c.incr_quota("tenant-123", "2026-04-27")
    assert count == 5
    c._redis.expire.assert_not_called()


@pytest.mark.asyncio
async def test_decr_quota():
    c = _make_cache_client()
    c._redis.decrby = AsyncMock()
    await c.decr_quota("tenant-123", "2026-04-27")
    c._redis.decrby.assert_called_once()


@pytest.mark.asyncio
async def test_invalidate():
    c = _make_cache_client()
    c._redis.delete = AsyncMock()
    await c.invalidate("public", "my query")
    c._redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_close():
    c = _make_cache_client()
    c._redis.aclose = AsyncMock()
    await c.close()
    c._redis.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_tenant_isolation_in_query_key():
    from dewie.storage.cache import _make_key

    k1 = _make_key("query", "tenant-A", "query text", "topics")
    k2 = _make_key("query", "tenant-B", "query text", "topics")
    assert k1 != k2
