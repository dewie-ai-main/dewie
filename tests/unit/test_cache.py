"""Tests for dewie.storage.cache — Redis cache client."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, patch

import pytest


def _make_redis_mock():
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.setex = AsyncMock()
    mock.delete = AsyncMock()
    mock.incr = AsyncMock(return_value=1)
    mock.decr = AsyncMock()
    mock.expire = AsyncMock()
    return mock


@pytest.fixture
def cache_and_redis():
    redis_mock = _make_redis_mock()
    with patch("dewie.storage.cache.aioredis.from_url", return_value=redis_mock):
        from dewie.storage.cache import CacheClient

        client = CacheClient(url="redis://localhost:6379", ttl=300)
    return client, redis_mock


@pytest.mark.asyncio
async def test_get_query_result_miss(cache_and_redis):
    client, redis = cache_and_redis
    redis.get.return_value = None
    result = await client.get_query_result("test query", "topic")
    assert result is None


@pytest.mark.asyncio
async def test_get_query_result_hit(cache_and_redis):
    client, redis = cache_and_redis
    data = {"results": [{"id": "abc", "score": 0.9}]}
    redis.get.return_value = json.dumps(data)
    result = await client.get_query_result("test query", "topic")
    assert result == data


@pytest.mark.asyncio
async def test_set_query_result(cache_and_redis):
    client, redis = cache_and_redis
    data = {"results": []}
    await client.set_query_result("my query", "topic", data, tenant_id="t1")
    redis.setex.assert_called_once()
    args = redis.setex.call_args[0]
    assert json.loads(args[2]) == data


@pytest.mark.asyncio
async def test_get_related_result_miss(cache_and_redis):
    client, redis = cache_and_redis
    redis.get.return_value = None
    result = await client.get_related_result("doc1", 2, "topic")
    assert result is None


@pytest.mark.asyncio
async def test_get_related_result_hit(cache_and_redis):
    client, redis = cache_and_redis
    data = {"nodes": []}
    redis.get.return_value = json.dumps(data)
    result = await client.get_related_result("doc1", 2, "topic", tenant_id="t99")
    assert result == data


@pytest.mark.asyncio
async def test_set_related_result(cache_and_redis):
    client, redis = cache_and_redis
    data = {"nodes": ["a", "b"]}
    await client.set_related_result("doc1", 3, "topic", data)
    redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_get_tenant_plan(cache_and_redis):
    client, redis = cache_and_redis
    redis.get.return_value = "pro"
    result = await client.get_tenant_plan("t1")
    assert result == "pro"


@pytest.mark.asyncio
async def test_set_tenant_plan(cache_and_redis):
    client, redis = cache_and_redis
    await client.set_tenant_plan("t1", "free")
    redis.setex.assert_called_once()
    args = redis.setex.call_args[0]
    assert args[2] == "free"


@pytest.mark.asyncio
async def test_incr_quota_first_call(cache_and_redis):
    client, redis = cache_and_redis
    redis.incrby.return_value = 1
    count = await client.incr_quota("t1", "2026-04-27")
    assert count == 1
    redis.expire.assert_called_once()


@pytest.mark.asyncio
async def test_incr_quota_subsequent(cache_and_redis):
    client, redis = cache_and_redis
    redis.incrby.return_value = 5
    count = await client.incr_quota("t1", "2026-04-27")
    assert count == 5
    redis.expire.assert_not_called()


@pytest.mark.asyncio
async def test_decr_quota(cache_and_redis):
    client, redis = cache_and_redis
    await client.decr_quota("t1", "2026-04-27")
    redis.decrby.assert_called_once()


@pytest.mark.asyncio
async def test_invalidate(cache_and_redis):
    client, redis = cache_and_redis
    await client.invalidate("t1", "my query", "topic")
    redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_close(cache_and_redis):
    client, redis = cache_and_redis
    redis.aclose = AsyncMock()
    await client.close()
    redis.aclose.assert_called_once()


def test_tenant_isolation_produces_different_keys():
    from dewie.storage.cache import _make_key

    key_a = _make_key("query", "tenant_a", "search", "topic")
    key_b = _make_key("query", "tenant_b", "search", "topic")
    assert key_a != key_b


def test_make_key_deterministic():
    from dewie.storage.cache import _make_key

    k1 = _make_key("query", "t1", "hello", "world")
    k2 = _make_key("query", "t1", "hello", "world")
    assert k1 == k2
    assert k1.startswith("dewie:query:")


@pytest.mark.asyncio
async def test_inprocess_cache_roundtrip_query_result():
    from dewie.storage.cache import InProcessCacheClient

    client = InProcessCacheClient(ttl=60)
    payload = {"results": [{"id": "doc-1"}]}

    await client.set_query_result("hello", "topic", payload, tenant_id="w1")
    cached = await client.get_query_result("hello", "topic", tenant_id="w1")

    assert cached == payload


@pytest.mark.asyncio
async def test_inprocess_cache_tenant_isolation():
    from dewie.storage.cache import InProcessCacheClient

    client = InProcessCacheClient(ttl=60)
    await client.set_query_result("hello", "topic", {"results": ["a"]}, tenant_id="tenant-a")

    miss = await client.get_query_result("hello", "topic", tenant_id="tenant-b")
    assert miss is None


@pytest.mark.asyncio
async def test_inprocess_cache_ttl_expiry():
    from dewie.storage.cache import InProcessCacheClient

    client = InProcessCacheClient(ttl=1)
    await client.set_related_result("doc-1", 1, "topic", {"nodes": ["x"]})
    assert await client.get_related_result("doc-1", 1, "topic") == {"nodes": ["x"]}

    with patch("dewie.storage.cache.time.time", return_value=time.time() + 5):
        assert await client.get_related_result("doc-1", 1, "topic") is None
