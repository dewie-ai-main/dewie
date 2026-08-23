# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Redis-backed cache for query results.

Keys are namespaced by query fingerprint + depth to allow partial reuse
across requests that share a common traversal prefix.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import redis.asyncio as aioredis

from dewie.config import settings


def _make_key(namespace: str, *parts: Any) -> str:
    """Build a deterministic cache key from arbitrary parts."""
    raw = ":".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"dewie:{namespace}:{digest}"


class CacheClient:
    """Async Redis client for caching query results and traversal nodes."""

    def __init__(
        self,
        url: str = settings.redis_url,
        ttl: int = settings.cache_ttl_seconds,
    ) -> None:
        self._redis = aioredis.from_url(url, decode_responses=True)
        self._ttl = ttl

    async def get_query_result(
        self, query: str, expand_by: str, tenant_id: str | None = None
    ) -> dict | None:  # type: ignore[type-arg]
        """Return a cached /query result, or None on cache miss.

        tenant_id must be included when auth is enabled so that results from
        one tenant's corpus are never served to another tenant.
        """
        key = _make_key("query", tenant_id or "public", query, expand_by)
        raw = await self._redis.get(key)
        return json.loads(raw) if raw else None

    async def set_query_result(
        self, query: str, expand_by: str, value: dict, tenant_id: str | None = None
    ) -> None:  # type: ignore[type-arg]
        """Cache a /query result with the configured TTL."""
        key = _make_key("query", tenant_id or "public", query, expand_by)
        await self._redis.setex(key, self._ttl, json.dumps(value))

    async def get_related_result(
        self, doc_id: str, depth: int, expand_by: str, tenant_id: str | None = None
    ) -> dict | None:  # type: ignore[type-arg]
        """Return a cached /query/related result, or None on cache miss.

        tenant_id must be included to prevent cross-tenant cache poisoning where
        Tenant B's /traverse response is incorrectly served to Tenant A.
        """
        key = _make_key("related", tenant_id or "public", doc_id, depth, expand_by)
        raw = await self._redis.get(key)
        return json.loads(raw) if raw else None

    async def set_related_result(
        self,
        doc_id: str,
        depth: int,
        expand_by: str,
        value: dict,  # type: ignore[type-arg]
        tenant_id: str | None = None,
    ) -> None:
        """Cache a /query/related result."""
        key = _make_key("related", tenant_id or "public", doc_id, depth, expand_by)
        await self._redis.setex(key, self._ttl, json.dumps(value))

    async def get_tenant_plan(self, tenant_id: str) -> str:
        """Return cached tenant plan string, or None on miss."""
        key = f"dewie:tenant_plan:{tenant_id}"
        return await self._redis.get(key)  # type: ignore[return-value]

    async def set_tenant_plan(self, tenant_id: str, plan: str) -> None:
        """Cache a tenant plan for 5 minutes."""
        key = f"dewie:tenant_plan:{tenant_id}"
        await self._redis.setex(key, 300, plan)

    async def incr_quota(self, tenant_id: str, date_str: str, cost: int = 1) -> int:
        """Atomically add `cost` credits to today's usage counter and return the new total."""
        key = f"dewie:quota:{tenant_id}:{date_str}"
        count = await self._redis.incrby(key, cost)
        if count == cost:  # key was just created
            await self._redis.expire(key, 90000)  # 25-hour TTL covers midnight rollover
        return count

    async def decr_quota(self, tenant_id: str, date_str: str, cost: int = 1) -> None:
        """Roll back a previously incremented quota counter (used on 429 rejection)."""
        key = f"dewie:quota:{tenant_id}:{date_str}"
        await self._redis.decrby(key, cost)

    async def invalidate_user_sessions(self, user_id: str) -> None:
        """
        Force all existing session JWTs for a user to be rejected.
        Stores the current timestamp as the minimum valid `iat` for this user.
        Any token issued before now will fail the middleware check.
        TTL matches the max JWT lifetime (30 days + 1 hour buffer).
        """
        import time as _time

        key = f"dewie:session_min_iat:{user_id}"
        await self._redis.setex(key, 30 * 86400 + 3600, int(_time.time()))

    async def get_session_min_iat(self, user_id: str) -> int | None:
        """Return the minimum valid iat for a user's session, or None if not set."""
        key = f"dewie:session_min_iat:{user_id}"
        val = await self._redis.get(key)
        return int(val) if val is not None else None

    async def invalidate(self, *parts: Any) -> None:
        """Delete a specific cache entry by its key parts."""
        key = _make_key("query", *parts)
        await self._redis.delete(key)

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._redis.aclose()


class InProcessCacheClient:
    """In-memory cache client with TTL semantics for single-process deployments."""

    def __init__(self, ttl: int = settings.cache_ttl_seconds) -> None:
        self._ttl = ttl
        self._store: dict[str, tuple[float, str]] = {}

    def _purge_if_expired(self, key: str) -> None:
        item = self._store.get(key)
        if item is None:
            return
        expires_at, _value = item
        if expires_at < time.time():
            self._store.pop(key, None)

    async def get_query_result(
        self, query: str, expand_by: str, tenant_id: str | None = None
    ) -> dict | None:  # type: ignore[type-arg]
        key = _make_key("query", tenant_id or "public", query, expand_by)
        self._purge_if_expired(key)
        item = self._store.get(key)
        if item is None:
            return None
        return json.loads(item[1])

    async def set_query_result(
        self, query: str, expand_by: str, value: dict, tenant_id: str | None = None
    ) -> None:  # type: ignore[type-arg]
        key = _make_key("query", tenant_id or "public", query, expand_by)
        self._store[key] = (time.time() + self._ttl, json.dumps(value))

    async def get_related_result(
        self, doc_id: str, depth: int, expand_by: str, tenant_id: str | None = None
    ) -> dict | None:  # type: ignore[type-arg]
        key = _make_key("related", tenant_id or "public", doc_id, depth, expand_by)
        self._purge_if_expired(key)
        item = self._store.get(key)
        if item is None:
            return None
        return json.loads(item[1])

    async def set_related_result(
        self,
        doc_id: str,
        depth: int,
        expand_by: str,
        value: dict,
        tenant_id: str | None = None,
    ) -> None:  # type: ignore[type-arg]
        key = _make_key("related", tenant_id or "public", doc_id, depth, expand_by)
        self._store[key] = (time.time() + self._ttl, json.dumps(value))

    async def get_tenant_plan(self, tenant_id: str) -> str | None:
        key = _make_key("tenant_plan", tenant_id)
        self._purge_if_expired(key)
        item = self._store.get(key)
        return item[1] if item is not None else None

    async def set_tenant_plan(self, tenant_id: str, plan: str) -> None:
        key = _make_key("tenant_plan", tenant_id)
        self._store[key] = (time.time() + 300, plan)

    async def incr_quota(self, tenant_id: str, date_str: str, cost: int = 1) -> int:
        key = _make_key("quota", tenant_id, date_str)
        self._purge_if_expired(key)
        item = self._store.get(key)
        current = int(item[1]) if item is not None else 0
        current += cost
        self._store[key] = (time.time() + 90000, str(current))
        return current

    async def decr_quota(self, tenant_id: str, date_str: str, cost: int = 1) -> None:
        key = _make_key("quota", tenant_id, date_str)
        self._purge_if_expired(key)
        item = self._store.get(key)
        if item is None:
            return
        current = max(0, int(item[1]) - cost)
        self._store[key] = (item[0], str(current))

    async def invalidate_user_sessions(self, user_id: str) -> None:
        key = _make_key("session_min_iat", user_id)
        self._store[key] = (time.time() + 30 * 86400 + 3600, str(int(time.time())))

    async def get_session_min_iat(self, user_id: str) -> int | None:
        key = _make_key("session_min_iat", user_id)
        self._purge_if_expired(key)
        item = self._store.get(key)
        return int(item[1]) if item is not None else None

    async def invalidate(self, *parts: Any) -> None:
        key = _make_key("query", *parts)
        self._store.pop(key, None)

    async def close(self) -> None:
        return None


__all__ = ["CacheClient", "InProcessCacheClient"]
