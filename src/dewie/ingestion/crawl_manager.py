# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Global crawl resource manager.

Tracks active ingest sources, enforces a shared rate-limit ceiling, and
allows per-source pause/resume.  All state is in-process; an optional JSON
status file at STATUS_FILE gives external observers a snapshot.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

STATUS_FILE = Path("/tmp/dewie-crawl-manager.json")


@dataclass
class SourceState:
    name: str
    paused: bool = False
    docs_ingested: int = 0
    last_seen: float = 0.0  # Unix timestamp
    error_count: int = 0


class CrawlManager:
    """
    Global crawl resource manager.

    Usage::

        manager = CrawlManager(max_rps=5.0)
        async with manager.acquire("rss:techcrunch"):
            doc = await fetch(url)
            manager.record_ingested("rss:techcrunch")
    """

    def __init__(self, max_rps: float = 5.0) -> None:
        self._max_rps = max_rps
        # Token bucket: capacity = max_rps * 2, starts full.
        self._capacity = max_rps * 2
        self._tokens: float = self._capacity
        self._last_refill: float = time.monotonic()
        self._bucket_lock = asyncio.Lock()

        self._sources: dict[str, SourceState] = {}
        # One asyncio.Event per source; set == not paused (running).
        self._resume_events: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Source registry
    # ------------------------------------------------------------------

    def register(self, source_name: str) -> None:
        """Register a crawl source (idempotent)."""
        if source_name not in self._sources:
            self._sources[source_name] = SourceState(name=source_name)
            event = asyncio.Event()
            event.set()  # not paused by default
            self._resume_events[source_name] = event

    def pause(self, source_name: str) -> None:
        """Pause a source. acquire() will block until resumed."""
        self.register(source_name)
        self._sources[source_name].paused = True
        self._resume_events[source_name].clear()

    def resume(self, source_name: str) -> None:
        """Resume a previously paused source."""
        self.register(source_name)
        self._sources[source_name].paused = False
        self._resume_events[source_name].set()

    def record_ingested(self, source_name: str, count: int = 1) -> None:
        """Increment ingested count and update last_seen."""
        self.register(source_name)
        self._sources[source_name].docs_ingested += count
        self._sources[source_name].last_seen = time.time()

    def record_error(self, source_name: str) -> None:
        """Increment error count for a source."""
        self.register(source_name)
        self._sources[source_name].error_count += 1

    def status(self) -> dict:
        """Return current state of all registered sources."""
        return {name: asdict(state) for name, state in self._sources.items()}

    def save_status(self) -> None:
        """Write status snapshot to STATUS_FILE (silently swallows write errors)."""
        try:
            STATUS_FILE.write_text(json.dumps(self.status(), indent=2))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Token-bucket rate limiter
    # ------------------------------------------------------------------

    async def _acquire_token(self) -> None:
        """Block until a token is available, then consume one."""
        while True:
            async with self._bucket_lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._max_rps)
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # How long until the next token arrives?
                wait = (1.0 - self._tokens) / self._max_rps

            await asyncio.sleep(wait)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def acquire(self, source_name: str) -> AsyncIterator[None]:
        """
        Async context manager.

        Blocks if the source is paused or the global rate limit is exceeded.
        Automatically registers unknown sources on first use.
        """
        self.register(source_name)

        # Block while paused.
        await self._resume_events[source_name].wait()

        # Consume a token (blocks if bucket is empty).
        await self._acquire_token()

        yield
