# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Crawl coordinator: spins up N worker tasks, each claiming jobs from QueueManager,
fetching pages via FetcherRouter, and feeding documents through the enrichment pipeline.

The coordinator no longer imports from ``api.routes`` — enrichment is delegated
to the injected ``MetadataProcessor``, which is topology-independent.  This
allows the coordinator to run as a standalone process separate from the API server.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse
from uuid import UUID

from dewie.crawler.extractor import extract_links
from dewie.crawler.fetcher import BaseCrawlFetcher, FetcherRouter
from dewie.crawler.queue import QueueManager
from dewie.enrichment.processor import MetadataProcessor
from dewie.storage.postgres import PostgresClient

logger = logging.getLogger(__name__)


@dataclass
class CrawlStats:
    """Accumulated statistics for a crawl session."""

    session_id: UUID
    pages_done: int = 0
    pages_failed: int = 0
    pages_needs_js: int = 0
    urls_enqueued: int = 0
    errors: list[str] = field(default_factory=list)


class CrawlCoordinator:
    """
    Orchestrates concurrent crawl workers for a single crawl session.

    Each worker claims jobs from the ``QueueManager``, fetches pages via the
    ``FetcherRouter``, enqueues discovered child links, and delegates
    enrichment to the ``MetadataProcessor``.

    The coordinator is topology-independent: it can run in-process with the
    API server (current default) or as a standalone worker process consuming
    from a shared queue (target architecture with RabbitMQ).

    Args:
        pg:               Async PostgreSQL client.
        queue:            ``QueueManager`` backed by the crawl_jobs table.
        fetcher:          ``FetcherRouter`` or ``BaseCrawlFetcher`` implementation.
        processor:        ``MetadataProcessor`` for enrichment.  Injected so the
                          coordinator has no dependency on the HTTP route layer.
        session_id:       UUID identifying this crawl session.
        max_depth:        Maximum crawl depth from the seed URL.
        max_pages:        Stop after this many successfully crawled pages.
        concurrency:      Number of concurrent worker tasks.
        politeness_delay: Seconds to wait between fetches per worker.
        same_domain:      If True, only follow links within the seed domain.
    """

    def __init__(
        self,
        pg: PostgresClient,
        queue: QueueManager,
        fetcher: BaseCrawlFetcher,
        processor: MetadataProcessor,
        session_id: UUID,
        max_depth: int,
        max_pages: int,
        concurrency: int,
        politeness_delay: float,
        same_domain: bool,
    ) -> None:
        self._pg = pg
        self._queue = queue
        self._fetcher = fetcher
        self._processor = processor
        self._session_id = session_id
        self._max_depth = max_depth
        self._max_pages = max_pages
        self._concurrency = concurrency
        self._politeness_delay = politeness_delay
        self._same_domain = same_domain
        self._stop_event = asyncio.Event()
        self._stats = CrawlStats(session_id=session_id)
        self._stats_lock = asyncio.Lock()

    async def run(self) -> CrawlStats:
        """Launch worker tasks and a monitor, wait until done."""
        workers = [asyncio.create_task(self._worker(worker_id=i)) for i in range(self._concurrency)]
        monitor = asyncio.create_task(self._monitor())

        await asyncio.gather(*workers)
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass

        return self._stats

    def request_stop(self) -> None:
        """Signal all workers to finish their current job and exit."""
        self._stop_event.set()

    # ── Monitor ───────────────────────────────────────────────────────────────

    async def _monitor(self) -> None:
        """Periodically check progress and set the stop event when done."""
        while True:
            await asyncio.sleep(2)
            done = await self._queue.count_done(self._session_id)
            pending = await self._queue.count_pending_and_processing(self._session_id)
            logger.info(
                "Session %s — done=%d pending/processing=%d",
                self._session_id,
                done,
                pending,
            )
            if done >= self._max_pages:
                logger.info("max_pages=%d reached, stopping.", self._max_pages)
                self._stop_event.set()
                return
            if pending == 0 and done > 0:
                logger.info("Queue drained, stopping.")
                self._stop_event.set()
                return

    # ── Worker ────────────────────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        """Claim and process crawl jobs until the stop event is set."""
        logger.debug("Worker %d started.", worker_id)
        while not self._stop_event.is_set():
            job = await self._queue.dequeue(self._session_id, self._max_depth)
            if job is None:
                # Nothing available yet — wait before retrying
                await asyncio.sleep(2)
                continue

            try:
                doc, html = await self._fetcher.fetch(job.url)
                doc.crawl_session = self._session_id

                # JS-rendered detection
                is_js = isinstance(self._fetcher, FetcherRouter) and self._fetcher.is_js_rendered(
                    doc.body, html
                )

                if is_js:
                    if isinstance(self._fetcher, FetcherRouter):
                        await self._fetcher.on_js_page(job.url, job.id)
                    await self._queue.mark_needs_js(job.id)
                    async with self._stats_lock:
                        self._stats.pages_needs_js += 1
                    continue

                # Enqueue child links before persisting (BFS order)
                if job.depth < self._max_depth:
                    links = extract_links(html, job.url, same_domain=self._same_domain)
                    domain = urlparse(job.url).netloc
                    enqueued = await self._queue.enqueue_batch(
                        links,
                        depth=job.depth + 1,
                        parent_url=job.url,
                        session_id=self._session_id,
                        domain=domain,
                    )
                    async with self._stats_lock:
                        self._stats.urls_enqueued += enqueued

                await self._pg.upsert(doc)
                # Enrichment is fully delegated to the processor — no route imports
                await self._processor.enrich_and_persist(doc, self._pg)
                await self._queue.mark_done(job.id)

                async with self._stats_lock:
                    self._stats.pages_done += 1

                if self._politeness_delay > 0:
                    await asyncio.sleep(self._politeness_delay)

            except Exception as exc:
                err = str(exc)[:2000]
                logger.error("Worker %d failed on %s: %s", worker_id, job.url, err)
                await self._queue.mark_failed(job.id, err)
                async with self._stats_lock:
                    self._stats.pages_failed += 1
                    self._stats.errors.append(f"{job.url}: {err}")

        logger.debug("Worker %d exiting.", worker_id)
