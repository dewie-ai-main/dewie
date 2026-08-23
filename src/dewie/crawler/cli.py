# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
CLI entry point for the Dewie crawler.

Usage
-----
    python -m dewie.crawler.cli \
        --seed https://en.wikipedia.org/wiki/Python_(programming_language) \
        --max-depth 2 \
        --max-pages 50 \
        --same-domain \
        --concurrency 3 \
        --delay 1.0

The fetcher strategy (static vs. JS detection) is selected automatically
by FetcherRouter — no flag is needed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from urllib.parse import urlparse
from uuid import uuid4

from dewie.config import settings
from dewie.crawler.coordinator import CrawlCoordinator
from dewie.crawler.fetcher import FetcherRouter
from dewie.crawler.queue import QueueManager
from dewie.crawler.schema import init_crawl_schema
from dewie.enrichment.processor import MetadataProcessor
from dewie.enrichment.registry import BackendRegistry
from dewie.enrichment.router import EnrichmentRouter
from dewie.storage.postgres import PostgresClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dewie.crawler")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dewie-crawler",
        description="Seed-and-crawl a website into the Dewie pipeline.",
    )
    p.add_argument("--seed", required=True, help="Starting URL for the crawl.")
    p.add_argument(
        "--max-depth",
        type=int,
        default=settings.crawler_max_depth,
        help=f"Maximum link depth to follow (default: {settings.crawler_max_depth}).",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=settings.crawler_max_pages,
        help=f"Stop after indexing this many pages (default: {settings.crawler_max_pages}).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=settings.crawler_concurrency,
        help=f"Number of parallel worker tasks (default: {settings.crawler_concurrency}).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=settings.crawler_politeness_delay,
        help=f"Seconds to sleep between requests per worker (default: {settings.crawler_politeness_delay}).",
    )
    p.add_argument(
        "--same-domain",
        action="store_true",
        default=settings.crawler_same_domain,
        help="Only follow links on the same domain as the seed URL.",
    )
    return p


async def _main(args: argparse.Namespace) -> int:
    pg = PostgresClient()
    registry = BackendRegistry.from_config(settings)
    router = EnrichmentRouter.from_config(settings, registry)
    processor = MetadataProcessor(router=router, registry=registry)

    try:
        logger.info("Initialising schemas…")
        await pg.init_schema()
        await init_crawl_schema(pg)

        session_id = uuid4()
        domain = urlparse(args.seed).netloc
        logger.info("Crawl session: %s", session_id)
        logger.info("Seed URL    : %s", args.seed)
        logger.info(
            "Settings    : max_depth=%d  max_pages=%d  concurrency=%d  delay=%.1fs  same_domain=%s",
            args.max_depth,
            args.max_pages,
            args.concurrency,
            args.delay,
            args.same_domain,
        )

        queue = QueueManager(pg)
        await queue.seed(args.seed, session_id, domain)

        coordinator: CrawlCoordinator | None = None

        def _handle_sigint(*_) -> None:  # type: ignore[no-untyped-def]
            logger.info("SIGINT received — finishing current jobs then stopping…")
            if coordinator is not None:
                coordinator.request_stop()

        signal.signal(signal.SIGINT, _handle_sigint)

        async with FetcherRouter() as fetcher:
            coordinator = CrawlCoordinator(
                pg=pg,
                queue=queue,
                fetcher=fetcher,
                processor=processor,
                session_id=session_id,
                max_depth=args.max_depth,
                max_pages=args.max_pages,
                concurrency=args.concurrency,
                politeness_delay=args.delay,
                same_domain=args.same_domain,
            )
            stats = await coordinator.run()

        print()
        print("═" * 60)
        print(f"  Session UUID : {stats.session_id}")
        print(f"  Pages indexed: {stats.pages_done}")
        print(f"  Pages failed : {stats.pages_failed}")
        print(f"  Needs JS     : {stats.pages_needs_js}")
        print(f"  URLs enqueued: {stats.urls_enqueued}")
        print("═" * 60)
        print()
        print("Cleanup commands:")
        print(f"  DELETE FROM documents  WHERE crawl_session = '{stats.session_id}';")
        print(f"  DELETE FROM crawl_jobs WHERE crawl_session = '{stats.session_id}';")

        return 0

    finally:
        await pg.close()


def cli() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    cli()
