# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING

from dewie.config import settings
from dewie.enrichment.processor import MetadataProcessor
from dewie.ingestion.rss import poll_rss_feed
from dewie.storage.postgres import PostgresClient

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def run_ingest_loop(
    pg: PostgresClient,
    processor: MetadataProcessor | None,
    stop: asyncio.Event | None = None
) -> None:
    """Ingest poller loop for monolith runtime mode."""
    sleep_secs = max(getattr(settings, "enrichment_sleep_secs", 30), 30)
    worker_id = str(uuid.uuid4())[:8]
    warned = False

    logger.info(
        "ingest_poller_loop started",
        extra={"worker_id": worker_id, "sleep_secs": sleep_secs},
    )

    tick = 0
    while not (stop and stop.is_set()):
        tick += 1
        task_id = f"{worker_id}-t{tick}"
        t0 = time.monotonic()
        try:
            # Only feeds whose poll_interval_minutes has elapsed since their
            # last poll — NOT every enabled feed on every tick. poll_rss_feed
            # stamps last_polled_at via mark_feed_polled, so a feed re-appears
            # here only after its interval passes.
            feeds = await pg.get_feeds_due_for_poll()

            if not feeds:
                logger.debug(
                    "ingest_poller_loop: no feeds due for poll",
                    extra={"task_id": task_id},
                )
            else:
                for feed in feeds:
                    await poll_rss_feed(feed, pg, processor)

            await asyncio.sleep(sleep_secs)
            elapsed = time.monotonic() - t0
            logger.debug(
                "ingest_poller_loop tick done",
                extra={"task_id": task_id, "tick": tick, "elapsed_ms": round(elapsed * 1000)},
            )
        except asyncio.CancelledError:
            logger.info(
                "ingest_poller_loop cancelled",
                extra={"worker_id": worker_id, "tick": tick},
            )
            return
        except Exception:
            logger.exception(
                "ingest_poller_loop error",
                extra={"task_id": task_id},
            )
            await asyncio.sleep(sleep_secs)
