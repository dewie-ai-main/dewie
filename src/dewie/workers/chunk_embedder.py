# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.
from __future__ import annotations

import asyncio
import logging
import time
import uuid

from dewie.enrichment.chunk_embedder import run_once
from dewie.storage.postgres import PostgresClient

logger = logging.getLogger(__name__)


async def run_chunk_embedder_loop(
    pg: PostgresClient,
    settings,
    stop: asyncio.Event | None = None,
) -> None:
    """Continuously chunk and embed unprocessed long-form documents."""
    batch_size = getattr(settings, "chunk_embedder_batch_size", 100)
    sleep_secs = getattr(settings, "chunk_embedder_sleep_secs", 10)
    worker_id = str(uuid.uuid4())[:8]

    logger.info(
        "chunk_embedder_loop started",
        extra={"worker_id": worker_id, "batch_size": batch_size, "sleep_secs": sleep_secs},
    )

    tick = 0
    while not (stop and stop.is_set()):
        tick += 1
        task_id = f"{worker_id}-t{tick}"
        t0 = time.monotonic()
        try:
            logger.debug(
                "chunk_embedder_loop tick started",
                extra={"task_id": task_id, "tick": tick},
            )
            processed = await run_once(pg, batch_size=batch_size)
            elapsed = time.monotonic() - t0
            logger.info(
                "chunk_embedder_loop tick done",
                extra={
                    "task_id": task_id,
                    "tick": tick,
                    "processed": processed,
                    "elapsed_ms": round(elapsed * 1000),
                },
            )
            await asyncio.sleep(sleep_secs)
        except asyncio.CancelledError:
            logger.info(
                "chunk_embedder_loop cancelled",
                extra={"worker_id": worker_id, "tick": tick},
            )
            return
        except Exception:
            elapsed = time.monotonic() - t0
            logger.exception(
                "chunk_embedder_loop tick failed",
                extra={"task_id": task_id, "tick": tick, "elapsed_ms": round(elapsed * 1000)},
            )
            await asyncio.sleep(sleep_secs)
