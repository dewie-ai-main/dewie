# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.
from __future__ import annotations

import asyncio
import logging

from dewie.storage.postgres import PostgresClient

logger = logging.getLogger(__name__)


async def run_edge_rebuild_loop(
    pg: PostgresClient,
    settings,
    stop: asyncio.Event | None = None,
) -> None:
    """Continuously maintain AQ tsvector and capability clusters."""
    sleep_secs = getattr(settings, "edge_rebuild_sleep_secs", 60)

    logger.info("edge rebuild worker started (sleep=%ds)", sleep_secs)

    while not (stop and stop.is_set()):
        try:
            await pg.backfill_aq_tsvec()
            await pg.rebuild_capability_clusters()
            await asyncio.sleep(sleep_secs)
        except asyncio.CancelledError:
            logger.info("edge rebuild worker cancelled")
            return
        except Exception:
            logger.exception("edge rebuild loop tick failed")
            await asyncio.sleep(sleep_secs)
