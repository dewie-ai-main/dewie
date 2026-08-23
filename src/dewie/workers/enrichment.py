# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from uuid import UUID

from sqlalchemy import text as _text

from dewie.enrichment.processor import MetadataProcessor
from dewie.models.content import ContentStatus
from dewie.storage.body_store import load_body
from dewie.storage.postgres import PostgresClient

_STUCK_TIMEOUT_HOURS = 24

logger = logging.getLogger(__name__)


async def run_enrichment_loop(
    pg: PostgresClient,
    processor: MetadataProcessor,
    settings,
    stop: asyncio.Event | None = None,
) -> None:
    """Continuously enrich pending documents in small batches.

    Serial mode: set ENRICHMENT_BATCH_SIZE=1 and ENRICHMENT_SLEEP_SECS=0.
    The worker claims one doc, enriches it fully, then immediately claims
    the next — no concurrency, no inter-batch wait, max throughput on a
    single local LLM.
    """
    batch_size = getattr(settings, "enrichment_batch_size", 2)
    sleep_secs = getattr(settings, "enrichment_sleep_secs", 30)
    serial = batch_size == 1 and sleep_secs == 0
    worker_id = str(uuid.uuid4())[:8]

    if serial:
        logger.info(
            "enrichment_loop started in SERIAL mode",
            extra={"worker_id": worker_id, "batch_size": 1, "sleep_secs": 0},
        )
    else:
        logger.info(
            "enrichment_loop started",
            extra={"worker_id": worker_id, "batch_size": batch_size, "sleep_secs": sleep_secs},
        )

    tick = 0
    while not (stop and stop.is_set()):
        tick += 1
        task_id = f"{worker_id}-t{tick}"
        t0 = time.monotonic()
        try:
            # Reap stuck processing docs — reset to pending after timeout.
            # Uses enriched_at IS NULL as a proxy for "never finished" since we
            # lack a processing_started_at column. ingested_at guards against
            # reaping docs that were only recently claimed.
            try:
                # Dialect-specific "N hours ago": Postgres INTERVAL vs SQLite datetime().
                if getattr(pg, "_is_sqlite", False):
                    cutoff_expr = f"datetime('now', '-{_STUCK_TIMEOUT_HOURS} hours')"
                else:
                    cutoff_expr = f"NOW() - INTERVAL '{_STUCK_TIMEOUT_HOURS} hours'"
                async with pg._engine.begin() as _conn:
                    reaped = await _conn.execute(_text(
                        "UPDATE documents SET status = 'pending'"
                        " WHERE status = 'processing'"
                        " AND enriched_at IS NULL"
                        f" AND ingested_at < {cutoff_expr}"
                        " RETURNING id"
                    ))
                    reaped_ids = [str(r[0]) for r in reaped.fetchall()]
                if reaped_ids:
                    logger.warning(
                        "enrichment_loop reaped %d stuck processing doc(s): %s",
                        len(reaped_ids),
                        reaped_ids,
                    )
            except NotImplementedError:
                pass  # SQLite does not support this syntax — single-worker, no stuck jobs
            except Exception:
                logger.warning("stuck-job reaper failed", exc_info=True)

            pending = await pg.get_pending_docs(limit=batch_size)
            if not pending:
                # Queue empty — always sleep before next poll to avoid busy-loop
                await asyncio.sleep(max(sleep_secs, 3))
                continue

            logger.debug(
                "enrichment_loop tick started",
                extra={"task_id": task_id, "tick": tick, "doc_count": len(pending)},
            )

            for doc_id_str in pending:
                if stop and stop.is_set():
                    break
                doc_task_id = f"{task_id}-{doc_id_str[:8]}"
                t1 = time.monotonic()
                try:
                    doc_id = UUID(doc_id_str)
                    doc = await pg.get_by_id(doc_id)
                    if doc is None:
                        continue

                    doc.body = load_body(str(doc.id)) or ""
                    logger.debug(
                        "enrichment_loop doc started",
                        extra={"task_id": doc_task_id, "doc_id": doc_id_str},
                    )
                    await processor.enrich_and_persist(doc, pg)
                    elapsed_doc = time.monotonic() - t1
                    logger.info(
                        "enrichment_loop doc done",
                        extra={
                            "task_id": doc_task_id,
                            "doc_id": doc_id_str,
                            "elapsed_ms": round(elapsed_doc * 1000),
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    elapsed_doc = time.monotonic() - t1
                    logger.exception(
                        "enrichment_loop doc failed",
                        extra={
                            "task_id": doc_task_id,
                            "doc_id": doc_id_str,
                            "elapsed_ms": round(elapsed_doc * 1000),
                        },
                    )
                    try:
                        await pg.mark_status(UUID(doc_id_str), ContentStatus.PENDING)
                    except Exception:
                        logger.warning(
                            "enrichment_loop could not reset doc to pending",
                            extra={"task_id": doc_task_id, "doc_id": doc_id_str},
                        )

            elapsed = time.monotonic() - t0
            logger.info(
                "enrichment_loop tick done",
                extra={
                    "task_id": task_id,
                    "tick": tick,
                    "doc_count": len(pending),
                    "elapsed_ms": round(elapsed * 1000),
                },
            )

            # Inter-batch sleep — skipped in serial mode (sleep_secs=0)
            if sleep_secs > 0:
                await asyncio.sleep(sleep_secs)
        except asyncio.CancelledError:
            logger.info(
                "enrichment_loop cancelled",
                extra={"worker_id": worker_id, "tick": tick},
            )
            return
        except Exception:
            elapsed = time.monotonic() - t0
            logger.exception(
                "enrichment_loop tick failed",
                extra={"task_id": task_id, "tick": tick, "elapsed_ms": round(elapsed * 1000)},
            )
            await asyncio.sleep(max(sleep_secs, 3))
