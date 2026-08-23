# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Data retention policy job for SOC 2 compliance.

This job marks documents as 'expired' when they exceed the retention period.
It is disabled by default (RETENTION_POLICY_ENABLED=False).
When disabled, run_retention_check is a no-op with zero runtime overhead.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)


async def run_retention_check(pg) -> int:
    """
    Run the data retention policy check.

    Queries for documents where ingested_at < NOW() - INTERVAL 'retention_days days'
    and marks them as status='expired'.

    Args:
        pg: PostgresClient instance

    Returns:
        Number of documents marked as expired.

    Note:
        This is a no-op if RETENTION_POLICY_ENABLED is False.
    """
    from dewie.config import settings

    if not settings.retention_policy_enabled:
        return 0

    retention_days = settings.RETENTION_DAYS

    try:
        async with pg._session_factory() as session:
            result = await session.execute(
                text("""
                    UPDATE documents
                    SET status = 'expired'
                    WHERE status NOT IN ('expired', 'failed')
                      AND ingested_at < NOW() - INTERVAL ':retention_days days'
                """),
                {"retention_days": retention_days},
            )
            await session.commit()
            expired_count = result.rowcount
            logger.info(
                "retention_job: marked %d documents as expired (retention_days=%d)",
                expired_count,
                retention_days,
            )
            return expired_count
    except Exception:
        logger.warning("retention_job.run_retention_check failed", exc_info=True)
        return 0