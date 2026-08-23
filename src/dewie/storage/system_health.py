# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
system_health KV store — lightweight health state persistence.

write_health_kv / read_health_kv write to the system_health table.
All writes are non-fatal (wrapped in try/except).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from dewie.storage.postgres import PostgresClient

logger = logging.getLogger(__name__)


async def write_health_kv(pg: PostgresClient, key: str, value: str) -> None:
    """Insert or update a key-value pair in system_health. Never raises."""
    try:
        is_sqlite = getattr(pg, "_is_sqlite", False)
        if is_sqlite:
            sql = text("""
                INSERT INTO system_health (key, value, updated_at)
                VALUES (:key, :value, datetime('now'))
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = datetime('now')
            """)
        else:
            sql = text("""
                INSERT INTO system_health (key, value, updated_at)
                VALUES (:key, :value, NOW())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            """)
        async with pg._session_factory() as session:
            await session.execute(sql, {"key": key, "value": value})
            await session.commit()
    except Exception:
        logger.warning("system_health.write_health_kv failed (key=%s)", key, exc_info=True)


async def read_health_kv(pg: PostgresClient, key: str) -> dict | None:  # type: ignore[type-arg]
    """
    Read a key from system_health.
    Returns {"value": str, "updated_at": str} or None if not found.
    """
    try:
        sql = text("SELECT value, updated_at FROM system_health WHERE key = :key")
        async with pg._session_factory() as session:
            row = (await session.execute(sql, {"key": key})).mappings().first()
        if row is None:
            return None
        return {
            "value": row["value"],
            "updated_at": row["updated_at"] if isinstance(row["updated_at"], str) else (row["updated_at"].isoformat() if row["updated_at"] else None),
        }
    except Exception:
        logger.warning("system_health.read_health_kv failed (key=%s)", key, exc_info=True)
        return None
