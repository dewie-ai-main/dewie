# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Audit logging service for SOC 2 Type I/II compliance.

All audit hooks are disabled by default (AUDIT_LOG_ENABLED=False).
When disabled, log_audit_event is a no-op with zero runtime overhead.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

if TYPE_CHECKING:
    from dewie.storage.postgres import PostgresClient

logger = logging.getLogger(__name__)


async def log_audit_event(
    pg: PostgresClient,
    tenant_id: uuid.UUID,
    actor_id: uuid.UUID,
    action: str,
    resource_type: str,
    resource_id: uuid.UUID,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Log an audit event to the audit_log table.

    This is a no-op if AUDIT_LOG_ENABLED is False (default).
    The audit_log table is append-only — rows are never updated or deleted.
    """
    from dewie.config import settings

    if not settings.audit_log_enabled:
        return

    if metadata is None:
        metadata = {}

    try:
        async with pg._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO audit_log
                        (tenant_id, actor_id, action, resource_type, resource_id, timestamp, metadata)
                    VALUES
                        (CAST(:tenant_id AS UUID),
                         CAST(:actor_id AS UUID),
                         :action,
                         :resource_type,
                         CAST(:resource_id AS UUID),
                         :timestamp,
                         CAST(:metadata AS JSONB))
                """),
                {
                    "tenant_id": str(tenant_id),
                    "actor_id": str(actor_id),
                    "action": action,
                    "resource_type": resource_type,
                    "resource_id": str(resource_id),
                    "timestamp": datetime.utcnow(),
                    "metadata": metadata,
                },
            )
            await session.commit()
    except Exception:
        logger.warning(
            "audit_service.log_audit_event failed (tenant=%s actor=%s action=%s resource=%s:%s)",
            tenant_id,
            actor_id,
            action,
            resource_type,
            resource_id,
            exc_info=True,
        )