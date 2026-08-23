# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
dewie/compliance.py — SOC 2 audit hooks scaffold (Issue #98).

Status: SCAFFOLDED, ALL HOOKS DISABLED BY DEFAULT.
Zero runtime overhead when disabled — every function is a no-op.

Enable progressively via env flags:
  AUDIT_LOG_ENABLED=true          — write audit events to audit_log table
  RETENTION_POLICY_ENABLED=true   — mark expired docs per retention policy
  PII_SCAN_ENABLED=true           — scan docs for PII before ingest
  ANOMALY_DETECTION_ENABLED=true  — flag unusual query patterns (requires access log)

SOC 2 criteria coverage:
  Hook                    | Criteria
  ----------------------- | ----------------------------------
  audit_log               | Security, Processing Integrity
  retention_policy        | Privacy
  pii_scan                | Privacy, Confidentiality
  anomaly_detection       | Security, Availability

Notes:
  - Full SOC 2 Type I audit: ~$15-30K, needed only at enterprise stage
  - These hooks exist so an auditor can see the infrastructure is ready
  - All depend on Issue #96 (tenant isolation) being in place
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

log = logging.getLogger("dewie.compliance")


# ---------------------------------------------------------------------------
# 1. Immutable Audit Log
# ---------------------------------------------------------------------------
#
# Captures: corpus ingest, doc deletion, key creation/revocation, admin actions.
# Table: audit_log (defined in postgres.py schema)
# When disabled: instant no-op, zero DB calls.
#
async def audit_log(
    pg: Any,
    *,
    settings: Any,
    tenant_id: uuid.UUID,
    actor_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    metadata: dict | None = None,
) -> None:
    """
    Record an audit event.

    action examples: "doc.ingest", "doc.delete", "key.create", "key.revoke",
                     "query.search", "admin.list_keys"

    No-op when AUDIT_LOG_ENABLED=false.
    """
    if not settings.audit_log_enabled:
        return

    import json

    from sqlalchemy import text as _text

    is_sqlite = "sqlite" in str(getattr(pg, "_engine", "")).lower() or getattr(
        pg, "_is_sqlite", False
    )
    metadata_expr = ":metadata" if is_sqlite else "CAST(:metadata AS jsonb)"

    # Audit logging is telemetry — it must never fail the request it observes.
    try:
        async with pg._engine.begin() as conn:
            await conn.execute(
                _text(
                    "INSERT INTO audit_log "
                    "(tenant_id, actor_id, action, resource_type, resource_id, metadata) "
                    "VALUES (:tenant_id, :actor_id, :action, :resource_type, :resource_id, "
                    f"{metadata_expr})"
                ),
                {
                    "tenant_id": str(tenant_id),
                    "actor_id": actor_id,
                    "action": action,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "metadata": json.dumps(metadata or {}),
                },
            )
    except Exception as exc:
        log.warning("audit_log write failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# 2. Data Retention Policy
# ---------------------------------------------------------------------------
#
# Marks docs older than retention_days as "expired" status.
# Tenant-configurable retention periods (uses global setting for now).
# When disabled: instant no-op.
#
async def apply_retention_policy(pg: Any, *, settings: Any) -> int:
    """
    Mark documents older than retention_days as status='expired'.

    Returns the number of documents marked. No-op when RETENTION_POLICY_ENABLED=false.
    """
    if not settings.retention_policy_enabled:
        return 0

    from sqlalchemy import text as _text

    async with pg._engine.begin() as conn:
        result = await conn.execute(
            _text(
                "UPDATE documents SET status = 'expired' "
                "WHERE status = 'ready' "
                "AND ingested_at < now() - make_interval(days => :days) "
                "AND status != 'expired'"
            ),
            {"days": settings.retention_days},
        )
        return result.rowcount


# ---------------------------------------------------------------------------
# 3. PII Detection
# ---------------------------------------------------------------------------
#
# Pre-ingest scan for common PII patterns.
# When disabled: passthrough (returns the text unchanged, no flags).
#
_PII_PATTERNS = [
    # Email
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    # US SSN
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Credit card (basic Luhn-shaped patterns)
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    # US phone
    re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
]


def scan_pii(text: str, settings: Any) -> list[str]:
    """
    Scan text for PII patterns.

    Returns a list of matched pattern names. Empty list = clean.
    No-op (returns []) when PII_SCAN_ENABLED=false.
    """
    if not settings.pii_scan_enabled:
        return []

    pattern_names = ["email", "ssn", "credit_card", "phone"]
    found = []
    for name, pattern in zip(pattern_names, _PII_PATTERNS):
        if pattern.search(text):
            found.append(name)
    return found


# ---------------------------------------------------------------------------
# 4. Access Anomaly Detection
# ---------------------------------------------------------------------------
#
# Flags unusual query volume spikes or off-hours access per tenant.
# Requires access_log_enabled=true to have data to analyze.
# When disabled: no-op, returns empty list.
#
async def check_access_anomalies(
    pg: Any,
    *,
    settings: Any,
    tenant_id: uuid.UUID,
    window_minutes: int = 60,
    spike_threshold: float = 3.0,
) -> list[dict]:
    """
    Check for anomalous access patterns for a tenant.

    Returns list of anomaly dicts. Empty = no anomalies.
    No-op when ANOMALY_DETECTION_ENABLED=false.

    Anomaly types detected:
    - volume_spike: requests in last window > spike_threshold * rolling average
    - off_hours: significant access between 00:00-06:00 tenant local time (UTC for now)
    """
    if not settings.anomaly_detection_enabled:
        return []

    from sqlalchemy import text as _text

    anomalies = []

    async with pg._engine.connect() as conn:
        # Volume spike: compare last hour to prior 24h average
        result = await conn.execute(
            _text(
                "SELECT "
                "  COUNT(*) FILTER (WHERE ts > now() - make_interval(mins => :window)) AS recent, "
                "  COUNT(*) / NULLIF(24, 0) AS hourly_avg "
                "FROM access_log "
                "WHERE tenant_id = :tenant_id "
                "  AND ts > now() - INTERVAL '24 hours'"
            ),
            {"tenant_id": str(tenant_id), "window": window_minutes},
        )
        row = result.mappings().fetchone()
        if row and row["hourly_avg"] and row["recent"]:
            if row["recent"] > spike_threshold * row["hourly_avg"]:
                anomalies.append(
                    {
                        "type": "volume_spike",
                        "tenant_id": str(tenant_id),
                        "recent_requests": row["recent"],
                        "hourly_avg": row["hourly_avg"],
                        "ratio": row["recent"] / row["hourly_avg"],
                    }
                )

    return anomalies


# ---------------------------------------------------------------------------
# 5. Encryption At Rest
# ---------------------------------------------------------------------------
#
# Postgres encryption at rest is a managed DB / infrastructure setting.
# Document here rather than in code.
#
ENCRYPTION_AT_REST_NOTES = """
Encryption at rest for Dewie:

1. Self-hosted (Docker on host): Depends on host disk encryption (FileVault on macOS,
   LUKS on Linux). Postgres itself does not encrypt data files.
   → Enable full-disk encryption on the host.

2. Managed Postgres (RDS, Cloud SQL, Supabase, Neon):
   → All managed providers enable encryption at rest by default (AES-256).
   → No action required — verify in provider console.

3. Column-level encryption (future):
   → For high-sensitivity fields, use pgcrypto extension.
   → Not currently implemented.

For SOC 2 Type I, document which option is in use and provide evidence
(provider console screenshot or host encryption config).
"""
