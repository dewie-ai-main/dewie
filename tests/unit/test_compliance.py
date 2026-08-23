"""Tests for dewie.compliance — SOC 2 audit hooks."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_settings(**kwargs):
    s = MagicMock()
    s.audit_log_enabled = False
    s.retention_policy_enabled = False
    s.pii_scan_enabled = False
    s.anomaly_detection_enabled = False
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


# ── audit_log ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_noop_when_disabled():
    from dewie.compliance import audit_log

    pg = MagicMock()
    settings = _make_settings(audit_log_enabled=False)
    tenant_id = uuid.uuid4()

    await audit_log(
        pg,
        settings=settings,
        tenant_id=tenant_id,
        actor_id="u1",
        action="doc.ingest",
        resource_type="doc",
        resource_id="r1",
    )
    pg._engine.begin.assert_not_called()


@pytest.mark.asyncio
async def test_audit_log_writes_when_enabled():
    from dewie.compliance import audit_log

    settings = _make_settings(audit_log_enabled=True)
    tenant_id = uuid.uuid4()

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_begin = MagicMock()
    mock_begin.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_begin.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._engine.begin.return_value = mock_begin

    await audit_log(
        pg,
        settings=settings,
        tenant_id=tenant_id,
        actor_id="u1",
        action="doc.delete",
        resource_type="doc",
        resource_id="doc-123",
        metadata={"reason": "test"},
    )

    mock_conn.execute.assert_called_once()
    params = mock_conn.execute.call_args[0][1]
    assert params["action"] == "doc.delete"
    assert params["actor_id"] == "u1"


# ── apply_retention_policy ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retention_policy_noop_when_disabled():
    from dewie.compliance import apply_retention_policy

    pg = MagicMock()
    settings = _make_settings(retention_policy_enabled=False)
    result = await apply_retention_policy(pg, settings=settings)
    assert result == 0
    pg._engine.begin.assert_not_called()


@pytest.mark.asyncio
async def test_retention_policy_executes_when_enabled():
    from dewie.compliance import apply_retention_policy

    settings = _make_settings(retention_policy_enabled=True, retention_days=90)

    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.rowcount = 5
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_begin = MagicMock()
    mock_begin.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_begin.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._engine.begin.return_value = mock_begin

    count = await apply_retention_policy(pg, settings=settings)
    assert count == 5
    params = mock_conn.execute.call_args[0][1]
    assert params["days"] == 90


# ── scan_pii ──────────────────────────────────────────────────────────────────


def test_scan_pii_noop_when_disabled():
    from dewie.compliance import scan_pii

    settings = _make_settings(pii_scan_enabled=False)
    result = scan_pii("SSN: 123-45-6789", settings)
    assert result == []


def test_scan_pii_detects_email():
    from dewie.compliance import scan_pii

    settings = _make_settings(pii_scan_enabled=True)
    result = scan_pii("Contact user@example.com for details.", settings)
    assert "email" in result


def test_scan_pii_detects_ssn():
    from dewie.compliance import scan_pii

    settings = _make_settings(pii_scan_enabled=True)
    result = scan_pii("SSN: 123-45-6789", settings)
    assert "ssn" in result


def test_scan_pii_detects_phone():
    from dewie.compliance import scan_pii

    settings = _make_settings(pii_scan_enabled=True)
    result = scan_pii("Call 555-867-5309 for help.", settings)
    assert "phone" in result


def test_scan_pii_clean_text():
    from dewie.compliance import scan_pii

    settings = _make_settings(pii_scan_enabled=True)
    result = scan_pii("This document discusses machine learning algorithms.", settings)
    assert result == []


def test_scan_pii_multiple_patterns():
    from dewie.compliance import scan_pii

    settings = _make_settings(pii_scan_enabled=True)
    result = scan_pii("Email user@test.com SSN 999-99-9999", settings)
    assert "email" in result
    assert "ssn" in result


# ── check_access_anomalies ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_access_anomalies_noop_when_disabled():
    from dewie.compliance import check_access_anomalies

    pg = MagicMock()
    settings = _make_settings(anomaly_detection_enabled=False)
    result = await check_access_anomalies(pg, settings=settings, tenant_id=uuid.uuid4())
    assert result == []


@pytest.mark.asyncio
async def test_check_access_anomalies_no_spike():
    from dewie.compliance import check_access_anomalies

    settings = _make_settings(anomaly_detection_enabled=True)

    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_row = {"recent": 5, "hourly_avg": 10}
    mock_result.mappings.return_value.fetchone.return_value = mock_row
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._engine.connect.return_value = mock_ctx

    result = await check_access_anomalies(pg, settings=settings, tenant_id=uuid.uuid4())
    assert result == []  # recent (5) < 3.0 * hourly_avg (10) = no spike


@pytest.mark.asyncio
async def test_check_access_anomalies_spike_detected():
    from dewie.compliance import check_access_anomalies

    settings = _make_settings(anomaly_detection_enabled=True)

    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_row = {"recent": 100, "hourly_avg": 5}
    mock_result.mappings.return_value.fetchone.return_value = mock_row
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._engine.connect.return_value = mock_ctx

    result = await check_access_anomalies(pg, settings=settings, tenant_id=uuid.uuid4())
    assert len(result) == 1
    assert result[0]["type"] == "volume_spike"


# ── ENCRYPTION_AT_REST_NOTES ──────────────────────────────────────────────────


def test_encryption_notes_exist():
    from dewie.compliance import ENCRYPTION_AT_REST_NOTES

    assert "encryption" in ENCRYPTION_AT_REST_NOTES.lower()
    assert "AES-256" in ENCRYPTION_AT_REST_NOTES
