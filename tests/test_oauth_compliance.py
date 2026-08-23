"""
tests/test_oauth_compliance.py — OAuth scaffold + SOC 2 compliance hook tests.

All unit tests — no live DB or external services required.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# OAuth scaffold (Issue #97)
# ---------------------------------------------------------------------------



@pytest.mark.asyncio
async def test_verify_jwt_noop_when_disabled():
    """verify_jwt returns None immediately when oauth_enabled=False."""
    from dewie.oauth import verify_jwt

    mock_settings = MagicMock()
    mock_settings.oauth_enabled = False

    result = await verify_jwt("some.jwt.token", mock_settings)
    assert result is None


@pytest.mark.asyncio
async def test_verify_bearer_noop_on_non_bearer():
    """verify_bearer_token returns None for non-Bearer authorization strings."""
    from dewie.oauth import verify_bearer_token

    mock_settings = MagicMock()
    mock_settings.oauth_enabled = False

    result = await verify_bearer_token("Basic dXNlcjpwYXNz", mock_settings)
    assert result is None


@pytest.mark.asyncio
async def test_verify_jwt_noop_when_enabled():
    """verify_jwt returns None even when oauth_enabled=True (Clerk integration removed)."""
    from dewie.oauth import verify_jwt

    mock_settings = MagicMock()
    mock_settings.oauth_enabled = True
    mock_settings.clerk_jwks_url = "https://example.clerk.dev/.well-known/jwks.json"

    result = await verify_jwt("header.payload.sig", mock_settings)
    assert result is None


def test_scope_constants_cover_all_dewie_scopes():
    from dewie.auth import ALL_SCOPES, SCOPE_ADMIN, SCOPE_INGEST, SCOPE_READ

    assert SCOPE_READ == "read"
    assert SCOPE_INGEST == "ingest"
    assert SCOPE_ADMIN == "admin"
    assert set(ALL_SCOPES) == {"read", "ingest", "admin"}


# ---------------------------------------------------------------------------
# SOC 2 compliance hooks (Issue #98)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_log_noop_when_disabled():
    """audit_log() is a complete no-op when audit_log_enabled=False."""
    from dewie.compliance import audit_log

    mock_settings = MagicMock()
    mock_settings.audit_log_enabled = False
    mock_pg = MagicMock()
    mock_pg._engine = MagicMock()

    # Should not call the engine at all
    await audit_log(
        mock_pg,
        settings=mock_settings,
        tenant_id=uuid.uuid4(),
        actor_id="test-actor",
        action="doc.ingest",
        resource_type="document",
        resource_id=str(uuid.uuid4()),
    )

    mock_pg._engine.begin.assert_not_called()


@pytest.mark.asyncio
async def test_retention_policy_noop_when_disabled():
    from dewie.compliance import apply_retention_policy

    mock_settings = MagicMock()
    mock_settings.retention_policy_enabled = False
    mock_pg = MagicMock()

    result = await apply_retention_policy(mock_pg, settings=mock_settings)
    assert result == 0


def test_pii_scan_noop_when_disabled():
    from dewie.compliance import scan_pii

    mock_settings = MagicMock()
    mock_settings.pii_scan_enabled = False

    result = scan_pii("my email is john@example.com and SSN 123-45-6789", mock_settings)
    assert result == []


def test_pii_scan_detects_email_when_enabled():
    from dewie.compliance import scan_pii

    mock_settings = MagicMock()
    mock_settings.pii_scan_enabled = True

    result = scan_pii("contact us at hello@example.com for more info", mock_settings)
    assert "email" in result


def test_pii_scan_detects_ssn_when_enabled():
    from dewie.compliance import scan_pii

    mock_settings = MagicMock()
    mock_settings.pii_scan_enabled = True

    result = scan_pii("SSN: 123-45-6789", mock_settings)
    assert "ssn" in result


def test_pii_scan_clean_text():
    from dewie.compliance import scan_pii

    mock_settings = MagicMock()
    mock_settings.pii_scan_enabled = True

    result = scan_pii("The quick brown fox jumps over the lazy dog.", mock_settings)
    assert result == []


@pytest.mark.asyncio
async def test_anomaly_detection_noop_when_disabled():
    from dewie.compliance import check_access_anomalies

    mock_settings = MagicMock()
    mock_settings.anomaly_detection_enabled = False
    mock_pg = MagicMock()

    result = await check_access_anomalies(mock_pg, settings=mock_settings, tenant_id=uuid.uuid4())
    assert result == []


def test_compliance_feature_flags_all_off_by_default():
    from dewie.config import Settings

    s = Settings(_env_file=None)
    assert s.audit_log_enabled is True  # audit logging on by default
    assert s.retention_policy_enabled is False
    assert s.PII_SCAN_ENABLED is False
    assert s.ANOMALY_DETECTION_ENABLED is False
    assert s.RETENTION_DAYS == 365


def test_encryption_at_rest_notes_documented():
    """Sanity check that the encryption notes string exists and covers key points."""
    from dewie.compliance import ENCRYPTION_AT_REST_NOTES

    assert "AES-256" in ENCRYPTION_AT_REST_NOTES
    assert "pgcrypto" in ENCRYPTION_AT_REST_NOTES
    assert "SOC 2" in ENCRYPTION_AT_REST_NOTES
