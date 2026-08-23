-- Migration: add_audit_log (Issue #98)
-- Creates the audit_log table for SOC 2 Type I/II audit readiness.
-- This table is append-only — application code must never UPDATE or DELETE rows.

CREATE TABLE IF NOT EXISTS audit_log (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     UUID        NOT NULL,
    actor_id      UUID        NOT NULL,
    action        VARCHAR(255) NOT NULL,
    resource_type VARCHAR(50)  NOT NULL,
    resource_id   UUID        NOT NULL,
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_audit_log_tenant_id  ON audit_log (tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp  ON audit_log (timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_actor_id   ON audit_log (actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action     ON audit_log (action);