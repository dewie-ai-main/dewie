-- Issue #34: Add retry_count column to documents table.
-- Retry state now lives on the document row so bulk error resolves cannot reset it.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS retry_count INT NOT NULL DEFAULT 0;
