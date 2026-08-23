-- Migration: add document_type classification field (issue #8)
-- Adds the document_type column to existing databases.
-- New databases already have this column via the CREATE TABLE statement
-- in postgres.py, so all statements use IF NOT EXISTS / IF EXISTS guards.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS document_type TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_document_type
    ON documents (document_type);
