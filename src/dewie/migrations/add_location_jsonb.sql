-- Add nullable location JSONB column to documents table for future geo-expansion.
-- This is a stub — no location-based search is active yet.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS location JSONB NULL;
