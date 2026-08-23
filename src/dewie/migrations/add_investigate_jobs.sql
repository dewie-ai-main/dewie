CREATE TABLE IF NOT EXISTS investigate_jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query       TEXT NOT NULL,
    strategy    TEXT NOT NULL DEFAULT 'subquestion',  -- 'subquestion' | 'matrix'
    context     TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | running | done | failed
    plan        JSONB,          -- {sub_questions: [...]} or {entity_list: [...], attributes: [...]}
    result      JSONB,          -- {report: "...", summary: "...", sources: [...], total_facts: N}
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at  TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS investigate_jobs_status_idx ON investigate_jobs(status);
CREATE INDEX IF NOT EXISTS investigate_jobs_created_idx ON investigate_jobs(created_at DESC);
