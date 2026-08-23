# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""baseline schema

Revision ID: 000000000000
Revises:
Create Date: 2026-06-02
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "000000000000"
down_revision = None
branch_labels = None
depends_on = None


def _embed_dim() -> int:
    """Vector dimension for embedding columns on a FRESH database.

    Sized from the configured embedding model so the default (in-process
    EmbeddingGemma, 768) works out of the box. All embedding columns below are
    created with ``IF NOT EXISTS``/``CREATE TABLE IF NOT EXISTS``, so existing
    databases (already at 1536) are untouched — this only affects the initial
    creation on a new install. Changing the model afterwards still requires a
    re-embed + ``ALTER COLUMN ... TYPE vector(<dims>)``.
    """
    import os

    for var in ("EMBED_DIMENSIONS", "EMBED_OUTPUT_DIMENSIONS"):
        v = os.environ.get(var)
        if v and v.strip().isdigit():
            return int(v)
    try:
        from dewie.config import settings
        from dewie.storage.postgres import _embed_dimensions_for_model

        return _embed_dimensions_for_model(settings.embed_model)
    except Exception:
        return 1536


_DIM = _embed_dim()


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id           UUID        PRIMARY KEY,
            url          TEXT        NOT NULL UNIQUE,
            title        TEXT        NOT NULL DEFAULT '',
            summary      TEXT        NOT NULL DEFAULT '',
            source       TEXT        NOT NULL DEFAULT '',
            ingested_at  TIMESTAMPTZ NOT NULL,
            status       TEXT        NOT NULL DEFAULT 'pending',
            topics       JSONB       NOT NULL DEFAULT '[]',
            keywords     JSONB       NOT NULL DEFAULT '[]',
            entities     JSONB       NOT NULL DEFAULT '[]',
            sentiment    REAL,
            crawl_session UUID,
            search_vec    TSVECTOR,
            user_id       TEXT,
            owner_user_id UUID,
            corpus_id     TEXT
        )
    """)

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS crawl_session UUID")
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS search_vec TSVECTOR")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_topics ON documents USING GIN (topics)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_keywords ON documents USING GIN (keywords)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_entities ON documents USING GIN (entities)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_crawl_session ON documents (crawl_session)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_search_vec ON documents USING GIN (search_vec)")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS enrichment_version INT NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_model TEXT")
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMPTZ")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_enrichment_version ON documents (enrichment_version)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS document_edges (
            source_id    UUID NOT NULL,
            target_id    UUID NOT NULL,
            rel_type     TEXT NOT NULL,
            weight       FLOAT NOT NULL DEFAULT 1.0,
            shared_attrs TEXT[] DEFAULT '{}',
            PRIMARY KEY (source_id, target_id, rel_type)
        )
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_edges_source_weight
        ON document_edges (source_id, weight DESC)
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_edges_target_weight
        ON document_edges (target_id, weight DESC)
    """)

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS author TEXT")
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS reading_level TEXT")
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type TEXT")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_document_type ON documents (document_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_reading_level ON documents (reading_level)")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS priority INT NOT NULL DEFAULT 0")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_priority ON documents (priority)")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embed_summary TEXT NOT NULL DEFAULT ''")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS tone TEXT")

    op.execute(f"ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding vector({_DIM})")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_embedding_ready ON documents (id) WHERE status = 'ready' AND embedding IS NOT NULL")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS aq_tsvec TSVECTOR")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_aq_tsvec_stored ON documents USING GIN (aq_tsvec)")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS answers_questions JSONB NOT NULL DEFAULT '[]'")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_aq_tsvec ON documents USING gin (to_tsvector('english', coalesce(answers_questions::text, ''))) WHERE answers_questions IS NOT NULL AND jsonb_array_length(answers_questions) > 0")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS alternate_terms JSONB NOT NULL DEFAULT '[]'")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_alternate_terms ON documents USING GIN (alternate_terms)")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS enrichment_quality_score SMALLINT")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_enrichment_quality_score ON documents (enrichment_quality_score)")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS retain_body BOOLEAN NOT NULL DEFAULT false")

    op.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id            BIGSERIAL PRIMARY KEY,
            ts            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            source        TEXT NOT NULL DEFAULT 'api',
            question      TEXT NOT NULL,
            model         TEXT,
            session_id    TEXT,
            hops          INTEGER DEFAULT 0,
            hop_trace     JSONB DEFAULT '[]',
            docs_returned JSONB DEFAULT '[]',
            full_results  JSONB DEFAULT NULL,
            answer        TEXT,
            correct       BOOLEAN,
            input_tokens  INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd      NUMERIC(12,8) DEFAULT 0,
            elapsed_ms    INTEGER DEFAULT 0
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS query_log_ts_idx ON query_log (ts DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS query_log_model_idx ON query_log (model)")

    op.execute("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS full_results JSONB DEFAULT NULL")

    op.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id        UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            step          TEXT NOT NULL,
            model         TEXT NOT NULL,
            prompt_hash   TEXT NOT NULL,
            raw_response  TEXT NOT NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (doc_id, step, model)
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_cache_doc_id ON llm_cache (doc_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_cache_step ON llm_cache (step)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS dewie_sources (
            id          UUID PRIMARY KEY,
            name        TEXT NOT NULL UNIQUE,
            type        TEXT NOT NULL,
            config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            enabled     BOOLEAN NOT NULL DEFAULT TRUE,
            created_by  UUID,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            tested_at   TIMESTAMPTZ,
            test_status TEXT,
            test_error  TEXT,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_dewie_sources_enabled ON dewie_sources (enabled)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_errors (
          id          SERIAL PRIMARY KEY,
          doc_id      UUID,
          step        TEXT NOT NULL,
          error_type  TEXT NOT NULL,
          message     TEXT NOT NULL,
          retry_count INT DEFAULT 0,
          resolved    BOOLEAN DEFAULT FALSE,
          created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_errors_created ON pipeline_errors (created_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_errors_step ON pipeline_errors (step)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS system_health (
          key        TEXT PRIMARY KEY,
          value      TEXT NOT NULL,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS body_text TEXT")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_published_at ON documents (published_at)")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS paywall_detected BOOLEAN NOT NULL DEFAULT false")
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS paywall_type TEXT NOT NULL DEFAULT 'none'")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_paywall ON documents (paywall_detected) WHERE paywall_detected = true")

    op.execute("""
        CREATE TABLE IF NOT EXISTS enrichment_benchmark_runs (
            id                  SERIAL PRIMARY KEY,
            ts                  TIMESTAMPTZ DEFAULT now(),
            run_id              UUID NOT NULL,
            doc_id              TEXT NOT NULL,
            doc_title           TEXT,
            doc_source          TEXT,
            model               TEXT NOT NULL,
            backend             TEXT NOT NULL,
            json_valid          BOOLEAN,
            all_fields_present  BOOLEAN,
            has_summary         BOOLEAN,
            has_keywords        BOOLEAN,
            has_entities        BOOLEAN,
            has_topics          BOOLEAN,
            has_sentiment       BOOLEAN,
            has_tone            BOOLEAN,
            has_document_type   BOOLEAN,
            has_reading_level   BOOLEAN,
            has_aq_questions    BOOLEAN,
            aq_question_count   INT,
            summary_word_count  INT,
            keyword_count       INT,
            entity_count        INT,
            latency_ms          INT,
            input_tokens        INT,
            output_tokens       INT,
            cost_usd            FLOAT,
            raw_output          JSONB,
            error               TEXT
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_enrichment_benchmark_run_id ON enrichment_benchmark_runs (run_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_enrichment_benchmark_ts ON enrichment_benchmark_runs (ts DESC)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS tenants (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name       TEXT NOT NULL,
            plan       TEXT NOT NULL DEFAULT 'free',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        INSERT INTO tenants (id, name, plan)
        VALUES ('00000000-0000-0000-0000-000000000001', 'default', 'free')
        ON CONFLICT (id) DO NOTHING
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id    UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            key_hash     TEXT NOT NULL UNIQUE,
            key_prefix   TEXT NOT NULL,
            scopes       TEXT[] NOT NULL DEFAULT '{read}',
            name         TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            revoked_at   TIMESTAMPTZ,
            last_used_at TIMESTAMPTZ
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys (key_prefix)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS access_log (
            id          BIGSERIAL PRIMARY KEY,
            tenant_id   UUID,
            key_id      UUID,
            endpoint    TEXT NOT NULL,
            method      TEXT NOT NULL,
            status_code INT,
            latency_ms  INT,
            ts          TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id            BIGSERIAL PRIMARY KEY,
            tenant_id     UUID,
            actor_id      TEXT,
            action        TEXT NOT NULL,
            resource_type TEXT,
            resource_id   TEXT,
            metadata      JSONB NOT NULL DEFAULT '{}',
            ts            TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log (action)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS document_aq (
            id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id     UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            aq_text    TEXT        NOT NULL,
            embedding  vector(1536),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """.replace("vector(1536)", f"vector({_DIM})"))

    op.execute("CREATE INDEX IF NOT EXISTS idx_document_aq_doc_id ON document_aq (doc_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_document_aq_embedding ON document_aq USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS bench_query_datasets (
            id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            name        TEXT        NOT NULL UNIQUE,
            description TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            metadata    JSONB       NOT NULL DEFAULT '{}'
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS bench_query_items (
            id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            dataset_id       UUID        NOT NULL REFERENCES bench_query_datasets(id) ON DELETE CASCADE,
            query            TEXT        NOT NULL,
            label            TEXT,
            expected_doc_ids JSONB       NOT NULL DEFAULT '[]',
            metadata         JSONB       NOT NULL DEFAULT '{}',
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS bench_query_items_dataset_idx ON bench_query_items (dataset_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS bench_runs (
            id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            name         TEXT        NOT NULL,
            dataset_id   UUID        NOT NULL REFERENCES bench_query_datasets(id),
            conditions   JSONB       NOT NULL DEFAULT '[]',
            description  TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at TIMESTAMPTZ,
            summary      JSONB       NOT NULL DEFAULT '{}'
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS bench_runs_dataset_idx ON bench_runs (dataset_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS bench_run_results (
            id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id            UUID        NOT NULL REFERENCES bench_runs(id) ON DELETE CASCADE,
            query_item_id     UUID        NOT NULL REFERENCES bench_query_items(id),
            condition         TEXT        NOT NULL,
            ranker            TEXT        NOT NULL,
            result_rank       INTEGER     NOT NULL,
            doc_id            TEXT,
            doc_title         TEXT,
            score             DOUBLE PRECISION,
            gap_signal_fired  BOOLEAN,
            gap_signal_text   TEXT,
            result_confidence JSONB,
            relevant          BOOLEAN,
            elapsed_ms        INTEGER,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS bench_run_results_run_condition_idx ON bench_run_results (run_id, condition, query_item_id)")
    op.execute("CREATE INDEX IF NOT EXISTS bench_run_results_query_idx ON bench_run_results (run_id, query_item_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id     UUID        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            email         TEXT        NOT NULL UNIQUE,
            name          TEXT,
            picture       TEXT,
            google_sub    TEXT        UNIQUE,
            apple_sub     TEXT        UNIQUE,
            is_admin      BOOLEAN     NOT NULL DEFAULT false,
            plan          TEXT        NOT NULL DEFAULT 'free',
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_login_at TIMESTAMPTZ
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_users_google_sub ON users (google_sub)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_apple_sub ON users (apple_sub)")

    op.execute("""
        INSERT INTO users (id, tenant_id, email, name, is_admin, plan)
        VALUES ('00000000-0000-0000-0000-000000000002',
                '00000000-0000-0000-0000-000000000001',
                'dev@dewie.ai', 'Dev (Internal)', true, 'enterprise')
        ON CONFLICT (id) DO NOTHING
    """)

    op.execute("""
        DO $$ BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'query_log' AND column_name = 'user_id'
          ) THEN
            ALTER TABLE query_log ADD COLUMN user_id UUID REFERENCES users(id);
            UPDATE query_log
            SET user_id = '00000000-0000-0000-0000-000000000002'
            WHERE user_id IS NULL;
          END IF;
        END $$
    """)

    op.execute("CREATE INDEX IF NOT EXISTS query_log_user_ts_idx ON query_log (user_id, ts DESC)")

    op.execute("""
        DO $$ BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'users' AND column_name = 'activation_status'
          ) THEN
            ALTER TABLE users ADD COLUMN activation_status TEXT NOT NULL DEFAULT 'pending';
            UPDATE users SET activation_status = 'approved'
            WHERE id = '00000000-0000-0000-0000-000000000002';
          END IF;
        END $$
    """)

    op.execute("""
        UPDATE users SET activation_status = 'approved'
        WHERE id = '00000000-0000-0000-0000-000000000002'
          AND activation_status != 'approved'
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_users_activation ON users (activation_status)")

    op.execute("""
        DO $$ BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'users' AND column_name = 'password_hash'
          ) THEN
            ALTER TABLE users ADD COLUMN password_hash TEXT;
          END IF;
        END $$
    """)

    op.execute("""
        DO $$ BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'users' AND column_name = 'password_reset_token'
          ) THEN
            ALTER TABLE users ADD COLUMN password_reset_token TEXT;
          END IF;
        END $$
    """)

    op.execute("""
        DO $$ BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'users' AND column_name = 'password_reset_expires'
          ) THEN
            ALTER TABLE users ADD COLUMN password_reset_expires TIMESTAMPTZ;
          END IF;
        END $$
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS invite_codes (
            id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            code        TEXT        NOT NULL UNIQUE,
            email       TEXT,
            note        TEXT,
            created_by  UUID        REFERENCES users(id),
            used_by     UUID        REFERENCES users(id),
            used_at     TIMESTAMPTZ,
            expires_at  TIMESTAMPTZ,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_invite_codes_code ON invite_codes (code)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_invite_codes_email ON invite_codes (email)")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_status TEXT NOT NULL DEFAULT 'none'")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_chunk_status ON documents (chunk_status) WHERE chunk_status != 'none'")

    op.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id      UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER     NOT NULL,
            text        TEXT        NOT NULL,
            token_count INTEGER,
            embedding   vector(1536),
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (doc_id, chunk_index)
        )
    """.replace("vector(1536)", f"vector({_DIM})"))

    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks (doc_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS search_queue (
            id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            query        TEXT        NOT NULL,
            category     TEXT,
            priority     INTEGER     NOT NULL DEFAULT 5,
            status       TEXT        NOT NULL DEFAULT 'pending',
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            processed_at TIMESTAMPTZ
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_search_queue_status ON search_queue (status, priority DESC, created_at)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS review_queue (
            id         SERIAL      PRIMARY KEY,
            doc_id     UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            reason     TEXT        NOT NULL,
            url        TEXT        NOT NULL,
            source     TEXT        NOT NULL DEFAULT '',
            notes      TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_review_queue_reason ON review_queue (reason)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_review_queue_source ON review_queue (source)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name                TEXT NOT NULL,
            parent_workspace_id UUID REFERENCES workspaces(id),
            sharing_tier        TEXT NOT NULL DEFAULT 'internal_only',
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        INSERT INTO workspaces (id, name, sharing_tier)
        VALUES ('00000000-0000-0000-0000-000000000010', 'org', 'internal_only')
        ON CONFLICT (id) DO NOTHING
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_workspaces_parent ON workspaces (parent_workspace_id) WHERE parent_workspace_id IS NOT NULL")

    op.execute("""
        CREATE TABLE IF NOT EXISTS corpora (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id  UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            name          TEXT NOT NULL,
            slug          TEXT NOT NULL UNIQUE,
            sharing_tier  TEXT NOT NULL DEFAULT 'internal_only',
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        INSERT INTO corpora (id, workspace_id, name, slug, sharing_tier)
        VALUES (
            '00000000-0000-0000-0000-000000000011',
            '00000000-0000-0000-0000-000000000010',
            'Default',
            'default',
            'internal_only'
        )
        ON CONFLICT (id) DO NOTHING
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_corpora_workspace ON corpora (workspace_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_corpora_slug ON corpora (slug)")

    # Migration cleanup: remove SaaS-only tenant infrastructure
    op.execute("DROP POLICY IF EXISTS tenant_isolation_documents ON documents")
    op.execute("DROP POLICY IF EXISTS tenant_isolation_edges ON document_edges")
    op.execute("DROP TABLE IF EXISTS invite_codes CASCADE")

    op.execute("""
        DO $$ BEGIN
          IF EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE table_name = 'api_keys' AND constraint_name = 'api_keys_tenant_id_fkey'
              AND constraint_type = 'FOREIGN KEY'
          ) THEN
            ALTER TABLE api_keys DROP CONSTRAINT api_keys_tenant_id_fkey;
          END IF;
        END $$
    """)

    op.execute("DROP INDEX IF EXISTS idx_api_keys_tenant")
    op.execute("DROP TABLE IF EXISTS tenants CASCADE")
    op.execute("DROP TABLE IF EXISTS access_log")
    op.execute("DROP TABLE IF EXISTS audit_log")

    op.execute("""
        DO $$ BEGIN
          IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'documents' AND column_name = 'tenant_id'
          ) THEN ALTER TABLE documents DROP COLUMN tenant_id; END IF;
        END $$
    """)

    op.execute("""
        DO $$ BEGIN
          IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'document_edges' AND column_name = 'tenant_id'
          ) THEN ALTER TABLE document_edges DROP COLUMN tenant_id; END IF;
        END $$
    """)

    op.execute("""
        DO $$ BEGIN
          IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'enrichment_benchmark_runs' AND column_name = 'tenant_id'
          ) THEN ALTER TABLE enrichment_benchmark_runs DROP COLUMN tenant_id; END IF;
        END $$
    """)

    op.execute("""
        DO $$ BEGIN
          IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'api_keys' AND column_name = 'tenant_id'
          ) THEN ALTER TABLE api_keys DROP COLUMN tenant_id; END IF;
        END $$
    """)

    op.execute("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS workspace_ids UUID[] NOT NULL DEFAULT '{}'")
    # auth.create_api_key inserts user_id; without this column every key
    # creation path (including /api/auth/signup) fails on a fresh install
    op.execute("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id)")

    op.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT")

    op.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS aq_text TEXT")
    op.execute(f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS aq_embedding vector({_DIM})")
    op.execute("CREATE INDEX IF NOT EXISTS document_chunks_aq_embedding_idx ON document_chunks USING hnsw (aq_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")

    op.execute("ALTER TABLE document_aq ADD COLUMN IF NOT EXISTS embedding_model TEXT")

    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS instance_id UUID")
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS shared_at TIMESTAMPTZ")

    op.execute("""
        CREATE TABLE IF NOT EXISTS rss_feeds (
            id                    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            url                   TEXT        NOT NULL,
            name                  TEXT        NOT NULL DEFAULT '',
            corpus_id             TEXT,
            tags                  JSONB       NOT NULL DEFAULT '[]',
            enabled               BOOLEAN     NOT NULL DEFAULT TRUE,
            poll_interval_minutes INTEGER     NOT NULL DEFAULT 60,
            last_polled_at        TIMESTAMPTZ,
            created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
            tenant_id             UUID        NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001'
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_rss_feeds_tenant ON rss_feeds (tenant_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rss_feeds_enabled ON rss_feeds (enabled) WHERE enabled = TRUE")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_rss_feeds_url_tenant ON rss_feeds (url, tenant_id)")


def downgrade() -> None:
    # Baseline migration — no safe downgrade path.
    # Downgrading from this would require dropping all tables in reverse
    # dependency order. In practice, upgrades only.
    pass
