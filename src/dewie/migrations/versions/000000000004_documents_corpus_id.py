# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""add documents.corpus_id

Revision ID: 000000000004
Revises: 000000000003
Create Date: 2026-06-30

corpus_id is a text tag used to group documents by dataset origin (e.g.
'beir:DATASET', 'customer:NAME', 'test:bench25'). The field exists on
ContentDocument but was never added to the Postgres migration chain.
"""

from alembic import op

revision = "000000000004"
down_revision = "000000000003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS corpus_id TEXT"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_corpus_id ON documents (corpus_id)"
    )


def downgrade() -> None:
    # Only drop the index — the column was also added in baseline (000000000000)
    # so dropping it here would break fresh installs that run downgrade.
    op.execute("DROP INDEX IF EXISTS idx_documents_corpus_id")
