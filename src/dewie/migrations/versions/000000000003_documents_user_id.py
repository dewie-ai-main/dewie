# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""add documents.user_id

Revision ID: 000000000003
Revises: 000000000002
Create Date: 2026-06-26

The user ingest route (POST /api/user/ingest) sets user_id on the document
so it can be retrieved via GET /api/user/uploads. The column was present in
the SQLite schema but was never added to the Postgres migration chain.
"""

from alembic import op

revision = "000000000003"
down_revision = "000000000002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id TEXT"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_documents_user_id")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS user_id")
