# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""add query_log.tenant_id

Revision ID: 000000000002
Revises: 000000000001
Create Date: 2026-06-20

query_logger.py's log_query() has always included tenant_id in its INSERT,
but the baseline schema's query_log table never got the matching column —
every query_log write has been silently failing (caught and logged as a
non-fatal warning) on any database created from the baseline migration.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "000000000002"
down_revision = "000000000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE query_log ADD COLUMN IF NOT EXISTS tenant_id UUID")
    op.execute("CREATE INDEX IF NOT EXISTS query_log_tenant_idx ON query_log (tenant_id, ts DESC)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS query_log_tenant_idx")
    op.execute("ALTER TABLE query_log DROP COLUMN IF EXISTS tenant_id")
