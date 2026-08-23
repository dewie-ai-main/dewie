# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""add documents.embedding_full (untruncated embedding, optional)

Revision ID: 000000000001
Revises: 000000000000
Create Date: 2026-06-20

Unindexed by design — pgvector's HNSW/IVFFlat indexes cap out at 2,000
dimensions, well below this model's native 4096, so this column exists
purely for exact-precision reranking of a small ANN candidate set (fetched
by id, not index-scanned), not for indexed similarity search. Only
populated when embed_store_full_vector is enabled in dewie.yml.

The width (4096) matches today's embed_model (Qwen3-Embedding-8B). A future
model with a larger native size would need another migration to widen it.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "000000000001"
down_revision = "000000000000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_full vector(4096)")


def downgrade() -> None:
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS embedding_full")
