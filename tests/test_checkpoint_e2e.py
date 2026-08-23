"""
Checkpoint E2E test — validates full enrichment pipeline on an existing doc.
Requires a running API and DB. Run with:
    pytest tests/test_checkpoint_e2e.py -v -m integration
"""

import asyncio
import os
import time

import aiohttp
import asyncpg
import pytest

API_URL = os.environ.get("DEWIE_API_URL", "http://localhost:8000")
DB_DSN = os.environ.get(
    "DEWIE_DB_URL", "postgresql://dewie:dewie@localhost:5432/dewie"
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enrichment_checkpoint():
    # 1. Find a ready doc with embedding
    try:
        conn = await asyncpg.connect(DB_DSN)
    except Exception:
        pytest.skip("Database not available")
    row = await conn.fetchrow(
        "SELECT id, url FROM documents WHERE status=$1 AND embedding IS NOT NULL LIMIT 1", "ready"
    )
    if row is None:
        pytest.skip("No ready doc with embedding found in DB")
    doc_id = str(row["id"])

    # 2. Reset to pending
    await conn.execute(
        """
        UPDATE documents SET
            status=$1, embedding=NULL, enriched_at=NULL,
            embed_summary=$2, answers_questions=$3::jsonb,
            topics=$4::jsonb, keywords=$5::jsonb,
            entities=$6::jsonb, summary=$7, tone=NULL
        WHERE id=$8
    """,
        "pending",
        "",
        "[]",
        "[]",
        "[]",
        "[]",
        "",
        row["id"],
    )
    await conn.execute("DELETE FROM pipeline_errors WHERE doc_id=$1", row["id"])
    await conn.close()

    # 3. Priority inject
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{API_URL}/pipeline/enrich/priority", json={"doc_id": doc_id}) as r:
            assert r.status == 200, f"Priority inject failed: {await r.text()}"

        # 4. Poll until ready
        start = time.time()
        result = None
        while time.time() - start < 120:
            await asyncio.sleep(5)
            async with s.get(f"{API_URL}/documents/{doc_id}/inspect") as r:
                if r.status != 200:
                    continue
                doc = await r.json()
                if doc.get("status") == "ready" and doc.get("has_embedding"):
                    result = doc
                    break
                if doc.get("status") == "failed":
                    pytest.fail(f"Doc failed enrichment: {doc}")

    assert result is not None, "Doc never reached ready+embedded within 120s"

    # 5. Assert all enrichment fields
    assert result["has_embedding"] is True
    assert len(result.get("embed_summary", "")) > 50, "embed_summary too short"
    assert len(result.get("answers_questions", [])) >= 2, "Need at least 2 AQ questions"
    assert len(result.get("topics", [])) >= 2, "Need at least 2 topics"
    assert len(result.get("keywords", [])) >= 3, "Need at least 3 keywords"
    assert result.get("sentiment") is not None, "sentiment must be set"
    assert -1.0 <= result["sentiment"] <= 1.0, "sentiment out of range"
    assert result.get("document_type"), "document_type must be set"

    # 6. Verify embedding stored in DB
    conn = await asyncpg.connect(DB_DSN)
    emb = await conn.fetchval("SELECT embedding IS NOT NULL FROM documents WHERE id=$1", row["id"])
    await conn.close()
    assert emb is True, "Embedding not found in DB after enrichment"
