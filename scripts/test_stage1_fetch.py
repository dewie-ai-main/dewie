#!/usr/bin/env python3
"""
Stage 1: Fetch test — download a URL and persist it.

Runs the real WebIngester against a live URL, writes the doc to Postgres,
saves the body to the flat-file body store, and dumps a JSON artifact for
inspection by later stages.

Usage:
    python scripts/test_stage1_fetch.py <url>
    python scripts/test_stage1_fetch.py https://arxiv.org/html/2506.17811v1

The artifact is written to:
    data/test_artifacts/stage1_fetch/<doc_id>.json

Pass that doc_id into test_stage3_enrich.py to force enrichment.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text as _text

from dewie.ingestion.web import WebIngester
from dewie.models.content import ContentStatus
from dewie.storage.body_store import save_body
from dewie.storage.postgres import PostgresClient

ARTIFACT_DIR = Path("data/test_artifacts/stage1_fetch")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    marker = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{marker}] {label}{suffix}")
    return condition


async def run(url: str) -> int:
    print(f"\nStage 1: Fetch\n  URL: {url}\n")
    pg = PostgresClient()
    docs = []

    t0 = time.monotonic()
    async with WebIngester() as ingester:
        async for doc in ingester.fetch(url):
            docs.append(doc)
    elapsed = time.monotonic() - t0

    if not check("fetched at least one document", len(docs) > 0, f"{len(docs)} doc(s)"):
        return 1

    doc = docs[0]
    print(f"\n  doc_id : {doc.id}")
    print(f"  title  : {doc.title[:80]}")
    print(f"  body   : {len(doc.body or '')} chars")
    print(f"  fetch  : {elapsed:.1f}s\n")

    ok = True
    ok &= check("title is non-empty", bool(doc.title and doc.title.strip()))
    ok &= check(
        "body is non-empty",
        bool(doc.body and len(doc.body.strip()) > 200),
        f"{len(doc.body or '')} chars",
    )
    ok &= check("url is set", bool(doc.url))

    # Persist to DB
    doc.status = ContentStatus.PENDING
    t1 = time.monotonic()
    await pg.upsert(doc)
    db_elapsed = time.monotonic() - t1
    ok &= check("upserted to postgres", True, f"{db_elapsed*1000:.0f}ms")

    # Resolve canonical id (ON CONFLICT keeps existing row's id)
    import uuid as _uuid
    async with pg._engine.begin() as conn:
        row = await conn.execute(_text("SELECT id FROM documents WHERE url = :url"), {"url": doc.url})
        result = row.fetchone()
    if result:
        doc.id = _uuid.UUID(str(result[0]))

    # Save body to flat file
    save_body(doc.id, doc.body or "")
    ok &= check("body saved to body_store", True)

    # Also persist body_text column (used by enrichment if body_store is empty)
    await pg.write_body_text(doc.id, doc.body or "")
    ok &= check("body_text written to postgres", True)

    # Verify round-trip from DB
    fetched = await pg.get_by_id(doc.id)
    ok &= check("doc readable from postgres", fetched is not None)
    if fetched:
        ok &= check(
            "title survives round-trip",
            fetched.title == doc.title,
            repr(fetched.title[:60]),
        )

    # Write artifact for subsequent stages
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "doc_id": str(doc.id),
        "url": doc.url,
        "title": doc.title,
        "source": doc.source,
        "body_chars": len(doc.body or ""),
        "status": doc.status.value,
        "fetch_secs": round(elapsed, 2),
    }
    artifact_path = ARTIFACT_DIR / f"{doc.id}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  artifact: {artifact_path}")
    print(f"\n  doc_id for downstream stages: {doc.id}\n")

    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_stage1_fetch.py <url>")
        sys.exit(1)
    sys.exit(asyncio.run(run(sys.argv[1])))
