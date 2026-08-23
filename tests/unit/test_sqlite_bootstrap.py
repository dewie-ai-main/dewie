from __future__ import annotations

from datetime import datetime

import pytest

from dewie.models.content import ContentDocument, ContentStatus
from dewie.storage.postgres import PostgresClient


@pytest.mark.asyncio
async def test_sqlite_init_and_basic_upsert_search(tmp_path):
    db_path = tmp_path / "dewie-test.db"
    dsn = f"sqlite+aiosqlite:///{db_path}"

    pg = PostgresClient(dsn=dsn)
    try:
        await pg.init_schema()

        async with pg._engine.connect() as conn:
            users_table = await conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
            )
            assert users_table.fetchone() is not None

        doc = ContentDocument(
            url="https://example.com/sqlite-test",
            title="SQLite startup test",
            summary="SQLite fallback should permit basic startup and search.",
            source="test",
            ingested_at=datetime.utcnow(),
            status=ContentStatus.READY,
            topics=["sqlite"],
            keywords=["startup", "fallback"],
        )
        await pg.upsert(doc)

        loaded = await pg.get_by_id(doc.id)
        assert loaded is not None
        assert loaded.url == doc.url
        assert loaded.status == ContentStatus.READY

        hits = await pg.search("startup", limit=5)
        assert any(str(d.id) == str(doc.id) for d, _score in hits)
    finally:
        await pg.close()
