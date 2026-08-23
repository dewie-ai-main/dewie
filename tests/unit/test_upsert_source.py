import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dewie.storage.postgres import PostgresClient


@pytest.fixture
def pg():
    pg = object.__new__(PostgresClient)
    pg._engine = MagicMock()
    pg._session_factory = MagicMock()
    pg._is_sqlite = False
    return pg

@pytest.mark.asyncio
async def test_upsert_source_no_validate(pg):
    # Mock the connection for the upsert
    conn = AsyncMock()
    result = MagicMock()
    # Mock the row returned by the SELECT after upsert
    row = {
        "id": str(uuid.uuid4()),
        "name": "test-source",
        "type": "postgres",
        "config_json": json.dumps({"dsn": "..."}),
        "enabled": True,
        "created_by": None,
        "created_at": "2025-01-01",
        "tested_at": None,
        "test_status": None,
        "test_error": None,
        "updated_at": "2025-01-01"
    }
    # For Postgres, the mapping needs to work
    row_mapping = MagicMock()
    row_mapping.__getitem__.side_effect = lambda k: row[k]
    row_mapping.get.side_effect = lambda k, default=None: row.get(k, default)
    
    # result.mappings().fetchone() returns the row
    result.mappings.return_value.fetchone.return_value = row_mapping
    
    conn.execute = AsyncMock(return_value=result)
    pg._engine.begin.return_value = MagicMock(__aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock(return_value=False))
    
    with patch.object(pg, '_normalize_source_row', return_value={"id": row["id"]}):
        source = await pg.upsert_source(name="test-source", source_type="postgres", config={"dsn": "..."})
        
        assert source["id"] == row["id"]

@pytest.mark.asyncio
async def test_upsert_source_with_postgres_validation_success(pg):
    conn = AsyncMock()
    result = MagicMock()
    row = {
        "id": str(uuid.uuid4()),
        "name": "test-postgres",
        "type": "postgres",
        "config_json": json.dumps({}),
        "enabled": True,
        "created_by": None,
        "created_at": "2025-01-01",
        "tested_at": "2025-01-01",
        "test_status": "ok",
        "test_error": None,
        "updated_at": "2025-01-01"
    }
    row_mapping = MagicMock()
    row_mapping.__getitem__.side_effect = lambda k: row[k]
    row_mapping.get.side_effect = lambda k, default=None: row.get(k, default)
    
    result.mappings.return_value.fetchone.return_value = row_mapping
    conn.execute = AsyncMock(return_value=result)
    
    # Mock begin() for the upsert
    begin_cm = MagicMock(__aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock(return_value=False))
    pg._engine.begin.return_value = begin_cm
    
    # Mock set_source_test_result
    pg.set_source_test_result = AsyncMock()
    
    # Mock connection test
    pg._test_postgres_connection = AsyncMock(return_value=(True, None))
    
    with patch.object(pg, '_normalize_source_row', return_value=row):
        source = await pg.upsert_source(name="test-postgres", source_type="postgres", config={}, validate=True)
        
        assert source["id"] == row["id"]
        pg._test_postgres_connection.assert_called_once_with({})
        pg.set_source_test_result.assert_called_once_with(row["id"], ok=True, error=None)

@pytest.mark.asyncio
async def test_upsert_source_with_mcp_validation_failure(pg):
    conn = AsyncMock()
    result = MagicMock()
    row = {
        "id": str(uuid.uuid4()),
        "name": "test-mcp",
        "type": "mcp",
        "config_json": json.dumps({}),
        "enabled": True,
        "created_by": None,
        "created_at": "2025-01-01",
        "tested_at": "2025-01-01",
        "test_status": "error",
        "test_error": "Connection failed",
        "updated_at": "2025-01-01"
    }
    row_mapping = MagicMock()
    row_mapping.__getitem__.side_effect = lambda k: row[k]
    row_mapping.get.side_effect = lambda k, default=None: row.get(k, default)
    
    result.mappings.return_value.fetchone.return_value = row_mapping
    conn.execute = AsyncMock(return_value=result)
    
    begin_cm = MagicMock(__aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock(return_value=False))
    pg._engine.begin.return_value = begin_cm
    
    pg.set_source_test_result = AsyncMock()
    pg._test_mcp_connection = AsyncMock(return_value=(False, "Connection failed"))
    
    with patch.object(pg, '_normalize_source_row', return_value=row):
        source = await pg.upsert_source(name="test-mcp", source_type="mcp", config={}, validate=True)
        
        assert source["id"] == row["id"]
        pg._test_mcp_connection.assert_called_once_with({})
        pg.set_source_test_result.assert_called_once_with(row["id"], ok=False, error="Connection failed")

@pytest.mark.asyncio
async def test_upsert_source_unsupported_type_skips_validation(pg):
    conn = AsyncMock()
    result = MagicMock()
    row = {
        "id": str(uuid.uuid4()),
        "name": "test-unsupported",
        "type": "unknown",
        "config_json": json.dumps({}),
        "enabled": True,
        "created_by": None,
        "created_at": "2025-01-01",
        "tested_at": None,
        "test_status": None,
        "test_error": None,
        "updated_at": "2025-01-01"
    }
    row_mapping = MagicMock()
    row_mapping.__getitem__.side_effect = lambda k: row[k]
    row_mapping.get.side_effect = lambda k, default=None: row.get(k, default)
    
    result.mappings.return_value.fetchone.return_value = row_mapping
    conn.execute = AsyncMock(return_value=result)
    
    begin_cm = MagicMock(__aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock(return_value=False))
    pg._engine.begin.return_value = begin_cm
    
    pg.set_source_test_result = AsyncMock()
    
    with patch.object(pg, '_normalize_source_row', return_value=row):
        source = await pg.upsert_source(name="test-unsupported", source_type="unknown", config={}, validate=True)
        
        assert source["id"] == row["id"]
        pg.set_source_test_result.assert_not_called()
