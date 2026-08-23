# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Run SQL migration files not yet applied.

Usage:
    python -m dewie.migrations.run_migration

Tracks applied migrations in a `schema_migrations` table (name, applied_at).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

MIGRATIONS_DIR = Path(__file__).parent


async def run() -> None:
    dsn = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://dewie:dewie@localhost:5432/dewie",
    )
    engine = create_async_engine(dsn)

    async with engine.begin() as conn:
        await conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name       TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        )

    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

    async with engine.begin() as conn:
        rows = await conn.execute(text("SELECT name FROM schema_migrations"))
        applied = {r[0] for r in rows.fetchall()}

    for path in sql_files:
        name = path.name
        if name in applied:
            print(f"  skip  {name}")
            continue

        sql = path.read_text()
        async with engine.begin() as conn:
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    await conn.execute(text(stmt))
            await conn.execute(
                text("INSERT INTO schema_migrations (name) VALUES (:name)"),
                {"name": name},
            )
        print(f"  apply {name}")

    py_files = sorted(MIGRATIONS_DIR.glob("*.py"))
    # Only process Python migration files that are not special files
    skip_names = {"run_migration.py", "env.py", "script.py.mako"}
    for path in py_files:
        name = path.name
        if name in skip_names:
            continue
        if name in applied:
            print(f"  skip  {name}")
            continue

        # Import the migration module dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO schema_migrations (name) VALUES (:name)"),
                {"name": name},
            )
        print(f"  apply {name}")

    await engine.dispose()
    print("Migrations done.")


if __name__ == "__main__":
    asyncio.run(run())
