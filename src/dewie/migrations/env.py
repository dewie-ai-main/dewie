# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""Alembic environment configuration for Dewie schema migrations."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add src/ to path so imports work when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override sqlalchemy.url from POSTGRES_DSN env var, or discrete POSTGRES_HOST/
# PORT/USER/PASSWORD/DB vars (assembled by Settings), if set.
postgres_dsn = os.environ.get("POSTGRES_DSN") or os.environ.get("POSTGRES_URL")
if not postgres_dsn:
    from dewie.config import settings as _settings

    postgres_dsn = _settings.postgres_dsn
if postgres_dsn:
    # Strip +asyncpg driver prefix — alembic env runs synchronously
    sync_dsn = postgres_dsn.replace("postgresql+asyncpg://", "postgresql://").replace("@localhost:", "@127.0.0.1:")
    # Strip query params psycopg2 doesn't understand (e.g. ?ssl=disable)
    if "?" in sync_dsn:
        sync_dsn = sync_dsn.split("?")[0]
    config.set_main_option("sqlalchemy.url", sync_dsn)

target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    ini_section = config.get_section(config.config_ini_section, {})
    # sqlalchemy.url may have been set via set_main_option from POSTGRES_DSN —
    # engine_from_config reads the ini section, not main_option, so promote it
    if "sqlalchemy.url" not in ini_section:
        ini_section["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")
    connectable = engine_from_config(
        ini_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
