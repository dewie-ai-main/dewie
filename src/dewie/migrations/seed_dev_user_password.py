# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""Seed dev@dewie.ai user password hash at migration time.

Generates a bcrypt hash for the 'password' plaintext at runtime instead of
hardcoding it in a SQL file. This avoids exposing the hash in source control.

Safe to run multiple times (idempotent WHERE guard in SQL).
"""

from __future__ import annotations

from sqlalchemy import text

from dewie.local_auth import hash_password

PASSWORD = "password"
DEV_USER_ID = "00000000-0000-0000-0000-000000000002"
HASH = hash_password(PASSWORD)


async def run(conn) -> None:
    await conn.execute(
        text("""
        UPDATE users
        SET password_hash = :hash
        WHERE id = :user_id
          AND (password_hash IS NULL OR password_hash = '')
        """),
        {"hash": HASH, "user_id": DEV_USER_ID},
    )
