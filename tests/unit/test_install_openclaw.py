# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dewie import install_openclaw as inst

# ── write_skill_file ──────────────────────────────────────────────────────────


def test_write_skill_file_writes_when_absent(tmp_path):
    result = inst.write_skill_file(tmp_path)
    dest = tmp_path / "skills" / "dewie" / "SKILL.md"
    assert result == "written"
    assert dest.read_text() == inst.SKILL_MD


def test_write_skill_file_noop_when_identical(tmp_path):
    inst.write_skill_file(tmp_path)
    dest = tmp_path / "skills" / "dewie" / "SKILL.md"
    mtime_before = dest.stat().st_mtime_ns

    result = inst.write_skill_file(tmp_path)

    assert result == "unchanged"
    assert dest.stat().st_mtime_ns == mtime_before


def test_write_skill_file_refuses_when_differing_without_force(tmp_path):
    dest = tmp_path / "skills" / "dewie" / "SKILL.md"
    dest.parent.mkdir(parents=True)
    dest.write_text("custom content")

    with pytest.raises(inst.InstallError, match="--force"):
        inst.write_skill_file(tmp_path)

    assert dest.read_text() == "custom content"


def test_write_skill_file_overwrites_with_force(tmp_path):
    dest = tmp_path / "skills" / "dewie" / "SKILL.md"
    dest.parent.mkdir(parents=True)
    dest.write_text("custom content")

    result = inst.write_skill_file(tmp_path, force=True)

    assert result == "overwritten"
    assert dest.read_text() == inst.SKILL_MD


def test_write_skill_file_dry_run_writes_nothing(tmp_path):
    result = inst.write_skill_file(tmp_path, dry_run=True)
    dest = tmp_path / "skills" / "dewie" / "SKILL.md"

    assert result == "written"
    assert not dest.exists()


# ── register_mcp_server ────────────────────────────────────────────────────────


def test_register_mcp_server_raises_when_binary_missing():
    with patch("shutil.which", return_value=None):
        with pytest.raises(inst.InstallError, match="not found on PATH"):
            inst.register_mcp_server("http://localhost:10946", "ck_live_x")


def test_register_mcp_server_builds_expected_command():
    with patch("shutil.which", return_value="/usr/bin/openclaw"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            cmd = inst.register_mcp_server("http://localhost:10946/", "ck_live_x")

    called_args = mock_run.call_args[0][0]
    assert called_args == [
        "/usr/bin/openclaw",
        "mcp",
        "add",
        "dewie",
        "--url",
        "http://localhost:10946/api/mcp-stream/mcp",
        "--transport",
        "streamable-http",
        "--header",
        "Authorization=Bearer ck_live_x",
    ]
    assert "dewie" in cmd


def test_register_mcp_server_dry_run_does_not_call_subprocess():
    with patch("shutil.which", return_value="/usr/bin/openclaw"):
        with patch("subprocess.run") as mock_run:
            inst.register_mcp_server("http://localhost:10946", "ck_live_x", dry_run=True)
    mock_run.assert_not_called()


def test_register_mcp_server_raises_on_nonzero_exit():
    with patch("shutil.which", return_value="/usr/bin/openclaw"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
            with pytest.raises(inst.InstallError, match="boom"):
                inst.register_mcp_server("http://localhost:10946", "ck_live_x")


# ── create_key ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_key_returns_raw_key_and_closes_client():
    fake_pg = MagicMock()
    fake_pg.close = AsyncMock(return_value=None)

    with patch("dewie.storage.postgres.PostgresClient", return_value=fake_pg):
        with patch("dewie.auth.create_api_key", AsyncMock(return_value=("ck_live_abc", {}))):
            raw = await inst.create_key("postgresql+asyncpg://x", "test-key")

    assert raw == "ck_live_abc"
    fake_pg.close.assert_awaited_once()
