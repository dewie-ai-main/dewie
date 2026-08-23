"""Shared fixtures for E2E tests.

E2E tests exercise the full FastAPI middleware chain (auth, rate-limit, quota)
against mocked storage backends — no live DB or Redis required.
"""

from __future__ import annotations

import os
from pathlib import Path

from tests.conftest import *  # noqa: F401,F403 — re-export shared fixtures


def _parse_env_file(path: Path) -> dict[str, str]:
	"""Parse a simple KEY=VALUE env file for local dev test configuration."""
	parsed: dict[str, str] = {}
	if not path.exists():
		return parsed

	for raw_line in path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key:
			parsed[key] = value
	return parsed


def load_dev_env_file(file_name: str = ".env.remote-catalog.local") -> dict[str, str]:
	"""Load local optional test env values without overriding real env vars."""
	repo_root = Path(__file__).resolve().parents[2]
	values = _parse_env_file(repo_root / file_name)
	for key, value in values.items():
		os.environ.setdefault(key, value)
	return values


def get_dev_api_base() -> str:
	"""Return API base URL for optional live dev suites."""
	load_dev_env_file()
	return os.environ.get("DEWIE_TEST_API_BASE", "http://localhost:8000/api").rstrip("/")
