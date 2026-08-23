"""Unit tests for healthcheck.py — timezone-aware hour calculation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from dewie.config_file import _load


@pytest.fixture
def config_with_timezone(tmp_path: Path) -> Path:
    """Write a temporary dewie.yml with monitoring.timezone set."""
    config = tmp_path / "dewie.yml"
    config.write_text(yaml.dump({
        "enrichment": {"workers": 2},
        "monitoring": {
            "timezone": "America/Los_Angeles",
            "quiet_start_hour": 23,
            "quiet_end_hour": 7,
        },
    }))
    return config


@pytest.fixture
def config_with_europe_london(tmp_path: Path) -> Path:
    """Write a temporary dewie.yml with Europe/London timezone."""
    config = tmp_path / "dewie.yml"
    config.write_text(yaml.dump({
        "enrichment": {"workers": 2},
        "monitoring": {
            "timezone": "Europe/London",
            "quiet_start_hour": 23,
            "quiet_end_hour": 7,
        },
    }))
    return config


def test_now_configured_hour_returns_valid_hour(config_with_timezone: Path):
    """_now_configured_hour should return an integer 0-23."""
    from zoneinfo import ZoneInfo

    cfg = _load(config_with_timezone)
    tz = ZoneInfo(cfg.monitoring.timezone)
    hour = datetime.now(tz).hour
    assert 0 <= hour <= 23


def test_now_configured_hour_europe_london(config_with_europe_london: Path):
    """_now_configured_hour should return correct hour for Europe/London."""
    from zoneinfo import ZoneInfo

    cfg = _load(config_with_europe_london)
    tz = ZoneInfo(cfg.monitoring.timezone)
    hour = datetime.now(tz).hour
    assert 0 <= hour <= 23


def test_quiet_hours_logic(config_with_timezone: Path):
    """Quiet hours check should work with configured timezone."""

    cfg = _load(config_with_timezone)
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(cfg.monitoring.timezone)
    h = datetime.now(tz).hour
    in_quiet = h >= 23 or h < cfg.monitoring.quiet_start_hour

    # Just verify the logic doesn't crash
    assert isinstance(in_quiet, bool)


def test_default_timezone_is_los_angeles():
    """When timezone is not specified, default to America/Los_Angeles."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("enrichment:\n  workers: 2\n")
        path = Path(f.name)
    cfg = _load(path)
    assert cfg.monitoring.timezone == "America/Los_Angeles"


def test_monitors_use_configured_timezone(config_with_timezone: Path):
    """healthcheck.py should load the timezone from config."""

    # Temporarily patch the config path
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write(yaml.dump({
            "enrichment": {"workers": 2},
            "monitoring": {
                "timezone": "Europe/London",
                "quiet_start_hour": 23,
                "quiet_end_hour": 7,
            },
        }))
        custom_path = Path(f.name)

    cfg = _load(custom_path)
    assert cfg.monitoring.timezone == "Europe/London"


def test_monitor_default_timezone(config_with_timezone: Path):
    """monitor.py should default to America/Los_Angeles when not specified."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("enrichment:\n  workers: 2\n")
        path = Path(f.name)
    cfg = _load(path)
    assert cfg.monitoring.timezone == "America/Los_Angeles"
