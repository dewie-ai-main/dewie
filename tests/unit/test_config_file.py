"""Unit tests for dewie.config_file — dewie.yml loader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from dewie.config_file import _load


def test_defaults_when_file_missing():
    cfg = _load(Path("/nonexistent/dewie.yml"))
    assert cfg.enrichment.workers == 3
    assert cfg.enrichment.auto_restart is True
    assert cfg.enrichment.max_restarts_per_hour == 3
    assert cfg.ingest.rss_enabled is True
    assert cfg.ingest.wiki_enabled is False
    assert cfg.monitoring.quiet_start_hour == 23
    assert cfg.monitoring.alert_dedup_seconds == 1800
    assert cfg.monitoring.timezone == "America/Los_Angeles"


def test_partial_override():
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("enrichment:\n  workers: 4\n")
        path = Path(f.name)
    cfg = _load(path)
    assert cfg.enrichment.workers == 4
    assert cfg.enrichment.auto_restart is True  # default preserved


def test_full_config():
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("""
enrichment:
  workers: 3
  auto_restart: false
  max_restarts_per_hour: 5
ingest:
  rss_enabled: false
  wiki_enabled: true
monitoring:
  quiet_start_hour: 22
  quiet_end_hour: 8
  alert_dedup_seconds: 900
""")
        path = Path(f.name)
    cfg = _load(path)
    assert cfg.enrichment.workers == 3
    assert cfg.enrichment.auto_restart is False
    assert cfg.enrichment.max_restarts_per_hour == 5
    assert cfg.ingest.rss_enabled is False
    assert cfg.ingest.wiki_enabled is True
    assert cfg.monitoring.quiet_start_hour == 22
    assert cfg.monitoring.alert_dedup_seconds == 900


def test_workers_clamped():
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("enrichment:\n  workers: 0\n")
        path = Path(f.name)
    with pytest.raises(Exception):  # pydantic validation error
        _load(path)


def test_quiet_hours_equal_rejected():
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("monitoring:\n  quiet_start_hour: 9\n  quiet_end_hour: 9\n")
        path = Path(f.name)
    with pytest.raises(Exception):  # pydantic validation error — equal = 24h suppression
        _load(path)


def test_quiet_hours_equal_rejected_v2():
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("monitoring:\n  quiet_start_hour: 9\n  quiet_end_hour: 9\n")
        path = Path(f.name)
    with pytest.raises(Exception):  # equal = 24h suppression, not allowed
        _load(path)


def test_timezone_default():
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("enrichment:\n  workers: 2\n")
        path = Path(f.name)
    cfg = _load(path)
    assert cfg.monitoring.timezone == "America/Los_Angeles"


def test_custom_timezone():
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("monitoring:\n  timezone: Europe/London\n")
        path = Path(f.name)
    cfg = _load(path)
    assert cfg.monitoring.timezone == "Europe/London"


def test_default_timezone_when_not_specified():
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("monitoring:\n  quiet_start_hour: 22\n  quiet_end_hour: 6\n")
        path = Path(f.name)
    cfg = _load(path)
    assert cfg.monitoring.timezone == "America/Los_Angeles"


def test_timezone_custom():
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("monitoring:\n  timezone: Europe/London\n  quiet_start_hour: 22\n  quiet_end_hour: 8\n")
        path = Path(f.name)
    cfg = _load(path)
    assert cfg.monitoring.timezone == "Europe/London"
    assert cfg.monitoring.quiet_start_hour == 22
    assert cfg.monitoring.quiet_end_hour == 8


def test_timezone_full_config():
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("""
monitoring:
  timezone: America/New_York
  quiet_start_hour: 22
  quiet_end_hour: 8
  alert_dedup_seconds: 600
""")
        path = Path(f.name)
    cfg = _load(path)
    assert cfg.monitoring.timezone == "America/New_York"
    assert cfg.monitoring.quiet_start_hour == 22
    assert cfg.monitoring.quiet_end_hour == 8
    assert cfg.monitoring.alert_dedup_seconds == 600


def test_full_config_with_timezone():
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        f.write("""
enrichment:
  workers: 3
  auto_restart: false
  max_restarts_per_hour: 5
monitoring:
  timezone: Asia/Tokyo
  quiet_start_hour: 22
  quiet_end_hour: 6
  alert_dedup_seconds: 900
""")
        path = Path(f.name)
    cfg = _load(path)
    assert cfg.monitoring.timezone == "Asia/Tokyo"
    assert cfg.monitoring.quiet_start_hour == 22
    assert cfg.monitoring.quiet_end_hour == 6
