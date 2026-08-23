"""Tests for dewie.utils.freshness — presets, custom ranges, and error handling."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from dewie.utils.freshness import (
    _CUSTOM_DATE_RANGE_REGEX,
    _PRESET_DAYS,
    DateRange,
    FreshnessPreset,
    parse_freshness,
)

# ── FreshnessPreset enum ──────────────────────────────────────────────────────


def test_freshness_preset_values():
    assert FreshnessPreset.PAST_DAY == "pd"
    assert FreshnessPreset.PAST_WEEK == "pw"
    assert FreshnessPreset.PAST_MONTH == "pm"
    assert FreshnessPreset.PAST_YEAR == "py"


def test_preset_days_mapping():
    assert _PRESET_DAYS[FreshnessPreset.PAST_DAY] == 1
    assert _PRESET_DAYS[FreshnessPreset.PAST_WEEK] == 7
    assert _PRESET_DAYS[FreshnessPreset.PAST_MONTH] == 30
    assert _PRESET_DAYS[FreshnessPreset.PAST_YEAR] == 365


# ── parse_freshness — presets ────────────────────────────────────────────────


def test_parse_pd():
    now = date(2025, 1, 15)
    result = parse_freshness("pd", now=now)
    assert isinstance(result, DateRange)
    assert result.start == now - timedelta(days=1)
    assert result.end == now


def test_parse_pw():
    now = date(2025, 1, 15)
    result = parse_freshness("pw", now=now)
    assert isinstance(result, DateRange)
    assert result.start == now - timedelta(days=7)
    assert result.end == now


def test_parse_pm():
    now = date(2025, 1, 15)
    result = parse_freshness("pm", now=now)
    assert isinstance(result, DateRange)
    assert result.start == now - timedelta(days=30)
    assert result.end == now


def test_parse_py():
    now = date(2025, 1, 15)
    result = parse_freshness("py", now=now)
    assert isinstance(result, DateRange)
    assert result.start == now - timedelta(days=365)
    assert result.end == now


def test_parse_preset_uppercase():
    """Presets are case-insensitive."""
    now = date(2025, 1, 15)
    assert parse_freshness("PD", now=now) == parse_freshness("pd", now=now)
    assert parse_freshness("Pw", now=now) == parse_freshness("pw", now=now)
    assert parse_freshness("PM", now=now) == parse_freshness("pm", now=now)
    assert parse_freshness("Py", now=now) == parse_freshness("py", now=now)


def test_parse_preset_with_whitespace():
    """Presets with leading/trailing whitespace are trimmed."""
    now = date(2025, 1, 15)
    assert parse_freshness("  pd  ", now=now) == parse_freshness("pd", now=now)


def test_parse_pd_default_now():
    """Without `now`, uses today's date."""
    result = parse_freshness("pd")
    assert isinstance(result, DateRange)
    assert result.end == date.today()
    assert result.start == date.today() - timedelta(days=1)


# ── parse_freshness — custom date ranges ─────────────────────────────────────


def test_parse_custom_range():
    result = parse_freshness("2024-01-01to2024-06-30")
    assert isinstance(result, DateRange)
    assert result.start == date(2024, 1, 1)
    assert result.end == date(2024, 6, 30)


def test_parse_custom_range_with_spaces():
    """Custom ranges allow whitespace around 'to'."""
    result = parse_freshness("2024-01-01 to 2024-06-30")
    assert isinstance(result, DateRange)
    assert result.start == date(2024, 1, 1)
    assert result.end == date(2024, 6, 30)


def test_parse_custom_range_with_hyphen_separator():
    """Custom ranges allow hyphen instead of 'to'."""
    result = parse_freshness("2024-01-01 - 2024-06-30")
    assert isinstance(result, DateRange)
    assert result.start == date(2024, 1, 1)
    assert result.end == date(2024, 6, 30)


def test_parse_custom_range_single_day():
    """Start and end date can be the same."""
    result = parse_freshness("2024-03-15to2024-03-15")
    assert isinstance(result, DateRange)
    assert result.start == result.end == date(2024, 3, 15)


def test_parse_custom_range_full_year():
    result = parse_freshness("2024-01-01to2024-12-31")
    assert isinstance(result, DateRange)
    assert result.start == date(2024, 1, 1)
    assert result.end == date(2024, 12, 31)


# ── parse_freshness — invalid inputs ─────────────────────────────────────────


def test_parse_none_returns_none():
    assert parse_freshness(None) is None


def test_parse_empty_string_returns_none():
    assert parse_freshness("") is None


def test_parse_whitespace_only_returns_none():
    assert parse_freshness("   ") is None


def test_parse_unknown_string_returns_none():
    assert parse_freshness("foobar") is None


def test_parse_invalid_custom_range_dates():
    """Non-date strings in custom range format return None."""
    assert parse_freshness("not-a-date to not-a-either") is None


def test_parse_start_after_end_raises():
    with pytest.raises(ValueError, match="start date.*after end date"):
        parse_freshness("2024-12-31to2024-01-01")


def test_parse_start_after_end_directly():
    with pytest.raises(ValueError, match="start date.*after end date"):
        parse_freshness("2024-12-31 to 2024-01-01")


# ── DateRange dataclass ──────────────────────────────────────────────────────


def test_date_range_is_frozen():
    dr = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))
    with pytest.raises(Exception):
        dr.start = date(2025, 1, 1)


def test_date_range_equality():
    dr1 = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))
    dr2 = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))
    dr3 = DateRange(start=date(2024, 1, 1), end=date(2025, 1, 1))
    assert dr1 == dr2
    assert dr1 != dr3


# ── Custom date range regex ──────────────────────────────────────────────────


def test_regex_matches_valid_range():
    assert _CUSTOM_DATE_RANGE_REGEX.match("2024-01-01to2024-06-30")
    assert _CUSTOM_DATE_RANGE_REGEX.match("2024-01-01 to 2024-06-30")
    assert _CUSTOM_DATE_RANGE_REGEX.match("2024-01-01 - 2024-06-30")


def test_regex_rejects_invalid():
    assert _CUSTOM_DATE_RANGE_REGEX.match("2024-01-01") is None
    assert _CUSTOM_DATE_RANGE_REGEX.match("not-a-date to not-a-either") is None
    assert _CUSTOM_DATE_RANGE_REGEX.match("pd") is None
    assert _CUSTOM_DATE_RANGE_REGEX.match("") is None
