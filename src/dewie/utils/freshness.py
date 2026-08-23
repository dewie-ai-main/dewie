# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Freshness filtering for search queries.

Provides preset-based (pd, pw, pm, py) and custom date-range freshness
filtering that maps to ``start_date`` / ``end_date`` parameters for the
search provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum


class FreshnessPreset(StrEnum):
    """Convenience presets for filtering results by recency."""

    PAST_DAY = "pd"
    PAST_WEEK = "pw"
    PAST_MONTH = "pm"
    PAST_YEAR = "py"


@dataclass(frozen=True)
class DateRange:
    """A closed date range for freshness filtering."""

    start: date
    end: date


# Number of days each preset covers going backward from today.
_PRESET_DAYS: dict[FreshnessPreset, int] = {
    FreshnessPreset.PAST_DAY: 1,
    FreshnessPreset.PAST_WEEK: 7,
    FreshnessPreset.PAST_MONTH: 30,
    FreshnessPreset.PAST_YEAR: 365,
}

# Regex for custom date ranges: YYYY-MM-DDtoYYYY-MM-DD (spaces around "to" are optional)
# Also supports hyphen/dash as a separator: YYYY-MM-DD - YYYY-MM-DD
_CUSTOM_DATE_RANGE_REGEX = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*(?:to|[-])\s*(\d{4}-\d{2}-\d{2})$"
)


def parse_freshness(freshness: str, *, now: date | None = None) -> DateRange | None:
    """
    Parse a freshness string into a :class:`DateRange`.

    Recognised formats:
        - Presets: ``"pd"``, ``"pw"``, ``"pm"``, ``"py"``
        - Custom range: ``"YYYY-MM-DDtoYYYY-MM-DD"`` (whitespace around "to" is allowed)

    Returns ``None`` when the freshness string is unrecognised, so callers can
    fall through to existing ``published_after`` / ``published_before`` fields
    (which already work when set directly).

    Args:
        freshness: A freshness preset or custom date range string.
        now: Reference date for calculating relative presets. Defaults to today.

    Returns:
        A :class:`DateRange` with ``start`` and ``end``, or ``None`` if
        *freshness* does not match any recognised pattern.

    Raises:
        ValueError: If the custom range dates are invalid or start > end.
    """
    if not freshness or not isinstance(freshness, str):
        return None

    freshness_lower = freshness.strip().lower()

    # Check presets first
    if freshness_lower in _PRESET_DAYS:
        ref = now or date.today()
        delta = _PRESET_DAYS[FreshnessPreset(freshness_lower)]
        return DateRange(start=ref - timedelta(days=delta), end=ref)

    # Check custom date range
    match = _CUSTOM_DATE_RANGE_REGEX.match(freshness_lower)
    if match:
        start_str, end_str = match.group(1), match.group(2)
        try:
            start = date.fromisoformat(start_str)
            end = date.fromisoformat(end_str)
        except ValueError:
            return None
        if start > end:
            raise ValueError(
                f"Invalid freshness range: start date ({start}) is after end date ({end})."
            )
        return DateRange(start=start, end=end)

    # Unrecognised — caller may still use published_after/published_before
    return None
