# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
source_filter.py — URL source filtering based on configurable blocklists.

Provides:
  - is_blocked_source(url) — check if a URL's hostname is in the blocklist
  - is_low_quality_source(url) — check if a URL's hostname is in the low-quality list

Usage:
    from dewie.ingestion.source_filter import is_blocked_source, is_low_quality_source
    if is_blocked_source(url):
        reject document
"""

from __future__ import annotations

from urllib.parse import urlparse

from dewie.config_file import cfg


def _get_hostname(url: str) -> str:
    """Extract hostname from a URL, stripping www. prefix."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return hostname.lstrip("www.")


def is_blocked_source(url: str) -> bool:
    """
    Check if the URL's hostname is in the configured blocked_sources list.

    Returns True if the source is blocked (should never be ingested),
    False otherwise.
    """
    if not cfg.ingest.blocked_sources:
        return False
    hostname = _get_hostname(url)
    blocked = cfg.ingest.blocked_sources
    # Direct match
    if hostname in blocked:
        return True
    # Match without www prefix — check both forms
    hostname_no_www = hostname.lstrip("www.")
    return any(source in hostname or hostname in source for source in blocked)


def is_low_quality_source(url: str) -> bool:
    """
    Check if the URL's hostname is in the configured low_quality_sources list.

    Returns True if the source is low quality, False otherwise.
    """
    if not cfg.ingest.low_quality_sources:
        return False
    hostname = _get_hostname(url)
    low_quality = cfg.ingest.low_quality_sources
    # Direct match
    if hostname in low_quality:
        return True
    # Subdomain match — check if any low-quality source is a suffix
    return any(hostname.endswith(source) or source.endswith(hostname) for source in low_quality)
