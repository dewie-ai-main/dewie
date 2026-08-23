"""Tests for the standalone status page at static/status.html.

Verifies:
  1. The back link to the admin page exists.
  2. The back link has the correct href.
  3. The back link styling is present.
  4. The page still has all expected structural elements.
"""

from __future__ import annotations

from pathlib import Path

_STATIC = Path(__file__).parents[2] / "static"


# ── Back button/link ──────────────────────────────────────────────────────────


def test_status_page_has_back_link():
    """status.html should have a 'Back to Admin' link."""
    content = (_STATIC / "status.html").read_text()
    assert '← Back to Admin' in content, "Missing '← Back to Admin' link text"


def test_status_page_back_link_href():
    """Back link should point to /ui/admin.html."""
    content = (_STATIC / "status.html").read_text()
    assert 'href="/ui/admin.html"' in content, (
        "Back link href should point to /ui/admin.html"
    )


def test_status_page_back_link_css_class():
    """Back link should use the back-link CSS class."""
    content = (_STATIC / "status.html").read_text()
    assert '.back-link' in content, "Missing .back-link CSS class definition"
    assert 'class="back-link"' in content, "Missing class=back-link on the link element"


def test_status_page_back_link_styled():
    """Back link CSS should include hover styling."""
    content = (_STATIC / "status.html").read_text()
    assert '.back-link:hover' in content, "Missing .back-link:hover CSS"


# ── Page structure integrity ──────────────────────────────────────────────────


def test_status_page_has_title():
    """status.html should have the correct title."""
    content = (_STATIC / "status.html").read_text()
    assert "Service Status" in content, "Missing 'Service Status' in page title"


def test_status_page_has_overall_indicator():
    """status.html should have the overall status indicator."""
    content = (_STATIC / "status.html").read_text()
    assert 'id="overall"' in content, "Missing overall status indicator"
    assert 'id="overall-dot"' in content, "Missing overall status dot"
    assert 'id="overall-text"' in content, "Missing overall status text"


def test_status_page_has_service_grid():
    """status.html should have the service grid container."""
    content = (_STATIC / "status.html").read_text()
    assert 'id="grid"' in content, "Missing service grid container"


def test_status_page_has_refresh_button():
    """status.html should have a refresh button."""
    content = (_STATIC / "status.html").read_text()
    assert "manualRefresh" in content, "Missing manualRefresh function"


def test_status_page_has_error_banner():
    """status.html should have an error banner."""
    content = (_STATIC / "status.html").read_text()
    assert 'id="error-banner"' in content, "Missing error banner"


def test_status_page_calls_service_status_endpoint():
    """status.html should fetch from /service-status."""
    content = (_STATIC / "status.html").read_text()
    assert "/service-status" in content, "Missing /service-status endpoint reference"


def test_status_page_all_status_colors_present():
    """status.html should define CSS for ok, degraded, and error states."""
    content = (_STATIC / "status.html").read_text()
    assert ".dot.ok" in content, "Missing .dot.ok CSS"
    assert ".dot.degraded" in content, "Missing .dot.degraded CSS"
    assert ".dot.error" in content, "Missing .dot.error CSS"
    assert ".dot.loading" in content, "Missing .dot.loading CSS"
