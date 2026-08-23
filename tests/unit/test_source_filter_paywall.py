"""Tests for source_filter and paywall_handler."""


import pytest

from dewie.ingestion.paywall_handler import apply_terminal_status, classify_paywall_document
from dewie.ingestion.source_filter import is_blocked_source, is_low_quality_source
from dewie.models.content import ContentDocument

# ── source_filter ──────────────────────────────────────────────────────────────


class TestSourceFilterWithConfig:
    """Source filtering against an explicit config.

    Previously asserted the contents of the developer's local dewie.yml —
    an environment-dependent test. Now patches the lists it relies on.
    """

    @pytest.fixture(autouse=True)
    def _filter_config(self, monkeypatch):
        from dewie.ingestion.source_filter import cfg

        monkeypatch.setattr(cfg.ingest, "blocked_sources", ["ft.com", "economist.com"])
        monkeypatch.setattr(cfg.ingest, "low_quality_sources", [".yahoo.com"])

    def test_blocked_ft_com(self):
        assert is_blocked_source("https://ft.com/article/123") is True

    def test_blocked_ft_subdomain(self):
        assert is_blocked_source("https://www.ft.com/content/abc") is True

    def test_blocked_economist(self):
        assert is_blocked_source("https://economist.com/finance/456") is True

    def test_blocked_economist_subdomain(self):
        assert is_blocked_source("https://www.economist.com/blogs/789") is True

    def test_not_blocked_cnn(self):
        assert is_blocked_source("https://cnn.com/world/news") is False

    def test_not_blocked_google(self):
        assert is_blocked_source("https://google.com/search") is False

    def test_low_quality_sports_yahoo(self):
        assert is_low_quality_source("https://sports.yahoo.com/nfl/news") is True

    def test_low_quality_finance_yahoo(self):
        assert is_low_quality_source("https://finance.yahoo.com/news/market") is True

    def test_not_low_quality_bbc(self):
        assert is_low_quality_source("https://bbc.com/news/world") is False


# ── paywall_handler ────────────────────────────────────────────────────────────


class TestPaywallClassification:

    def test_normal_no_paywall(self):
        doc = ContentDocument(url="https://example.com/article", body="Full text here")
        assert classify_paywall_document(doc) == "normal"

    def test_terminal_no_body_empty(self):
        doc = ContentDocument(url="https://example.com/article", paywall_detected=True, body="")
        assert classify_paywall_document(doc) == "terminal_no_body"

    def test_terminal_no_body_whitespace(self):
        doc = ContentDocument(url="https://example.com/article", paywall_detected=True, body="   \n  ")
        assert classify_paywall_document(doc) == "terminal_no_body"

    def test_terminal_stub_short_body(self):
        doc = ContentDocument(url="https://example.com/article", paywall_detected=True, body="A" * 499)
        assert classify_paywall_document(doc) == "terminal_stub"

    def test_enrich_normal_substantial_body(self):
        doc = ContentDocument(url="https://example.com/article", paywall_detected=True, body="A" * 500)
        assert classify_paywall_document(doc) == "enrich_normal"

    def test_enrich_normal_large_body(self):
        doc = ContentDocument(url="https://example.com/article", paywall_detected=True, body="A" * 5000)
        assert classify_paywall_document(doc) == "enrich_normal"


class TestApplyTerminalStatus:

    def test_terminal_no_body_sets_skip_reason(self):
        doc = ContentDocument(url="https://example.com/article", paywall_detected=True, body="")
        apply_terminal_status(doc, "terminal_no_body")
        assert doc.status == "terminal"
        assert doc.skip_reason == "paywall_no_body"

    def test_terminal_stub_sets_skip_reason(self):
        doc = ContentDocument(url="https://example.com/article", paywall_detected=True, body="A" * 100)
        apply_terminal_status(doc, "terminal_stub")
        assert doc.status == "terminal"
        assert doc.skip_reason == "stub"

    def test_enrich_normal_does_not_change_status(self):
        doc = ContentDocument(url="https://example.com/article", paywall_detected=True, body="A" * 600)
        apply_terminal_status(doc, "enrich_normal")
        assert doc.status != "terminal"
