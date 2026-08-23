"""Acceptance tests for issue #348 — API keys moved from query page to account page."""

from __future__ import annotations

import re

import pytest


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


# ── app.html must not contain API keys ────────────────────────────────────────


class TestQueryPageNoApiKeySection:
    """static/app.html must not contain API key UI, endpoints, or JS functions."""

    @pytest.fixture(autouse=True)
    def _html(self):
        self.html = _read("static/app.html")

    def _assert_no(self, label: str, pattern: str):
        matches = re.findall(pattern, self.html, re.IGNORECASE)
        assert not matches, f"{label} still present: {matches}"

    def _assert_no_text(self, label: str, text: str):
        assert text not in self.html, f"{label} text still present"

    def _assert_no_css(self, label: str, cls: str):
        assert cls not in self.html, f"{label} class '{cls}' still present"

    def no_key_section(self):
        """No HTML section with id or comment referencing API keys."""
        self._assert_no_text(
            "API keys section comment",
            "<!-- API KEYS -->",
        )
        assert 'id="keys-list"' not in self.html, "keys-list element still in app.html"

    def no_key_endpoints(self):
        """No references to /admin/keys or /user/api-keys in app.html."""
        self._assert_no(
            "admin/keys endpoint reference",
            r"/admin/keys",
        )
        self._assert_no(
            "user/api-keys endpoint reference",
            r"/user/api-keys",
        )

    def no_key_js_functions(self):
        """No loadKeys, createKey, or revokeKey JS functions."""
        self._assert_no(
            "loadKeys function",
            r"(?:const|let|var)\s+loadKeys\s*[=\(]",
        )
        self._assert_no(
            "createKey function",
            r"(?:const|let|var)\s+createKey\s*[=\(]",
        )
        self._assert_no(
            "revokeKey function",
            r"(?:const|let|var)\s+revokeKey\s*[=\(]",
        )

    def no_key_css(self):
        """No API key CSS classes."""
        for cls in (".keys-grid", ".key-row", ".btn-danger"):
            self._assert_no_css("key CSS", cls)

    def no_key_inputs(self):
        """No key name input fields."""
        assert 'name="key_name"' not in self.html, "key_name input still in app.html"

    def no_key_load_init(self):
        """No loadKeys() call on page init."""
        assert (
            "loadKeys()" not in self.html
        ), "loadKeys() init call still in app.html"


# ── account.html must contain API keys ────────────────────────────────────────


class TestAccountPageHasApiKeySection:
    """static/account.html must have the API keys section."""

    @pytest.fixture(autouse=True)
    def _html(self):
        self.html = _read("static/account.html")

    def test_keys_list_element(self):
        assert 'id="keys-list"' in self.html

    def test_keys_endpoint(self):
        assert "/user/api-keys" in self.html

    def test_keys_load_function(self):
        assert "loadKeys" in self.html
