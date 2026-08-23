"""
Browser E2E tests — test the live admin panel at localhost:10946.
Run with: pytest tests/browser/ -v --tb=short

These tests require the dev server to be running:
  bash restart.sh
"""
from __future__ import annotations

import os

import pytest

BASE_URL = os.environ.get("BROWSER_TEST_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("BROWSER_TEST_API_KEY", "")
LOGIN_EMAIL = os.environ.get("BROWSER_TEST_EMAIL", "admin@example.com")
LOGIN_PASSWORD = os.environ.get("BROWSER_TEST_PASSWORD", "")


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def api_key():
    return API_KEY


@pytest.fixture
def admin_page(page, base_url, api_key):
    """A Playwright page pre-authenticated with API key in localStorage."""
    # Navigate to the origin first so localStorage is scoped correctly
    page.goto(base_url + "/health")
    page.evaluate(f"localStorage.setItem('dewie_api_key', '{api_key}')")
    return page
