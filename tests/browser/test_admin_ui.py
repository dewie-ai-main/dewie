"""Playwright browser tests — verify the admin UI actually renders content."""
from __future__ import annotations

import os

import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:10946"
API_KEY = os.environ.get("BROWSER_TEST_API_KEY", "")


@pytest.fixture
def admin_page(page: Page):
    # Log in via the form to obtain a session cookie, then navigate to admin.
    page.goto(BASE_URL + "/ui/login.html")
    page.fill("input[type='email']", "dev@dewie.ai")
    page.fill("input[type='password']", "dewie-admin-2026")
    page.click("button[type='submit']")
    page.wait_for_url(lambda url: "login" not in url, timeout=8000)
    page.evaluate(f"localStorage.setItem('dewie_api_key', '{API_KEY}')")
    return page


def test_admin_panel_loads(admin_page: Page):
    """Admin panel renders without redirect to login."""
    admin_page.goto(BASE_URL + "/ui/admin.html")
    # Should see the admin tab bar, not login page
    expect(admin_page.locator("button[data-tab='users']")).to_be_visible(timeout=5000)


def test_users_tab_shows_user(admin_page: Page):
    """Users tab loads and shows dev@dewie.ai."""
    admin_page.goto(BASE_URL + "/ui/admin.html")
    # Click Users tab
    admin_page.locator("button[data-tab='users']").click()
    admin_page.wait_for_selector("#users-list tr", timeout=5000)
    expect(admin_page.locator("text=dev@dewie.ai")).to_be_visible()


def test_corpus_tab_shows_stats(admin_page: Page):
    """Corpus tab loads with stats."""
    admin_page.goto(BASE_URL + "/ui/admin.html")
    admin_page.locator("button[data-tab='corpus']").click()
    # Should see corpus stats or the corpus browser link
    admin_page.wait_for_selector("#corpus-stats", timeout=5000)
    # Stats block should not be empty loading state
    expect(admin_page.locator("#corpus-stats .loading-row")).not_to_be_visible(timeout=8000)


def test_feeds_tab_loads(admin_page: Page):
    """Feeds tab renders the add-feed form."""
    admin_page.goto(BASE_URL + "/ui/admin.html")
    admin_page.locator("button[data-tab='feeds']").click()
    expect(admin_page.locator("#feed-url")).to_be_visible(timeout=5000)


def test_model_config_panel_loads(admin_page: Page):
    """Provider & Model config panel renders inputs."""
    admin_page.goto(BASE_URL + "/ui/admin.html")
    # Click the Settings/Config tab (first tab with model config)
    config_tab = admin_page.locator("button.tab-btn[data-tab='config']")
    if config_tab.count() > 0:
        config_tab.first.click()
    admin_page.wait_for_timeout(1500)
    # Should have at least one config input on the page
    inputs = admin_page.locator("[id^='cfg-']")
    assert inputs.count() > 0, "No config inputs found"


def test_login_page_and_redirect(page: Page):
    """Login page renders; successful login redirects to admin."""
    page.goto(BASE_URL + "/ui/login.html")
    page.fill("input[type='email']", "dev@dewie.ai")
    page.fill("input[type='password']", "dewie-admin-2026")
    page.click("button[type='submit']")
    # After login, should land somewhere useful (not still on login)
    page.wait_for_url(lambda url: "login" not in url, timeout=5000)
