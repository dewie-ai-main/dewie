"""Tests for the feeds admin panel — add, list, delete."""
from __future__ import annotations

import os

import httpx
import pytest

BASE_URL = "http://localhost:10946"
API_KEY = os.environ.get("BROWSER_TEST_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
TEST_FEED_URL = "https://test.example.com/feed.xml"
TEST_FEED_NAME = "E2E Test Feed"


@pytest.fixture(autouse=True)
def cleanup_test_feed():
    """Remove any leftover test feeds before and after each test."""
    _delete_test_feeds()
    yield
    _delete_test_feeds()


def _delete_test_feeds():
    feeds = httpx.get(f"{BASE_URL}/admin/feeds", headers=HEADERS).json()
    for feed in feeds:
        if feed.get("url") == TEST_FEED_URL or feed.get("name") == TEST_FEED_NAME:
            httpx.delete(f"{BASE_URL}/admin/feeds/{feed['id']}", headers=HEADERS)


def test_feeds_list_is_empty_or_list():
    r = httpx.get(f"{BASE_URL}/admin/feeds", headers=HEADERS)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_add_feed():
    """POST /admin/feeds creates a new feed."""
    r = httpx.post(
        f"{BASE_URL}/admin/feeds",
        headers=HEADERS,
        json={"url": TEST_FEED_URL, "name": TEST_FEED_NAME, "poll_interval_minutes": 60},
    )
    assert r.status_code in (200, 201), f"Expected 200/201, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["url"] == TEST_FEED_URL
    assert data["name"] == TEST_FEED_NAME
    return data["id"]


def test_feed_appears_in_list():
    """After adding a feed, it shows up in GET /admin/feeds."""
    httpx.post(
        f"{BASE_URL}/admin/feeds",
        headers=HEADERS,
        json={"url": TEST_FEED_URL, "name": TEST_FEED_NAME, "poll_interval_minutes": 60},
    )
    r = httpx.get(f"{BASE_URL}/admin/feeds", headers=HEADERS)
    assert r.status_code == 200
    feeds = r.json()
    urls = [f["url"] for f in feeds]
    assert TEST_FEED_URL in urls, f"Test feed not found in {urls}"


def test_delete_feed():
    """DELETE /admin/feeds/{id} removes the feed."""
    # Create
    create = httpx.post(
        f"{BASE_URL}/admin/feeds",
        headers=HEADERS,
        json={"url": TEST_FEED_URL, "name": TEST_FEED_NAME, "poll_interval_minutes": 60},
    )
    assert create.status_code in (200, 201)
    feed_id = create.json()["id"]

    # Delete
    r = httpx.delete(f"{BASE_URL}/admin/feeds/{feed_id}", headers=HEADERS)
    assert r.status_code in (200, 204), f"Delete failed: {r.status_code}: {r.text}"

    # Confirm gone
    feeds = httpx.get(f"{BASE_URL}/admin/feeds", headers=HEADERS).json()
    ids = [f["id"] for f in feeds]
    assert feed_id not in ids


def test_update_feed():
    """PATCH /admin/feeds/{id} updates feed properties."""
    create = httpx.post(
        f"{BASE_URL}/admin/feeds",
        headers=HEADERS,
        json={"url": TEST_FEED_URL, "name": TEST_FEED_NAME, "poll_interval_minutes": 60},
    )
    assert create.status_code in (200, 201)
    feed_id = create.json()["id"]

    r = httpx.patch(
        f"{BASE_URL}/admin/feeds/{feed_id}",
        headers=HEADERS,
        json={"poll_interval_minutes": 120},
    )
    assert r.status_code in (200, 204), f"Patch failed: {r.status_code}: {r.text}"
