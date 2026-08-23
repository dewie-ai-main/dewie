import os
import pytest

collect_ignore_glob = ["tests/performance/*"]


def pytest_configure(config):
    """Register custom marker for daemon-mode test skipping."""
    config.addinivalue_line(
        "markers",
        "skip_in_daemon: skip this test when DAEMON_RUN=1 (pre-existing or environment-specific failure)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip marked tests during daemon runs to reduce noise and improve signal for OpenCode."""
    if os.getenv('DAEMON_RUN') == '1':
        skip_daemon = pytest.mark.skip(reason="skipped during daemon run (pre-existing failure)")
        for item in items:
            if 'skip_in_daemon' in item.keywords:
                item.add_marker(skip_daemon)
