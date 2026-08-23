# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Shared logging configuration and redaction for the Dewie API.

Provides:
  - `get_logger(name)` — returns a logger with request-aware format
  - `redact_sensitive(data)` — recursively redacts secrets from dicts/lists
  - `truncate_body(body, limit=1000)` — truncates long request bodies
  - `set_request_id(request_id)` / `get_request_id()` — contextvar helpers
"""

from __future__ import annotations

import contextvars
import logging
import re
from typing import Any

# Context variable to carry request_id across async boundaries.
_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)

# Fields/patterns that should never appear in logs.
_SENSITIVE_PATTERNS: tuple[str, ...] = (
    "api_key",
    "password",
    "token",
    "authorization",
)
_REDACTED = "***REDACTED***"
_BODY_TRUNCATE = 1000


# ── Context helpers ───────────────────────────────────────────────────────────


def set_request_id(request_id: str) -> contextvars.Token[str]:
    """Set the request-id in the current async context."""
    return _request_id_ctx.set(request_id)


def reset_request_id(token: contextvars.Token[str]) -> None:
    """Restore the previous request-id value."""
    _request_id_ctx.reset(token)


def get_request_id() -> str:
    """Return the current request-id from context, or empty string."""
    return _request_id_ctx.get()


# ── Redaction ─────────────────────────────────────────────────────────────────


def redact_sensitive(data: Any) -> Any:
    """Recursively redact sensitive fields from a data structure.

    Matches keys containing any of ``api_key``, ``password``, ``token``,
    or ``authorization`` (case-insensitive).  Dict values are replaced with
    ``***REDACTED***``.  Nested dicts/lists are handled recursively.
    """
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for k, v in data.items():
            key_lower = k.lower()
            if any(pat in key_lower for pat in _SENSITIVE_PATTERNS):
                out[k] = _REDACTED
            else:
                out[k] = redact_sensitive(v)
        return out
    if isinstance(data, (list, tuple)):
        result = [redact_sensitive(item) for item in data]
        return result if isinstance(data, list) else tuple(result)
    if isinstance(data, str):
        # Also scan string values for Authorization header fragments
        auth_pat = re.compile(
            r"(?i)(bearer\s+)([A-Za-z0-9._~+/=-]+)", re.IGNORECASE
        )
        data = auth_pat.sub(r"\1***REDACTED***", data)
    return data


def _redact_dict_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Redact values of known-sensitive keys in a single dict layer."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key_lower = k.lower()
        if any(pat in key_lower for pat in _SENSITIVE_PATTERNS):
            out[k] = _REDACTED
        else:
            out[k] = v
    return out


# ── Body truncation ───────────────────────────────────────────────────────────


def truncate_body(body: Any, limit: int = _BODY_TRUNCATE) -> str:
    """Truncate a request body to *limit* characters for logging."""
    text = str(body)
    if len(text) > limit:
        text = text[:limit] + f"... (truncated, {len(body)} chars total)"
    return text


# ── Logger factory ────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] [%(request_id)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _RequestIDFilter(logging.Filter):
    """Inject ``request_id`` from contextvars into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        # An explicit extra={"request_id": ...} from the call site wins over
        # the contextvar — don't stomp it with an empty default.
        if not getattr(record, "request_id", ""):
            record.request_id = _request_id_ctx.get() or ""
        return True


def _configure_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name*."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
    )
    handler.setFormatter(formatter)
    logger.addFilter(_RequestIDFilter())
    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured for the Dewie API.

    Use ``dewie.api`` for public API modules and ``dewie.admin`` for admin.
    """
    return _configure_logger(name)
