# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
FastAPI middleware: rate limiting, API key auth, request-ID, and error handling.

Auth behaviour (controlled by feature flags):
  AUTH_ENABLED=false (default)  → all requests pass through
  AUTH_ENABLED=true             → X-API-Key header required on non-exempt routes

Public API:
  - ``limiter`` — InProcessLimiter instance
  - ``register_middleware(app)`` — attach auth + rate-limit middleware
  - ``register_api_middleware(app)`` — attach request-ID + error handlers
  - ``rate_limit(rpm)`` — produce a rate-limit string
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI, Request, Response

from dewie.config import settings

from .middleware.error_handler import ErrorHandlerMiddleware
from .middleware.request_id import RequestIDMiddleware

# ── Rate limiting ─────────────────────────────────────────────────────────────


def _rate_limit_key(request: Request) -> str:
    """
    Rate limit key: use API key prefix when present, else real IP.
    Keying on API key isolates users from each other and prevents the benchmark
    runner (localhost) from sharing a bucket with browser/agent traffic.
    """
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return f"key:{api_key[:16]}"
    return _real_ip(request)


def _parse_rate_limit(limit: str) -> int:
    """Parse a "<rpm>/minute" style rate limit string into integer rpm."""
    try:
        head = limit.split("/", 1)[0].strip()
        return max(int(head), 0)
    except Exception:
        return settings.rate_limit_rpm


def _coerce_rpm(value: object) -> int:
    try:
        return max(int(value), 0)
    except Exception:
        try:
            return max(int(settings.rate_limit_rpm), 0)
        except Exception:
            return 60


class InProcessLimiter:
    """Minimal decorator-compatible limiter to replace slowapi in OSS mode."""

    def __init__(self, key_func: Callable[[Request], str]) -> None:
        self._key_func = key_func
        self.enabled = True
        # Compatibility fields expected by SlowAPIMiddleware in tests.
        self._auto_check = False
        self._exempt_routes: set[str] = set()
        self._route_limits: set[str] = set()

    def limit(self, limit_value: str):
        rpm = _parse_rate_limit(limit_value)

        def decorator(func):
            func._dewie_rate_limit_rpm = rpm
            self._route_limits.add(getattr(func, "__name__", ""))
            return func

        return decorator

    def _inject_headers(self, response: Response, _view_rate_limit=None) -> Response:  # noqa: ANN001
        return response

    def _check_request_limit(self, request: Request, handler=None, in_middleware=True) -> None:  # noqa: ANN001
        return None


limiter = InProcessLimiter(key_func=_rate_limit_key)


@dataclass
class _Bucket:
    window_epoch_minute: int
    count: int


_BUCKETS: dict[tuple[str, int], _Bucket] = {}
_BUCKET_LOCK = asyncio.Lock()


def _limit_for_request(request: Request) -> int:
    endpoint = request.scope.get("endpoint")
    endpoint_limit = getattr(endpoint, "_dewie_rate_limit_rpm", None) if endpoint else None
    if endpoint_limit is not None:
        return _coerce_rpm(endpoint_limit)
    return _coerce_rpm(settings.rate_limit_rpm)


async def _rate_limit_middleware(request: Request, call_next: Callable) -> Response:
    if not limiter.enabled:
        return await call_next(request)

    rpm = _limit_for_request(request)
    if rpm <= 0:
        return await call_next(request)

    key = _rate_limit_key(request)
    now_window = int(time.time() // 60)
    bucket_key = (key, rpm)

    async with _BUCKET_LOCK:
        bucket = _BUCKETS.get(bucket_key)
        if bucket is None or bucket.window_epoch_minute != now_window:
            _BUCKETS[bucket_key] = _Bucket(window_epoch_minute=now_window, count=1)
        elif bucket.count >= rpm:
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )
        else:
            bucket.count += 1

    return await call_next(request)


# ── Auth helpers ──────────────────────────────────────────────────────────────


def _has_scope(scopes: list[str], required: str) -> bool:
    """Return True when required scope is present, or admin scope is present."""
    return required in scopes or "admin" in scopes


def get_remote_address(request: Request) -> str:
    """Compatibility helper retained for test fixtures and older call sites."""
    client = request.client
    return client.host if client else "unknown"


def _real_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Forwarded-For only from trusted (private/loopback) proxies."""
    import ipaddress as _ipaddress
    client_ip = get_remote_address(request)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for and client_ip not in ("unknown",):
        try:
            ip = _ipaddress.ip_address(client_ip)
            if ip.is_private or ip.is_loopback:
                return forwarded_for.split(",")[0].strip()
        except ValueError:
            pass
    return client_ip


# ── Auth ──────────────────────────────────────────────────────────────────────

_AUTH_EXEMPT_PREFIXES = (
    "/health",
    "/service-status",
    "/api/service-status",
    "/pipeline/health",
    "/api/pipeline/health",
    "/favicon",
    "/ui/",  # static assets only; API calls still require auth
    "/documents/shared/",  # explicit public tokenized sharing route
    # Auth routes — both legacy (no prefix) and /api prefixed forms
    "/auth/google",
    "/auth/google/callback",
    "/auth/login",
    "/auth/signup",
    "/auth/me",
    "/auth/logout",
    "/auth/signout",
    "/auth/apple",
    "/auth/apple/callback",
    "/api/auth/google",
    "/api/auth/google/callback",
    "/api/auth/login",
    "/api/auth/signup",
    "/api/auth/me",
    "/api/auth/logout",
    "/api/auth/signout",
    "/api/auth/apple",
    "/api/auth/apple/callback",
)


_SCOPE_PREFIX_RULES: tuple[tuple[str, str], ...] = (
    ("/api/admin", "admin"),
    # Read-only corpus/worker-status sub-routes are accessible with 'read' scope;
    # must come before the broader '/api/pipeline' admin rule (longest-prefix wins).
    ("/api/pipeline/corpus", "read"),
    ("/api/pipeline/workers/status", "read"),
    ("/api/pipeline", "admin"),
    ("/api/ingest", "ingest"),
    ("/api/query", "read"),
    ("/api/documents/ingest", "ingest"),
    ("/api/documents", "read"),
    ("/api/graph", "read"),
    ("/api/traverse", "read"),
    ("/api/capabilities", "read"),
    ("/api/search-queue", "read"),
    ("/api/research", "read"),
    ("/api/investigate", "read"),
    ("/api/stats", "read"),
    ("/api/mcp", "read"),
)


async def _api_key_middleware(request: Request, call_next: Callable) -> Response:
    """
    Authentication middleware.

    Sets request.state.workspace_ids (list[UUID]) and request.state.key_id.
    Empty workspace_ids means access to all workspaces (no restriction).
    """
    path = request.url.path

    # Always set defaults so downstream code never has to check AttributeError
    request.state.workspace_ids = []
    request.state.key_id = None
    request.state.key_scopes = []
    request.state.is_admin = False
    request.state.user_id = None
    request.state.email = None
    request.state.activation_status = "approved"

    if getattr(settings, "local_auth_enabled", False) is True:
        is_admin = bool(getattr(settings, "local_auth_is_admin", True))
        request.state.user_id = getattr(
            settings, "local_auth_user_id", "00000000-0000-0000-0000-000000000002"
        )
        request.state.email = getattr(settings, "local_auth_email", "Dewie Local Catalog")
        request.state.key_id = None
        request.state.key_scopes = ["read", "ingest", "admin"] if is_admin else ["read", "ingest"]
        request.state.is_admin = is_admin
        return await call_next(request)

    if not settings.auth_enabled:
        return await call_next(request)

    # /ui/admin.html must require auth even though /ui/ is otherwise exempt.
    # Direct navigation to the admin panel should redirect to login when
    # the user is not authenticated, not serve the page unauthenticated.
    if path == "/ui/admin.html":
        pass  # fall through to full auth check below
    elif path == "/" or any(path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES):
        return await call_next(request)

    pg = request.app.state.postgres

    # ── Session JWT (from cookies) ─────────────────────────────────────────────
    session_token = request.cookies.get("dewie_session", "").strip()
    if session_token:
        from dewie.local_auth import verify_session_token

        payload = verify_session_token(session_token)
        if payload:
            # Check if the session token has been revoked
            if pg:
                try:
                    from dewie.local_auth import is_session_revoked
                    if await is_session_revoked(pg, session_token):
                        if path == "/ui/admin.html":
                            from fastapi.responses import RedirectResponse
                            return RedirectResponse(url="/ui/login.html", status_code=302)
                        return Response(
                            content='{"detail": "Session revoked. Please log in again."}',
                            status_code=401,
                            media_type="application/json",
                        )
                except Exception:
                    pass  # Best-effort: don't block auth if revocation check fails
            request.state.user_id = payload.get("sub")
            request.state.email = payload.get("email")
            request.state.is_admin = payload.get("is_admin", False)
            scopes = ["admin", "ingest", "read"] if request.state.is_admin else ["read", "ingest"]
            request.state.key_scopes = scopes
            activation = payload.get("activation_status", "approved")
            request.state.activation_status = activation
            if activation in ("pending", "rejected"):
                return Response(
                    content='{"detail": "Account pending approval"}',
                    status_code=403,
                    media_type="application/json",
                )
            return await call_next(request)

    # For admin HTML page: redirect to login rather than returning 401 JSON
    if path == "/ui/admin.html":
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/ui/login.html", status_code=302)

    # ── API key ────────────────────────────────────────────────────────────────
    raw_key = request.headers.get("X-API-Key", "").strip()
    if path in ("/admin/keys", "/api/admin/keys") and request.method == "POST" and not raw_key:
        # Allow the request through if either:
        # 1) No API keys exist yet (bootstrap path for fresh installs), OR
        # 2) A valid session JWT is present (session-authenticated admin creating keys).
        session_ok = False
        session_tok = request.cookies.get("dewie_session", "").strip()
        if session_tok:
            from dewie.local_auth import verify_session_token

            payload = verify_session_token(session_tok)
            if payload:
                session_ok = True
        if session_ok:
            # Session-authenticated admin — the route handler also checks
            # is_admin, so grant it here rather than 403 downstream.
            request.state.is_admin = True
            request.state.key_scopes = ["read", "ingest", "admin"]
            return await call_next(request)
        from sqlalchemy import text as _text

        async with pg._engine.connect() as conn:
            count = (
                await conn.execute(
                    _text("SELECT COUNT(*) FROM api_keys WHERE revoked_at IS NULL")
                )
            ).scalar()
        if int(count or 0) == 0:
            # Fresh-install bootstrap: no keys exist, so this first key
            # creation must be treated as admin for the route guard too.
            # The flag tells the handler to grant the founder key full scopes.
            request.state.is_admin = True
            request.state.key_scopes = ["read", "ingest", "admin"]
            request.state.bootstrap_founder = True
            return await call_next(request)
    if not raw_key:
        return Response(
            content='{"detail": "Missing X-API-Key header"}',
            status_code=401,
            media_type="application/json",
        )

    from dewie.auth import verify_api_key

    key_record = await verify_api_key(raw_key, pg)
    if key_record is None:
        return Response(
            content='{"detail": "Invalid or revoked API key"}',
            status_code=403,
            media_type="application/json",
        )

    request.state.workspace_ids = key_record.get("workspace_ids") or []
    request.state.key_id = key_record["id"]
    request.state.key_scopes = key_record.get("scopes") or []
    request.state.is_admin = _has_scope(request.state.key_scopes, "admin")
    if key_record.get("user_id"):
        request.state.user_id = key_record["user_id"]

    # Scope enforcement by route family (defense in depth).
    # Uses longest-prefix-wins: the most specific matching rule takes precedence.
    # Route handlers should still enforce any stricter rules they require.
    matched_prefix: str | None = None
    matched_scope: str | None = None
    for prefix, required_scope in _SCOPE_PREFIX_RULES:
        if path.startswith(prefix):
            if matched_prefix is None or len(prefix) > len(matched_prefix):
                matched_prefix = prefix
                matched_scope = required_scope
    if matched_scope is not None and not _has_scope(request.state.key_scopes, matched_scope):
        return Response(
            content=f'{{"detail": "Insufficient scope: {matched_scope} required"}}',
            status_code=403,
            media_type="application/json",
        )

    return await call_next(request)


# ── Registration ──────────────────────────────────────────────────────────────


def register_middleware(app: FastAPI) -> None:
    """Attach rate limiting, auth middleware, and error handlers."""
    app.state.limiter = limiter
    app.middleware("http")(_rate_limit_middleware)
    app.middleware("http")(_api_key_middleware)


def register_api_middleware(app: FastAPI) -> None:
    """Add request-id and error-handling middleware to *app*."""
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RequestIDMiddleware)


def rate_limit(rpm: int = settings.rate_limit_rpm) -> str:
    """Return a slowapi rate-limit string for the given requests-per-minute."""
    return f"{rpm}/minute"
