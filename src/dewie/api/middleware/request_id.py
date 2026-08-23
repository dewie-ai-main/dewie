# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
X-Request-ID middleware for the Dewie API.

Reads the ``X-Request-ID`` header from the request, generates a uuid4()
if absent, stores it on ``request.state.request_id``, adds it to response
headers, and attaches it to the logging context via ``contextvars``.
"""

from __future__ import annotations

import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from dewie.api.logging_config import get_logger, reset_request_id, set_request_id

_logger = get_logger("dewie.api.middleware.request_id")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request ID to every request/response cycle."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id

        # Attach request_id to the logging context via contextvars
        token = set_request_id(request_id)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            reset_request_id(token)
