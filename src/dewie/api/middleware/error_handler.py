# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Structured error handler middleware.

Catches unhandled exceptions, logs a full traceback at ERROR level, and
returns a JSON ``{"detail": "...", "request_id": "..."}`` response.
Stack traces are never exposed to the client.
"""

from __future__ import annotations

import traceback

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from dewie.api.logging_config import get_logger, get_request_id

logger = get_logger("dewie.api.middleware.error_handler")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and return structured JSON errors."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        try:
            return await call_next(request)
        except HTTPException:
            raise
        except Exception as exc:
            request_id = get_request_id() or getattr(request.state, "request_id", "unknown")

            # Log full traceback for internal debugging — never send to client.
            logger.error(
                "Unhandled exception: %s\n%s",
                exc,
                traceback.format_exc(),
            )

            detail = str(exc) if str(exc) else "An unexpected error occurred"

            return JSONResponse(
                status_code=500,
                content={"detail": detail, "request_id": request_id},
            )
