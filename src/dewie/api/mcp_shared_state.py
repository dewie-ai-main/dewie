# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Shared resource holder for the in-process MCP Streamable HTTP transport.

Starlette's `Mount` re-runs `scope["app"] = self` for the mounted sub-app
(the FastMCP-internal Starlette instance), so `request.app` inside an
`@mcp.tool()` handler is NOT Dewie's main FastAPI app — `request.app.state`
is unusable there. `pg`/`processor` are stashed here instead, set once from
`main.py`'s lifespan, alongside (not instead of) `app.state.postgres` /
`app.state.processor`, which other routes still use normally.
"""

from __future__ import annotations

from typing import Any

_state: dict[str, Any] = {}


def configure(pg: Any, processor: Any) -> None:
    _state["pg"] = pg
    _state["processor"] = processor


def get_pg() -> Any:
    return _state["pg"]


def get_processor() -> Any:
    return _state.get("processor")
