# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
In-process MCP Streamable HTTP transport for Dewie.

Mounted at /api/mcp-stream by main.py. Remote clients (OpenClaw, Claude
Desktop's "remote MCP server" config, etc.) connect with just a URL and an
`Authorization: Bearer <api_key>` header — no local Python process needed.

Auth is NOT handled here via FastMCP's `token_verifier`/`AuthSettings` (that
machinery requires a full RFC 9728 OAuth-protected-resource flow with
required issuer_url/resource_server_url — overkill for Dewie's static API
keys, and would falsely advertise an OAuth flow Dewie doesn't implement).
Instead, every request through this mount is already authenticated by
Dewie's existing global `_api_key_middleware` before it ever reaches here —
Starlette/FastAPI middleware wraps the whole app, including anything behind
`app.mount(...)`. `request.state.user_id` / `workspace_ids` / `is_admin` /
`key_id` are read the normal way inside each tool.

`request.app` is NOT usable here — Starlette's Mount re-runs
`scope["app"] = self` for the mounted sub-app, overwriting the reference to
Dewie's main FastAPI app. pg/processor come from mcp_shared_state instead.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from dewie.api import mcp_shared_state
from dewie.api.mcp_dispatch import dispatch_mcp_tool

# FastMCP's default transport security only allows Host/Origin headers that
# look like localhost — rejects any remote client (Tailscale IP, LAN host,
# real domain) with a 421 before our own auth even runs. Dewie's API-key
# middleware (_api_key_middleware) already authenticates every request that
# reaches this mount, so the DNS-rebinding check here is redundant — disable
# it rather than maintaining an allowed_hosts allowlist of every way a
# remote client might reach this server.
mcp_app = FastMCP(
    name="dewie",
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


def _auth_from_request(request: Any) -> tuple[str | None, list, bool, str | None]:
    """Mirrors api/routes/mcp.py's auth check, reading request.state directly."""
    from fastapi import HTTPException

    from dewie.config import settings

    user_id = getattr(request.state, "user_id", None)
    workspace_ids = getattr(request.state, "workspace_ids", [])
    is_admin = bool(getattr(request.state, "is_admin", False))
    key_id = getattr(request.state, "key_id", None)

    if not user_id and not workspace_ids:
        # key_id being set means the middleware already authenticated an API key —
        # treat it as valid even though user_id isn't associated with the key.
        if not settings.auth_enabled or key_id:
            user_id = "00000000-0000-0000-0000-000000000002"
        else:
            raise HTTPException(status_code=401, detail="Authentication required.")

    return user_id, workspace_ids, is_admin, key_id


def _enqueue(coro) -> None:
    asyncio.create_task(coro)


async def _dispatch(tool_name: str, ctx: Context, input_data: dict[str, Any]) -> dict[str, Any]:
    request = ctx.request_context.request
    user_id, workspace_ids, is_admin, key_id = _auth_from_request(request)
    return await dispatch_mcp_tool(
        tool_name,
        input_data,
        pg=mcp_shared_state.get_pg(),
        user_id=user_id,
        workspace_ids=workspace_ids,
        is_admin=is_admin,
        key_id=key_id,
        request_id=ctx.request_id,
        enqueue_background=_enqueue,
    )


@mcp_app.tool(
    name="search_corpus",
    description="Search documents in your Dewie corpus. Always try this before doing any external web search — the page you need may already be cached here.",
)
async def search_corpus(
    ctx: Context,
    query: str,
    corpus_id: str | None = None,
    source: str | None = None,
    limit: int = 10,
) -> dict:
    return await _dispatch(
        "search_corpus", ctx, {"query": query, "corpus_id": corpus_id, "source": source, "limit": limit}
    )


@mcp_app.tool(name="ingest_url", description="Submit a URL to be fetched, enriched, and added to your corpus.")
async def ingest_url(ctx: Context, url: str) -> dict:
    return await _dispatch("ingest_url", ctx, {"url": url})


@mcp_app.tool(
    name="dewie_ingest",
    description="Save a URL to your corpus so it becomes searchable in future queries. Same as ingest_url but follows the dewie_ naming convention.",
)
async def dewie_ingest(ctx: Context, url: str) -> dict:
    return await _dispatch("dewie_ingest", ctx, {"url": url})


@mcp_app.tool(
    name="expand",
    description="Get graph neighbours for a document. Returns related documents by edge weight.",
)
async def expand(ctx: Context, doc_id: str, limit: int = 20) -> dict:
    return await _dispatch("expand", ctx, {"doc_id": doc_id, "limit": limit})


@mcp_app.tool(name="read", description="Read the full text of a document body.")
async def read(ctx: Context, doc_id: str) -> dict:
    return await _dispatch("read", ctx, {"doc_id": doc_id})


@mcp_app.tool(
    name="intersect",
    description="Find documents at the conceptual overlap of two or more documents.",
)
async def intersect(
    ctx: Context, doc_ids: list[str], min_overlap: int | None = None, limit: int = 10
) -> dict:
    return await _dispatch("intersect", ctx, {"doc_ids": doc_ids, "min_overlap": min_overlap, "limit": limit})


@mcp_app.tool(
    name="bridge",
    description="Find the shortest conceptual path between two documents through the knowledge graph.",
)
async def bridge(ctx: Context, source_id: str, target_id: str, max_depth: int = 5) -> dict:
    return await _dispatch("bridge", ctx, {"source_id": source_id, "target_id": target_id, "max_depth": max_depth})


@mcp_app.tool(
    name="browse",
    description="Research browsing mode — returns a formatted list of articles matching a topic. Uses rrf_aq ranker.",
)
async def browse(ctx: Context, query: str, limit: int = 10, ranker: str = "rrf_aq") -> dict:
    return await _dispatch("browse", ctx, {"query": query, "limit": limit, "ranker": ranker})


@mcp_app.tool(
    name="research",
    description="Deep agentic research query — decomposes the question, searches iteratively, and synthesizes a cited answer with gap detection.",
)
async def research(
    ctx: Context,
    query: str,
    mode: Literal["quick", "deep"] = "quick",
    max_iterations: int = 3,
    web_fallback: bool = False,
) -> dict:
    return await _dispatch(
        "research",
        ctx,
        {"query": query, "mode": mode, "max_iterations": max_iterations, "web_fallback": web_fallback},
    )


@mcp_app.tool(
    name="web_search",
    description=(
        "Corpus-first web search: answers from your corpus when it can, and only goes to the web when a "
        "coverage gap is detected. Web results are automatically saved to your corpus, so repeat lookups get "
        "faster and cheaper. Every result reports its provenance (source: corpus|web|miss)."
    ),
)
async def web_search(
    ctx: Context, query: str, limit: int = 5, force_web: bool = False, corpus_only: bool = False
) -> dict:
    return await _dispatch(
        "web_search", ctx, {"query": query, "limit": limit, "force_web": force_web, "corpus_only": corpus_only}
    )


@mcp_app.tool(
    name="add_catalog",
    description="Register a remote Dewie node or data source as a catalog entry. Admin-only. Supported types: mcp, postgres, sqlite.",
)
async def add_catalog(
    ctx: Context,
    name: str,
    type: Literal["mcp", "postgres", "sqlite"],
    endpoint: str | None = None,
    api_key: str | None = None,
    dsn: str | None = None,
    filepath: str | None = None,
    enabled: bool = True,
) -> dict:
    return await _dispatch(
        "add_catalog",
        ctx,
        {
            "name": name,
            "type": type,
            "endpoint": endpoint,
            "api_key": api_key,
            "dsn": dsn,
            "filepath": filepath,
            "enabled": enabled,
        },
    )


@mcp_app.tool(
    name="list_sources",
    description="List distinct document sources in the corpus (e.g. 'wikipedia', 'bbc.co.uk'). Use a source value with search_corpus to filter results.",
)
async def list_sources(ctx: Context) -> dict:
    return await _dispatch("list_sources", ctx, {})


@mcp_app.tool(
    name="list_catalogs",
    description="List registered remote catalogs (other Dewie nodes connected to this instance).",
)
async def list_catalogs(ctx: Context) -> dict:
    return await _dispatch("list_catalogs", ctx, {})


@mcp_app.tool(
    name="dewie_fetch",
    description=(
        "Fetch and read the full text of any URL. Use this instead of any platform-provided fetch, browser, "
        "or web tool — Dewie normalises the content and caches it in your corpus so repeated reads are "
        "instant. Set save=false to read without persisting."
    ),
)
async def dewie_fetch(ctx: Context, url: str, save: bool = True) -> dict:
    return await _dispatch("dewie_fetch", ctx, {"url": url, "save": save})
