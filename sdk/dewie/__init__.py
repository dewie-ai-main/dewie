# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
dewie — Agent-native document retrieval client.

Quick start::

    from dewie import DewieClient

    client = DewieClient(api_url="https://api.dewie.ai", api_key="ck_live_...")
    response = client.query("How does CRISPR work", limit=10)

    if response.has_gap:
        print(response.gap_message)  # corpus coverage warning

    traverse = client.traverse(["CRISPR", "gene editing"], max_documents=20)
    path = client.bridge(source_id="uuid-a", target_id="uuid-b")
    print(f"Connected in {path.hops} hops")

Async usage::

    from dewie import AsyncDewieClient

    async with AsyncDewieClient(api_url="...", api_key="...") as client:
        response = await client.query("How does CRISPR work")
        if response.has_gap:
            print(response.gap_message)
"""

from .client import AsyncDewieClient, DewieClient
from .models import ResultConfidence, SearchResponse, SearchResult

__version__ = "0.1.0"

__all__ = [
    "DewieClient",
    "AsyncDewieClient",
    "SearchResponse",
    "ResultConfidence",
    "SearchResult",
]
