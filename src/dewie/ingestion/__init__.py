# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.
from dewie.ingestion.base import BaseIngester
from dewie.ingestion.markitdown_ingester import MarkItDownIngester
from dewie.ingestion.rss import RSSIngester
from dewie.ingestion.source_router import SourceRouter
from dewie.ingestion.web import WebIngester

__all__ = [
    "BaseIngester",
    "MarkItDownIngester",
    "RSSIngester",
    "SourceRouter",
    "WebIngester",
]
