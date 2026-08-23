# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Built-in enrichment backends.

Available backends
------------------
- ``HttpBackend``        — Generic HTTP POST (Ollama, OpenAI-compat, custom).
- ``PassthroughBackend`` — Returns a static JSON string.  For tests and CI.

All backends satisfy the ``EnrichmentBackend`` Protocol defined in
``dewie.enrichment.base``.  Custom backends do not need to import from
this package — they only need to satisfy the Protocol.

Note: SpaCy backend has been removed. LLM failures fail hard with no silent fallback.
"""

from dewie.enrichment.backends.agent import AgentBackend
from dewie.enrichment.backends.http import HttpBackend
from dewie.enrichment.backends.passthrough import PassthroughBackend

__all__ = ["AgentBackend", "HttpBackend", "PassthroughBackend"]
