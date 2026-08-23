# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Dewie enrichment package.

Public surface
--------------
- ``EnrichmentBackend``  — Protocol every backend must satisfy.
- ``ExtractionResult``   — Pydantic model for the structured extraction output.
- ``BackendError``       — Exception raised by backends on failure.
- ``EnrichmentPass``     — ABC every enrichment pass must implement.
- ``MetadataProcessor``  — Orchestrates enrichment using the router + registry.
- ``BackendRegistry``    — Named lookup for registered backends.
- ``PassRegistry``       — Named lookup for registered enrichment passes.
- ``EnrichmentRouter``   — Selects a backend for a given document.
- ``build_extraction_prompt`` — Builds the prompt injected into LLM backends.
- ``MetadataPass``       — LLM metadata extraction pass.
- ``EmbedPass``          — Document embedding pass.
- ``ChunkPass``          — Document chunking pass.
- ``CURRENT_ENRICHMENT_VERSION`` — Bump to force re-enrichment on next worker run.

Typical startup sequence
------------------------
::

    from dewie.enrichment import BackendRegistry, EnrichmentRouter, MetadataProcessor
    from dewie.enrichment.backends.http import HttpBackend
    from dewie.enrichment.backends.passthrough import PassthroughBackend

    registry = BackendRegistry()
    registry.register(PassthroughBackend())
    registry.register(HttpBackend(name="ollama_3b", ...))
    registry.set_default("passthrough")

    router = EnrichmentRouter(registry, rules=[...])
    processor = MetadataProcessor(router=router, registry=registry)

    enriched_doc = await processor.enrich(doc)

Pass-based pipeline
-------------------
::

    from dewie.enrichment import PassRegistry
    from dewie.enrichment.passes import MetadataPass, EmbedPass, ChunkPass

    registry = PassRegistry()
    registry.register(MetadataPass(router, backend_registry))
    registry.register(EmbedPass())
    registry.register(ChunkPass())

    for pass_instance in registry.get_ordered():
        await pass_instance.run(doc, pg)
"""

# Bump this constant to force re-enrichment of all docs on next worker run.
# Workers should skip docs where enrichment_version >= CURRENT_ENRICHMENT_VERSION.
CURRENT_ENRICHMENT_VERSION: int = 1

from dewie.enrichment.base import BackendError, EnrichmentBackend, EnrichmentPass, ExtractionResult
from dewie.enrichment.passes import ChunkPass, EmbedPass, MetadataPass
from dewie.enrichment.processor import MetadataProcessor
from dewie.enrichment.registry import BackendRegistry, PassRegistry
from dewie.enrichment.router import EnrichmentRouter
from dewie.enrichment.schema import build_extraction_prompt

__all__ = [
    "BackendError",
    "BackendRegistry",
    "ChunkPass",
    "CURRENT_ENRICHMENT_VERSION",
    "EmbedPass",
    "EnrichmentBackend",
    "EnrichmentPass",
    "EnrichmentRouter",
    "ExtractionResult",
    "MetadataPass",
    "MetadataProcessor",
    "PassRegistry",
    "build_extraction_prompt",
]
