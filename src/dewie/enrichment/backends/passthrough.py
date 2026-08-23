# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
PassthroughBackend — static JSON stub for testing and CI.

Returns a pre-configured JSON string without performing any computation or
network I/O.  Useful for:

- Unit tests that need a deterministic enrichment result.
- CI pipelines where no LLM or spaCy model is available.
- Development environments where enrichment quality is not important.
- Baseline benchmarks comparing enrichment backends.

The ``response_json`` must be a valid ``ExtractionResult`` JSON string.
If it is not, ``MetadataProcessor`` will log the parse error and apply its
standard fallback chain.

Usage
-----
::

    from dewie.enrichment.backends.passthrough import PassthroughBackend

    backend = PassthroughBackend(
        name="test_stub",
        response_json=json.dumps({
            "document_type": "blog_post",
            "keywords": ["test", "stub"],
            "themes": ["testing"],
            "entities": [],
            "summary": "A test document.",
            "enrichment_quality_score": 50,
            "sentiment": 0.0,
            "tone": "neutral",
            "language": "en",
        }),
    )
"""

from __future__ import annotations


class PassthroughBackend:
    """
    Enrichment backend that returns a static pre-configured JSON string.

    All calls to ``complete()`` return the same ``response_json`` regardless
    of the prompt.  This makes test behaviour fully deterministic and removes
    any dependency on external services or local models.

    Args:
        name:          Registry identifier for this backend.  Must be unique.
        response_json: A valid JSON string matching the ``ExtractionResult``
                       schema.  Returned verbatim by every ``complete()`` call.
    """

    def __init__(self, name: str = "passthrough", response_json: str = "") -> None:
        self._name = name
        self._response_json = response_json or _DEFAULT_RESPONSE

    @property
    def name(self) -> str:
        """Registry identifier for this backend."""
        return self._name

    async def complete(self, prompt: str) -> str:  # noqa: ARG002
        """
        Return the static response JSON string.

        Args:
            prompt: Ignored.  Present for ``EnrichmentBackend`` Protocol conformance.

        Returns:
            The ``response_json`` string provided at construction time.
        """
        return self._response_json


_DEFAULT_RESPONSE: str = (
    '{"document_type": "other", "keywords": [], "themes": [], "entities": [], '
    '"summary": "", "enrichment_quality_score": 0, "sentiment": 0.0, "tone": "neutral", '
    '"language": "en"}'
)
