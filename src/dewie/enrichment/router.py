# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
EnrichmentRouter — selects an enrichment backend for a given document.

The router sits between the ``MetadataProcessor`` and the ``BackendRegistry``.
It applies a prioritised list of rules to each document and returns the first
matching backend.  When no rule matches, it returns the registry default.

Rules are declarative dicts loaded from ``settings.enrichment_routing_rules``
(a JSON array).  This keeps routing logic in config, not code, making it easy
to tune without redeployment.

Supported rule predicates
--------------------------
- ``if_body_shorter_than: int``  → match documents with body length < value
- ``if_body_longer_than: int``   → match documents with body length ≥ value
- ``if_document_type: str``      → match documents where ``document_type``
                                   equals the given value (post-classification)
- ``default: str``               → always matches; used as final rule

Example config (``ENRICHMENT_ROUTING_RULES`` in ``.env``)::

    ENRICHMENT_ROUTING_RULES='[
      {"if_body_longer_than":  5000, "use_backend": "ollama_3b"},
      {"default":                    "ollama_3b"}
    ]'

Custom routers
--------------
The router interface is ``select(doc: ContentDocument) -> EnrichmentBackend``.
Any callable or class satisfying this signature can be injected into
``MetadataProcessor`` instead of ``EnrichmentRouter``.  This allows entirely
custom routing logic (ML classifiers, A/B experiments, etc.) without changing
the processor.

Example custom router::

    class MyRouter:
        def select(self, doc: ContentDocument) -> EnrichmentBackend:
            if "arxiv.org" in doc.url:
                return registry.get("claude_sonnet")
            return registry.default()

    processor = MetadataProcessor(router=MyRouter(), registry=registry)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from dewie.enrichment.base import EnrichmentBackend

if TYPE_CHECKING:
    from dewie.config import Settings
    from dewie.enrichment.registry import BackendRegistry
    from dewie.models.content import ContentDocument

logger = logging.getLogger(__name__)

# Type alias for a routing rule dict
RoutingRule = dict[str, Any]


class EnrichmentRouter:
    """
    Rule-based enrichment backend selector.

    Rules are evaluated in order.  The first rule that matches the document
    determines the backend.  If no rule matches, the registry default is used.

    Args:
        registry: The ``BackendRegistry`` containing all available backends.
        rules:    Ordered list of routing rule dicts.  Each rule must contain
                  exactly one predicate key (``if_body_shorter_than``,
                  ``if_body_longer_than``, ``if_document_type``, or ``default``)
                  and a ``use_backend`` value (except ``default``, which uses
                  its value directly as the backend name).
    """

    def __init__(
        self,
        registry: BackendRegistry,
        rules: list[RoutingRule] | None = None,
    ) -> None:
        self._registry = registry
        self._rules: list[RoutingRule] = rules or []

    def select(self, doc: ContentDocument) -> EnrichmentBackend:
        """
        Select an enrichment backend for the given document.

        Evaluates routing rules in order and returns the first matching backend.
        Falls back to the registry default if no rule matches.

        Args:
            doc: The document to be enriched.  Field values available at the
                 time of selection: ``url``, ``title``, ``body``, ``source``,
                 ``document_type`` (if pre-classified), ``status``.

        Returns:
            The selected ``EnrichmentBackend`` instance.
        """
        body_len = len(doc.body)

        for rule in self._rules:
            backend_name = _resolve_backend_name(rule)
            if backend_name is None:
                continue

            if "if_body_shorter_than" in rule:
                threshold = int(rule["if_body_shorter_than"])
                if body_len < threshold:
                    logger.debug(
                        "Router: body_len=%d < %d → backend=%s (doc=%s)",
                        body_len,
                        threshold,
                        backend_name,
                        doc.url,
                    )
                    return self._safe_get(backend_name)

            elif "if_body_longer_than" in rule:
                threshold = int(rule["if_body_longer_than"])
                if body_len >= threshold:
                    logger.debug(
                        "Router: body_len=%d ≥ %d → backend=%s (doc=%s)",
                        body_len,
                        threshold,
                        backend_name,
                        doc.url,
                    )
                    return self._safe_get(backend_name)

            elif "if_document_type" in rule:
                doc_type = str(rule["if_document_type"])
                if doc.document_type is not None and str(doc.document_type) == doc_type:
                    logger.debug(
                        "Router: document_type=%s → backend=%s (doc=%s)",
                        doc_type,
                        backend_name,
                        doc.url,
                    )
                    return self._safe_get(backend_name)

            elif "default" in rule:
                logger.debug("Router: default → backend=%s (doc=%s)", backend_name, doc.url)
                return self._safe_get(backend_name)

        # No rule matched
        logger.debug("Router: no rule matched for doc=%s; using registry default.", doc.url)
        return self._registry.default()

    def _safe_get(self, name: str) -> EnrichmentBackend:
        """
        Retrieve a backend by name, falling back to the registry default on error.

        This prevents a misconfigured routing rule from crashing enrichment.
        """
        try:
            return self._registry.get(name)
        except KeyError:
            logger.warning("Router rule references unknown backend '%s'; using default.", name)
            return self._registry.default()

    # ── Config-driven construction ────────────────────────────────────────────

    @classmethod
    def from_config(cls, settings: Settings, registry: BackendRegistry) -> EnrichmentRouter:
        """
        Build an ``EnrichmentRouter`` from application settings.

        Parses ``settings.enrichment_routing_rules`` (a JSON array string).

        Args:
            settings: Application settings singleton.
            registry: Fully populated ``BackendRegistry``.

        Returns:
            A configured ``EnrichmentRouter``.
        """
        rules: list[RoutingRule] = []
        raw = getattr(settings, "enrichment_routing_rules", "[]")
        if raw and raw.strip() not in ("", "[]"):
            try:
                rules = json.loads(raw)
            except json.JSONDecodeError:
                logger.error("ENRICHMENT_ROUTING_RULES is not valid JSON; using no rules.")
        return cls(registry=registry, rules=rules)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_backend_name(rule: RoutingRule) -> str | None:
    """Extract the backend name from a rule dict."""
    if "use_backend" in rule:
        return str(rule["use_backend"])
    if "default" in rule:
        return str(rule["default"])
    return None
