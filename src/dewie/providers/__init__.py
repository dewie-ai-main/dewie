# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Provider abstraction for Dewie's LLM and embedding backends.

Supported chat providers:   anthropic, openrouter, openai, ollama
Supported embed providers:  openai, ollama

Usage:
    from dewie.providers import get_chat_provider, get_embedding_provider

    chat = get_chat_provider("aq_generation")
    result = await chat.complete(messages=[...])

    embed = get_embedding_provider()
    vectors = await embed.embed(["text one", "text two"])
"""

from .base import ChatProvider, EmbeddingProvider
from .factory import get_chat_provider, get_embedding_provider

__all__ = [
    "ChatProvider",
    "EmbeddingProvider",
    "get_chat_provider",
    "get_embedding_provider",
]
