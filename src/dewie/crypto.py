# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""Fernet encryption for server registry API keys at rest."""

from __future__ import annotations

from cryptography.fernet import Fernet

from .config import settings


def _fernet() -> Fernet:
    key = settings.encryption_master_key
    if not key:
        raise RuntimeError("ENCRYPTION_MASTER_KEY is not set — required to store literal server API keys")
    return Fernet(key.encode())


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    return _fernet().decrypt(ciphertext.encode()).decode()
