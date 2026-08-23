"""Tests for dewie.crypto — Fernet encryption for server registry API keys."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet


def test_encrypt_decrypt_round_trip(monkeypatch):
    from dewie.config import settings as _settings
    from dewie.crypto import decrypt, encrypt

    monkeypatch.setattr(_settings, "encryption_master_key", Fernet.generate_key().decode())
    ciphertext = encrypt("sk-some-secret")
    assert ciphertext != "sk-some-secret"
    assert decrypt(ciphertext) == "sk-some-secret"


def test_encrypt_without_master_key_raises(monkeypatch):
    from dewie.config import settings as _settings
    from dewie.crypto import encrypt

    monkeypatch.setattr(_settings, "encryption_master_key", "")
    with pytest.raises(RuntimeError, match="ENCRYPTION_MASTER_KEY"):
        encrypt("sk-some-secret")


def test_decrypt_with_wrong_key_raises(monkeypatch):
    from dewie.config import settings as _settings
    from dewie.crypto import decrypt, encrypt

    monkeypatch.setattr(_settings, "encryption_master_key", Fernet.generate_key().decode())
    ciphertext = encrypt("sk-some-secret")

    monkeypatch.setattr(_settings, "encryption_master_key", Fernet.generate_key().decode())
    with pytest.raises(Exception):
        decrypt(ciphertext)
