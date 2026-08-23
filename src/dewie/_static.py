# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""Locate the UI static directory in both installed and checkout layouts."""

from __future__ import annotations

from pathlib import Path

_PKG = Path(__file__).resolve().parent


def static_dir() -> Path | None:
    """Return the static/ directory, or None if the UI is not available.

    Installed wheel: dewie/static (force-included by hatch).
    Source checkout: <repo>/static (two levels up from src/dewie/).
    """
    for candidate in (_PKG / "static", _PKG.parents[1] / "static"):
        if candidate.is_dir():
            return candidate
    return None
