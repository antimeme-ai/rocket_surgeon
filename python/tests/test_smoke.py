"""Smoke test to verify the package structure works."""

from __future__ import annotations

import rocket_surgeon


def test_import() -> None:
    assert rocket_surgeon.__version__ == "0.1.0"
