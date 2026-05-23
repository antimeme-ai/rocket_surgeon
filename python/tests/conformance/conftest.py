"""Shared fixtures for model conformance tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO_ROOT / "tests"))
from e2e_harness import (  # noqa: E402
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

__all__ = [
    "REPO_ROOT",
    "build_binaries",
    "make_request",
    "recv_message",
    "send_message",
    "spawn_daemon",
]
