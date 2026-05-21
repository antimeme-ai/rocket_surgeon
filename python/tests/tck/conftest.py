"""TCK test harness — pytest-bdd fixtures and shared step imports.

All scenarios are xfail until Phase 1 provides a real server implementation.
The harness validates that:
  1. All .feature files parse correctly (Gherkin syntax)
  2. All step patterns have matching definitions (no StepNotFound)
  3. Fixture infrastructure is wired up
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from .steps.common import *  # noqa: F403 — register all step defs
from .steps.kv import *  # noqa: F403 — register KV-cache step defs

TCK_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "tck"


@pytest.fixture
def tck_root() -> pathlib.Path:
    return TCK_ROOT


@pytest.fixture
def daemon_process() -> Any:
    """Stub: starts and stops the rocket_surgeon daemon process.

    Phase 1 replaces this with a real subprocess over a Unix socket.
    """
    return None


@pytest.fixture
def unix_socket_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "rocket_surgeon.sock"


@pytest.fixture
def rpc_client(daemon_process: Any, unix_socket_path: pathlib.Path) -> Any:
    """Stub: JSON-RPC client connected to the daemon over Unix socket."""
    return _StubRpcClient()


@pytest.fixture
def session_state() -> dict[str, Any]:
    """Mutable dict tracking the latest response state across steps."""
    return {}


class _StubRpcClient:
    """Placeholder JSON-RPC client. All calls raise NotImplementedError."""

    def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError("stub: no server implementation yet")

    def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        raise NotImplementedError("stub: no server implementation yet")
