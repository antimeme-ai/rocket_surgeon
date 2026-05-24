"""TCK test harness — real daemon subprocess over stdin/stdout JSON-RPC.

Spawns the three-process architecture (daemon → orchestrator → worker),
provides a real RPC client to step definitions, and manages lifecycle.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
from e2e_harness import (  # noqa: E402
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

from .steps.common import *  # noqa: E402, F403 — register all step defs
from .steps.kv import *  # noqa: E402, F403 — register KV-cache step defs

TCK_ROOT = REPO_ROOT / "tck"


def pytest_configure(config: pytest.Config) -> None:
    build_binaries()
    config.addinivalue_line(
        "markers",
        "deferred: scenario deferred — server feature not yet implemented",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    skip = pytest.mark.skip(reason="deferred: server feature not yet implemented")
    for item in items:
        if "deferred" in {m.name for m in item.iter_markers()}:
            item.add_marker(skip)


@pytest.fixture
def tck_root() -> pathlib.Path:
    return TCK_ROOT


@pytest.fixture
def daemon_proc() -> Generator[subprocess.Popen, None, None]:
    proc = spawn_daemon()
    yield proc
    proc.stdin.close()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture
def rpc(daemon_proc: subprocess.Popen) -> RpcClient:
    return RpcClient(daemon_proc)


@pytest.fixture
def session_state() -> dict[str, Any]:
    return {}


@pytest.fixture
def saved_values() -> dict[str, Any]:
    return {}


class RpcClient:
    """Real JSON-RPC client over Content-Length framed stdin/stdout."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self._req_id = 0
        self.last_response: dict[str, Any] = {}
        self.last_error: dict[str, Any] | None = None
        self.notifications: list[dict[str, Any]] = []

    def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._req_id += 1
        send_message(self._proc, make_request(method, params, self._req_id))
        while True:
            resp = recv_message(self._proc)
            if "id" not in resp and "method" in resp:
                self.notifications.append(resp)
                continue
            break
        self.last_response = resp
        if "error" in resp:
            self.last_error = resp["error"]
        else:
            self.last_error = None
        return resp

    def result_data(self) -> dict[str, Any]:
        return self.last_response.get("result", {}).get("data", {})

    def result_state(self) -> dict[str, Any]:
        return self.last_response.get("result", {}).get("state", {})

    def status(self) -> str | None:
        return self.result_state().get("status")

    def is_error(self) -> bool:
        return "error" in self.last_response

    def error_data(self) -> dict[str, Any]:
        if self.last_error is None:
            return {}
        return self.last_error.get("data", {})
