"""Integration test: attach with adapter resolution returns component vocabulary.

Spawns the worker binary directly and exercises _host/attach to verify
that adapter resolution produces canonical component names.

Usage:
    PYTHONPATH=python pytest tests/test_e2e_adapter.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sysconfig
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "target" / "debug"
WORKER_BIN = TARGET_DIR / "rs-worker"
PYTHON_DIR = REPO_ROOT / "python"
TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TIMEOUT_SEC = 60


def _worker_env() -> dict[str, str]:
    python_libdir = sysconfig.get_config_var("LIBDIR") or ""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_DIR)
    env["DYLD_LIBRARY_PATH"] = python_libdir
    env["LD_LIBRARY_PATH"] = python_libdir
    return env


def send_message(proc: subprocess.Popen[bytes], msg: dict[str, Any]) -> dict[str, Any]:
    """Send a Content-Length framed JSON-RPC message and read the response."""
    body = json.dumps(msg).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    assert proc.stdin is not None
    proc.stdin.write(header + body)
    proc.stdin.flush()

    assert proc.stdout is not None
    deadline = time.monotonic() + TIMEOUT_SEC

    header_line = b""
    while not header_line.endswith(b"\r\n\r\n"):
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for response header")
        byte = proc.stdout.read(1)
        if not byte:
            raise RuntimeError("Worker closed stdout")
        header_line += byte

    content_length = int(header_line.decode().split(":")[1].strip().split("\r\n")[0])
    body_bytes = proc.stdout.read(content_length)
    result: dict[str, Any] = json.loads(body_bytes)
    return result


@pytest.fixture
def worker() -> Generator[subprocess.Popen[bytes], None, None]:
    if not WORKER_BIN.is_file():
        pytest.skip(f"Worker binary not found: {WORKER_BIN}")

    proc = subprocess.Popen(
        [str(WORKER_BIN), "--log-level", "warn"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_worker_env(),
    )
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def test_attach_returns_component_vocabulary(worker: subprocess.Popen[bytes]) -> None:
    resp = send_message(
        worker,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "_host/attach",
            "params": {
                "model_source": TINY_MODEL,
                "model_family": "llama",
                "device": "cpu",
                "rank": 0,
            },
        },
    )
    assert resp.get("error") is None, f"Attach failed: {resp.get('error')}"
    result = resp["result"]
    assert "component_vocabulary" in result
    assert isinstance(result["component_vocabulary"], list)
    assert len(result["component_vocabulary"]) > 0
    assert "q_proj" in result["component_vocabulary"]
    assert "k_proj" in result["component_vocabulary"]
    assert "v_proj" in result["component_vocabulary"]
    assert "o_proj" in result["component_vocabulary"]
    assert result["model_type"] == "llama"
    assert result["num_layers"] > 0


def test_attach_returns_module_tree(worker: subprocess.Popen[bytes]) -> None:
    resp = send_message(
        worker,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "_host/attach",
            "params": {
                "model_source": TINY_MODEL,
                "model_family": "llama",
                "device": "cpu",
                "rank": 0,
            },
        },
    )
    assert resp.get("error") is None, f"Attach failed: {resp.get('error')}"
    result = resp["result"]
    assert "module_tree" in result
    assert len(result["module_tree"]) > 0
    has_layer = any("layers.0" in m for m in result["module_tree"])
    assert has_layer
