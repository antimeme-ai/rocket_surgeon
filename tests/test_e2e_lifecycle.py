"""End-to-end lifecycle test for the rocket_surgeon three-process architecture.

Spawns the real daemon binary, which in turn spawns the orchestrator and worker.
Exercises the full initialize -> attach -> detach lifecycle over the
Content-Length-framed JSON-RPC protocol on stdin/stdout.

Usage:
    PYTHONPATH=python python tests/test_e2e_lifecycle.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sysconfig
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "target" / "debug"
PYTHON_DIR = REPO_ROOT / "python"

DAEMON_BIN = TARGET_DIR / "rocket-surgeon"
ORCHESTRATOR_BIN = TARGET_DIR / "rs-orchestrator"
WORKER_BIN = TARGET_DIR / "rs-worker"

TIMEOUT_SEC = 60  # generous for model download on first run

# Tiny HuggingFace model — CPU-only, ~few MB
MODEL_SOURCE = "hf-internal-testing/tiny-random-LlamaForCausalLM"
MODEL_FAMILY = "llama"


# ---------------------------------------------------------------------------
# Content-Length framing helpers
# ---------------------------------------------------------------------------


def send_message(proc: subprocess.Popen, body: dict) -> None:
    """Send a Content-Length framed JSON-RPC message to the process's stdin."""
    payload = json.dumps(body).encode("utf-8")
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
    proc.stdin.write(header + payload)
    proc.stdin.flush()


def recv_message(proc: subprocess.Popen, timeout: float = TIMEOUT_SEC) -> dict:
    """Read a Content-Length framed JSON-RPC response from the process's stdout.

    Reads header lines until blank line, extracts Content-Length, then reads
    that many bytes of body.
    """
    deadline = time.monotonic() + timeout

    # Read headers until empty line
    content_length = None
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for response header")
        line = proc.stdout.readline()
        if not line:
            raise EOFError("Daemon stdout closed unexpectedly")
        line_str = line.decode("utf-8", errors="replace")
        stripped = line_str.rstrip("\r\n")
        if stripped == "":
            # End of headers
            break
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            if key.strip().lower() == "content-length":
                content_length = int(value.strip())

    if content_length is None:
        raise ValueError("Missing Content-Length header in response")

    # Read body
    body_bytes = b""
    while len(body_bytes) < content_length:
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for response body")
        chunk = proc.stdout.read(content_length - len(body_bytes))
        if not chunk:
            raise EOFError("Daemon stdout closed while reading body")
        body_bytes += chunk

    return json.loads(body_bytes.decode("utf-8"))


def make_request(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    """Build a JSON-RPC 2.0 request."""
    msg: dict = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


# ---------------------------------------------------------------------------
# Build step
# ---------------------------------------------------------------------------


def build_binaries() -> None:
    """Build all workspace binaries, handling PyO3 feature conflict."""
    print("[build] Building workspace (excluding PyO3 crates)...")
    subprocess.run(
        [
            "cargo", "build",
            "--workspace",
            "--exclude", "rocket-surgeon-python",
            "--exclude", "rocket-surgeon-worker",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    print("[build] Building worker separately (PyO3 auto-initialize)...")
    python_libdir = sysconfig.get_config_var("LIBDIR") or ""
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = python_libdir
    env["LD_LIBRARY_PATH"] = python_libdir
    subprocess.run(
        ["cargo", "build", "-p", "rocket-surgeon-worker"],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )

    # Verify all three binaries exist
    for binary in (DAEMON_BIN, ORCHESTRATOR_BIN, WORKER_BIN):
        if not binary.is_file():
            raise FileNotFoundError(f"Expected binary not found: {binary}")
    print("[build] All binaries built successfully.")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_test() -> None:
    """Execute the full e2e lifecycle test."""
    # Environment for the daemon — it needs PYTHONPATH for the worker's PyO3
    # skin, and DYLD_LIBRARY_PATH for libpython on macOS (SIP strips it).
    python_libdir = sysconfig.get_config_var("LIBDIR") or ""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_DIR)
    env["DYLD_LIBRARY_PATH"] = python_libdir
    env["LD_LIBRARY_PATH"] = python_libdir

    print(f"[env] PYTHONPATH={PYTHON_DIR}")
    print(f"[env] DYLD_LIBRARY_PATH={python_libdir}")
    print(f"[daemon] {DAEMON_BIN}")
    print(f"[daemon] --orchestrator-bin {ORCHESTRATOR_BIN}")
    print(f"[daemon] --worker-bin {WORKER_BIN}")

    proc = subprocess.Popen(
        [
            str(DAEMON_BIN),
            "--orchestrator-bin", str(ORCHESTRATOR_BIN),
            "--worker-bin", str(WORKER_BIN),
            "--log-level", "debug",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # inherit — daemon logs to stderr
        env=env,
    )

    try:
        # ------------------------------------------------------------------
        # Step 1: initialize
        # ------------------------------------------------------------------
        print("\n[test] Step 1: initialize")
        send_message(proc, make_request("initialize", {
            "client_name": "e2e-test",
            "protocol_version": "0.1.0",
        }, req_id=1))
        resp = recv_message(proc)
        print(f"  response id: {resp.get('id')}")

        assert resp.get("jsonrpc") == "2.0", f"Bad jsonrpc version: {resp}"
        assert resp.get("id") == 1, f"Bad response id: {resp}"
        assert resp.get("error") is None, f"Unexpected error: {resp.get('error')}"

        result = resp["result"]
        state = result["state"]
        assert state["status"] == "initialized", f"Expected initialized, got: {state['status']}"
        assert state["session_id"], "session_id should be non-empty"
        assert state["model_id"] is None, f"model_id should be null, got: {state['model_id']}"

        session_id = state["session_id"]
        print(f"  status: {state['status']}")
        print(f"  session_id: {session_id}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 2: attach (tiny HF model)
        # ------------------------------------------------------------------
        print("\n[test] Step 2: attach")
        send_message(proc, make_request("attach", {
            "model_path": MODEL_SOURCE,
            "model_family": MODEL_FAMILY,
            "device": "cpu",
            "num_ranks": 1,
        }, req_id=2))
        resp = recv_message(proc)
        print(f"  response id: {resp.get('id')}")

        assert resp.get("jsonrpc") == "2.0", f"Bad jsonrpc version: {resp}"
        assert resp.get("id") == 2, f"Bad response id: {resp}"
        assert resp.get("error") is None, f"Unexpected error: {resp.get('error')}"

        result = resp["result"]
        state = result["state"]
        assert state["status"] == "stopped", f"Expected stopped, got: {state['status']}"
        assert state["session_id"] == session_id, "session_id should be stable across calls"
        assert state["model_id"] is not None, "model_id should be set after attach"

        data = result["data"]
        assert data["model_family"] == MODEL_FAMILY, f"Expected {MODEL_FAMILY}, got: {data['model_family']}"
        assert data["num_layers"] == 32, f"Expected 32 stub layers, got: {data['num_layers']}"
        assert data["num_heads"] == 32, f"Expected 32 stub heads, got: {data['num_heads']}"
        assert data["hidden_dim"] == 4096, f"Expected 4096 stub hidden_dim, got: {data['hidden_dim']}"

        model_id = state["model_id"]
        print(f"  status: {state['status']}")
        print(f"  model_id: {model_id}")
        print(f"  model_family: {data['model_family']}")
        print(f"  num_layers: {data['num_layers']}, num_heads: {data['num_heads']}, hidden_dim: {data['hidden_dim']}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 3: detach
        # ------------------------------------------------------------------
        print("\n[test] Step 3: detach")
        send_message(proc, make_request("detach", {}, req_id=3))
        resp = recv_message(proc)
        print(f"  response id: {resp.get('id')}")

        assert resp.get("jsonrpc") == "2.0", f"Bad jsonrpc version: {resp}"
        assert resp.get("id") == 3, f"Bad response id: {resp}"
        assert resp.get("error") is None, f"Unexpected error: {resp.get('error')}"

        result = resp["result"]
        state = result["state"]
        assert state["status"] == "initialized", f"Expected initialized, got: {state['status']}"
        assert state["session_id"] == session_id, "session_id should be stable across calls"
        assert state["model_id"] is None, f"model_id should be null after detach, got: {state['model_id']}"

        data = result["data"]
        assert data["detached_model_id"] == model_id, (
            f"Expected detached_model_id={model_id}, got: {data['detached_model_id']}"
        )

        print(f"  status: {state['status']}")
        print(f"  model_id: {state['model_id']}")
        print(f"  detached_model_id: {data['detached_model_id']}")
        print("  PASS")

    finally:
        # ------------------------------------------------------------------
        # Cleanup: close stdin and wait for daemon to exit
        # ------------------------------------------------------------------
        print("\n[cleanup] Closing daemon stdin and waiting for exit...")
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
            print(f"[cleanup] Daemon exited with code {proc.returncode}")
        except subprocess.TimeoutExpired:
            print("[cleanup] Daemon did not exit in time, killing...")
            proc.kill()
            proc.wait()

    print("\n" + "=" * 60)
    print("PASS — full e2e lifecycle (initialize -> attach -> detach)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    build_binaries()
    run_test()
