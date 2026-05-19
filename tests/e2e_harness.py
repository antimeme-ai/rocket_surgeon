"""Shared E2E test harness for the rocket_surgeon three-process architecture.

Provides Content-Length framed JSON-RPC communication, common assertions,
build helpers, and daemon spawn logic used by the lifecycle and stepping tests.

NOT used by test_e2e_adapter.py — that test talks directly to the worker
with a slightly different protocol pattern.
"""

from __future__ import annotations

import json
import os
import select
import site
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


def _has_buffered_data(stream) -> bool:
    """Return True if *stream*'s ``BufferedReader`` already holds unread data.

    ``peek(1)`` would be the documented way, but it **blocks** when the
    internal buffer is empty (it calls ``raw.read()`` to refill).  Instead
    we temporarily flip the underlying fd to non-blocking so ``peek`` can
    return immediately.
    """
    if not hasattr(stream, "peek") or not hasattr(stream, "fileno"):
        return False
    import os
    fd = stream.fileno()
    was_blocking = os.get_blocking(fd)
    try:
        os.set_blocking(fd, False)
        data = stream.peek(1)
        return bool(data)
    except BlockingIOError:
        return False
    finally:
        os.set_blocking(fd, was_blocking)


def _wait_readable(stream, remaining: float) -> None:
    """Block until *stream* is readable or *remaining* seconds elapse.

    Checks Python's internal buffer first (non-blocking peek) so that
    data already read from the OS fd by a previous ``readline``/``read``
    is not missed, then falls back to ``select``.
    """
    if remaining <= 0:
        msg = "Timed out waiting for data on stdout"
        raise TimeoutError(msg)
    if _has_buffered_data(stream):
        return
    readable, _, _ = select.select([stream], [], [], remaining)
    if not readable:
        msg = "Timed out waiting for data on stdout"
        raise TimeoutError(msg)


def recv_message(proc: subprocess.Popen, timeout: float = TIMEOUT_SEC) -> dict:
    """Read a Content-Length framed JSON-RPC response from *proc.stdout*.

    Uses ``select.select`` before every blocking read so the call never hangs
    past *timeout* seconds, even on partial writes from the daemon.
    """
    deadline = time.monotonic() + timeout

    # --- Read headers until blank line ---
    content_length = None
    while True:
        remaining = deadline - time.monotonic()
        _wait_readable(proc.stdout, remaining)
        line = proc.stdout.readline()
        if not line:
            msg = "Daemon stdout closed unexpectedly"
            raise EOFError(msg)
        stripped = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if stripped == "":
            break
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            if key.strip().lower() == "content-length":
                content_length = int(value.strip())

    if content_length is None:
        msg = "Missing Content-Length header in response"
        raise ValueError(msg)

    # --- Read body ---
    body_bytes = b""
    while len(body_bytes) < content_length:
        remaining = deadline - time.monotonic()
        _wait_readable(proc.stdout, remaining)
        chunk = proc.stdout.read(content_length - len(body_bytes))
        if not chunk:
            msg = "Daemon stdout closed while reading body"
            raise EOFError(msg)
        body_bytes += chunk

    return json.loads(body_bytes.decode("utf-8"))


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


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
# Common assertions
# ---------------------------------------------------------------------------


def assert_jsonrpc(resp: dict, req_id: int) -> None:
    """Validate JSON-RPC 2.0 envelope basics."""
    assert resp.get("jsonrpc") == "2.0", f"Bad jsonrpc version: {resp.get('jsonrpc')}"
    assert resp.get("id") == req_id, f"Expected id={req_id}, got {resp.get('id')}"


def assert_session_id(resp: dict, expected: str) -> None:
    """Validate session_id stability across responses."""
    actual = resp["result"]["state"]["session_id"]
    assert actual == expected, f"session_id drift: {expected} -> {actual}"


def assert_envelope_fields(state: dict) -> None:
    """Validate that the session state envelope has all required fields."""
    required = {
        "session_id": str,
        "status": str,
        "tick_id": int,
        "available_actions": list,
    }
    for field, typ in required.items():
        assert field in state, f"Missing envelope field: {field}"
        assert isinstance(state[field], typ), (
            f"Envelope field {field}: expected {typ.__name__}, got {type(state[field]).__name__}"
        )


# ---------------------------------------------------------------------------
# Build step
# ---------------------------------------------------------------------------


def build_binaries() -> None:
    """Build all workspace binaries, handling PyO3 feature conflict."""
    print("[build] Building workspace (excluding PyO3 crates)...")
    subprocess.run(
        [
            "cargo",
            "build",
            "--workspace",
            "--exclude",
            "rocket-surgeon-python",
            "--exclude",
            "rocket-surgeon-worker",
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
            msg = f"Expected binary not found: {binary}"
            raise FileNotFoundError(msg)
    print("[build] All binaries built successfully.")


# ---------------------------------------------------------------------------
# Daemon spawning
# ---------------------------------------------------------------------------


def spawn_daemon(env_extras: dict[str, str] | None = None) -> subprocess.Popen:
    """Spawn the daemon binary with standard arguments.

    Returns the Popen handle with stdin/stdout pipes.
    *env_extras* is merged into the base environment (PYTHONPATH, lib paths).
    """
    python_libdir = sysconfig.get_config_var("LIBDIR") or ""
    env = os.environ.copy()
    # Worker uses PyO3 auto-initialize: the embedded interpreter derives
    # PYTHONHOME from libpython's location (the uv-managed Python), not from
    # the venv. Extend PYTHONPATH with the venv site-packages so torch and
    # other venv-installed packages are visible.
    env["PYTHONPATH"] = os.pathsep.join([str(PYTHON_DIR), *site.getsitepackages()])
    env["DYLD_LIBRARY_PATH"] = python_libdir
    env["LD_LIBRARY_PATH"] = python_libdir
    if env_extras:
        env.update(env_extras)

    print(f"[env] PYTHONPATH={env['PYTHONPATH']}")
    print(f"[env] DYLD_LIBRARY_PATH={env['DYLD_LIBRARY_PATH']}")
    print(f"[daemon] {DAEMON_BIN}")
    print(f"[daemon] --orchestrator-bin {ORCHESTRATOR_BIN}")
    print(f"[daemon] --worker-bin {WORKER_BIN}")

    return subprocess.Popen(
        [
            str(DAEMON_BIN),
            "--orchestrator-bin",
            str(ORCHESTRATOR_BIN),
            "--worker-bin",
            str(WORKER_BIN),
            "--log-level",
            "debug",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # inherit — daemon logs to stderr
        env=env,
    )
