"""End-to-end stepping test for the rocket_surgeon three-process architecture.

Spawns the real daemon binary, attaches a tiny model, then exercises
rocket/step with different counts and granularities.

Usage:
    PYTHONPATH=python python tests/test_e2e_stepping.py
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

TIMEOUT_SEC = 60
MODEL_SOURCE = "hf-internal-testing/tiny-random-LlamaForCausalLM"
MODEL_FAMILY = "llama"


# ---------------------------------------------------------------------------
# Content-Length framing helpers
# ---------------------------------------------------------------------------


def send_message(proc: subprocess.Popen, body: dict) -> None:
    """Serialize *body* as JSON and send with Content-Length framing."""
    payload = json.dumps(body).encode("utf-8")
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
    proc.stdin.write(header + payload)
    proc.stdin.flush()


def recv_message(proc: subprocess.Popen, timeout: float = TIMEOUT_SEC) -> dict:
    """Read one Content-Length-framed JSON-RPC message from *proc.stdout*."""
    deadline = time.monotonic() + timeout
    content_length = None
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for response header")
        line = proc.stdout.readline()
        if not line:
            raise EOFError("Daemon stdout closed unexpectedly")
        stripped = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if stripped == "":
            break
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            if key.strip().lower() == "content-length":
                content_length = int(value.strip())
    if content_length is None:
        raise ValueError("Missing Content-Length header")
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
    """Build a JSON-RPC 2.0 request dict."""
    msg: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


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
# Build helpers
# ---------------------------------------------------------------------------


def build_binaries() -> None:
    """Build all workspace binaries."""
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
    for binary in (DAEMON_BIN, ORCHESTRATOR_BIN, WORKER_BIN):
        if not binary.is_file():
            raise FileNotFoundError(f"Expected binary not found: {binary}")
    print("[build] All binaries built successfully.")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_test() -> None:  # noqa: C901
    python_libdir = sysconfig.get_config_var("LIBDIR") or ""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_DIR)
    env["DYLD_LIBRARY_PATH"] = python_libdir
    env["LD_LIBRARY_PATH"] = python_libdir

    print(f"[env] PYTHONPATH={env['PYTHONPATH']}")
    print(f"[env] DYLD_LIBRARY_PATH={env['DYLD_LIBRARY_PATH']}")
    print(f"[env] DAEMON_BIN={DAEMON_BIN}")
    print(f"[env] ORCHESTRATOR_BIN={ORCHESTRATOR_BIN}")
    print(f"[env] WORKER_BIN={WORKER_BIN}")

    proc = subprocess.Popen(
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
        stderr=None,
        env=env,
    )

    try:
        # ------------------------------------------------------------------
        # Step 1: initialize
        # ------------------------------------------------------------------
        print("\n[test] Step 1: initialize")
        send_message(
            proc,
            make_request(
                "initialize",
                {
                    "client_name": "e2e-stepping-test",
                    "protocol_version": "0.1.0",
                },
                req_id=1,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None, f"Initialize error: {resp.get('error')}"
        assert resp["result"]["state"]["status"] == "initialized"
        session_id = resp["result"]["state"]["session_id"]
        print(f"  session_id: {session_id}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 2: attach
        # ------------------------------------------------------------------
        print("\n[test] Step 2: attach")
        send_message(
            proc,
            make_request(
                "attach",
                {
                    "model_path": MODEL_SOURCE,
                    "model_family": MODEL_FAMILY,
                    "device": "cpu",
                    "num_ranks": 1,
                },
                req_id=2,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 2)
        assert resp.get("error") is None, f"Attach error: {resp.get('error')}"
        assert resp["result"]["state"]["status"] == "stopped"
        assert_session_id(resp, session_id)
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 3: first step (count=1, component granularity)
        # ------------------------------------------------------------------
        print("\n[test] Step 3: step forward count=1 component")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "component",
                },
                req_id=3,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        result = resp["result"]
        state = result["state"]
        data = result["data"]
        assert_session_id(resp, session_id)
        assert_envelope_fields(state)
        assert state["status"] == "stopped", f"Expected stopped, got: {state['status']}"
        assert data["ticks_executed"] == 1, f"Expected 1 tick, got: {data['ticks_executed']}"
        assert "stopped_at" in data
        stopped_at = data["stopped_at"]
        assert isinstance(stopped_at["tick_id"], int), "tick_id must be integer"
        assert isinstance(stopped_at["layer"], int), "layer must be integer"
        assert isinstance(stopped_at["component"], str), "component must be string"
        assert isinstance(stopped_at["event"], str), "event must be string"
        assert isinstance(stopped_at["direction"], str), "direction must be string"
        assert stopped_at["direction"] == "forward"
        assert stopped_at["event"] in ("input", "output"), (
            f"event must be input or output, got: {stopped_at['event']}"
        )
        assert stopped_at["layer"] == 0, (
            f"First step should start at layer 0, got: {stopped_at['layer']}"
        )
        tick_1 = state["tick_id"]
        print(f"  ticks_executed: {data['ticks_executed']}")
        print(f"  tick_id: {tick_1}")
        print(
            f"  stopped_at: layer={stopped_at['layer']}"
            f" component={stopped_at['component']}"
            f" event={stopped_at['event']}"
        )
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 4: second step — tick_id must increase
        # ------------------------------------------------------------------
        print("\n[test] Step 4: second step — tick_id monotonicity")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "component",
                },
                req_id=4,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        assert_session_id(resp, session_id)
        tick_2 = resp["result"]["state"]["tick_id"]
        assert tick_2 > tick_1, f"tick_id should increase: {tick_1} -> {tick_2}"
        print(f"  tick_id: {tick_1} -> {tick_2} (monotonic: OK)")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 5: third step — three-step monotonicity (TCK requires a < b < c)
        # ------------------------------------------------------------------
        print("\n[test] Step 5: third step — three-step monotonicity")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "component",
                },
                req_id=5,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 5)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        tick_3 = resp["result"]["state"]["tick_id"]
        assert tick_3 > tick_2, f"tick_id should increase: {tick_2} -> {tick_3}"
        print(f"  tick_id: {tick_1} < {tick_2} < {tick_3} (monotonic: OK)")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 6: multi-tick step (count=3)
        # ------------------------------------------------------------------
        print("\n[test] Step 6: step forward count=3")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 3,
                    "granularity": "component",
                },
                req_id=6,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 6)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        data_6 = resp["result"]["data"]
        assert data_6["ticks_executed"] == 3, (
            f"Expected 3 ticks, got: {data_6['ticks_executed']}"
        )
        tick_after_multi = resp["result"]["state"]["tick_id"]
        assert tick_after_multi > tick_3, "tick_id must advance after multi-step"
        print(f"  ticks_executed: {data_6['ticks_executed']}")
        print(f"  tick_id: {tick_after_multi}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 7: layer granularity step
        # ------------------------------------------------------------------
        print("\n[test] Step 7: step forward count=1 layer granularity")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "layer",
                },
                req_id=7,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 7)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        data_7 = resp["result"]["data"]
        assert data_7["ticks_executed"] == 1, (
            f"Expected 1 layer tick, got: {data_7['ticks_executed']}"
        )
        tick_after_layer = resp["result"]["state"]["tick_id"]
        assert tick_after_layer > tick_after_multi, "tick_id must advance after layer step"
        print(f"  ticks_executed (layer): {data_7['ticks_executed']}")
        print(f"  tick_id: {tick_after_layer}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 8: backward step → CAPABILITY_NOT_SUPPORTED
        # ------------------------------------------------------------------
        print("\n[test] Step 8: backward step returns error")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "backward",
                    "count": 1,
                    "granularity": "component",
                },
                req_id=8,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 8)
        assert resp.get("error") is not None, "Expected error for backward step"
        error = resp["error"]
        assert "data" in error, "Error should include structured data"
        assert error["data"]["error_code"] == "CAPABILITY_NOT_SUPPORTED", (
            f"Expected CAPABILITY_NOT_SUPPORTED, got: {error['data'].get('error_code')}"
        )
        assert error["data"]["severity"] == "recoverable", (
            f"Expected recoverable severity, got: {error['data'].get('severity')}"
        )
        print(f"  error_code: {error['data']['error_code']}")
        print(f"  severity: {error['data']['severity']}")
        print(f"  message: {error['message']}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 9: detach
        # ------------------------------------------------------------------
        print("\n[test] Step 9: detach")
        send_message(proc, make_request("detach", {}, req_id=9))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 9)
        assert resp.get("error") is None, f"Detach error: {resp.get('error')}"
        assert resp["result"]["state"]["status"] == "initialized"
        assert_session_id(resp, session_id)
        print("  PASS")

    finally:
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
    print("PASS — e2e stepping")
    print("  initialize -> attach -> step x3 (monotonicity)")
    print("  -> step count=3 -> step layer -> backward error -> detach")
    print("=" * 60)


if __name__ == "__main__":
    build_binaries()
    run_test()
