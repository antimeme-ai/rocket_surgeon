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

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "target" / "debug"
PYTHON_DIR = REPO_ROOT / "python"

DAEMON_BIN = TARGET_DIR / "rocket-surgeon"
ORCHESTRATOR_BIN = TARGET_DIR / "rs-orchestrator"
WORKER_BIN = TARGET_DIR / "rs-worker"

TIMEOUT_SEC = 60
MODEL_SOURCE = "hf-internal-testing/tiny-random-LlamaForCausalLM"
MODEL_FAMILY = "llama"


def send_message(proc: subprocess.Popen, body: dict) -> None:
    payload = json.dumps(body).encode("utf-8")
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
    proc.stdin.write(header + payload)
    proc.stdin.flush()


def recv_message(proc: subprocess.Popen, timeout: float = TIMEOUT_SEC) -> dict:
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
    msg: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def build_binaries() -> None:
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


def run_test() -> None:
    python_libdir = sysconfig.get_config_var("LIBDIR") or ""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_DIR)
    env["DYLD_LIBRARY_PATH"] = python_libdir
    env["LD_LIBRARY_PATH"] = python_libdir
    env["HF_HUB_OFFLINE"] = "1"

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
        # Step 1: initialize
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
        assert resp.get("error") is None, f"Initialize error: {resp.get('error')}"
        assert resp["result"]["state"]["status"] == "initialized"
        session_id = resp["result"]["state"]["session_id"]
        print(f"  session_id: {session_id}")
        print("  PASS")

        # Step 2: attach
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
        assert resp.get("error") is None, f"Attach error: {resp.get('error')}"
        assert resp["result"]["state"]["status"] == "stopped"
        print("  PASS")

        # Step 3: first step (count=1, component granularity)
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
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        result = resp["result"]
        state = result["state"]
        data = result["data"]
        assert state["status"] == "stopped", f"Expected stopped, got: {state['status']}"
        assert data["ticks_executed"] == 1, f"Expected 1 tick, got: {data['ticks_executed']}"
        assert "stopped_at" in data
        stopped_at = data["stopped_at"]
        assert "tick_id" in stopped_at
        assert "layer" in stopped_at
        assert "component" in stopped_at
        assert "event" in stopped_at
        assert "direction" in stopped_at
        assert stopped_at["direction"] == "forward"
        tick_1 = state["tick_id"]
        print(f"  ticks_executed: {data['ticks_executed']}")
        print(f"  tick_id: {tick_1}")
        print(
            f"  stopped_at: layer={stopped_at['layer']}"
            f" component={stopped_at['component']}"
            f" event={stopped_at['event']}"
        )
        print("  PASS")

        # Step 4: second step — tick_id must increase
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
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        tick_2 = resp["result"]["state"]["tick_id"]
        assert tick_2 > tick_1, f"tick_id should increase: {tick_1} -> {tick_2}"
        print(f"  tick_id: {tick_1} -> {tick_2} (monotonic: OK)")
        print("  PASS")

        # Step 5: backward step → CAPABILITY_NOT_SUPPORTED
        print("\n[test] Step 5: backward step returns error")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "backward",
                    "count": 1,
                    "granularity": "component",
                },
                req_id=5,
            ),
        )
        resp = recv_message(proc)
        assert resp.get("error") is not None, "Expected error for backward step"
        print(f"  error: {resp['error']['message']}")
        print("  PASS")

        # Step 6: detach
        print("\n[test] Step 6: detach")
        send_message(proc, make_request("detach", {}, req_id=6))
        resp = recv_message(proc)
        assert resp.get("error") is None, f"Detach error: {resp.get('error')}"
        assert resp["result"]["state"]["status"] == "initialized"
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
    print("PASS — e2e stepping (initialize -> attach -> step x2 -> detach)")
    print("=" * 60)


if __name__ == "__main__":
    build_binaries()
    run_test()
