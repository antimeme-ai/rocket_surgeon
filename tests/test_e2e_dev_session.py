"""Smoke test for the dev-session driver.

Verifies that scripts/dev-session.py reaches canonical state with a model
attached, dispatches a status request, and recovers from :reset.

This is the "coverage on us" piece: every change to the daemon, worker,
or harness must keep the dev loop alive.

Usage:
    PYTHONPATH=python python tests/test_e2e_dev_session.py
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = REPO_ROOT / "scripts" / "dev-session.py"
PYTHON_DIR = REPO_ROOT / "python"

READY_BANNER = "[dev] ready"
SETUP_TIMEOUT_SEC = 120  # build + attach + step
RESPONSE_TIMEOUT_SEC = 60


def _read_until(stream, predicate, timeout: float, tag: str) -> list[str]:
    """Collect stderr lines until predicate(line) is true or timeout elapses."""
    deadline = time.monotonic() + timeout
    collected: list[str] = []
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        ready, _, _ = select.select([stream], [], [], min(1.0, remaining))
        if not ready:
            continue
        raw = stream.readline()
        if not raw:
            msg = f"{tag}: stream closed before predicate matched"
            raise EOFError(msg)
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        collected.append(line)
        print(f"  [driver stderr] {line}", flush=True)
        if predicate(line):
            return collected
    joined = "\n".join(collected[-20:])
    msg = f"{tag}: timed out after {timeout}s. Last lines:\n{joined}"
    raise TimeoutError(msg)


def _read_json_response(stream, timeout: float) -> dict:
    """Read one JSON object from the driver's stdout.

    The driver pretty-prints responses with indent=2, so the closing brace
    appears alone on a line at column 0. We accumulate lines until we can
    parse a complete JSON object.
    """
    deadline = time.monotonic() + timeout
    buf = ""
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        ready, _, _ = select.select([stream], [], [], min(1.0, remaining))
        if not ready:
            continue
        chunk = stream.readline()
        if not chunk:
            msg = "driver stdout closed before response complete"
            raise EOFError(msg)
        buf += chunk.decode("utf-8", errors="replace")
        try:
            return json.loads(buf)
        except json.JSONDecodeError:
            continue
    msg = f"timed out waiting for JSON response. Buffer:\n{buf}"
    raise TimeoutError(msg)


def _send(proc: subprocess.Popen, line: str) -> None:
    print(f"  [test -> driver] {line}", flush=True)
    proc.stdin.write((line + "\n").encode("utf-8"))
    proc.stdin.flush()


def run_test() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_DIR)

    print(f"[test] spawning {DRIVER}")
    proc = subprocess.Popen(
        [sys.executable, str(DRIVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(REPO_ROOT),
    )

    try:
        # --- Step 1: wait for ready banner -------------------------------
        print("\n[test] Step 1: wait for driver ready")
        _read_until(
            proc.stderr,
            lambda line: line.startswith(READY_BANNER),
            SETUP_TIMEOUT_SEC,
            "ready banner",
        )
        print("  PASS")

        # --- Step 2: :state shortcut returns a status envelope -----------
        print("\n[test] Step 2: :state returns status envelope")
        _send(proc, ":state")
        resp = _read_json_response(proc.stdout, RESPONSE_TIMEOUT_SEC)
        assert resp.get("jsonrpc") == "2.0", f"bad jsonrpc: {resp}"
        assert resp.get("error") is None, f"status error: {resp.get('error')}"
        state = resp["result"]["state"]
        assert state["status"] == "stopped", f"expected stopped, got {state['status']}"
        assert state["tick_id"] is not None, "tick_id should be populated after step"
        first_session = state["session_id"]
        print(f"  session_id: {first_session}  tick_id: {state['tick_id']}")
        print("  PASS")

        # --- Step 3: raw JSON dispatch via _method -----------------------
        print("\n[test] Step 3: raw JSON request dispatch")
        _send(proc, '{"_method": "rocket/status"}')
        resp = _read_json_response(proc.stdout, RESPONSE_TIMEOUT_SEC)
        assert resp.get("error") is None, f"raw dispatch error: {resp.get('error')}"
        assert resp["result"]["state"]["session_id"] == first_session
        print("  PASS")

        # --- Step 4: :reset respawns daemon and returns to ready ---------
        print("\n[test] Step 4: :reset respawns + reaches ready again")
        _send(proc, ":reset")
        _read_until(
            proc.stderr,
            lambda line: line.startswith(READY_BANNER),
            SETUP_TIMEOUT_SEC,
            "ready banner after reset",
        )
        _send(proc, ":state")
        resp = _read_json_response(proc.stdout, RESPONSE_TIMEOUT_SEC)
        assert resp.get("error") is None
        new_session = resp["result"]["state"]["session_id"]
        assert new_session != first_session, (
            f"session_id should change across :reset (was {first_session})"
        )
        print(f"  new session_id: {new_session}")
        print("  PASS")

        # --- Step 5: :quit closes cleanly --------------------------------
        print("\n[test] Step 5: :quit closes cleanly")
        _send(proc, ":quit")
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            msg = "driver did not exit within 30s of :quit"
            raise AssertionError(msg) from None
        assert rc == 0, f"driver exited non-zero: {rc}"
        print("  PASS")

        print("\n[test] all steps passed")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    run_test()
