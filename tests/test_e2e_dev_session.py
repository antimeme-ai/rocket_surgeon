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
import re
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = REPO_ROOT / "scripts" / "dev-session.py"
PYTHON_DIR = REPO_ROOT / "python"

READY_PATTERN = re.compile(r"^\[dev\] ready ")
SETUP_TIMEOUT_SEC = 120  # build + attach + step
RESPONSE_TIMEOUT_SEC = 60


class StderrDrainer:
    """Background thread that continuously drains driver stderr.

    Echoes every line to the test's own stderr so the user sees daemon logs,
    and signals `ready_event` when the canonical "[dev] ready ..." banner
    appears. Call `arm()` to clear the event before each phase that expects
    a fresh ready banner (i.e. before `:reset`).
    """

    def __init__(self, stream) -> None:
        self.stream = stream
        self.ready_event = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def arm(self) -> None:
        self.ready_event.clear()

    def wait_ready(self, timeout: float) -> None:
        if not self.ready_event.wait(timeout=timeout):
            msg = f"ready banner not seen within {timeout}s"
            raise TimeoutError(msg)

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            raw = self.stream.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            print(f"  [driver stderr] {line}", file=sys.stderr, flush=True)
            if READY_PATTERN.match(line):
                self.ready_event.set()


def _read_json_response(stream, timeout: float) -> dict:
    """Read one JSON object from the driver's stdout.

    Reads raw bytes via os.read on the underlying fd to bypass Python's
    BufferedReader — `select` only sees the kernel buffer, so a mix of
    `select` + `readline` deadlocks when the buffered reader has bytes
    that select doesn't know about. The driver writes JSON in one
    `print(..., flush=True)` call, but the bytes can still arrive in
    multiple chunks; accumulate until `json.loads` succeeds.
    """
    fd = stream.fileno()
    deadline = time.monotonic() + timeout
    buf = b""
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        ready, _, _ = select.select([fd], [], [], min(1.0, remaining))
        if not ready:
            continue
        chunk = os.read(fd, 8192)
        if not chunk:
            msg = "driver stdout closed before response complete"
            raise EOFError(msg)
        buf += chunk
        try:
            return json.loads(buf.decode("utf-8"))
        except json.JSONDecodeError:
            continue
    msg = f"timed out waiting for JSON response. Buffer:\n{buf.decode('utf-8', errors='replace')}"
    raise TimeoutError(msg)


def _send(proc: subprocess.Popen, line: str) -> None:
    print(f"  [test -> driver] {line}", flush=True)
    proc.stdin.write((line + "\n").encode("utf-8"))
    proc.stdin.flush()


def run_test() -> None:  # noqa: PLR0915
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

    drainer = StderrDrainer(proc.stderr)
    drainer.start()

    try:
        # --- Step 1: wait for ready banner -------------------------------
        print("\n[test] Step 1: wait for driver ready")
        drainer.wait_ready(SETUP_TIMEOUT_SEC)
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
        drainer.arm()
        _send(proc, ":reset")
        drainer.wait_ready(SETUP_TIMEOUT_SEC)
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
        drainer.stop()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    run_test()
