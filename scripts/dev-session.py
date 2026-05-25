#!/usr/bin/env python3
"""Long-lived dev driver — owns a daemon, holds canonical state, exposes a REPL.

Spawns the rocket-surgeon daemon as a child, runs canonical setup
(initialize -> attach tiny-llama -> capture-all probe -> step 1), then drops
to a JSON-RPC REPL. On daemon death, respawns and replays setup.

Usage:
    PYTHONPATH=python python scripts/dev-session.py

REPL grammar (one entry per line):
    {"_method": "rocket/inspect", "target": "llama:0:0:attn.o_proj:output"}
    :state
    :reset
    :setup
    :script path/to/file.txt
    :help
    :quit

Stdout = JSON responses. Stderr = banners and warnings.
"""

from __future__ import annotations

import contextlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from e2e_harness import (  # noqa: E402
    MODEL_FAMILY,
    MODEL_SOURCE,
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)


PROBE_ID = "dev-capture-all"
SETUP_TIMEOUT_SEC = 60


class DriverState:
    """Mutable holder for the per-daemon lifecycle."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.session_id: str | None = None
        self.last_status: str | None = None
        self.last_tick: int | None = None
        self.req_seq = 1000  # leave low ids for setup

    def next_id(self) -> int:
        self.req_seq += 1
        return self.req_seq


def _banner(line: str) -> None:
    print(f"[dev] {line}", file=sys.stderr, flush=True)


def _send(state: DriverState, method: str, params: dict | None, req_id: int) -> dict:
    if state.proc is None:
        msg = "daemon is not running"
        raise RuntimeError(msg)
    send_message(state.proc, make_request(method, params, req_id=req_id))
    return recv_message(state.proc, timeout=SETUP_TIMEOUT_SEC)


def _update_state_from(state: DriverState, resp: dict) -> None:
    result = resp.get("result") or {}
    envelope = result.get("state") or {}
    if envelope.get("session_id"):
        state.session_id = envelope["session_id"]
    if envelope.get("status"):
        state.last_status = envelope["status"]
    if envelope.get("tick_id") is not None:
        state.last_tick = envelope["tick_id"]


def _setup(state: DriverState) -> None:
    """Run the canonical setup sequence against the current daemon."""
    _banner("initialize ...")
    resp = _send(
        state,
        "initialize",
        {"client_name": "dev-session", "protocol_version": "0.3.0"},
        req_id=1,
    )
    if resp.get("error"):
        msg = f"initialize failed: {resp['error']}"
        raise RuntimeError(msg)
    _update_state_from(state, resp)

    _banner(f"attach {MODEL_SOURCE} ...")
    resp = _send(
        state,
        "attach",
        {
            "model_path": MODEL_SOURCE,
            "model_family": MODEL_FAMILY,
            "device": "cpu",
            "num_ranks": 1,
        },
        req_id=2,
    )
    if resp.get("error"):
        msg = f"attach failed: {resp['error']}"
        raise RuntimeError(msg)
    _update_state_from(state, resp)

    _banner("define capture-all probe ...")
    resp = _send(
        state,
        "rocket/probe",
        {
            "action": "define",
            "probe": {
                "id": PROBE_ID,
                "point": "*:*:*:*:*:*",
                "action": "capture",
                "config": {"summary": True, "capture_tensor": False},
                "enabled": True,
                "priority": 0,
            },
        },
        req_id=3,
    )
    if resp.get("error"):
        # Re-definition against a live daemon will collide; not fatal.
        _banner(f"probe define non-fatal: {resp['error'].get('message', resp['error'])}")
    else:
        _update_state_from(state, resp)

    _banner("step forward count=1 ...")
    resp = _send(
        state,
        "rocket/step",
        {"direction": "forward", "count": 1, "granularity": "component"},
        req_id=4,
    )
    if resp.get("error"):
        msg = f"step failed: {resp['error']}"
        raise RuntimeError(msg)
    _update_state_from(state, resp)


def _spawn_and_setup(state: DriverState) -> None:
    # spawn_daemon() prints env/path info to stdout (built for e2e tests that
    # own stdout). The driver reserves stdout for JSON responses, so redirect.
    with contextlib.redirect_stdout(sys.stderr):
        state.proc = spawn_daemon()
    _setup(state)
    _banner(
        f"ready · session={state.session_id} status={state.last_status} tick={state.last_tick}"
    )
    _banner("type :help for commands. JSON requests use _method to name the verb.")


def _kill(state: DriverState) -> None:
    if state.proc is None:
        return
    try:
        state.proc.terminate()
        state.proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        state.proc.kill()
        state.proc.wait(timeout=5)
    state.proc = None
    state.session_id = None
    state.last_status = None
    state.last_tick = None


def _print_response(resp: dict) -> None:
    print(json.dumps(resp, indent=2), flush=True)


def _cmd_help() -> None:
    print(
        """Commands:
  {"_method": "<verb>", ...params}   send JSON-RPC request, _method names the verb
  :state                              shortcut for rocket/status
  :setup                              re-run canonical setup against current daemon
  :reset                              kill daemon, respawn, replay setup
  :script <path>                      run newline-delimited commands from file
  :help                               this text
  :quit                               clean shutdown

Examples:
  {"_method": "rocket/status"}
  {"_method": "rocket/inspect", "target": "llama:0:0:attn.o_proj:output"}
  {"_method": "rocket/step", "direction": "forward", "count": 3, "granularity": "component"}
""",
        file=sys.stderr,
        flush=True,
    )


def _dispatch_json(state: DriverState, payload: dict) -> None:
    method = payload.pop("_method", None)
    if not method:
        _banner('error: request missing "_method"')
        return
    try:
        resp = _send(state, method, payload or None, req_id=state.next_id())
    except (EOFError, BrokenPipeError) as exc:
        _banner(f"daemon died ({exc}) — respawning")
        _kill(state)
        _spawn_and_setup(state)
        return
    _update_state_from(state, resp)
    _print_response(resp)


def _dispatch_command(state: DriverState, line: str) -> bool:
    """Handle a `:command` line. Returns False if the loop should exit."""
    parts = shlex.split(line)
    cmd = parts[0]
    args = parts[1:]
    if cmd == ":quit":
        return False
    if cmd == ":help":
        _cmd_help()
        return True
    if cmd == ":state":
        _dispatch_json(state, {"_method": "rocket/status"})
        return True
    if cmd == ":setup":
        _setup(state)
        _banner(f"setup re-run · status={state.last_status} tick={state.last_tick}")
        return True
    if cmd == ":reset":
        _kill(state)
        _spawn_and_setup(state)
        return True
    if cmd == ":script":
        if not args:
            _banner("error: :script requires a path")
            return True
        path = Path(args[0])
        if not path.is_file():
            _banner(f"error: not a file: {path}")
            return True
        for script_line in path.read_text().splitlines():
            stripped = script_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            _run_line(state, stripped)
        return True
    _banner(f"unknown command: {cmd}  (try :help)")
    return True


def _run_line(state: DriverState, line: str) -> bool:
    """Dispatch one stripped line. Returns False if loop should exit."""
    if line.startswith(":"):
        return _dispatch_command(state, line)
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        _banner(f"invalid JSON: {exc}")
        return True
    if not isinstance(payload, dict):
        _banner("error: JSON payload must be an object")
        return True
    _dispatch_json(state, payload)
    return True


def main() -> int:
    _banner("building binaries ...")
    # build_binaries() prints progress to stdout (built for e2e tests that own
    # stdout). The driver reserves stdout for JSON responses, so redirect.
    with contextlib.redirect_stdout(sys.stderr):
        build_binaries()
    state = DriverState()
    try:
        _spawn_and_setup(state)
    except Exception as exc:
        _banner(f"fatal during setup: {exc}")
        _kill(state)
        return 1

    try:
        # NB: explicit readline() loop instead of `for line in sys.stdin`,
        # which uses readahead buffering on pipes and would delay dispatch
        # until the buffer fills.
        while True:
            raw_line = sys.stdin.readline()
            if not raw_line:
                break
            stripped = raw_line.strip()
            if not stripped:
                continue
            if not _run_line(state, stripped):
                break
    except KeyboardInterrupt:
        _banner("interrupt")
    finally:
        _kill(state)
        _banner("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
