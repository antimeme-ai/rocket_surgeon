"""End-to-end lifecycle test for the rocket_surgeon three-process architecture.

Spawns the real daemon binary, which in turn spawns the orchestrator and worker.
Exercises the full initialize -> attach -> detach lifecycle over the
Content-Length-framed JSON-RPC protocol on stdin/stdout.

Usage:
    PYTHONPATH=python python tests/test_e2e_lifecycle.py
"""

from __future__ import annotations

import subprocess

from e2e_harness import (
    MODEL_FAMILY,
    MODEL_SOURCE,
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_test() -> None:  # noqa: PLR0915
    """Execute the full e2e lifecycle test."""
    proc = spawn_daemon()

    try:
        # ------------------------------------------------------------------
        # Step 1: initialize
        # ------------------------------------------------------------------
        print("\n[test] Step 1: initialize")
        send_message(
            proc,
            make_request(
                "initialize",
                {"client_name": "e2e-test", "protocol_version": "0.1.0"},
                req_id=1,
            ),
        )
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
        assert data["model_family"] == MODEL_FAMILY, (
            f"Expected {MODEL_FAMILY}, got: {data['model_family']}"
        )
        assert data["num_layers"] == 32, f"Expected 32 stub layers, got: {data['num_layers']}"
        assert data["num_heads"] == 32, f"Expected 32 stub heads, got: {data['num_heads']}"
        assert data["hidden_dim"] == 4096, (
            f"Expected 4096 stub hidden_dim, got: {data['hidden_dim']}"
        )

        model_id = state["model_id"]
        print(f"  status: {state['status']}")
        print(f"  model_id: {model_id}")
        print(f"  model_family: {data['model_family']}")
        print(
            f"  num_layers: {data['num_layers']},"
            f" num_heads: {data['num_heads']},"
            f" hidden_dim: {data['hidden_dim']}"
        )
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
        assert state["model_id"] is None, (
            f"model_id should be null after detach, got: {state['model_id']}"
        )

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
