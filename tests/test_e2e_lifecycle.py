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


def run_test() -> None:
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
                {"client_name": "e2e-test", "protocol_version": "0.3.0"},
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
        # BEAD-0008: attach response now carries real model metadata from
        # the worker, not per-family stubs. tiny-random-LlamaForCausalLM
        # config: 2 hidden layers, 4 attention heads, hidden_size 16.
        assert data["num_layers"] == 2, f"Expected 2 real layers, got: {data['num_layers']}"
        assert data["num_heads"] == 4, f"Expected 4 real heads, got: {data['num_heads']}"
        assert data["hidden_dim"] == 16, f"Expected 16 real hidden_dim, got: {data['hidden_dim']}"

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

        # ------------------------------------------------------------------
        # Step 4: BEAD-0008 — attach with broken backend returns
        # BACKEND_ATTACH_FAILED, session stays in initialized state
        # ------------------------------------------------------------------
        print("\n[test] Step 4: attach with broken backend returns error")
        send_message(
            proc,
            make_request(
                "attach",
                {
                    "model_path": "/models/does-not-exist-anywhere",
                    "model_family": "llama",
                    "device": "cpu",
                    "num_ranks": 1,
                },
                req_id=4,
            ),
        )
        resp = recv_message(proc)
        print(f"  response id: {resp.get('id')}")

        err = resp.get("error")
        assert err is not None, f"Expected error, got success: {resp}"
        err_data = err.get("data") or {}
        assert err_data.get("error_code") == "BACKEND_ATTACH_FAILED", (
            f"Expected BACKEND_ATTACH_FAILED, got: {err_data.get('error_code')}"
        )
        assert err_data.get("severity") == "recoverable", (
            f"Expected recoverable, got: {err_data.get('severity')}"
        )
        ctx = err_data.get("context") or {}
        assert "backend_error" in ctx, f"Expected backend_error in context, got: {ctx}"

        # Confirm session state did not mutate — a follow-up status call
        # should still show initialized (unit test covers this too, but the
        # e2e proves the wiring all the way through).
        send_message(proc, make_request("rocket/status", {}, req_id=5))
        status_resp = recv_message(proc)
        status_state = status_resp["result"]["state"]
        assert status_state["status"] == "initialized", (
            f"Session must remain initialized, got: {status_state['status']}"
        )
        assert status_state["model_id"] is None, (
            f"model_id must be null, got: {status_state['model_id']}"
        )

        print(f"  error_code: {err_data.get('error_code')}")
        print(f"  backend_error: {ctx.get('backend_error')}")
        print(f"  session.status after failure: {status_state['status']}")
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
