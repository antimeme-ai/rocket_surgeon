"""E2e test: layer-granularity stepping completes full layers.

After stepping N layers, the layer entered at the Nth boundary must be
fully processed — inspect must return tensor data for late components
(like down_proj) at that layer.

Usage:
    PYTHONPATH=python python tests/test_e2e_layer_step.py
"""

from __future__ import annotations

from e2e_harness import (
    MODEL_FAMILY,
    MODEL_SOURCE,
    assert_jsonrpc,
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

TIMEOUT = 60


def run_test() -> None:
    """Step 1 layer, then inspect a late component at the entered layer.

    Without the drain fix, only the first component of layer 1 is processed.
    Inspecting down_proj at layer 1 should fail (no tensor data).
    With the fix, the entire layer 1 is drained, so down_proj has data.
    """
    proc = spawn_daemon()

    try:
        # Initialize
        send_message(
            proc,
            make_request(
                "initialize",
                {
                    "client_name": "layer-step-test",
                    "protocol_version": "0.3.0",
                },
                req_id=1,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None

        # Attach
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
        assert resp.get("error") is None

        # Step 1 layer — crosses boundary 0→1
        print("\n[test] Step 1 layer")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "layer",
                },
                req_id=3,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None, f"step error: {resp.get('error')}"
        data = resp["result"]["data"]
        stopped = data["stopped_at"]
        print(f"  stopped_at: layer={stopped['layer']} component={stopped['component']}")

        # Inspect down_proj at layer 1 (a late component in the layer).
        # Without the drain fix, only the first component of layer 1
        # is processed, so down_proj has no data.
        print("\n[test] Inspect down_proj at layer 1 (the entered layer)")
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "*:0:1:down_proj:output",
                    "detail": "summary",
                },
                req_id=4,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None, (
            f"inspect down_proj at layer 1 failed: {resp.get('error')}"
        )
        tensors = resp["result"]["data"]["tensors"]
        assert len(tensors) >= 1, (
            f"Expected tensor data for down_proj at layer 1, got {len(tensors)} tensors"
        )
        stats = tensors[0].get("stats", {})
        std = stats.get("std", 0.0)
        assert std > 0.0, f"Expected non-zero std for down_proj at layer 1, got {std}"
        print(f"  tensors: {len(tensors)}, std={std:.6f}")
        print("  PASS")

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()


def run_auto_reset_test() -> None:
    """After forward_complete, a new step starts a fresh forward pass."""
    proc = spawn_daemon()

    try:
        # Initialize + Attach
        send_message(
            proc,
            make_request(
                "initialize",
                {
                    "client_name": "auto-reset-test",
                    "protocol_version": "0.3.0",
                },
                req_id=1,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None

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
        assert resp.get("error") is None

        # First forward pass — step to completion
        print("\n[test] First forward pass to completion")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 200,
                    "granularity": "layer",
                },
                req_id=3,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None
        data = resp["result"]["data"]
        stopped = data["stopped_at"]
        assert stopped["component"] != "", "first pass should have real stopped_at"
        print(f"  first pass: layer={stopped['layer']} comp={stopped['component']}")

        # Second forward pass — step again, should auto-reset and run
        # Without auto-reset, the worker times out (30s) and the daemon
        # returns a synthetic response with stopped_at.component == "".
        print("\n[test] Second forward pass (auto-reset)")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 200,
                    "granularity": "layer",
                },
                req_id=4,
            ),
        )
        resp = recv_message(proc, timeout=TIMEOUT)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None, f"second step error: {resp.get('error')}"
        data = resp["result"]["data"]
        stopped = data["stopped_at"]
        assert stopped["component"] != "", (
            "second forward pass has empty stopped_at.component — "
            "forward pass was not auto-reset after completion"
        )
        print(f"  second pass: layer={stopped['layer']} comp={stopped['component']}")
        print("  PASS")

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    build_binaries()
    run_test()
    run_auto_reset_test()
    print("\nAll layer step tests passed!")
