"""E2e test: head-level ablation via bracket notation.

Ablates a single attention head (o_proj[0]) and verifies:
1. The intervention fires
2. The downstream lm_head output differs from baseline

Usage:
    PYTHONPATH=python python tests/test_e2e_head_ablation.py
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
    proc = spawn_daemon()

    try:
        # Initialize + Attach
        send_message(
            proc,
            make_request(
                "initialize",
                {
                    "client_name": "head-ablation-test",
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
        num_heads = resp["result"]["data"]["num_heads"]
        hidden_dim = resp["result"]["data"]["hidden_dim"]
        head_dim = hidden_dim // num_heads
        print(
            f"  num_heads={num_heads}, hidden_dim={hidden_dim}, "
            f"head_dim={head_dim}"
        )

        # Baseline: full forward pass without intervention
        print("\n[test] Baseline forward pass")
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

        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "*:0:*:lm_head:output",
                    "detail": "summary",
                },
                req_id=4,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None
        baseline_tensors = resp["result"]["data"]["tensors"]
        assert len(baseline_tensors) >= 1
        baseline_mean = baseline_tensors[0]["stats"]["mean"]
        baseline_std = baseline_tensors[0]["stats"]["std"]
        print(f"  baseline lm_head: mean={baseline_mean:.6f} std={baseline_std:.6f}")

        # Set head ablation
        print("\n[test] Register ablation on *:0:0:o_proj[0]:output")
        send_message(
            proc,
            make_request(
                "rocket/intervene",
                {
                    "action": "set",
                    "recipe": {
                        "id": "ablate-head-0",
                        "type": "ablate",
                        "target": "*:0:0:o_proj[0]:output",
                        "params": {"mode": "zero"},
                        "priority": 0,
                    },
                },
                req_id=5,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 5)
        assert resp.get("error") is None

        # Ablated forward pass (auto-reset starts new pass)
        print("\n[test] Ablated forward pass")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 200,
                    "granularity": "layer",
                },
                req_id=6,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 6)
        assert resp.get("error") is None
        data = resp["result"]["data"]
        fired = data.get("fired_interventions", [])
        assert "ablate-head-0" in fired, (
            f"Expected ablate-head-0 to fire, got: {fired}"
        )
        print(f"  interventions fired: {fired}")

        # Inspect lm_head after ablation
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "*:0:*:lm_head:output",
                    "detail": "summary",
                },
                req_id=7,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 7)
        assert resp.get("error") is None
        ablated_tensors = resp["result"]["data"]["tensors"]
        assert len(ablated_tensors) >= 1
        ablated_mean = ablated_tensors[0]["stats"]["mean"]
        ablated_std = ablated_tensors[0]["stats"]["std"]
        print(f"  ablated lm_head: mean={ablated_mean:.6f} std={ablated_std:.6f}")

        # Ablation should change the downstream output
        assert baseline_mean != ablated_mean or baseline_std != ablated_std, (
            "Ablating head 0 should change lm_head output, "
            f"but baseline ({baseline_mean:.6f}, {baseline_std:.6f}) "
            f"== ablated ({ablated_mean:.6f}, {ablated_std:.6f})"
        )
        print("  lm_head output changed after ablation")
        print("  PASS")

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()


def run_inspect_head_test() -> None:
    """Inspect a specific head returns only that head's data."""
    proc = spawn_daemon()

    try:
        send_message(
            proc,
            make_request(
                "initialize",
                {
                    "client_name": "inspect-head-test",
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
        num_heads = resp["result"]["data"]["num_heads"]
        hidden_dim = resp["result"]["data"]["hidden_dim"]
        head_dim = hidden_dim // num_heads

        # Step 1 layer to populate data
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
        assert resp.get("error") is None

        # Inspect full o_proj
        print("\n[test] Inspect full o_proj at layer 0")
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "*:0:0:o_proj:output",
                    "detail": "summary",
                },
                req_id=4,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None
        full_shape = resp["result"]["data"]["tensors"][0]["shape"]
        print(f"  full shape: {full_shape}")

        # Inspect head 0 of o_proj
        print("\n[test] Inspect o_proj[0] at layer 0")
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "*:0:0:o_proj[0]:output",
                    "detail": "summary",
                },
                req_id=5,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 5)
        assert resp.get("error") is None, (
            f"inspect head error: {resp.get('error')}"
        )
        head_tensors = resp["result"]["data"]["tensors"]
        assert len(head_tensors) >= 1
        head_shape = head_tensors[0]["shape"]
        print(f"  head shape: {head_shape}")

        # Head shape last dim should be head_dim, not hidden_dim
        assert head_shape[-1] == head_dim, (
            f"Expected last dim = head_dim ({head_dim}), "
            f"got {head_shape[-1]}"
        )
        print("  PASS — head inspect returns correct shape")

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
    run_inspect_head_test()
    print("\nAll head ablation tests passed!")
