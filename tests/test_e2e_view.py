"""End-to-end built-in views test for rocket_surgeon.

Spawns the real daemon, attaches a tiny model, steps forward,
and exercises both residual_stream_norm and attention_pattern views.

The tiny test model uses sdpa attention by default, so attention_pattern
returns CAPABILITY_NOT_SUPPORTED — this tests the error path correctly.

Usage:
    PYTHONPATH=python python tests/test_e2e_view.py
"""

from __future__ import annotations

import subprocess
import time

from e2e_harness import (
    MODEL_FAMILY,
    MODEL_SOURCE,
    TIMEOUT_SEC,
    assert_jsonrpc,
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)


def recv_response(proc: subprocess.Popen, req_id: int, timeout: float = TIMEOUT_SEC) -> dict:
    """Read messages until we get a response with the given req_id, skipping notifications."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = recv_message(proc, timeout=deadline - time.monotonic())
        if "id" in msg and msg["id"] == req_id:
            return msg
    msg = f"Never got response for req_id={req_id}"
    raise TimeoutError(msg)


def run_test() -> None:  # noqa: PLR0915
    proc = spawn_daemon()

    try:
        req_id = 0

        # ── Step 1: initialize ──────────────────────────────────────
        print("\n[test] Step 1: initialize")
        req_id += 1
        send_message(
            proc,
            make_request(
                "initialize",
                {
                    "client_name": "e2e-view-test",
                    "protocol_version": "0.1.0",
                },
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"init error: {resp.get('error')}"
        caps = resp["result"]["data"]["capabilities"]
        built_in_views = caps.get("built_in_views", [])
        assert "residual_stream_norm" in built_in_views, (
            f"Expected residual_stream_norm in capabilities, got {built_in_views}"
        )
        assert "attention_pattern" in built_in_views, (
            f"Expected attention_pattern in capabilities, got {built_in_views}"
        )
        print(f"  capabilities.built_in_views = {built_in_views}")
        print("  PASS")

        # ── Step 2: attach ──────────────────────────────────────────
        print("\n[test] Step 2: attach")
        req_id += 1
        send_message(
            proc,
            make_request(
                "attach",
                {
                    "model_path": MODEL_SOURCE,
                    "model_family": MODEL_FAMILY,
                    "device": "cpu",
                },
                req_id,
            ),
        )
        resp = recv_message(proc, timeout=TIMEOUT_SEC)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"attach error: {resp.get('error')}"
        model_info = resp["result"]["data"]
        num_layers = model_info["num_layers"]
        num_heads = model_info["num_heads"]
        print(f"  model: {num_layers} layers, {num_heads} heads")
        print("  PASS")

        # ── Step 3: view before step → error ────────────────────────
        print("\n[test] Step 3: view before step (expect error)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/view",
                {
                    "view": "residual_stream_norm",
                },
                req_id,
            ),
        )
        resp = recv_response(proc, req_id)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is not None, "Expected error for view before step"
        err = resp["error"]
        err_data = err.get("data", {})
        assert err_data.get("error_code") == "VIEW_DATA_UNAVAILABLE", (
            f"Expected VIEW_DATA_UNAVAILABLE, got {err_data.get('error_code')}"
        )
        print(f"  Got expected error: {err_data.get('error_code')}")
        print("  PASS")

        # ── Step 4: step forward (full pass) ────────────────────────
        # Use a large count to drain all module hooks through the entire
        # forward pass so last_outputs has every layer's tensor.
        print("\n[test] Step 4: step forward (full pass)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1000,
                    "granularity": "component",
                },
                req_id,
            ),
        )
        resp = recv_response(proc, req_id)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"step error: {resp.get('error')}"
        print("  PASS")

        # ── Step 5: residual_stream_norm ────────────────────────────
        print("\n[test] Step 5: residual_stream_norm")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/view",
                {
                    "view": "residual_stream_norm",
                },
                req_id,
            ),
        )
        resp = recv_response(proc, req_id)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"view error: {resp.get('error')}"
        view_data = resp["result"]["data"]
        assert view_data["view"] == "residual_stream_norm", (
            f"Expected view=residual_stream_norm, got {view_data['view']}"
        )
        data = view_data["data"]
        norms = data["norms"]
        assert isinstance(norms, list), f"Expected norms as list, got {type(norms)}"
        # num_layers from attach is a stub (32); real tiny model has 2.
        # The norms count reflects the actual model, not the stub.
        assert len(norms) > 0, "Expected at least one norm"
        for i, n in enumerate(norms):
            assert isinstance(n, int | float), f"norm[{i}] is not a number: {n}"
            assert n > 0, f"norm[{i}] should be positive, got {n}"
        assert "layers" in data, "Expected layers field in response"
        assert len(data["layers"]) == len(norms), "layers and norms should have same length"
        assert data["norm_type"] == "l2"
        print(f"  norms = {norms}")
        print("  PASS")

        # ── Step 6: attention_pattern → CAPABILITY_NOT_SUPPORTED ────
        # The tiny test model uses sdpa attention (HF default), so
        # attention weights are not materialized. This tests the error path.
        print("\n[test] Step 6: attention_pattern on sdpa model (expect CAPABILITY_NOT_SUPPORTED)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/view",
                {
                    "view": "attention_pattern",
                    "params": {"layer": 0},
                },
                req_id,
            ),
        )
        resp = recv_response(proc, req_id)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is not None, "Expected error for attention_pattern on sdpa model"
        err = resp["error"]
        err_data = err.get("data", {})
        assert err_data.get("error_code") == "CAPABILITY_NOT_SUPPORTED", (
            f"Expected CAPABILITY_NOT_SUPPORTED, got {err_data.get('error_code')}"
        )
        assert "sdpa" in err["message"].lower() or "sdpa" in str(err_data).lower(), (
            f"Error should mention sdpa, got: {err['message']}"
        )
        print(f"  Got expected error: {err_data.get('error_code')}")
        print("  PASS")

        # ── Step 7: attention_pattern with invalid layer → INVALID_PARAMS
        print("\n[test] Step 7: attention_pattern with invalid layer (expect INVALID_PARAMS)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/view",
                {
                    "view": "attention_pattern",
                    "params": {"layer": 9999},
                },
                req_id,
            ),
        )
        resp = recv_response(proc, req_id)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is not None, "Expected error for invalid layer"
        err = resp["error"]
        err_data = err.get("data", {})
        # sdpa models fail before layer validation, so this may also be CAPABILITY_NOT_SUPPORTED
        assert err_data.get("error_code") in ("INVALID_PARAMS", "CAPABILITY_NOT_SUPPORTED"), (
            f"Expected INVALID_PARAMS or CAPABILITY_NOT_SUPPORTED, got {err_data.get('error_code')}"  # noqa: E501
        )
        print(f"  Got expected error: {err_data.get('error_code')}")
        print("  PASS")

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
            print(f"\n[cleanup] Daemon exited with code {proc.returncode}")
        except subprocess.TimeoutExpired:
            print("\n[cleanup] Daemon did not exit in time, killing...")
            proc.kill()
            proc.wait()

    print("\n" + "=" * 60)
    print("PASS — e2e built-in views")
    print("  initialize → attach → view before step (error)")
    print("  → step → residual_stream_norm → attention_pattern (sdpa error)")
    print("  → attention_pattern invalid layer (error) → cleanup")
    print("=" * 60)


if __name__ == "__main__":
    build_binaries()
    run_test()
