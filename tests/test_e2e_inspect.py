"""End-to-end inspect test for the rocket_surgeon three-process architecture.

Spawns the real daemon binary, attaches a tiny model, steps forward,
then exercises rocket/inspect with different targets and detail levels.

Usage:
    PYTHONPATH=python python tests/test_e2e_inspect.py
"""

from __future__ import annotations

import subprocess

from e2e_harness import (
    MODEL_FAMILY,
    MODEL_SOURCE,
    assert_envelope_fields,
    assert_jsonrpc,
    assert_session_id,
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)


def run_test() -> None:  # noqa: PLR0915
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
                {
                    "client_name": "e2e-inspect-test",
                    "protocol_version": "0.3.0",
                },
                req_id=1,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None, f"Initialize error: {resp.get('error')}"
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
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 3: step forward to populate last_outputs
        # ------------------------------------------------------------------
        print("\n[test] Step 3: step forward count=3 to populate tensors")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 3,
                    "granularity": "component",
                },
                req_id=3,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        assert resp["result"]["data"]["ticks_executed"] == 3
        print(f"  ticks_executed: {resp['result']['data']['ticks_executed']}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 4: inspect with wildcard target — summary
        # ------------------------------------------------------------------
        print("\n[test] Step 4: inspect with wildcard target (summary)")
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "model:0:*:*:*:fwd",
                    "detail": "summary",
                },
                req_id=4,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None, f"Inspect error: {resp.get('error')}"
        result = resp["result"]
        state = result["state"]
        data = result["data"]
        assert_session_id(resp, session_id)
        assert_envelope_fields(state)
        assert state["status"] == "stopped"
        assert isinstance(data["tensors"], list), "tensors must be array"
        assert len(data["tensors"]) >= 1, f"Expected at least 1 tensor, got {len(data['tensors'])}"
        tensor = data["tensors"][0]
        assert "tensor_id" in tensor, "tensor must have tensor_id"
        assert len(tensor["tensor_id"]) == 64, (
            f"tensor_id must be 64 hex chars, got {len(tensor['tensor_id'])}"
        )
        assert "shape" in tensor, "tensor must have shape"
        assert "dtype" in tensor, "tensor must have dtype"
        assert "stats" in tensor, "tensor must have stats"
        stats = tensor["stats"]
        for field in ("mean", "std", "min", "max", "abs_max", "sparsity", "l2_norm"):
            assert field in stats, f"stats must have {field}"
        assert "histogram" in stats, "stats must have histogram"
        assert data.get("slice_data") is None, "summary should not include slice_data"
        assert data.get("view_result") is None, "summary should not include view_result"
        print(f"  tensors returned: {len(data['tensors'])}")
        print(f"  first tensor_id: {tensor['tensor_id'][:16]}...")
        print(f"  shape: {tensor['shape']}")
        print(f"  dtype: {tensor['dtype']}")
        print(f"  mean: {stats['mean']}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 5: inspect with detail=slice
        # ------------------------------------------------------------------
        print("\n[test] Step 5: inspect with detail=slice")
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "model:0:*:*:*:fwd",
                    "detail": "slice",
                    "slices": [[0, 8]],
                },
                req_id=5,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 5)
        assert resp.get("error") is None, f"Inspect slice error: {resp.get('error')}"
        data = resp["result"]["data"]
        assert data.get("slice_data") is not None, "slice_data must be present"
        assert isinstance(data["slice_data"], str), "slice_data must be string"
        assert len(data["slice_data"]) > 0, "slice_data must not be empty"
        print(f"  slice_data length: {len(data['slice_data'])}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 6: inspect nonexistent target — INVALID_TARGET
        # ------------------------------------------------------------------
        print("\n[test] Step 6: inspect nonexistent target")
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "model:0:0:nonexistent_component_xyz:0:fwd",
                    "detail": "summary",
                },
                req_id=6,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 6)
        assert resp.get("error") is not None, "Expected error for nonexistent target"
        error = resp["error"]
        assert "data" in error, "Error should include structured data"
        assert error["data"]["error_code"] == "INVALID_TARGET", (
            f"Expected INVALID_TARGET, got: {error['data'].get('error_code')}"
        )
        assert error["data"]["severity"] == "recoverable"
        print(f"  error_code: {error['data']['error_code']}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 7: inspect with slice out of bounds — SLICE_OUT_OF_BOUNDS
        # ------------------------------------------------------------------
        print("\n[test] Step 7: inspect slice out of bounds")
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "model:0:*:*:*:fwd",
                    "detail": "slice",
                    "slices": [[0, 999999999]],
                },
                req_id=7,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 7)
        assert resp.get("error") is not None, "Expected error for OOB slice"
        error = resp["error"]
        assert error["data"]["error_code"] == "SLICE_OUT_OF_BOUNDS", (
            f"Expected SLICE_OUT_OF_BOUNDS, got: {error['data'].get('error_code')}"
        )
        print(f"  error_code: {error['data']['error_code']}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 8: inspect default detail (omit detail field)
        # ------------------------------------------------------------------
        print("\n[test] Step 8: inspect default detail")
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "model:0:*:*:*:fwd",
                },
                req_id=8,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 8)
        assert resp.get("error") is None, f"Inspect default error: {resp.get('error')}"
        data = resp["result"]["data"]
        assert len(data["tensors"]) >= 1
        assert data.get("slice_data") is None
        assert data.get("view_result") is None
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 9: tensor_id is BLAKE3 hash (64 hex chars)
        # ------------------------------------------------------------------
        print("\n[test] Step 9: tensor_id format")
        tensor_id = resp["result"]["data"]["tensors"][0]["tensor_id"]
        assert len(tensor_id) == 64, f"tensor_id must be 64 chars, got {len(tensor_id)}"
        assert all(c in "0123456789abcdef" for c in tensor_id), (
            f"tensor_id must be hex, got: {tensor_id}"
        )
        print(f"  tensor_id: {tensor_id}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 10: inspect response envelope has full SessionState
        # ------------------------------------------------------------------
        print("\n[test] Step 10: response envelope")
        state = resp["result"]["state"]
        assert_envelope_fields(state)
        assert state["status"] == "stopped"
        assert state.get("model_id") is not None
        assert state.get("position") is not None or state.get("tick_id") is not None
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 11: detach
        # ------------------------------------------------------------------
        print("\n[test] Step 11: detach")
        send_message(proc, make_request("detach", {}, req_id=300))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 300)
        assert resp.get("error") is None, f"Detach error: {resp.get('error')}"
        assert resp["result"]["state"]["status"] == "initialized"
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 12: inspect before attach returns error
        # ------------------------------------------------------------------
        print("\n[test] Step 12: inspect before attach returns error")
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "model:0:0:q_proj:output",
                    "detail": "summary",
                },
                req_id=301,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 301)
        assert resp.get("error") is not None, "Expected error when inspecting without attach"
        print(f"  error: {resp['error']['message']}")
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
    print("PASS — e2e inspect")
    print("  initialize -> attach -> step x3")
    print("  -> inspect summary -> inspect slice -> nonexistent target")
    print("  -> slice OOB -> default detail -> tensor_id format")
    print("  -> response envelope -> detach -> inspect-before-attach")
    print("=" * 60)


if __name__ == "__main__":
    build_binaries()
    run_test()
