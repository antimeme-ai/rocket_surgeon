"""E2E tests for the shared memory data plane (WU 1.8).

Verifies:
1. Capabilities advertise shared_memory_supported
2. Tensor transfer works (regression — same behavior, shm transport)
3. Shared memory regions are cleaned up on detach

Usage:
    PYTHONPATH=python python tests/test_e2e_shm.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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


def run_test() -> None:
    proc = spawn_daemon()

    try:
        req_id = 0

        # ------------------------------------------------------------------
        # Step 1: initialize — check shared_memory_supported capability
        # ------------------------------------------------------------------
        print("\n[test] Step 1: initialize")
        req_id += 1
        send_message(
            proc,
            make_request(
                "initialize",
                {
                    "client_name": "e2e-shm-test",
                    "protocol_version": "0.3.0",
                },
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"init error: {resp.get('error')}"
        caps = resp["result"]["data"]["capabilities"]
        assert caps.get("shared_memory_supported") is True, (
            f"Expected shared_memory_supported=true, got {caps.get('shared_memory_supported')}"
        )
        session_id = resp["result"]["state"]["session_id"]
        print(f"  session_id: {session_id}")
        print(f"  shared_memory_supported: {caps['shared_memory_supported']}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 2: attach
        # ------------------------------------------------------------------
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
        assert resp["result"]["state"]["status"] == "stopped"
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 3: step forward to populate last_outputs
        # ------------------------------------------------------------------
        print("\n[test] Step 3: step forward")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 3,
                    "granularity": "component",
                },
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"step error: {resp.get('error')}"
        assert resp["result"]["data"]["ticks_executed"] == 3
        print(f"  ticks_executed: {resp['result']['data']['ticks_executed']}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 4: inspect summary — regression test (transport-transparent)
        # ------------------------------------------------------------------
        print("\n[test] Step 4: inspect summary (regression)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "*:0:*:*:*:output",
                    "detail": "summary",
                },
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"inspect error: {resp.get('error')}"
        data = resp["result"]["data"]
        tensors = data["tensors"]
        assert len(tensors) >= 1, f"Expected at least 1 tensor, got {len(tensors)}"
        t = tensors[0]
        assert "tensor_id" in t, "tensor must have tensor_id"
        assert len(t["tensor_id"]) == 64, (
            f"tensor_id must be 64 hex chars, got {len(t['tensor_id'])}"
        )
        assert "stats" in t, "tensor must have stats"
        assert "mean" in t["stats"], "stats must have mean"
        assert "std" in t["stats"], "stats must have std"
        print(f"  tensors returned: {len(tensors)}")
        print(f"  first tensor_id: {t['tensor_id'][:16]}...")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 5: inspect slice — regression test
        # ------------------------------------------------------------------
        print("\n[test] Step 5: inspect slice (regression)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {
                    "target": "*:0:*:*:*:output",
                    "detail": "slice",
                    "slices": [[0, 8]],
                },
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"inspect slice error: {resp.get('error')}"
        data = resp["result"]["data"]
        assert data.get("slice_data") is not None, "slice_data must be present"
        assert isinstance(data["slice_data"], str), "slice_data must be string"
        assert len(data["slice_data"]) > 0, "slice_data must not be empty"
        print(f"  slice_data length: {len(data['slice_data'])}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 6: detach — verify shm cleanup
        # ------------------------------------------------------------------
        print("\n[test] Step 6: detach + shm cleanup")
        req_id += 1
        send_message(proc, make_request("detach", {}, req_id))
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"detach error: {resp.get('error')}"
        assert resp["result"]["state"]["status"] == "initialized"

        if Path("/dev/shm").is_dir():
            rs_regions = [f.name for f in Path("/dev/shm").iterdir() if f.name.startswith("rs-")]
            assert len(rs_regions) == 0, f"stale shm regions found: {rs_regions}"
            print("  /dev/shm clean — no stale rs- regions")
        else:
            print("  (macOS — /dev/shm not available, skipping region check)")
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
    print("PASS — e2e shared memory data plane")
    print("  initialize (shared_memory_supported) → attach → step x3")
    print("  → inspect summary → inspect slice → detach + cleanup")
    print("=" * 60)


if __name__ == "__main__":
    build_binaries()
    run_test()
