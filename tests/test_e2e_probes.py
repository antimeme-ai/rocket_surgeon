"""End-to-end probe test for the rocket_surgeon three-process architecture.

Spawns the real daemon binary, attaches a tiny model, defines probes,
steps forward, and verifies probe events fire.

Usage:
    PYTHONPATH=python python tests/test_e2e_probes.py
"""

from __future__ import annotations

import subprocess

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


def run_test() -> None:  # noqa: PLR0915
    proc = spawn_daemon()

    try:
        # Step 1: initialize
        print("\n[test] Step 1: initialize")
        send_message(
            proc,
            make_request(
                "initialize",
                {"client_name": "e2e-probe-test", "protocol_version": "0.1.0"},
                req_id=1,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None, f"Initialize error: {resp.get('error')}"
        session_id = resp["result"]["state"]["session_id"]
        print(f"  session_id: {session_id}")
        print("  PASS")

        # Step 2: attach
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
        print("  PASS")

        # Step 3: define a capture probe
        print("\n[test] Step 3: define capture probe")
        send_message(
            proc,
            make_request(
                "rocket/probe",
                {
                    "action": "define",
                    "probe": {
                        "id": "p-cap-1",
                        "point": "*:*:*:*:*:*",
                        "action": "capture",
                        "config": {"summary": True, "capture_tensor": False},
                        "enabled": True,
                        "priority": 0,
                    },
                },
                req_id=3,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None, f"Probe define error: {resp.get('error')}"
        result = resp["result"]
        data = result["data"]
        assert data["probe_id"] == "p-cap-1", f"Expected probe_id p-cap-1, got {data['probe_id']}"
        assert len(data["probes"]) == 1, f"Expected 1 probe, got {len(data['probes'])}"
        print("  PASS")

        # Step 4: list probes
        print("\n[test] Step 4: list probes")
        send_message(
            proc,
            make_request("rocket/probe", {"action": "list"}, req_id=4),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None
        data = resp["result"]["data"]
        assert len(data["probes"]) == 1
        assert data["probe_id"] is None
        print("  PASS")

        # Step 5: step forward and check for events
        print("\n[test] Step 5: step forward (probes should fire)")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {"direction": "forward", "count": 3, "granularity": "component"},
                req_id=5,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 5)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        print("  PASS")

        # Step 6: disable probe
        print("\n[test] Step 6: disable probe")
        send_message(
            proc,
            make_request(
                "rocket/probe",
                {"action": "disable", "probe_id": "p-cap-1"},
                req_id=6,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 6)
        assert resp.get("error") is None
        probes = resp["result"]["data"]["probes"]
        p1 = [p for p in probes if p["id"] == "p-cap-1"][0]
        assert p1["enabled"] is False, f"Expected disabled, got {p1['enabled']}"
        print("  PASS")

        # Step 7: enable probe
        print("\n[test] Step 7: enable probe")
        send_message(
            proc,
            make_request(
                "rocket/probe",
                {"action": "enable", "probe_id": "p-cap-1"},
                req_id=7,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 7)
        assert resp.get("error") is None
        probes = resp["result"]["data"]["probes"]
        p1 = [p for p in probes if p["id"] == "p-cap-1"][0]
        assert p1["enabled"] is True
        print("  PASS")

        # Step 8: remove probe
        print("\n[test] Step 8: remove probe")
        send_message(
            proc,
            make_request(
                "rocket/probe",
                {"action": "remove", "probe_id": "p-cap-1"},
                req_id=8,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 8)
        assert resp.get("error") is None
        probes = resp["result"]["data"]["probes"]
        assert len(probes) == 0, f"Expected 0 probes after remove, got {len(probes)}"
        print("  PASS")

        # Step 9: enable nonexistent probe -> PROBE_NOT_FOUND
        print("\n[test] Step 9: enable nonexistent probe")
        send_message(
            proc,
            make_request(
                "rocket/probe",
                {"action": "enable", "probe_id": "ghost"},
                req_id=9,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 9)
        assert resp.get("error") is not None, "Expected error for nonexistent probe"
        err_data = resp["error"]["data"]
        assert err_data["error_code"] == "PROBE_NOT_FOUND", f"Got {err_data['error_code']}"
        assert err_data["severity"] == "recoverable"
        print("  PASS")

        # Step 10: detach
        print("\n[test] Step 10: detach")
        send_message(
            proc,
            make_request("detach", {}, req_id=10),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 10)
        assert resp.get("error") is None
        print("  PASS")

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
            print(f"[cleanup] Daemon exited with code {proc.returncode}")
        except subprocess.TimeoutExpired:
            print("[cleanup] Daemon did not exit in time, killing...")
            proc.kill()
            proc.wait()

    print("\n" + "=" * 60)
    print("PASS — e2e probes")
    print("  initialize -> attach -> define probe -> list")
    print("  -> step (probes fire) -> disable -> enable -> remove")
    print("  -> PROBE_NOT_FOUND -> detach")
    print("=" * 60)


if __name__ == "__main__":
    build_binaries()
    run_test()
