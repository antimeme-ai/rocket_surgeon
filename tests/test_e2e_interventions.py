"""End-to-end test: intervention execution during forward pass.

Validates the full stack: register intervention via protocol, step,
verify tensor modification via fired_interventions in response.

Usage:
    PYTHONPATH=python python tests/test_e2e_interventions.py
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


def run_test() -> None:  # noqa: PLR0915
    build_binaries()
    proc = spawn_daemon()

    try:
        # Initialize
        print("\n[test] Step 1: initialize")
        send_message(
            proc,
            make_request(
                "initialize",
                {
                    "client_name": "e2e-interventions-test",
                    "protocol_version": "0.3.0",
                },
                req_id=1,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None, f"initialize error: {resp.get('error')}"
        print("  PASS")

        # Attach model
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
        assert resp.get("error") is None, f"attach error: {resp.get('error')}"
        print("  PASS")

        # Register scale intervention on all components (wildcard target)
        print("\n[test] Step 3: register scale intervention")
        send_message(
            proc,
            make_request(
                "rocket/intervene",
                {
                    "action": "set",
                    "recipe": {
                        "id": "iv-scale-all",
                        "type": "scale",
                        "target": "*:*:*:*:fwd",
                        "params": {"factor": 0.001},
                        "priority": 0,
                    },
                },
                req_id=3,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None, f"intervene error: {resp.get('error')}"
        data = resp["result"]["data"]
        assert data["applied"] is True
        assert len(data["active_interventions"]) == 1
        print("  PASS")

        # Step forward
        print("\n[test] Step 4: step forward (3 component ticks)")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 3,
                },
                req_id=4,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None, f"step error: {resp.get('error')}"

        # Verify interventions fired
        data = resp["result"]["data"]
        fired = data.get("fired_interventions", [])
        assert len(fired) > 0, "expected at least one intervention to fire"
        assert all(f == "iv-scale-all" for f in fired), f"unexpected fired IDs: {fired}"
        print(f"  {len(fired)} interventions fired across 3 component steps")
        print("  PASS")

        # Clear intervention and verify no more firing
        print("\n[test] Step 5: clear intervention, step again")
        send_message(
            proc,
            make_request(
                "rocket/intervene",
                {"action": "clear", "intervention_id": "iv-scale-all"},
                req_id=5,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 5)
        assert resp.get("error") is None

        send_message(
            proc,
            make_request(
                "rocket/step",
                {"direction": "forward", "count": 1},
                req_id=6,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 6)
        assert resp.get("error") is None, f"step error: {resp.get('error')}"
        data = resp["result"]["data"]
        fired = data.get("fired_interventions", [])
        assert len(fired) == 0, f"expected no interventions after clear, got: {fired}"
        print("  PASS")

    finally:
        proc.stdin.close()
        proc.wait(timeout=15)


if __name__ == "__main__":
    run_test()
    print("\nAll intervention e2e tests passed!")
