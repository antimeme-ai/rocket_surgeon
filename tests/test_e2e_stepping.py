"""End-to-end stepping test for the rocket_surgeon three-process architecture.

Spawns the real daemon binary, attaches a tiny model, then exercises
rocket/step with different counts and granularities.

Usage:
    PYTHONPATH=python python tests/test_e2e_stepping.py
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

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_test() -> None:
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
                    "client_name": "e2e-stepping-test",
                    "protocol_version": "0.3.0",
                },
                req_id=1,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None, f"Initialize error: {resp.get('error')}"
        assert resp["result"]["state"]["status"] == "initialized"
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
        assert_session_id(resp, session_id)
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 3: first step (count=1, component granularity)
        # ------------------------------------------------------------------
        print("\n[test] Step 3: step forward count=1 component")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "component",
                },
                req_id=3,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        result = resp["result"]
        state = result["state"]
        data = result["data"]
        assert_session_id(resp, session_id)
        assert_envelope_fields(state)
        assert state["status"] == "stopped", f"Expected stopped, got: {state['status']}"
        assert data["ticks_executed"] == 1, f"Expected 1 tick, got: {data['ticks_executed']}"
        assert "stopped_at" in data
        stopped_at = data["stopped_at"]
        assert isinstance(stopped_at["tick_id"], int), "tick_id must be integer"
        assert isinstance(stopped_at["layer"], int), "layer must be integer"
        assert isinstance(stopped_at["component"], str), "component must be string"
        assert isinstance(stopped_at["event"], str), "event must be string"
        assert isinstance(stopped_at["direction"], str), "direction must be string"
        assert stopped_at["direction"] == "forward"
        assert stopped_at["event"] in ("input", "output"), (
            f"event must be input or output, got: {stopped_at['event']}"
        )
        assert stopped_at["layer"] == 0, (
            f"First step should start at layer 0, got: {stopped_at['layer']}"
        )
        tick_1 = state["tick_id"]
        print(f"  ticks_executed: {data['ticks_executed']}")
        print(f"  tick_id: {tick_1}")
        print(
            f"  stopped_at: layer={stopped_at['layer']}"
            f" component={stopped_at['component']}"
            f" event={stopped_at['event']}"
        )
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 4: second step — tick_id must increase
        # ------------------------------------------------------------------
        print("\n[test] Step 4: second step — tick_id monotonicity")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "component",
                },
                req_id=4,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        assert_session_id(resp, session_id)
        tick_2 = resp["result"]["state"]["tick_id"]
        assert tick_2 > tick_1, f"tick_id should increase: {tick_1} -> {tick_2}"
        print(f"  tick_id: {tick_1} -> {tick_2} (monotonic: OK)")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 5: third step — three-step monotonicity (TCK requires a < b < c)
        # ------------------------------------------------------------------
        print("\n[test] Step 5: third step — three-step monotonicity")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "component",
                },
                req_id=5,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 5)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        tick_3 = resp["result"]["state"]["tick_id"]
        assert tick_3 > tick_2, f"tick_id should increase: {tick_2} -> {tick_3}"
        print(f"  tick_id: {tick_1} < {tick_2} < {tick_3} (monotonic: OK)")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 6: multi-tick step (count=3)
        # ------------------------------------------------------------------
        print("\n[test] Step 6: step forward count=3")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 3,
                    "granularity": "component",
                },
                req_id=6,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 6)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        data_6 = resp["result"]["data"]
        assert data_6["ticks_executed"] == 3, f"Expected 3 ticks, got: {data_6['ticks_executed']}"
        tick_after_multi = resp["result"]["state"]["tick_id"]
        assert tick_after_multi > tick_3, "tick_id must advance after multi-step"
        print(f"  ticks_executed: {data_6['ticks_executed']}")
        print(f"  tick_id: {tick_after_multi}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 7: layer granularity step
        # ------------------------------------------------------------------
        print("\n[test] Step 7: step forward count=1 layer granularity")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "layer",
                },
                req_id=7,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 7)
        assert resp.get("error") is None, f"Step error: {resp.get('error')}"
        data_7 = resp["result"]["data"]
        assert data_7["ticks_executed"] == 1, (
            f"Expected 1 layer tick, got: {data_7['ticks_executed']}"
        )
        tick_after_layer = resp["result"]["state"]["tick_id"]
        assert tick_after_layer > tick_after_multi, "tick_id must advance after layer step"
        print(f"  ticks_executed (layer): {data_7['ticks_executed']}")
        print(f"  tick_id: {tick_after_layer}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 8: backward step -> CAPABILITY_NOT_SUPPORTED
        # ------------------------------------------------------------------
        print("\n[test] Step 8: backward step returns error")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "backward",
                    "count": 1,
                    "granularity": "component",
                },
                req_id=8,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 8)
        assert resp.get("error") is not None, "Expected error for backward step"
        error = resp["error"]
        assert "data" in error, "Error should include structured data"
        assert error["data"]["error_code"] == "CAPABILITY_NOT_SUPPORTED", (
            f"Expected CAPABILITY_NOT_SUPPORTED, got: {error['data'].get('error_code')}"
        )
        assert error["data"]["severity"] == "recoverable", (
            f"Expected recoverable severity, got: {error['data'].get('severity')}"
        )
        print(f"  error_code: {error['data']['error_code']}")
        print(f"  severity: {error['data']['severity']}")
        print(f"  message: {error['message']}")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 9: 10-step tick_id uniqueness
        # ------------------------------------------------------------------
        print("\n[test] Step 9: 10-step tick_id uniqueness")
        tick_ids: list[int] = []
        for i in range(10):
            send_message(
                proc,
                make_request(
                    "rocket/step",
                    {
                        "direction": "forward",
                        "count": 1,
                        "granularity": "component",
                    },
                    req_id=100 + i,
                ),
            )
            resp = recv_message(proc)
            assert_jsonrpc(resp, 100 + i)
            assert resp.get("error") is None, f"Step {i} error: {resp.get('error')}"
            tick_ids.append(resp["result"]["state"]["tick_id"])
        assert len(tick_ids) == len(set(tick_ids)), (
            f"tick_ids must all be unique, got duplicates: {tick_ids}"
        )
        print(f"  collected {len(tick_ids)} tick_ids, all unique: OK")
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 10: layer-granularity step completes the entered layer
        # (detach + re-attach for a fresh forward pass — the tiny model
        # only has 2 layers, so count=1 layer step drains the entered
        # layer and forward-completes the whole pass)
        # ------------------------------------------------------------------
        print("\n[test] Step 10: layer-granularity step completes entered layer")
        send_message(proc, make_request("detach", {}, req_id=190))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 190)
        assert resp.get("error") is None, f"Pre-layer detach error: {resp.get('error')}"

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
                req_id=191,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 191)
        assert resp.get("error") is None, f"Pre-layer re-attach error: {resp.get('error')}"

        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "layer",
                },
                req_id=200,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 200)
        assert resp.get("error") is None, f"Layer step A error: {resp.get('error')}"
        stopped_a = resp["result"]["data"]["stopped_at"]
        layer_a = stopped_a["layer"]
        comp_a = stopped_a["component"]
        assert comp_a != "", "layer step should produce a real stopped_at"
        print(f"  layer_a={layer_a}, comp_a={comp_a}")

        # Second step: auto-resets after forward_complete, runs again
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "layer",
                },
                req_id=201,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 201)
        assert resp.get("error") is None, f"Layer step B error: {resp.get('error')}"
        stopped_b = resp["result"]["data"]["stopped_at"]
        comp_b = stopped_b["component"]
        assert comp_b != "", (
            "second layer step should auto-reset and produce real stopped_at"
        )
        print(f"  layer_b={stopped_b['layer']}, comp_b={comp_b}")
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
        assert_session_id(resp, session_id)
        print("  PASS")

        # ------------------------------------------------------------------
        # Step 12: step before attach returns error
        # ------------------------------------------------------------------
        print("\n[test] Step 12: step before attach returns error")
        send_message(
            proc,
            make_request(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 1,
                    "granularity": "component",
                },
                req_id=301,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 301)
        assert resp.get("error") is not None, "Expected error when stepping without attach"
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
    print("PASS — e2e stepping")
    print("  initialize -> attach -> step x3 (monotonicity)")
    print("  -> step count=3 -> step layer -> backward error")
    print("  -> 10-step uniqueness -> layer advances -> detach")
    print("  -> step-before-attach error")
    print("=" * 60)


if __name__ == "__main__":
    build_binaries()
    run_test()
