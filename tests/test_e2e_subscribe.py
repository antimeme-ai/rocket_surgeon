"""End-to-end subscribe + event delivery test for rocket_surgeon.

Spawns the real daemon, attaches a tiny model, subscribes for events,
steps forward, and verifies notifications are delivered.

Usage:
    PYTHONPATH=python python tests/test_e2e_subscribe.py
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


def assert_notification(msg: dict, method: str) -> None:
    """Validate a JSON-RPC 2.0 Notification."""
    assert msg.get("jsonrpc") == "2.0", f"Bad jsonrpc: {msg.get('jsonrpc')}"
    assert "id" not in msg, f"Notification must not have 'id', got {msg.get('id')}"
    assert msg.get("method") == method, f"Expected method={method}, got {msg.get('method')}"
    assert "params" in msg, "Notification missing 'params'"
    assert isinstance(msg["params"].get("seq"), int), f"Missing or non-int seq: {msg['params'].get('seq')}"


def recv_response(proc: subprocess.Popen, req_id: int, timeout: float = TIMEOUT_SEC) -> dict:
    """Read messages until we get a response with the given req_id, skipping notifications."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = recv_message(proc, timeout=deadline - time.monotonic())
        if "id" in msg and msg["id"] == req_id:
            return msg
    msg = f"Never got response for req_id={req_id}"
    raise TimeoutError(msg)


def try_recv_notification(proc: subprocess.Popen, timeout: float = 2.0) -> dict | None:
    """Try to read a notification within timeout. Returns None on timeout."""
    try:
        msg = recv_message(proc, timeout=timeout)
        if "id" not in msg:
            return msg
        return msg  # unexpected response — return it anyway for debugging
    except (TimeoutError, EOFError):
        return None


def run_test() -> None:  # noqa: PLR0915
    proc = spawn_daemon()

    try:
        # Step 1: initialize
        print("\n[test] Step 1: initialize")
        send_message(proc, make_request("initialize", {"client_name": "e2e-subscribe-test", "protocol_version": "0.1.0"}, req_id=1))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None, f"Initialize error: {resp.get('error')}"
        print("  PASS")

        # Step 2: attach
        print("\n[test] Step 2: attach")
        send_message(proc, make_request("attach", {"model_path": MODEL_SOURCE, "model_family": MODEL_FAMILY, "device": "cpu", "num_ranks": 1}, req_id=2))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 2)
        assert resp.get("error") is None, f"Attach error: {resp.get('error')}"
        print("  PASS")

        # Step 3: step without subscribe — no notifications
        print("\n[test] Step 3: step without subscribe (no notifications expected)")
        send_message(proc, make_request("rocket/step", {"direction": "forward", "count": 1, "granularity": "component"}, req_id=3))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None
        notif = try_recv_notification(proc, timeout=1.0)
        assert notif is None, f"Got unexpected notification before subscribe: {notif}"
        print("  PASS")

        # Step 4: subscribe
        print("\n[test] Step 4: subscribe")
        send_message(proc, make_request("rocket/subscribe", {}, req_id=4))
        resp = recv_response(proc, 4)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None, f"Subscribe error: {resp.get('error')}"
        data = resp["result"]["data"]
        assert "tick.stopped" in data["available_events"]
        assert "tick.heartbeat" in data["available_events"]
        assert "probe.fired" in data["available_events"]
        assert data["status"] == "stopped"
        print("  PASS")

        # Step 5: step with subscribe — expect tick.stopped notification
        print("\n[test] Step 5: step with events enabled")
        send_message(proc, make_request("rocket/step", {"direction": "forward", "count": 1, "granularity": "component"}, req_id=5))
        resp = recv_response(proc, 5)
        assert_jsonrpc(resp, 5)
        assert resp.get("error") is None

        notif = try_recv_notification(proc, timeout=5.0)
        assert notif is not None, "Expected tick.stopped notification after step"
        assert_notification(notif, "tick.stopped")
        first_seq = notif["params"]["seq"]
        print(f"  tick.stopped seq={first_seq}")
        print("  PASS")

        # Step 6: subscribe is idempotent
        print("\n[test] Step 6: subscribe again (idempotent)")
        send_message(proc, make_request("rocket/subscribe", {}, req_id=6))
        resp = recv_response(proc, 6)
        assert_jsonrpc(resp, 6)
        assert resp.get("error") is None
        print("  PASS")

        # Step 7: step again — verify seq is monotonically increasing
        print("\n[test] Step 7: step again, check monotonic seq")
        send_message(proc, make_request("rocket/step", {"direction": "forward", "count": 1, "granularity": "component"}, req_id=7))
        resp = recv_response(proc, 7)
        assert_jsonrpc(resp, 7)
        assert resp.get("error") is None

        notif = try_recv_notification(proc, timeout=5.0)
        assert notif is not None, "Expected tick.stopped notification"
        assert_notification(notif, "tick.stopped")
        second_seq = notif["params"]["seq"]
        assert second_seq > first_seq, f"seq not monotonic: {first_seq} -> {second_seq}"
        print(f"  tick.stopped seq={second_seq} (> {first_seq})")
        print("  PASS")

        # Step 8: unsubscribe
        print("\n[test] Step 8: unsubscribe")
        send_message(proc, make_request("rocket/unsubscribe", {}, req_id=8))
        resp = recv_response(proc, 8)
        assert_jsonrpc(resp, 8)
        assert resp.get("error") is None
        assert resp["result"]["data"]["status"] == "stopped"
        print("  PASS")

        # Step 9: step after unsubscribe — no notifications
        print("\n[test] Step 9: step after unsubscribe (no notifications)")
        send_message(proc, make_request("rocket/step", {"direction": "forward", "count": 1, "granularity": "component"}, req_id=9))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 9)
        assert resp.get("error") is None
        notif = try_recv_notification(proc, timeout=1.0)
        assert notif is None, f"Got notification after unsubscribe: {notif}"
        print("  PASS")

        # Step 10: unsubscribe is idempotent
        print("\n[test] Step 10: unsubscribe again (idempotent)")
        send_message(proc, make_request("rocket/unsubscribe", {}, req_id=10))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 10)
        assert resp.get("error") is None
        print("  PASS")

        # Step 11: re-subscribe for heartbeat test
        print("\n[test] Step 11: subscribe for heartbeat test")
        send_message(proc, make_request("rocket/subscribe", {}, req_id=11))
        resp = recv_response(proc, 11)
        assert_jsonrpc(resp, 11)
        assert resp.get("error") is None

        print("  Waiting ~3.5s for heartbeats...")
        time.sleep(3.5)

        heartbeats = []
        for _ in range(10):
            msg = try_recv_notification(proc, timeout=0.5)
            if msg is None:
                break
            if msg.get("method") == "tick.heartbeat":
                heartbeats.append(msg)

        assert len(heartbeats) >= 2, f"Expected >= 2 heartbeats, got {len(heartbeats)}"
        for hb in heartbeats:
            assert_notification(hb, "tick.heartbeat")
            assert "position" in hb["params"]
            assert "uptime_seconds" in hb["params"]

        seqs = [hb["params"]["seq"] for hb in heartbeats]
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], f"Heartbeat seq not monotonic: {seqs}"
        print(f"  Got {len(heartbeats)} heartbeats, seqs={seqs}")
        print("  PASS")

        # Step 12: notification wire format check
        print("\n[test] Step 12: notification wire format")
        send_message(proc, make_request("rocket/step", {"direction": "forward", "count": 1, "granularity": "component"}, req_id=12))
        resp = recv_response(proc, 12)
        assert_jsonrpc(resp, 12)

        notif = try_recv_notification(proc, timeout=5.0)
        assert notif is not None
        assert notif.get("jsonrpc") == "2.0"
        assert "id" not in notif
        assert isinstance(notif.get("method"), str)
        assert isinstance(notif.get("params"), dict)
        assert isinstance(notif["params"].get("seq"), int)
        print("  PASS")

        # Step 13: unsubscribe and detach
        print("\n[test] Step 13: cleanup")
        send_message(proc, make_request("rocket/unsubscribe", {}, req_id=13))
        resp = recv_response(proc, 13)
        assert resp.get("error") is None

        send_message(proc, make_request("detach", {}, req_id=14))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 14)
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
    print("PASS — e2e subscribe + event delivery")
    print("  initialize -> attach -> step (no events)")
    print("  -> subscribe -> step (tick.stopped) -> idempotent subscribe")
    print("  -> step (monotonic seq) -> unsubscribe -> step (no events)")
    print("  -> idempotent unsubscribe -> heartbeat -> wire format")
    print("  -> cleanup")
    print("=" * 60)


if __name__ == "__main__":
    build_binaries()
    run_test()
