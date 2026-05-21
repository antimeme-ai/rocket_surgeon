"""End-to-end checkpoint test for rocket_surgeon.

Spawns the real daemon, attaches a tiny model, steps forward, and exercises
the full rocket/checkpoint surface: create (activation + full_snapshot),
list, restore, delete, bookmark, and the CHECKPOINT_NOT_FOUND error paths.

Behavioural verification for tck/protocol/checkpoint.feature. The Gherkin
spec uses literal checkpoint ids ("ckpt-a"); the daemon mints UUIDs, so this
test creates checkpoints and captures the minted ids rather than assuming
literals.

Usage:
    PYTHONPATH=python python tests/test_e2e_checkpoint.py
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

        # ── Step 1: initialize — supports_checkpointing advertised ──
        print("\n[test] Step 1: initialize")
        req_id += 1
        send_message(
            proc,
            make_request(
                "initialize",
                {"client_name": "e2e-checkpoint-test", "protocol_version": "0.3.0"},
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"init error: {resp.get('error')}"
        caps = resp["result"]["data"]["capabilities"]
        assert caps.get("supports_checkpointing") is True, (
            f"Expected supports_checkpointing=true, got {caps.get('supports_checkpointing')}"
        )
        print("  capabilities.supports_checkpointing = True")
        print("  PASS")

        # ── Step 2: attach ──────────────────────────────────────────
        print("\n[test] Step 2: attach")
        req_id += 1
        send_message(
            proc,
            make_request(
                "attach",
                {"model_path": MODEL_SOURCE, "model_family": MODEL_FAMILY, "device": "cpu"},
                req_id,
            ),
        )
        resp = recv_message(proc, timeout=TIMEOUT_SEC)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"attach error: {resp.get('error')}"
        print("  PASS")

        # ── Step 3: step forward so a tick position exists ──────────
        print("\n[test] Step 3: step forward (full pass)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/step",
                {"direction": "forward", "count": 1000, "granularity": "component"},
                req_id,
            ),
        )
        resp = recv_response(proc, req_id)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"step error: {resp.get('error')}"
        stepped_tick = resp["result"]["state"]["tick_id"]
        print(f"  stepped to tick_id={stepped_tick}")
        print("  PASS")

        # ── Step 4: create activation checkpoint ────────────────────
        print("\n[test] Step 4: checkpoint create (activation)")
        req_id += 1
        send_message(
            proc,
            make_request("rocket/checkpoint", {"action": "create", "tier": "activation"}, req_id),
        )
        resp = recv_response(proc, req_id)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"create error: {resp.get('error')}"
        data = resp["result"]["data"]
        activation_id = data["checkpoint_id"]
        assert isinstance(activation_id, str) and activation_id, "expected non-empty checkpoint_id"
        entry = next(c for c in data["checkpoints"] if c["checkpoint_id"] == activation_id)
        assert entry["tier"] == "activation", f"expected tier=activation, got {entry['tier']}"
        assert entry["tick_id"] == stepped_tick, "checkpoint tick_id should match stepped position"
        print(f"  created activation checkpoint {activation_id}")
        print("  PASS")

        # ── Step 5: create full_snapshot checkpoint ─────────────────
        print("\n[test] Step 5: checkpoint create (full_snapshot)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/checkpoint", {"action": "create", "tier": "full_snapshot"}, req_id
            ),
        )
        resp = recv_response(proc, req_id)
        assert resp.get("error") is None, f"create error: {resp.get('error')}"
        data = resp["result"]["data"]
        snapshot_id = data["checkpoint_id"]
        entry = next(c for c in data["checkpoints"] if c["checkpoint_id"] == snapshot_id)
        assert entry["tier"] == "full_snapshot", f"expected full_snapshot, got {entry['tier']}"
        print(f"  created full_snapshot checkpoint {snapshot_id}")
        print("  PASS")

        # ── Step 6: list — entries carry all required fields ────────
        print("\n[test] Step 6: checkpoint list")
        req_id += 1
        send_message(proc, make_request("rocket/checkpoint", {"action": "list"}, req_id))
        resp = recv_response(proc, req_id)
        assert resp.get("error") is None, f"list error: {resp.get('error')}"
        checkpoints = resp["result"]["data"]["checkpoints"]
        assert len(checkpoints) == 2, f"expected 2 checkpoints, got {len(checkpoints)}"
        for c in checkpoints:
            for field, typ in (
                ("checkpoint_id", str),
                ("tick_id", int),
                ("layer_idx", int),
                ("tier", str),
                ("created_at", str),
            ):
                assert isinstance(c.get(field), typ), f"checkpoint.{field} bad: {c.get(field)!r}"
        print(f"  listed {len(checkpoints)} checkpoints with full metadata")
        print("  PASS")

        # ── Step 7: restore by id moves position ────────────────────
        print("\n[test] Step 7: checkpoint restore")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/checkpoint", {"action": "restore", "checkpoint_id": activation_id}, req_id
            ),
        )
        resp = recv_response(proc, req_id)
        assert resp.get("error") is None, f"restore error: {resp.get('error')}"
        result = resp["result"]
        assert result["data"]["restored_to"]["tick_id"] == stepped_tick, "restored_to mismatch"
        assert result["state"]["position"]["tick_id"] == stepped_tick, "state position not moved"
        print(f"  restored to tick_id={stepped_tick}")
        print("  PASS")

        # ── Step 8: delete removes the checkpoint ───────────────────
        print("\n[test] Step 8: checkpoint delete")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/checkpoint", {"action": "delete", "checkpoint_id": snapshot_id}, req_id
            ),
        )
        resp = recv_response(proc, req_id)
        assert resp.get("error") is None, f"delete error: {resp.get('error')}"
        remaining = resp["result"]["data"]["checkpoints"]
        assert len(remaining) == 1, f"expected 1 checkpoint after delete, got {len(remaining)}"
        assert all(c["checkpoint_id"] != snapshot_id for c in remaining), "deleted id still present"
        print("  PASS")

        # ── Step 9: bookmark a tick_id ──────────────────────────────
        print("\n[test] Step 9: checkpoint bookmark")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/checkpoint",
                {"action": "bookmark", "tick_id": stepped_tick, "name": "before-intervention"},
                req_id,
            ),
        )
        resp = recv_response(proc, req_id)
        assert resp.get("error") is None, f"bookmark error: {resp.get('error')}"
        checkpoints = resp["result"]["data"]["checkpoints"]
        assert any(
            c["tick_id"] == stepped_tick and c.get("bookmark") == "before-intervention"
            for c in checkpoints
        ), "bookmark not found on any checkpoint entry"
        print("  PASS")

        # ── Step 10: restore nonexistent → CHECKPOINT_NOT_FOUND ─────
        print("\n[test] Step 10: restore nonexistent (expect error)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/checkpoint", {"action": "restore", "checkpoint_id": "nonexistent"}, req_id
            ),
        )
        resp = recv_response(proc, req_id)
        assert resp.get("error") is not None, "expected error for nonexistent checkpoint"
        err_data = resp["error"].get("data", {})
        assert err_data.get("error_code") == "CHECKPOINT_NOT_FOUND", (
            f"expected CHECKPOINT_NOT_FOUND, got {err_data.get('error_code')}"
        )
        assert err_data.get("severity") == "recoverable", f"bad severity: {err_data.get('severity')}"
        assert err_data.get("suggestion"), "expected non-empty suggestion"
        print(f"  Got expected error: {err_data.get('error_code')}")
        print("  PASS")

        # ── Step 11: delete nonexistent → CHECKPOINT_NOT_FOUND ──────
        print("\n[test] Step 11: delete nonexistent (expect error)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/checkpoint", {"action": "delete", "checkpoint_id": "nonexistent"}, req_id
            ),
        )
        resp = recv_response(proc, req_id)
        assert resp.get("error") is not None, "expected error for nonexistent checkpoint"
        err_data = resp["error"].get("data", {})
        assert err_data.get("error_code") == "CHECKPOINT_NOT_FOUND", (
            f"expected CHECKPOINT_NOT_FOUND, got {err_data.get('error_code')}"
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
    print("PASS — e2e checkpoint")
    print("  initialize → attach → step → create (activation + full_snapshot)")
    print("  → list → restore → delete → bookmark → error paths → cleanup")
    print("=" * 60)


if __name__ == "__main__":
    build_binaries()
    run_test()
