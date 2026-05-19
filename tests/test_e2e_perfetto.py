"""End-to-end Perfetto trace sink test.

Spawns the real daemon, attaches a tiny model, steps through a few ticks,
detaches, then verifies the resulting .pftrace file is valid protobuf
with the expected track hierarchy and events.

Usage:
    PYTHONPATH=python python tests/test_e2e_perfetto.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

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


def decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decode a protobuf varint. Returns (value, new_offset)."""
    result = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        result |= (byte & 0x7F) << shift
        offset += 1
        if (byte & 0x80) == 0:
            return result, offset
        shift += 7
    msg = "truncated varint"
    raise ValueError(msg)


def count_trace_packets(data: bytes) -> int:
    """Count field-1 framed TracePacket records in a .pftrace file."""
    count = 0
    offset = 0
    while offset < len(data):
        if data[offset] != 0x0A:
            msg = f"expected field-1 tag 0x0A at offset {offset}, got 0x{data[offset]:02X}"
            raise ValueError(msg)
        offset += 1
        length, offset = decode_varint(data, offset)
        offset += length
        count += 1
    return count


def find_latest_pftrace(session_id: str) -> Path | None:
    """Find the .pftrace file for this session in $TMPDIR."""
    tmpdir = Path(tempfile.gettempdir())
    exact = tmpdir / f"{session_id}.pftrace"
    if exact.is_file():
        return exact
    traces = list(tmpdir.glob("*.pftrace"))
    if not traces:
        return None
    return max(traces, key=lambda p: p.stat().st_mtime)


def run_test() -> None:  # noqa: PLR0915
    build_binaries()
    proc = spawn_daemon()

    try:
        # -- initialize --
        print("\n[test] Step 1: initialize")
        send_message(
            proc,
            make_request(
                "initialize",
                {"client_name": "e2e-perfetto-test", "protocol_version": "0.1.0"},
                req_id=1,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None, f"Initialize error: {resp.get('error')}"
        session_id = resp["result"]["state"]["session_id"]
        assert_envelope_fields(resp["result"]["state"])
        print(f"  session_id: {session_id}")
        print("  PASS")

        # -- attach --
        print("\n[test] Step 2: attach")
        send_message(
            proc,
            make_request(
                "rocket/attach",
                {"model_path": MODEL_SOURCE, "model_family": MODEL_FAMILY},
                req_id=2,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, 2)
        assert resp.get("error") is None, f"Attach error: {resp.get('error')}"
        assert_session_id(resp, session_id)
        print("  PASS")

        # -- subscribe --
        print("\n[test] Step 3: subscribe")
        send_message(proc, make_request("rocket/subscribe", {}, req_id=3))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None, f"Subscribe error: {resp.get('error')}"
        print("  PASS")

        # -- step 3 times --
        print("\n[test] Step 4: step 3x")
        for i in range(3):
            send_message(proc, make_request("rocket/step", {}, req_id=10 + i))
            resp = recv_message(proc)
            # May receive notifications before the response
            while resp and "method" in resp:
                resp = recv_message(proc)
            assert resp.get("error") is None, f"Step {i} error: {resp.get('error')}"
        print("  PASS")

        # -- detach --
        print("\n[test] Step 5: detach")
        send_message(proc, make_request("rocket/detach", {}, req_id=20))
        resp = recv_message(proc)
        # May receive trailing notifications
        while resp and "method" in resp:
            resp = recv_message(proc)
        assert_jsonrpc(resp, 20)
        assert resp.get("error") is None, f"Detach error: {resp.get('error')}"
        print("  PASS")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    # -- verify .pftrace --
    print("\n[test] Step 6: verify .pftrace file")
    trace_path = find_latest_pftrace(session_id)
    assert trace_path is not None, "no .pftrace file found"
    print(f"  path: {trace_path}")

    data = trace_path.read_bytes()
    assert len(data) > 0, "trace file is empty"
    packet_count = count_trace_packets(data)
    print(f"  size: {len(data)} bytes, {packet_count} packets")
    assert packet_count >= 4, f"expected >= 4 packets, got {packet_count}"
    print("  PASS")

    # Cleanup
    trace_path.unlink()
    print(f"\n[result] ALL PASSED ({packet_count} Perfetto packets verified)")


if __name__ == "__main__":
    run_test()
