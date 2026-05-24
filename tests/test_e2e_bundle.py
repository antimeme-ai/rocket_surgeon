"""End-to-end test: session bundle export produces valid tar.gz.

Validates: attach, step, export — verify bundle contains core artifacts.

Usage:
    PYTHONPATH=python python tests/test_e2e_bundle.py
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

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

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = str(Path(tmpdir) / "test-session.tar.gz")

        try:
            # Initialize
            print("\n[test] Step 1: initialize")
            send_message(
                proc,
                make_request(
                    "initialize",
                    {
                        "client_name": "e2e-bundle-test",
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

            # Step forward (so trace has some activity)
            print("\n[test] Step 3: step forward")
            send_message(
                proc,
                make_request(
                    "rocket/step",
                    {
                        "direction": "forward",
                        "count": 2,
                    },
                    req_id=3,
                ),
            )
            resp = recv_message(proc)
            assert_jsonrpc(resp, 3)
            assert resp.get("error") is None, f"step error: {resp.get('error')}"
            print("  PASS")

            # Export bundle
            print("\n[test] Step 4: export bundle")
            send_message(
                proc,
                make_request(
                    "rocket/session.export",
                    {
                        "path": bundle_path,
                        "include_tensors": False,
                    },
                    req_id=4,
                ),
            )
            resp = recv_message(proc)
            assert_jsonrpc(resp, 4)
            assert resp.get("error") is None, f"export error: {resp.get('error')}"

            data = resp["result"]["data"]
            assert data["path"] == bundle_path
            assert data["size_bytes"] > 0
            assert data["artifact_count"] >= 8
            print(f"  {data['artifact_count']} artifacts, {data['size_bytes']} bytes")
            print("  PASS")

            # Validate tar.gz contents
            print("\n[test] Step 5: validate bundle contents")
            assert Path(bundle_path).is_file(), f"bundle file not found: {bundle_path}"
            with tarfile.open(bundle_path, "r:gz") as tar:
                names = tar.getnames()

                required = [
                    "manifest.json",
                    "interventions.json",
                    "protocol-trace.jsonl",
                    "env.json",
                    "model-info.json",
                    "prompt.json",
                    "bookmarks.json",
                ]
                for artifact in required:
                    assert artifact in names, f"missing {artifact} in {names}"

                manifest_member = tar.getmember("manifest.json")
                manifest_file = tar.extractfile(manifest_member)
                assert manifest_file is not None
                manifest_data = manifest_file.read()
                manifest = json.loads(manifest_data)
                assert "session_id" in manifest
                assert "protocol_version" in manifest
                assert manifest["protocol_version"] == "0.1.0"

                env_file = tar.extractfile(tar.getmember("env.json"))
                assert env_file is not None
                env_data = json.loads(env_file.read())
                assert "torch_version" in env_data
                assert "python_version" in env_data

                model_file = tar.extractfile(tar.getmember("model-info.json"))
                assert model_file is not None
                model_data = json.loads(model_file.read())
                assert "model_family" in model_data
                assert "num_layers" in model_data
                assert model_data["num_layers"] > 0

                bookmarks_file = tar.extractfile(tar.getmember("bookmarks.json"))
                assert bookmarks_file is not None
                bookmarks_data = json.loads(bookmarks_file.read())
                assert bookmarks_data == []

            print(f"  {len(required)} required artifacts present")
            print("  PASS")

        finally:
            proc.stdin.close()
            proc.wait(timeout=15)


if __name__ == "__main__":
    run_test()
    print("\nAll bundle e2e tests passed!")
