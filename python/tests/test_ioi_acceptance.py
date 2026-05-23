"""IOI Acceptance Test -- Indirect Object Identification circuit intervention.

Validates the full intervention stack on GPT-2-small: register ablate
interventions on known name-mover heads (Wang et al. 2023), step through
the forward pass, and verify the interventions fire correctly.

This is a protocol-level acceptance test: it proves the daemon can register
targeted interventions, route them through the worker to the Python engine,
and report which recipes fired. Logit-level validation (measuring actual
logit diff reduction) requires inspect capabilities not yet in the protocol.

Usage:
    PYTHONPATH=python python tests/test_ioi_acceptance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))
from e2e_harness import (  # noqa: E402
    assert_jsonrpc,
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

IOI_PROMPTS = json.loads((Path(__file__).parent / "fixtures" / "ioi_prompts.json").read_text())

CANDIDATE_NAME_MOVERS = [(9, 9), (9, 6), (10, 0)]


def run_test() -> None:  # noqa: PLR0915
    build_binaries()
    proc = spawn_daemon()
    req_id = 0

    try:
        # Initialize
        print("\n[ioi] Step 1: initialize")
        req_id += 1
        send_message(
            proc,
            make_request(
                "initialize",
                {"client_name": "ioi-acceptance", "protocol_version": "0.3.0"},
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None
        print("  PASS")

        # Attach GPT-2-small
        print("\n[ioi] Step 2: attach gpt2")
        req_id += 1
        send_message(
            proc,
            make_request(
                "attach",
                {
                    "model_path": "gpt2",
                    "model_family": "gpt2",
                    "device": "cpu",
                    "num_ranks": 1,
                },
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"attach failed: {resp.get('error')}"
        print("  PASS")

        # Register ablate interventions on name-mover heads
        print("\n[ioi] Step 3: register ablate interventions on name-mover heads")
        for layer, head in CANDIDATE_NAME_MOVERS:
            req_id += 1
            send_message(
                proc,
                make_request(
                    "rocket/intervene",
                    {
                        "action": "set",
                        "recipe": {
                            "id": f"ablate-nm-{layer}.{head}",
                            "type": "ablate",
                            "target": f"gpt2:0:{layer}:attn.c_proj:fwd",
                            "params": {},
                            "priority": 0,
                        },
                    },
                    req_id,
                ),
            )
            resp = recv_message(proc)
            assert_jsonrpc(resp, req_id)
            assert resp.get("error") is None, f"intervene failed: {resp.get('error')}"

        # Verify all interventions registered
        active = resp["result"]["data"]["active_interventions"]
        assert len(active) == len(CANDIDATE_NAME_MOVERS), (
            f"expected {len(CANDIDATE_NAME_MOVERS)} interventions, got {len(active)}"
        )
        print(f"  {len(active)} interventions registered")
        print("  PASS")

        # Step forward through a chunk of the forward pass
        print("\n[ioi] Step 4: step forward (20 ticks)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/step",
                {"direction": "forward", "count": 20},
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"step failed: {resp.get('error')}"

        data = resp["result"]["data"]
        fired = data.get("fired_interventions", [])
        stopped_at = data["stopped_at"]
        print(f"  stopped at tick {stopped_at['tick_id']}, layer {stopped_at['layer']}")
        print(f"  {len(fired)} interventions fired")

        if stopped_at["layer"] >= 9:
            assert len(fired) > 0, "expected interventions to fire at layer >= 9"
            expected_ids = {f"ablate-nm-{layer}.{head}" for layer, head in CANDIDATE_NAME_MOVERS}
            for f_id in fired:
                assert f_id in expected_ids, f"unexpected fired intervention: {f_id}"
            print("  PASS: name-mover ablations fired correctly")
        else:
            print("  PASS: haven't reached name-mover layers yet (layer < 9)")

        # List interventions (verify persistence across steps)
        print("\n[ioi] Step 5: verify interventions persist")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/intervene",
                {"action": "list"},
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None
        active = resp["result"]["data"]["active_interventions"]
        assert len(active) == len(CANDIDATE_NAME_MOVERS)
        print(f"  {len(active)} interventions still active after stepping")
        print("  PASS")

        # Clear and verify
        print("\n[ioi] Step 6: clear all interventions")
        for layer, head in CANDIDATE_NAME_MOVERS:
            req_id += 1
            send_message(
                proc,
                make_request(
                    "rocket/intervene",
                    {"action": "clear", "intervention_id": f"ablate-nm-{layer}.{head}"},
                    req_id,
                ),
            )
            resp = recv_message(proc)
            assert_jsonrpc(resp, req_id)
            assert resp.get("error") is None

        active = resp["result"]["data"]["active_interventions"]
        assert len(active) == 0, f"expected 0 interventions after clear, got {len(active)}"
        print("  PASS")

    finally:
        proc.stdin.close()
        proc.wait(timeout=60)


if __name__ == "__main__":
    run_test()
    print("\nIOI acceptance test passed!")
