"""IOI Acceptance Test -- logit difference measurement.

Validates that ablating name-mover heads (Wang et al. 2023) reduces the
indirect object logit difference by at least 50% on GPT-2-small.

Two-pass approach:
  1. Clean baseline: step to completion, inspect lm_head logits
  2. Ablated: register ablations on name-mover heads, step, inspect
  3. Assert >= 50% reduction in (logit_IO - logit_S)

Usage:
    PYTHONPATH=python python tests/test_ioi_acceptance.py
"""

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path
from typing import Any

try:
    from transformers import AutoTokenizer

    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))
from e2e_harness import (  # type: ignore[import-not-found]  # noqa: E402
    assert_jsonrpc,
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

CANDIDATE_NAME_MOVERS = [(9, 9), (9, 6), (10, 0)]
FIXTURES = REPO_ROOT / "python" / "tests" / "fixtures" / "ioi_prompts.json"


def decode_f32_logits(b64_data: str, shape: list[int]) -> list[list[float]]:
    """Decode base64 little-endian f32 tensor to [seq_len][vocab_size]."""
    raw = base64.b64decode(b64_data)
    total = 1
    for s in shape:
        total *= s
    floats = struct.unpack(f"<{total}f", raw)
    seq_len = shape[-2] if len(shape) >= 2 else 1
    vocab_size = shape[-1]
    result: list[list[float]] = []
    for i in range(seq_len):
        start = i * vocab_size
        row = list(floats[start : start + vocab_size])
        result.append(row)
    return result


def step_to_completion(
    proc: object, req_id: int, token_ids: list[int]
) -> tuple[int, dict[str, Any]]:
    """Step forward pass to completion. Returns (new_req_id, response)."""
    req_id += 1
    send_message(
        proc,
        make_request(
            "rocket/step",
            {
                "direction": "forward",
                "count": 1,
                "run_to": "completion",
                "tokens": token_ids,
            },
            req_id,
        ),
    )
    resp = recv_message(proc)
    assert_jsonrpc(resp, req_id)
    assert resp.get("error") is None, f"step failed: {resp.get('error')}"
    return req_id, resp


def inspect_lm_head(proc: object, req_id: int) -> tuple[int, str, list[int]]:
    """Inspect lm_head output. Returns (req_id, base64_data, shape)."""
    req_id += 1
    send_message(
        proc,
        make_request(
            "rocket/inspect",
            {
                "target": "gpt2:0:0:lm_head:output",
                "detail": "slice",
            },
            req_id,
        ),
    )
    resp = recv_message(proc)
    assert_jsonrpc(resp, req_id)
    assert resp.get("error") is None, f"inspect failed: {resp.get('error')}"
    data = resp["result"]["data"]
    slice_data = data["slice_data"]
    shape = data["tensors"][0]["shape"]
    return req_id, slice_data, shape


def verify_token_ids(prompt: dict[str, Any]) -> None:
    """Cross-check fixture token IDs against HF tokenizer if available."""
    if not HAS_TOKENIZER:
        print("[ioi] Skipping tokenizer verification (transformers not available)")
        return

    tok = AutoTokenizer.from_pretrained("gpt2")
    expected = tok.encode(prompt["text"])
    assert prompt["token_ids"] == expected, (
        f"Fixture token_ids mismatch: {prompt['token_ids']} vs {expected}"
    )
    io_tokens = tok.encode(" " + prompt["io"])
    s_tokens = tok.encode(" " + prompt["s"])
    assert io_tokens[0] == prompt["io_token_id"], (
        f"io_token_id mismatch: {prompt['io_token_id']} vs {io_tokens[0]}"
    )
    assert s_tokens[0] == prompt["s_token_id"], (
        f"s_token_id mismatch: {prompt['s_token_id']} vs {s_tokens[0]}"
    )
    print("[ioi] Token IDs verified against HF tokenizer")


def run_test() -> None:  # noqa: PLR0915
    build_binaries()

    prompts = json.loads(FIXTURES.read_text())
    prompt = prompts[0]
    token_ids: list[int] = prompt["token_ids"]
    io_token_id: int = prompt["io_token_id"]
    s_token_id: int = prompt["s_token_id"]

    verify_token_ids(prompt)

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
                {"client_name": "ioi-logit", "protocol_version": "0.3.0"},
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None
        print("  PASS")

        # Attach GPT-2
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

        # === CLEAN BASELINE PASS ===
        print("\n[ioi] Step 3: clean forward pass")
        req_id, _ = step_to_completion(proc, req_id, token_ids)
        print("  stepped to completion")

        print("\n[ioi] Step 4: inspect lm_head (clean)")
        req_id, b64_data, shape = inspect_lm_head(proc, req_id)
        logits = decode_f32_logits(b64_data, shape)
        last_pos = logits[-1]
        clean_io = last_pos[io_token_id]
        clean_s = last_pos[s_token_id]
        clean_diff = clean_io - clean_s
        print(f"  logit_IO={clean_io:.4f}, logit_S={clean_s:.4f}, diff={clean_diff:.4f}")
        print("  PASS")

        # === DETACH + RE-ATTACH ===
        print("\n[ioi] Step 5: detach + re-attach")
        req_id += 1
        send_message(proc, make_request("detach", {}, req_id))
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None

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
        assert resp.get("error") is None
        print("  PASS")

        # === ABLATED PASS ===
        print("\n[ioi] Step 6: register ablate interventions")
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
                            "target": f"gpt2:0:{layer}:attn.o_proj:fwd",
                            "params": {},
                            "priority": 0,
                        },
                    },
                    req_id,
                ),
            )
            resp = recv_message(proc)
            assert_jsonrpc(resp, req_id)
            assert resp.get("error") is None
        print(f"  {len(CANDIDATE_NAME_MOVERS)} interventions registered")
        print("  PASS")

        print("\n[ioi] Step 7: ablated forward pass")
        req_id, _ = step_to_completion(proc, req_id, token_ids)
        print("  stepped to completion")

        print("\n[ioi] Step 8: inspect lm_head (ablated)")
        req_id, b64_data, shape = inspect_lm_head(proc, req_id)
        logits = decode_f32_logits(b64_data, shape)
        last_pos = logits[-1]
        ablated_io = last_pos[io_token_id]
        ablated_s = last_pos[s_token_id]
        ablated_diff = ablated_io - ablated_s
        print(f"  logit_IO={ablated_io:.4f}, logit_S={ablated_s:.4f}, diff={ablated_diff:.4f}")
        print("  PASS")

        # === LOGIT DIFF REDUCTION ===
        print("\n[ioi] Step 9: verify logit diff reduction")
        reduction = (clean_diff - ablated_diff) / clean_diff
        print(f"  clean_diff={clean_diff:.4f}")
        print(f"  ablated_diff={ablated_diff:.4f}")
        print(f"  reduction={reduction:.2%}")
        assert reduction >= 0.50, f"Expected >= 50% logit diff reduction, got {reduction:.2%}"
        print("  PASS: logit diff reduced by >= 50%")

        # === EXPORT BUNDLE ===
        print("\n[ioi] Step 10: export session bundle")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/export",
                {"output_dir": "/tmp/ioi-bundle", "include_tensors": False},
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"export failed: {resp.get('error')}"
        print("  PASS")

    finally:
        proc.stdin.close()
        proc.wait(timeout=60)


if __name__ == "__main__":
    run_test()
    print("\nIOI logit measurement test passed!")
