# Sub-project C: IOI Logit Measurement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the IOI acceptance test to measure logit difference reduction after name-mover head ablation, validating >= 50% reduction.

**Architecture:** Two-pass test — clean baseline forward pass captures lm_head logits, then ablated forward pass after re-attach. Compares logit_IO - logit_S between passes.

**Tech Stack:** Python test using e2e_harness, struct module for f32 decode, base64 for tensor data.

---

### Task 1: Add token IDs to IOI fixtures

**Files:**
- Modify: `python/tests/fixtures/ioi_prompts.json`

- [ ] **Step 1: Look up GPT-2 token IDs for the IOI prompt**

The first fixture prompt is: `"When Mary and John went to the store, John gave a drink to"`

GPT-2 tokenizer encodes this. We need the IO token (" Mary" = 5335) and the S token (" John" = 1757). The full token sequence is needed for the `tokens` field in `rocket/step`.

GPT-2 token IDs (via tiktoken/HF tokenizer):
- "When" = 2437
- " Mary" = 5335
- " and" = 290
- " John" = 1757
- " went" = 1816
- " to" = 284
- " the" = 262
- " store" = 3650
- "," = 11
- " John" = 1757
- " gave" = 2921
- " a" = 257
- " drink" = 4144
- " to" = 284

Sequence: [2437, 5335, 290, 1757, 1816, 284, 262, 3650, 11, 1757, 2921, 257, 4144, 284]

- [ ] **Step 2: Update fixture with token IDs**

```json
[
  {
    "text": "When Mary and John went to the store, John gave a drink to",
    "io": "Mary",
    "s": "John",
    "template": "ABB",
    "token_ids": [2437, 5335, 290, 1757, 1816, 284, 262, 3650, 11, 1757, 2921, 257, 4144, 284],
    "io_token_id": 5335,
    "s_token_id": 1757
  },
  {
    "text": "When Alice and Bob went to the park, Bob gave a ball to",
    "io": "Alice",
    "s": "Bob",
    "template": "ABB",
    "token_ids": [2437, 14862, 290, 5765, 1816, 284, 262, 3952, 11, 5765, 2921, 257, 2613, 284],
    "io_token_id": 14862,
    "s_token_id": 5765
  }
]
```

Note: The exact token IDs need verification. The test itself should verify by checking that the tokenizer agrees, or we can derive them at test runtime using tiktoken. Safer approach: compute at runtime.

- [ ] **Step 3: Commit**

```bash
git add python/tests/fixtures/ioi_prompts.json
git commit -m "feat(test): add token IDs to IOI fixture prompts"
```

---

### Task 2: Rewrite IOI acceptance test with logit measurement

**Files:**
- Modify: `python/tests/test_ioi_acceptance.py`

- [ ] **Step 1: Rewrite the test**

Replace the entire test with a two-pass logit measurement approach.

The test flow:
1. Initialize + attach GPT-2
2. **Clean pass**: Step with `run_to: "completion"` and `tokens: <token_ids>` to run full forward pass. Then inspect `gpt2:0:0:lm_head:output` with `detail: "slice"` to get full logit tensor as base64. Decode the f32 bytes, extract `logit_IO = logits[-1][io_token_id]` and `logit_S = logits[-1][s_token_id]`. Compute `clean_diff = logit_IO - logit_S`.
3. Detach + re-attach to reset forward pass state.
4. **Ablated pass**: Register ablate interventions on name-mover heads (9.9, 9.6, 10.0). Step with same tokens + `run_to: "completion"`. Inspect lm_head. Compute `ablated_diff`.
5. Assert `(clean_diff - ablated_diff) / clean_diff >= 0.50`.
6. Export session bundle (validates Sub-project A).

```python
"""IOI Acceptance Test -- logit difference measurement.

Validates that ablating name-mover heads (Wang et al. 2023) reduces the
indirect object logit difference by at least 50% on GPT-2-small.

Usage:
    PYTHONPATH=python python tests/test_ioi_acceptance.py
"""

from __future__ import annotations

import base64
import json
import struct
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

CANDIDATE_NAME_MOVERS = [(9, 9), (9, 6), (10, 0)]
FIXTURES = REPO_ROOT / "python" / "tests" / "fixtures" / "ioi_prompts.json"


def decode_f32_logits(b64_data: str, shape: list[int]) -> list[list[float]]:
    """Decode base64 f32 tensor to nested list [seq_len, vocab_size]."""
    raw = base64.b64decode(b64_data)
    total = 1
    for s in shape:
        total *= s
    floats = struct.unpack(f"<{total}f", raw)
    # Reshape to [batch, seq_len, vocab_size] -> take [0] for batch dim
    seq_len = shape[-2] if len(shape) == 3 else shape[0]
    vocab_size = shape[-1]
    result = []
    for i in range(seq_len):
        row = list(floats[i * vocab_size : (i + 1) * vocab_size])
        result.append(row)
    return result


def step_to_completion(proc, req_id: int, token_ids: list[int]) -> int:
    """Step the forward pass to completion with given tokens. Returns new req_id."""
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
    return req_id


def inspect_lm_head(proc, req_id: int) -> tuple[int, str, list[int]]:
    """Inspect lm_head output and return (req_id, base64_data, shape)."""
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


def run_test() -> None:
    build_binaries()

    prompts = json.loads(FIXTURES.read_text())
    prompt = prompts[0]
    token_ids = prompt["token_ids"]
    io_token_id = prompt["io_token_id"]
    s_token_id = prompt["s_token_id"]

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
        req_id = step_to_completion(proc, req_id, token_ids)
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
        req_id = step_to_completion(proc, req_id, token_ids)
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
        assert reduction >= 0.50, (
            f"Expected >= 50% logit diff reduction, got {reduction:.2%}"
        )
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
```

- [ ] **Step 2: Run ruff + mypy**

Run: `ruff check python/tests/test_ioi_acceptance.py && mypy python/tests/test_ioi_acceptance.py`
Expected: Clean

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_ioi_acceptance.py
git commit -m "feat(test): rewrite IOI test with logit diff measurement"
```

---

### Task 3: Verify token IDs at runtime

**Files:**
- Modify: `python/tests/test_ioi_acceptance.py`

The hardcoded token IDs in the fixture may be wrong. Add a runtime check at the start of the test that validates token IDs against the HuggingFace tokenizer (if available) or tiktoken.

- [ ] **Step 1: Add token ID validation**

At the start of `run_test()`, after loading the fixture:

```python
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    expected = tok.encode(prompt["text"])
    assert token_ids == expected, (
        f"Fixture token IDs don't match tokenizer: {token_ids} vs {expected}"
    )
    assert tok.encode(" " + prompt["io"])[0] == io_token_id
    assert tok.encode(" " + prompt["s"])[0] == s_token_id
    print("[ioi] Token IDs verified against HF tokenizer")
except ImportError:
    print("[ioi] Skipping tokenizer verification (transformers not available)")
```

- [ ] **Step 2: Commit**

```bash
git add python/tests/test_ioi_acceptance.py
git commit -m "feat(test): add runtime token ID verification in IOI test"
```

---

### Task 4: Full verification

- [ ] **Step 1: Run ruff + mypy**

Run: `ruff check python/tests/test_ioi_acceptance.py && mypy python/tests/test_ioi_acceptance.py`
Expected: Clean

- [ ] **Step 2: Run Rust tests (sanity check)**

Run: `cargo test --workspace --quiet`
Expected: All pass

- [ ] **Step 3: Push and create PR**

```bash
git push -u origin HEAD
gh pr create --base master --title "feat: IOI logit diff measurement (Sub-project C)" --body "..."
```
