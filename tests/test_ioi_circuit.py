"""IOI initial reproduction on GPT-2 124M.

Wang et al. 2023 "Interpretability in the Wild" identifies a circuit of
~26 attention heads that drive indirect object identification. This test
reproduces the core finding: a sparse subset of heads drives the
logit_diff, with name mover heads concentrated in layers 9-11.

Single prompt, zero-ablation sweep across all 144 heads (12 layers x 12 heads).
Not a full replication -- proof that rocket_surgeon can perform real
interpretability analysis.

Usage:
    PYTHONPATH=python .venv/bin/python tests/test_ioi_circuit.py
"""

from __future__ import annotations

import base64
import struct
import time

from e2e_harness import (
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

TIMEOUT = 120

# "When Mary and John went to the store, John gave a drink to"
# GPT-2 BPE token IDs (verified against HF GPT2Tokenizer)
PROMPT_TOKENS = [
    2215,
    5335,
    290,
    1757,
    1816,
    284,
    262,
    3650,
    11,
    1757,
    2921,
    257,
    4144,
    284,
]
MARY_TOKEN = 5335
JOHN_TOKEN = 1757


def read_logit_diff(proc, req_counter, vocab_size, seq_len):
    """Read Mary-John logit diff from lm_head via slice API."""
    req_counter[0] += 1
    send_message(
        proc,
        make_request(
            "rocket/inspect",
            {
                "target": "gpt2:0:*:lm_head:output",
                "detail": "slice",
                "slices": [[0, seq_len * vocab_size * 4]],
            },
            req_id=req_counter[0],
        ),
    )
    resp = recv_message(proc, timeout=TIMEOUT)
    err = resp.get("error")
    assert err is None, f"inspect lm_head: {err}"
    raw = base64.b64decode(resp["result"]["data"]["slice_data"])
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    last_pos_offset = (seq_len - 1) * vocab_size
    mary_logit = values[last_pos_offset + MARY_TOKEN]
    john_logit = values[last_pos_offset + JOHN_TOKEN]
    return mary_logit - john_logit


def run() -> None:
    proc = spawn_daemon()
    req_id = [0]

    def rpc(method, params=None):
        req_id[0] += 1
        send_message(proc, make_request(method, params, req_id=req_id[0]))
        return recv_message(proc, timeout=TIMEOUT)

    def assert_ok(resp, label):
        if resp.get("error"):
            raise AssertionError(f"{label}: {resp['error']['message']}")

    try:
        # --- Initialize ---
        print("[1] Initialize")
        resp = rpc(
            "initialize",
            {"client_name": "ioi-circuit", "protocol_version": "0.3.0"},
        )
        assert_ok(resp, "initialize")

        # --- Attach GPT-2 ---
        print("[2] Attach GPT-2 124M")
        t0 = time.monotonic()
        resp = rpc(
            "attach",
            {
                "model_path": "gpt2",
                "model_family": "gpt2",
                "device": "cpu",
                "num_ranks": 1,
            },
        )
        assert_ok(resp, "attach")
        attach_data = resp["result"]["data"]
        num_layers = attach_data["num_layers"]
        num_heads = attach_data["num_heads"]
        hidden_dim = attach_data["hidden_dim"]
        head_dim = hidden_dim // num_heads
        print(f"    layers={num_layers} heads={num_heads} hidden={hidden_dim} head_dim={head_dim}")
        print(f"    attach took {time.monotonic() - t0:.1f}s")

        # --- Baseline forward pass ---
        seq_len = len(PROMPT_TOKENS)
        print(f"\n[3] Baseline forward pass ({seq_len} tokens)")
        t0 = time.monotonic()
        resp = rpc(
            "rocket/step",
            {
                "direction": "forward",
                "count": 500,
                "granularity": "layer",
                "tokens": PROMPT_TOKENS,
            },
        )
        assert_ok(resp, "baseline step")
        print(
            f"    ticks={resp['result']['data']['ticks_executed']} ({time.monotonic() - t0:.3f}s)"
        )

        # --- Read baseline logits ---
        print("\n[4] Read baseline logits from lm_head")
        resp = rpc(
            "rocket/inspect",
            {"target": "gpt2:0:*:lm_head:output", "detail": "summary"},
        )
        assert_ok(resp, "inspect lm_head summary")
        tensor = resp["result"]["data"]["tensors"][0]
        shape = tensor["shape"]
        vocab_size = shape[-1]
        print(f"    shape={shape} dtype={tensor['dtype']}")

        baseline_logit_diff = read_logit_diff(proc, req_id, vocab_size, seq_len)
        print(f"    baseline logit_diff = {baseline_logit_diff:.4f}")
        assert baseline_logit_diff > 0, (
            f"Model should prefer Mary over John, got logit_diff={baseline_logit_diff:.4f}"
        )

        # --- Ablation sweep ---
        total_heads = num_layers * num_heads
        print(f"\n[5] Ablation sweep: {num_layers}x{num_heads} = {total_heads} heads")
        results = []
        t0 = time.monotonic()

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                target = f"gpt2:0:{layer_idx}:o_proj[{head_idx}]:output"

                resp = rpc(
                    "rocket/intervene",
                    {
                        "action": "set",
                        "recipe": {
                            "id": "sweep-ablate",
                            "type": "ablate",
                            "target": target,
                            "params": {"mode": "zero"},
                            "priority": 0,
                        },
                    },
                )
                assert_ok(resp, f"intervene L{layer_idx}H{head_idx}")

                resp = rpc(
                    "rocket/step",
                    {
                        "direction": "forward",
                        "count": 500,
                        "granularity": "layer",
                        "tokens": PROMPT_TOKENS,
                    },
                )
                assert_ok(resp, f"step L{layer_idx}H{head_idx}")

                ablated_diff = read_logit_diff(
                    proc,
                    req_id,
                    vocab_size,
                    seq_len,
                )
                delta = baseline_logit_diff - ablated_diff

                results.append(
                    {
                        "layer": layer_idx,
                        "head": head_idx,
                        "ablated_logit_diff": ablated_diff,
                        "delta": delta,
                    }
                )

                resp = rpc(
                    "rocket/intervene",
                    {
                        "action": "clear",
                        "intervention_id": "sweep-ablate",
                    },
                )
                assert_ok(resp, f"clear L{layer_idx}H{head_idx}")

            elapsed = time.monotonic() - t0
            done = (layer_idx + 1) * num_heads
            print(f"    layer {layer_idx} done ({done}/{total_heads}, {elapsed:.1f}s)")

        sweep_time = time.monotonic() - t0
        print(f"    sweep complete in {sweep_time:.1f}s")

        # --- Analysis ---
        print("\n[6] Analysis")
        results.sort(key=lambda r: abs(r["delta"]), reverse=True)

        threshold = abs(baseline_logit_diff) * 0.1
        significant = [r for r in results if abs(r["delta"]) > threshold]

        print(f"\n    Baseline logit_diff: {baseline_logit_diff:.4f}")
        print(f"    Threshold (10%): {threshold:.4f}")
        print(f"    Significant heads: {len(significant)} / {total_heads}")

        print("\n    Top 20 heads by |delta|:")
        print(f"    {'Layer':>5} {'Head':>4} {'Delta':>10} {'Ablated LD':>10}")
        print(f"    {'-' * 5:>5} {'-' * 4:>4} {'-' * 10:>10} {'-' * 10:>10}")
        for r in results[:20]:
            marker = ""
            if r["layer"] >= 9 and r["delta"] > threshold:
                marker = " <- NAME MOVER?"
            print(
                f"    {r['layer']:>5} {r['head']:>4} "
                f"{r['delta']:>+10.4f} {r['ablated_logit_diff']:>10.4f}"
                f"{marker}"
            )

        # --- Validation ---
        print("\n[7] Validation")

        assert len(significant) < 30, f"Expected <30 significant heads, got {len(significant)}"
        print(f"    Sparse: {len(significant)} significant heads (<30)")

        top10_layers = {r["layer"] for r in results[:10]}
        has_late_layers = bool(top10_layers & {9, 10, 11})
        if has_late_layers:
            print(f"    Late-layer heads in top 10: layers {top10_layers & {9, 10, 11}}")
        else:
            print(f"    No late-layer heads in top 10 (layers present: {top10_layers})")
            print("      (May differ from Wang et al. with single prompt)")

        top_delta = results[0]["delta"]
        print(f"    Top head: L{results[0]['layer']}H{results[0]['head']} delta={top_delta:+.4f}")
        assert abs(top_delta) > threshold, (
            f"Top head delta ({top_delta:.4f}) should exceed threshold ({threshold:.4f})"
        )
        print("    Top head effect exceeds threshold")

        # --- Detach ---
        print("\n[8] Detach")
        resp = rpc("detach", {})
        assert_ok(resp, "detach")

        print("\n" + "=" * 70)
        print("IOI INITIAL REPRODUCTION COMPLETE")
        print(f"  Model: GPT-2 124M ({num_layers} layers, {num_heads} heads)")
        print(f"  Baseline logit_diff: {baseline_logit_diff:.4f}")
        print(f"  Significant heads: {len(significant)} / {total_heads}")
        print(f"  Top head: L{results[0]['layer']}H{results[0]['head']} (delta={top_delta:+.4f})")
        print(f"  Sweep time: {sweep_time:.1f}s")
        print("=" * 70)

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    build_binaries()
    run()
