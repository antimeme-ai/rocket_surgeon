"""End-to-end Phase 1 smoke test — proves the debugger's numbers are correct.

Attaches a real GPT-2 (124M) model — not the tiny random Llama the other e2e
tests use — steps the full forward pass, and verifies the daemon's reported
per-layer residual-stream L2 norms against a direct PyTorch computation on the
same deterministic input.

This is the Phase 1 exit-gate correctness check (WU 1.16). It is also the only
e2e coverage of the GPT-2 adapter — every other e2e test uses Llama.

Usage:
    python tests/test_e2e_phase1.py
"""

from __future__ import annotations

import math
import subprocess
import time

import torch
from e2e_harness import (
    TIMEOUT_SEC,
    assert_jsonrpc,
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)
from transformers import AutoModelForCausalLM

GPT2_MODEL = "gpt2"
GPT2_FAMILY = "gpt2"
GPT2_NUM_LAYERS = 12
GPT2_NUM_HEADS = 12
GPT2_HIDDEN_DIM = 768

# The worker runs the forward pass on this exact input — see
# crates/rocket-surgeon-worker/src/bridge.rs::run_forward (torch.zeros((1, 2))).
FORWARD_INPUT_SHAPE = (1, 2)

# attach downloads GPT-2 (~500MB) from HuggingFace on a cold cache. This test
# runs in the pre-push gate, so the attach response needs a generous timeout to
# absorb a first-run download; steady-state runs hit the cache and are fast.
ATTACH_TIMEOUT_SEC = 600

# Both sides take torch.norm of the same f32 forward pass; read-only capture
# hooks do not perturb the math, so the only divergence is f32 accumulation.
NORM_REL_TOL = 1e-4
NORM_ABS_TOL = 1e-3


def recv_response(proc: subprocess.Popen, req_id: int, timeout: float = TIMEOUT_SEC) -> dict:
    """Read messages until the response with *req_id*, skipping notifications."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = recv_message(proc, timeout=deadline - time.monotonic())
        if msg.get("id") == req_id:
            return msg
    raise TimeoutError(f"never got response for req_id={req_id}")


def reference_residual_norms() -> list[float]:
    """Per-layer residual-stream L2 norms computed directly with PyTorch.

    Mirrors the daemon exactly: a forward hook on every decoder block captures
    the block's output tensor, and the norm is ``torch.norm(t.float(), p=2)``.
    The input is the same deterministic ``zeros`` tensor the worker uses, so
    the two computations see an identical forward pass.
    """
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL, torch_dtype=torch.float32)
    model.eval()

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for idx, block in enumerate(model.transformer.h):

        def hook(_module: object, _inp: object, output: object, idx: int = idx) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            captured[idx] = tensor.detach()

        handles.append(block.register_forward_hook(hook))

    try:
        input_ids = torch.zeros(FORWARD_INPUT_SHAPE, dtype=torch.long)
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    n_blocks = len(model.transformer.h)
    return [torch.norm(captured[i].float(), p=2).item() for i in range(n_blocks)]


def run_test() -> None:  # noqa: PLR0915
    proc = spawn_daemon()
    req_id = 0

    try:
        # ── Step 1: initialize ──────────────────────────────────────────
        print("\n[test] Step 1: initialize")
        req_id += 1
        send_message(
            proc,
            make_request(
                "initialize",
                {"client_name": "e2e-phase1-test", "protocol_version": "0.1.0"},
                req_id,
            ),
        )
        resp = recv_response(proc, req_id)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"initialize error: {resp.get('error')}"
        views = resp["result"]["data"]["capabilities"].get("built_in_views", [])
        assert "residual_stream_norm" in views, f"residual_stream_norm missing: {views}"
        print("  PASS")

        # ── Step 2: attach real GPT-2 ───────────────────────────────────
        print("\n[test] Step 2: attach GPT-2 (downloads ~500MB on first run)")
        req_id += 1
        send_message(
            proc,
            make_request(
                "attach",
                {
                    "model_path": GPT2_MODEL,
                    "model_family": GPT2_FAMILY,
                    "device": "cpu",
                    "num_ranks": 1,
                },
                req_id,
            ),
        )
        resp = recv_response(proc, req_id, timeout=ATTACH_TIMEOUT_SEC)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"attach error: {resp.get('error')}"
        state = resp["result"]["state"]
        assert state["status"] == "stopped", f"expected stopped, got {state['status']}"
        info = resp["result"]["data"]
        assert info["num_layers"] == GPT2_NUM_LAYERS, f"num_layers: {info['num_layers']}"
        assert info["num_heads"] == GPT2_NUM_HEADS, f"num_heads: {info['num_heads']}"
        assert info["hidden_dim"] == GPT2_HIDDEN_DIM, f"hidden_dim: {info['hidden_dim']}"
        print(
            f"  {info['num_layers']} layers, {info['num_heads']} heads, "
            f"hidden_dim {info['hidden_dim']}"
        )
        print("  PASS")

        # ── Step 3: step the full forward pass ──────────────────────────
        print("\n[test] Step 3: step the full forward pass")
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
        ticks = resp["result"]["data"]["ticks_executed"]
        assert ticks > 0, f"expected ticks_executed > 0, got {ticks}"
        print(f"  ticks_executed: {ticks}")
        print("  PASS")

        # ── Step 4: residual_stream_norm view ───────────────────────────
        print("\n[test] Step 4: residual_stream_norm view")
        req_id += 1
        send_message(
            proc,
            make_request("rocket/view", {"view": "residual_stream_norm"}, req_id),
        )
        resp = recv_response(proc, req_id)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"view error: {resp.get('error')}"
        view = resp["result"]["data"]
        assert view["view"] == "residual_stream_norm", f"view: {view['view']}"
        daemon_norms = view["data"]["norms"]
        layers = view["data"]["layers"]
        assert view["data"]["norm_type"] == "l2", f"norm_type: {view['data']['norm_type']}"
        assert len(daemon_norms) == GPT2_NUM_LAYERS, (
            f"expected {GPT2_NUM_LAYERS} norms, got {len(daemon_norms)} — "
            f"residual_stream_norm did not find the GPT-2 decoder blocks"
        )
        assert layers == list(range(GPT2_NUM_LAYERS)), f"layers: {layers}"
        print(f"  daemon norms: {[round(n, 4) for n in daemon_norms]}")
        print("  PASS")

        # ── Step 5: inspect — GPT-2 tensor/stats pipeline ───────────────
        print("\n[test] Step 5: inspect captured tensors")
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/inspect",
                {"target": "model:0:*:*:*:fwd", "detail": "summary"},
                req_id,
            ),
        )
        resp = recv_response(proc, req_id)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"inspect error: {resp.get('error')}"
        tensors = resp["result"]["data"]["tensors"]
        assert len(tensors) > 0, "inspect returned no tensors"
        for tensor in tensors:
            stats = tensor["stats"]
            assert len(tensor["shape"]) > 0, f"empty shape: {tensor}"
            for field in ("mean", "std", "l2_norm"):
                value = stats[field]
                assert math.isfinite(value), f"stats.{field} not finite: {value}"
            assert stats["std"] >= 0, f"negative std: {stats['std']}"
            assert stats["l2_norm"] >= 0, f"negative l2_norm: {stats['l2_norm']}"
        print(f"  {len(tensors)} tensors, all stats finite")
        print("  PASS")

        # ── Step 6: detach ──────────────────────────────────────────────
        print("\n[test] Step 6: detach")
        req_id += 1
        send_message(proc, make_request("detach", {}, req_id))
        resp = recv_response(proc, req_id)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"detach error: {resp.get('error')}"
        assert resp["result"]["state"]["status"] == "initialized"
        print("  PASS")

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
            print(f"\n[cleanup] daemon exited with code {proc.returncode}")
        except subprocess.TimeoutExpired:
            print("\n[cleanup] daemon did not exit in time, killing...")
            proc.kill()
            proc.wait()

    # ── Step 7: verify daemon norms against direct PyTorch ──────────────
    print("\n[test] Step 7: verify residual norms against direct PyTorch")
    ref_norms = reference_residual_norms()
    assert len(ref_norms) == len(daemon_norms), (
        f"layer count mismatch: daemon {len(daemon_norms)}, torch {len(ref_norms)}"
    )
    max_rel_err = 0.0
    for layer, (daemon, ref) in enumerate(zip(daemon_norms, ref_norms, strict=True)):
        rel_err = abs(daemon - ref) / ref if ref != 0 else abs(daemon - ref)
        max_rel_err = max(max_rel_err, rel_err)
        print(
            f"  [verify] layer {layer:2d}: daemon={daemon:.6f} "
            f"torch={ref:.6f} rel_err={rel_err:.2e}"
        )
        assert math.isclose(daemon, ref, rel_tol=NORM_REL_TOL, abs_tol=NORM_ABS_TOL), (
            f"layer {layer}: daemon norm {daemon} != torch norm {ref} "
            f"(rel_err {rel_err:.2e}, tol {NORM_REL_TOL})"
        )
    print(f"  max rel_err across {len(ref_norms)} layers: {max_rel_err:.2e}")
    print("  PASS")

    print("\n" + "=" * 60)
    print("PASS — e2e Phase 1 smoke test")
    print("  initialize -> attach GPT-2 -> step full pass")
    print("  -> residual_stream_norm view -> inspect")
    print("  -> detach -> verify norms vs direct PyTorch")
    print("=" * 60)


if __name__ == "__main__":
    build_binaries()
    run_test()
