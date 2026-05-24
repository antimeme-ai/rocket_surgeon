"""Intervention overhead benchmark.

Measures the wall-clock overhead of active interventions compared to
baseline stepping. Asserts <= 2% regression per the Phase 2 exit criteria.

Approach:
  1. Attach model, step N ticks (baseline wall time)
  2. Detach, re-attach with interventions, step N ticks (intervention wall time)
  3. Compute overhead = (intervention - baseline) / baseline
  4. Assert overhead <= 0.02

Usage:
    python tests/bench_intervention_overhead.py
"""

from __future__ import annotations

import time

from e2e_harness import (
    assert_jsonrpc,
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

STEP_COUNT = 20
WARMUP_STEPS = 5
INTERVENTION_LAYERS = [0, 1, 2, 3, 4]
MAX_OVERHEAD = 0.02


def init_and_attach(proc: object, req_id: int) -> int:
    """Initialize and attach the default model. Returns updated req_id."""
    req_id += 1
    send_message(
        proc,
        make_request(
            "initialize",
            {"client_name": "bench-overhead", "protocol_version": "0.3.0"},
            req_id,
        ),
    )
    resp = recv_message(proc)
    assert_jsonrpc(resp, req_id)
    assert resp.get("error") is None, f"initialize: {resp.get('error')}"
    return req_id


def attach_model(proc: object, req_id: int) -> int:
    """Attach the default test model. Returns updated req_id."""
    req_id += 1
    send_message(
        proc,
        make_request(
            "attach",
            {
                "model_path": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "model_family": "llama",
                "device": "cpu",
                "num_ranks": 1,
            },
            req_id,
        ),
    )
    resp = recv_message(proc)
    assert_jsonrpc(resp, req_id)
    assert resp.get("error") is None, f"attach: {resp.get('error')}"
    return req_id


def step_n(proc: object, req_id: int, count: int) -> int:
    """Step forward `count` ticks. Returns updated req_id."""
    req_id += 1
    send_message(
        proc,
        make_request(
            "rocket/step",
            {"direction": "forward", "count": count},
            req_id,
        ),
    )
    resp = recv_message(proc)
    assert_jsonrpc(resp, req_id)
    assert resp.get("error") is None, f"step: {resp.get('error')}"
    return req_id


def detach(proc: object, req_id: int) -> int:
    """Detach the model. Returns updated req_id."""
    req_id += 1
    send_message(proc, make_request("detach", {}, req_id))
    resp = recv_message(proc)
    assert_jsonrpc(resp, req_id)
    assert resp.get("error") is None, f"detach: {resp.get('error')}"
    return req_id


def register_interventions(proc: object, req_id: int) -> int:
    """Register scale interventions on multiple layers. Returns updated req_id."""
    for layer in INTERVENTION_LAYERS:
        req_id += 1
        send_message(
            proc,
            make_request(
                "rocket/intervene",
                {
                    "action": "set",
                    "recipe": {
                        "id": f"scale-layer-{layer}",
                        "type": "scale",
                        "target": f"llama:0:{layer}:attn.o_proj:fwd",
                        "params": {"factor": 1.0},
                        "priority": 0,
                    },
                },
                req_id,
            ),
        )
        resp = recv_message(proc)
        assert_jsonrpc(resp, req_id)
        assert resp.get("error") is None, f"intervene: {resp.get('error')}"
    return req_id


def timed_steps(proc: object, req_id: int, count: int) -> tuple[int, float]:
    """Step `count` ticks one at a time, return (req_id, elapsed_seconds)."""
    start = time.perf_counter()
    for _ in range(count):
        req_id = step_n(proc, req_id, 1)
    elapsed = time.perf_counter() - start
    return req_id, elapsed


def run_bench() -> None:
    build_binaries()
    proc = spawn_daemon()
    req_id = 0

    try:
        req_id = init_and_attach(proc, req_id)

        # === BASELINE ===
        req_id = attach_model(proc, req_id)

        print(f"\n[bench] Warmup: {WARMUP_STEPS} steps")
        req_id = step_n(proc, req_id, WARMUP_STEPS)

        print(f"[bench] Baseline: {STEP_COUNT} steps (no interventions)")
        req_id, baseline_time = timed_steps(proc, req_id, STEP_COUNT)
        ms_per = baseline_time / STEP_COUNT * 1000
        print(f"  baseline = {baseline_time:.4f}s ({ms_per:.2f}ms/step)")

        # === WITH INTERVENTIONS ===
        req_id = detach(proc, req_id)
        req_id = attach_model(proc, req_id)

        print(f"\n[bench] Registering {len(INTERVENTION_LAYERS)} interventions")
        req_id = register_interventions(proc, req_id)

        print(f"[bench] Warmup: {WARMUP_STEPS} steps (with interventions)")
        req_id = step_n(proc, req_id, WARMUP_STEPS)

        n_iv = len(INTERVENTION_LAYERS)
        print(f"[bench] Intervention: {STEP_COUNT} steps ({n_iv} interventions)")
        req_id, intervention_time = timed_steps(proc, req_id, STEP_COUNT)
        print(
            f"  intervention = {intervention_time:.4f}s "
            f"({intervention_time / STEP_COUNT * 1000:.2f}ms/step)"
        )

        # === RESULT ===
        overhead = (intervention_time - baseline_time) / baseline_time
        print(f"\n[bench] Overhead: {overhead:.4%}")
        print(f"[bench] Threshold: <= {MAX_OVERHEAD:.0%}")

        assert overhead <= MAX_OVERHEAD, (
            f"Intervention overhead {overhead:.4%} exceeds {MAX_OVERHEAD:.0%} threshold. "
            f"Baseline={baseline_time:.4f}s, Intervention={intervention_time:.4f}s"
        )
        print("[bench] PASS")

    finally:
        proc.stdin.close()
        proc.wait(timeout=60)


if __name__ == "__main__":
    run_bench()
    print("\nIntervention overhead benchmark passed!")
