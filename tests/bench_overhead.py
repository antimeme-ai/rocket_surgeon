"""Benchmark: rocket_surgeon hook overhead vs bare PyTorch.

Measures the per-forward-pass cost of the stepping architecture
(mailbox barriers, hook callbacks, tensor capture) compared to a
direct PyTorch forward pass with no hooks.

Usage:
    PYTHONPATH=python .venv/bin/python tests/bench_overhead.py
"""

from __future__ import annotations

import statistics
import time

import torch
from e2e_harness import (
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)
from transformers import GPT2LMHeadModel

TOKENS = [
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
N_PASSES = 20


def bench_rocket_surgeon() -> list[float]:
    proc = spawn_daemon()
    req_id = 0

    def rpc(method, params=None):
        nonlocal req_id
        req_id += 1
        send_message(proc, make_request(method, params, req_id=req_id))
        return recv_message(proc, timeout=120)

    try:
        rpc(
            "initialize",
            {"client_name": "overhead-bench", "protocol_version": "0.3.0"},
        )
        resp = rpc(
            "attach",
            {
                "model_path": "gpt2",
                "model_family": "gpt2",
                "device": "cpu",
                "num_ranks": 1,
            },
        )
        data = resp["result"]["data"]
        print(f"  Model: GPT-2 124M, {data['num_layers']} layers, {data['num_heads']} heads")

        # Warm-up
        rpc(
            "rocket/step",
            {
                "direction": "forward",
                "count": 500,
                "granularity": "layer",
                "tokens": TOKENS,
            },
        )

        times = []
        for _ in range(N_PASSES):
            t0 = time.perf_counter()
            resp = rpc(
                "rocket/step",
                {
                    "direction": "forward",
                    "count": 500,
                    "granularity": "layer",
                    "tokens": TOKENS,
                },
            )
            elapsed = time.perf_counter() - t0
            assert resp.get("error") is None
            times.append(elapsed)

        rpc("detach", {})
        return times

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()


def bench_bare_pytorch() -> list[float]:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    input_ids = torch.tensor([TOKENS])

    # Warm-up
    with torch.no_grad():
        model(input_ids)

    times = []
    for _ in range(N_PASSES):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    return times


def report(label: str, times: list[float]) -> None:
    mean_ms = statistics.mean(times) * 1000
    std_ms = statistics.stdev(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    print(f"  mean={mean_ms:.1f}ms  std={std_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")
    return mean_ms


if __name__ == "__main__":
    build_binaries()

    print(f"\n[1] Bare PyTorch forward pass (N={N_PASSES})")
    bare_times = bench_bare_pytorch()
    bare_mean = report("bare", bare_times)

    print(f"\n[2] Rocket Surgeon forward pass (N={N_PASSES})")
    rs_times = bench_rocket_surgeon()
    rs_mean = report("rs", rs_times)

    overhead_pct = ((rs_mean - bare_mean) / bare_mean) * 100
    print(f"\n{'=' * 60}")
    print("OVERHEAD MEASUREMENT")
    print(f"  Bare PyTorch:     {bare_mean:.1f}ms")
    print(f"  Rocket Surgeon:   {rs_mean:.1f}ms")
    print(f"  Overhead:         {overhead_pct:+.1f}%")
    print(f"  Absolute delta:   {rs_mean - bare_mean:.1f}ms")
    threshold = "PASS (<5%)" if overhead_pct < 5.0 else "FAIL (>=5%)"
    print(f"  10th Dentist #2:  {threshold}")
    print(f"{'=' * 60}")
