"""Llama model conformance: validate component vocabulary and firing order.

Uses hf-internal-testing/tiny-random-LlamaForCausalLM (2 layers, 16 hidden)
for fast CI. Validates discover vocabulary and step-by-step firing order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tests"))
from e2e_harness import (  # noqa: E402
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

LLAMA_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
LLAMA_FAMILY = "llama"
LLAMA_NUM_LAYERS = 2

DISCOVER_PER_LAYER_COMPONENTS = {
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "attn.scores",
    "mlp",
    "residual_post",
}
DISCOVER_GLOBAL_COMPONENTS = {"lm_head"}


def _init_and_attach(proc: object, req_id: int) -> int:
    """Initialize and attach Llama. Returns updated req_id."""
    req_id += 1
    send_message(
        proc,
        make_request(
            "initialize",
            {"client_name": "conformance-llama", "protocol_version": "0.3.0"},
            req_id,
        ),
    )
    resp = recv_message(proc)
    assert resp.get("error") is None, f"initialize: {resp.get('error')}"

    req_id += 1
    send_message(
        proc,
        make_request(
            "attach",
            {
                "model_path": LLAMA_MODEL,
                "model_family": LLAMA_FAMILY,
                "device": "cpu",
                "num_ranks": 1,
            },
            req_id,
        ),
    )
    resp = recv_message(proc)
    assert resp.get("error") is None, f"attach: {resp.get('error')}"
    return req_id


@pytest.mark.slow
class TestLlamaConformance:
    def test_component_vocabulary(self) -> None:
        """Attach Llama, discover all components, validate vocabulary."""
        build_binaries()
        proc = spawn_daemon()
        req_id = 0

        try:
            req_id = _init_and_attach(proc, req_id)

            req_id += 1
            send_message(
                proc,
                make_request("rocket/discover", {"pattern": "*:*:*:*:*"}, req_id),
            )
            resp = recv_message(proc)
            assert resp.get("error") is None, f"discover: {resp.get('error')}"

            matches = resp["result"]["data"]["matches"]
            canonicals = [m["canonical"] for m in matches]
            unique_canonicals = set(canonicals)

            all_expected = DISCOVER_PER_LAYER_COMPONENTS | DISCOVER_GLOBAL_COMPONENTS
            for expected in all_expected:
                assert expected in unique_canonicals, (
                    f"missing component {expected}, got: {sorted(unique_canonicals)}"
                )

            expected_total = LLAMA_NUM_LAYERS * len(DISCOVER_PER_LAYER_COMPONENTS) + len(
                DISCOVER_GLOBAL_COMPONENTS
            )
            assert len(canonicals) == expected_total, (
                f"expected {expected_total} entries ({LLAMA_NUM_LAYERS} layers x "
                f"{len(DISCOVER_PER_LAYER_COMPONENTS)} + "
                f"{len(DISCOVER_GLOBAL_COMPONENTS)} global), got {len(canonicals)}"
            )

            print(
                f"PASS: {len(canonicals)} components discovered, "
                f"all {LLAMA_NUM_LAYERS} layers complete"
            )

        finally:
            proc.stdin.close()
            proc.wait(timeout=30)

    def test_probe_firing_order(self) -> None:
        """Step one tick at a time, validate firing order properties."""
        build_binaries()
        proc = spawn_daemon()
        req_id = 0

        try:
            req_id = _init_and_attach(proc, req_id)

            ticks: list[tuple[int, str]] = []
            max_ticks = 100

            for _ in range(max_ticks):
                req_id += 1
                send_message(
                    proc,
                    make_request(
                        "rocket/step",
                        {"direction": "forward", "count": 1},
                        req_id,
                    ),
                )
                resp = recv_message(proc)
                assert resp.get("error") is None, f"step: {resp.get('error')}"

                data = resp["result"]["data"]
                stopped = data["stopped_at"]
                layer = stopped["layer"]
                component = stopped["component"]
                ticks.append((layer, component))

                if component == "lm_head":
                    break

            assert len(ticks) > 0, "no ticks recorded"

            layers_seen = [t[0] for t in ticks]
            non_final = layers_seen[:-1]
            for i in range(1, len(non_final)):
                assert non_final[i] >= non_final[i - 1], (
                    f"layer order not monotonic at tick {i}: {non_final[i - 1]} -> {non_final[i]}"
                )

            components_by_layer: dict[int, list[str]] = {}
            for layer, component in ticks:
                components_by_layer.setdefault(layer, []).append(component)

            for layer_idx in range(LLAMA_NUM_LAYERS):
                layer_comps = components_by_layer.get(layer_idx, [])
                assert len(layer_comps) > 0, f"layer {layer_idx} has no components"
                assert len(layer_comps) == len(set(layer_comps)), (
                    f"layer {layer_idx} has duplicate components: {layer_comps}"
                )

            print(
                f"PASS: {len(ticks)} ticks across {LLAMA_NUM_LAYERS} layers, "
                f"layers monotonic, no duplicates"
            )

        finally:
            proc.stdin.close()
            proc.wait(timeout=120)
