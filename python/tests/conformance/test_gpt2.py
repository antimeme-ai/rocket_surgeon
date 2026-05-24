"""GPT-2 model conformance: validate component vocabulary and firing order.

Verifies that hook installation correctly discovers all canonical GPT-2
components, that the component ordering is sane, and that probe firing
order during a forward pass follows the expected pattern.

Requires GPT-2 model download (~500MB) on first run.
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

GPT2_MODEL = "gpt2"
GPT2_NUM_LAYERS = 12
PER_LAYER_COMPONENTS = {
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "attn.scores",
    "mlp",
    "residual_post",
}
GLOBAL_COMPONENTS = {"lm_head"}
ALL_COMPONENTS = PER_LAYER_COMPONENTS | GLOBAL_COMPONENTS


def _init_and_attach(proc: object, req_id: int) -> int:
    """Initialize and attach GPT-2. Returns updated req_id."""
    req_id += 1
    send_message(
        proc,
        make_request(
            "initialize",
            {"client_name": "conformance", "protocol_version": "0.3.0"},
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
                "model_path": GPT2_MODEL,
                "model_family": "gpt2",
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
class TestGpt2Conformance:
    def test_component_vocabulary(self) -> None:
        """Attach GPT-2, discover all components, validate vocabulary."""
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

            for expected in ALL_COMPONENTS:
                assert expected in unique_canonicals, (
                    f"missing component {expected}, got: {sorted(unique_canonicals)}"
                )

            expected_total = GPT2_NUM_LAYERS * len(PER_LAYER_COMPONENTS) + len(GLOBAL_COMPONENTS)
            assert len(canonicals) == expected_total, (
                f"expected {expected_total} entries ({GPT2_NUM_LAYERS} layers x "
                f"{len(PER_LAYER_COMPONENTS)} + {len(GLOBAL_COMPONENTS)} global), "
                f"got {len(canonicals)}"
            )

            print(f"PASS: {len(canonicals)} components discovered, all layers complete")

        finally:
            proc.stdin.close()
            proc.wait(timeout=30)

    def test_probe_firing_order(self) -> None:
        """Step one tick at a time, validate component firing order."""
        build_binaries()
        proc = spawn_daemon()
        req_id = 0

        try:
            req_id = _init_and_attach(proc, req_id)

            ticks: list[tuple[int, str]] = []
            max_ticks = GPT2_NUM_LAYERS * len(PER_LAYER_COMPONENTS) + 50

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

            for layer_idx in range(GPT2_NUM_LAYERS):
                layer_components = set(components_by_layer.get(layer_idx, []))
                for expected in PER_LAYER_COMPONENTS:
                    assert expected in layer_components, (
                        f"layer {layer_idx} missing component {expected}, "
                        f"got: {sorted(layer_components)}"
                    )

            for layer_idx in range(GPT2_NUM_LAYERS):
                layer_comps = components_by_layer.get(layer_idx, [])
                assert len(layer_comps) == len(set(layer_comps)), (
                    f"layer {layer_idx} has duplicate components: {layer_comps}"
                )

            print(
                f"PASS: {len(ticks)} ticks, layers monotonic, "
                f"all components present, no duplicates"
            )

        finally:
            proc.stdin.close()
            proc.wait(timeout=120)

    def test_step_through_layer(self) -> None:
        """Step through a few ticks, validate stopped_at position advances."""
        build_binaries()
        proc = spawn_daemon()
        req_id = 0

        try:
            req_id = _init_and_attach(proc, req_id)

            req_id += 1
            send_message(
                proc,
                make_request(
                    "rocket/step",
                    {"direction": "forward", "count": 4},
                    req_id,
                ),
            )
            resp = recv_message(proc)
            assert resp.get("error") is None, f"step: {resp.get('error')}"

            data = resp["result"]["data"]
            stopped_at = data["stopped_at"]
            assert stopped_at["tick_id"] > 0
            assert isinstance(stopped_at["layer"], int)
            assert isinstance(stopped_at["component"], str)
            assert len(stopped_at["component"]) > 0

            print(
                f"PASS: stepped to tick {stopped_at['tick_id']}, "
                f"layer {stopped_at['layer']}, component {stopped_at['component']}"
            )

        finally:
            proc.stdin.close()
            proc.wait(timeout=60)
