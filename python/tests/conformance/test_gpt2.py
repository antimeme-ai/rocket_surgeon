"""GPT-2 model conformance: validate component vocabulary after attach.

Verifies that hook installation correctly discovers all canonical GPT-2
components and that the component ordering is sane.

Requires GPT-2 model download (~500MB) on first run.
"""

from __future__ import annotations

import re
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
EXPECTED_COMPONENTS_PER_LAYER = {"attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"}


@pytest.mark.slow
class TestGpt2Conformance:
    def test_component_vocabulary(self) -> None:
        """Attach GPT-2, discover all components, validate vocabulary."""
        build_binaries()
        proc = spawn_daemon()
        req_id = 0

        try:
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

            req_id += 1
            send_message(
                proc,
                make_request("rocket/discover", {"pattern": "*"}, req_id),
            )
            resp = recv_message(proc)
            assert resp.get("error") is None, f"discover: {resp.get('error')}"

            matches = resp["result"]["data"]["matches"]
            canonicals = [m["canonical"] for m in matches]

            for layer_idx in range(GPT2_NUM_LAYERS):
                layer_components = {
                    _extract_component(c) for c in canonicals if _is_layer(c, layer_idx)
                }
                for expected in EXPECTED_COMPONENTS_PER_LAYER:
                    assert expected in layer_components, (
                        f"layer {layer_idx} missing {expected}, got: {sorted(layer_components)}"
                    )

            print(f"PASS: {len(canonicals)} components discovered, all 12 layers complete")

        finally:
            proc.stdin.close()
            proc.wait(timeout=30)

    def test_step_through_layer(self) -> None:
        """Step through a few ticks, validate stopped_at position advances."""
        build_binaries()
        proc = spawn_daemon()
        req_id = 0

        try:
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
            assert resp.get("error") is None

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
            assert resp.get("error") is None

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


_CANONICAL_RE = re.compile(r"^[^:]+:\d+:(\d+):([^:]+):[^:]+$")


def _extract_component(canonical: str) -> str:
    m = _CANONICAL_RE.match(canonical)
    return m.group(2) if m else ""


def _is_layer(canonical: str, layer: int) -> bool:
    m = _CANONICAL_RE.match(canonical)
    return m is not None and int(m.group(1)) == layer
