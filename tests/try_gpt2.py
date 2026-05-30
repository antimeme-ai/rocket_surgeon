"""Try rocket_surgeon on real GPT-2 124M.

Smoke test: initialize, attach, step through layers, inspect activations,
checkpoint, replay with divergence detection. CPU-only.
"""

from __future__ import annotations

import json
import time

from e2e_harness import (
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

TIMEOUT = 120


def run() -> None:
    proc = spawn_daemon()
    req_id = 0

    def rpc(method, params=None):
        nonlocal req_id
        req_id += 1
        send_message(proc, make_request(method, params, req_id=req_id))
        resp = recv_message(proc, timeout=TIMEOUT)
        if resp.get("error"):
            print(f"  ERROR [{method}]: {resp['error']['message']}")
        return resp

    try:
        # --- Initialize ---
        print("\n[1] initialize")
        resp = rpc("initialize", {"client_name": "gpt2-smoke", "protocol_version": "0.3.0"})
        print(f"    status: {resp['result']['state']['status']}")

        # --- Attach GPT-2 ---
        print("\n[2] attach gpt2")
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
        elapsed = time.monotonic() - t0
        if resp.get("error"):
            print(f"    ATTACH FAILED after {elapsed:.1f}s")
            return

        state = resp["result"]["state"]
        data = resp["result"]["data"]
        print(f"    status: {state['status']}")
        nl = data["num_layers"]
        nh = data["num_heads"]
        hd = data["hidden_dim"]
        print(f"    num_layers: {nl}, num_heads: {nh}, hidden_dim: {hd}")
        print(f"    attach took {elapsed:.1f}s")

        # --- Step forward one component ---
        print("\n[3] step forward (component, count=1)")
        t0 = time.monotonic()
        resp = rpc("rocket/step", {"direction": "forward", "count": 1, "granularity": "component"})
        elapsed = time.monotonic() - t0
        if resp.get("error"):
            return
        stopped = resp["result"]["data"]["stopped_at"]
        tid = resp["result"]["state"]["tick_id"]
        ly = stopped["layer"]
        comp = stopped["component"]
        print(f"    tick_id={tid} layer={ly} component={comp} ({elapsed:.3f}s)")

        # --- Step through 3 layers ---
        print("\n[4] step forward 3 layers")
        t0 = time.monotonic()
        resp = rpc("rocket/step", {"direction": "forward", "count": 3, "granularity": "layer"})
        elapsed = time.monotonic() - t0
        if resp.get("error"):
            return
        stopped = resp["result"]["data"]["stopped_at"]
        tid = resp["result"]["state"]["tick_id"]
        ly = stopped["layer"]
        comp = stopped["component"]
        print(f"    tick_id={tid} layer={ly} component={comp} ({elapsed:.3f}s)")

        # --- Inspect MLP output ---
        print("\n[5] inspect down_proj output at current layer")
        layer = stopped["layer"]
        resp = rpc(
            "rocket/inspect",
            {
                "target": f"gpt2:0:{layer}:down_proj:output",
                "detail": "summary",
            },
        )
        if not resp.get("error"):
            idata = resp["result"].get("data", resp["result"])
            print(f"    {json.dumps(idata, indent=2)[:600]}")
        else:
            # try alternate formats
            resp = rpc(
                "rocket/inspect",
                {
                    "target": f"gpt2:{layer}:down_proj:output",
                    "detail": "summary",
                },
            )
            if not resp.get("error"):
                idata = resp["result"].get("data", resp["result"])
                print(f"    {json.dumps(idata, indent=2)[:600]}")

        # --- Inspect attention output ---
        print("\n[6] inspect o_proj output")
        resp = rpc(
            "rocket/inspect",
            {
                "target": f"gpt2:0:{layer}:o_proj:output",
                "detail": "summary",
            },
        )
        if not resp.get("error"):
            idata = resp["result"].get("data", resp["result"])
            print(f"    {json.dumps(idata, indent=2)[:600]}")

        # --- Create checkpoint ---
        print("\n[7] create checkpoint")
        resp = rpc("rocket/checkpoint", {"action": "create", "tier": "activation"})
        if not resp.get("error"):
            ckpt_data = resp["result"].get("data", {})
            print(f"    checkpoint: {json.dumps(ckpt_data, indent=2)[:300]}")
            ckpt_id = ckpt_data.get("checkpoint_id", "")
        else:
            ckpt_id = None

        # --- Step to end ---
        print("\n[8] step to end of forward pass")
        t0 = time.monotonic()
        resp = rpc("rocket/step", {"direction": "forward", "count": 200, "granularity": "layer"})
        elapsed = time.monotonic() - t0
        if not resp.get("error"):
            data = resp["result"]["data"]
            print(f"    ticks_executed: {data['ticks_executed']}")
            print(f"    forward_complete: {data.get('forward_complete', '?')}")
            print(f"    run-to-end took {elapsed:.3f}s")
        else:
            print("    (step returned error, likely forward complete)")

        # --- Replay from checkpoint ---
        if ckpt_id:
            print(f"\n[9] replay from checkpoint {ckpt_id}")
            t0 = time.monotonic()
            resp = rpc(
                "rocket/replay",
                {
                    "from_checkpoint": ckpt_id,
                    "verify": True,
                    "cosine_threshold": 0.999,
                    "mre_threshold": 0.05,
                },
            )
            elapsed = time.monotonic() - t0
            if not resp.get("error"):
                rdata = resp["result"].get("data", resp["result"])
                print(f"    {json.dumps(rdata, indent=2)[:500]}")
                print(f"    replay took {elapsed:.3f}s")
            else:
                print(f"    replay took {elapsed:.3f}s")

        # --- Status check ---
        print("\n[10] final status")
        resp = rpc("rocket/status")
        if not resp.get("error"):
            st = resp["result"]["state"]
            print(f"    status: {st['status']}")
            print(f"    tick_id: {st.get('tick_id')}")
            pos = st.get("position", {})
            if pos:
                print(f"    position: layer={pos.get('layer')} component={pos.get('component')}")

        # --- Detach ---
        print("\n[11] detach")
        resp = rpc("detach")
        print(f"    status: {resp['result']['state']['status']}")

        print("\n" + "=" * 60)
        print("GPT-2 124M smoke test complete")
        print("=" * 60)

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
            print(f"[cleanup] exit code {proc.returncode}")
        except Exception:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    run()
