# IOI Initial Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the Wang et al. 2023 IOI circuit analysis on GPT-2 124M using rocket_surgeon's stepping architecture, answering the 10th Dentist's condition #3.

**Architecture:** Three independent fixes (layer step boundary, forward pass auto-reset, head-level bracket semantics) unblock a scripted 144-head zero-ablation sweep. Each fix is small and self-contained. The IOI test orchestrates the full workflow via the existing protocol.

**Tech Stack:** Rust (dispatch.rs, step_driver.rs, bridge.rs), Python (matching.py, engine.py), pytest, e2e harness

**Spec:** `docs/superpowers/specs/2026-05-30-ioi-initial-reproduction-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/rocket-surgeon-worker/src/dispatch.rs` | Modify | Layer drain logic, forward_complete auto-reset, head-slice tensor pre/post-processing in `try_apply_interventions`, head-slice in `collect_tensors` |
| `crates/rocket-surgeon-worker/src/step_driver.rs` | Modify | Add `DrainState` tracking for layer-granularity drain-to-boundary |
| `python/rocket_surgeon/host/interventions/matching.py` | Modify | Strip bracket notation from component segment before comparing |
| `tests/test_e2e_layer_step.py` | Create | E2e test: step N layers, verify stopped_at is last completed layer |
| `tests/test_e2e_head_ablation.py` | Create | E2e test: ablate one head, verify only that head's slice is zeroed |
| `tests/test_ioi_circuit.py` | Create | IOI initial reproduction: 144-head ablation sweep |

---

## Task 1: Layer Step Boundary — Drain to Completion

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/step_driver.rs`
- Modify: `crates/rocket-surgeon-worker/src/dispatch.rs:719-730`
- Create: `tests/test_e2e_layer_step.py`

### Understanding the current code

The step loop in `dispatch.rs:668-738` processes ticks from the forward-pass mailbox. For layer granularity (line 719-726):

```rust
if plan.granularity == rocket_surgeon_protocol::types::TickGranularity::Layer {
    if step_driver::is_layer_boundary(tracking_layer, layer) {
        ticks_consumed += 1;
    }
    tracking_layer = Some(layer);
} else {
    ticks_consumed += 1;
}

if ticks_consumed >= plan.ticks_to_drain {
    break;
}
```

The bug: when `ticks_consumed` reaches `ticks_to_drain`, the loop breaks immediately. At that instant, only the first component of the new layer has been processed. The layer is incomplete.

- [ ] **Step 1: Write the failing e2e test**

Create `tests/test_e2e_layer_step.py`:

```python
"""E2e test: layer-granularity stepping completes full layers.

After stepping N layers, the stopped_at position must be at the LAST
component of the Nth completed layer, and inspect must return tensor
data for components at that layer.
"""

from __future__ import annotations

from e2e_harness import (
    MODEL_FAMILY,
    MODEL_SOURCE,
    assert_jsonrpc,
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

TIMEOUT = 60


def run_test() -> None:
    build_binaries()
    proc = spawn_daemon()

    try:
        # Initialize
        send_message(proc, make_request("initialize", {
            "client_name": "layer-step-test",
            "protocol_version": "0.3.0",
        }, req_id=1))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None

        # Attach
        send_message(proc, make_request("attach", {
            "model_path": MODEL_SOURCE,
            "model_family": MODEL_FAMILY,
            "device": "cpu",
            "num_ranks": 1,
        }, req_id=2))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 2)
        assert resp.get("error") is None

        # Step 2 layers
        print("\n[test] Step 2 layers")
        send_message(proc, make_request("rocket/step", {
            "direction": "forward",
            "count": 2,
            "granularity": "layer",
        }, req_id=3))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None, f"step error: {resp.get('error')}"
        data = resp["result"]["data"]
        stopped = data["stopped_at"]

        # Key assertion: stopped_at layer must be 1 (0-indexed, 2 layers completed = layers 0,1)
        # and the component must NOT be the first component of layer 2
        print(f"  stopped_at: layer={stopped['layer']} component={stopped['component']}")
        assert stopped["layer"] <= 1, (
            f"Expected stopped_at.layer <= 1 after stepping 2 layers, "
            f"got layer={stopped['layer']} component={stopped['component']}"
        )

        # Inspect a component at the completed layer — should have tensor data
        layer = stopped["layer"]
        print(f"\n[test] Inspect down_proj at completed layer {layer}")
        send_message(proc, make_request("rocket/inspect", {
            "target": f"*:0:{layer}:down_proj:output",
            "detail": "summary",
        }, req_id=4))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None, (
            f"inspect at completed layer failed: {resp.get('error')}"
        )
        tensors = resp["result"]["data"]["tensors"]
        assert len(tensors) >= 1, (
            f"Expected at least 1 tensor at completed layer, got {len(tensors)}"
        )
        print(f"  tensors returned: {len(tensors)}")
        print("  PASS")

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    run_test()
    print("\nPASS — layer step boundary")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_layer_step.py 2>&1`

Expected: FAIL — inspect at the completed layer returns "no tensors captured" or the stopped_at layer is 2 instead of 1, because the current code breaks at the first component of layer 2 when stepping 2 layers.

- [ ] **Step 3: Fix step_driver to support drain-to-boundary**

Modify `crates/rocket-surgeon-worker/src/step_driver.rs`:

```rust
use rocket_surgeon_protocol::types::TickGranularity;

pub struct StepPlan {
    pub ticks_to_drain: u32,
    pub granularity: TickGranularity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainState {
    Counting,
    Draining,
}

pub fn plan_step(count: u32, granularity: Option<TickGranularity>) -> StepPlan {
    StepPlan {
        ticks_to_drain: count,
        granularity: granularity.unwrap_or(TickGranularity::Component),
    }
}

pub fn is_layer_boundary(current_layer: Option<u32>, new_layer: u32) -> bool {
    match current_layer {
        None => false,
        Some(prev) => new_layer != prev,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_step_defaults_to_component() {
        let plan = plan_step(3, None);
        assert_eq!(plan.ticks_to_drain, 3);
        assert_eq!(plan.granularity, TickGranularity::Component);
    }

    #[test]
    fn plan_step_respects_explicit_granularity() {
        let plan = plan_step(1, Some(TickGranularity::Layer));
        assert_eq!(plan.granularity, TickGranularity::Layer);
    }

    #[test]
    fn is_layer_boundary_detects_change() {
        assert!(is_layer_boundary(Some(0), 1));
        assert!(!is_layer_boundary(Some(0), 0));
        assert!(!is_layer_boundary(None, 0));
    }

    #[test]
    fn drain_state_transitions() {
        assert_eq!(DrainState::Counting, DrainState::Counting);
        assert_ne!(DrainState::Counting, DrainState::Draining);
    }
}
```

- [ ] **Step 4: Fix the step loop in dispatch.rs**

In `dispatch.rs`, modify the layer-granularity section of `run_step_loop` (lines ~719-730). Replace:

```rust
if plan.granularity == rocket_surgeon_protocol::types::TickGranularity::Layer {
    if step_driver::is_layer_boundary(tracking_layer, layer) {
        ticks_consumed += 1;
    }
    tracking_layer = Some(layer);
} else {
    ticks_consumed += 1;
}

if ticks_consumed >= plan.ticks_to_drain {
    break;
}
```

With:

```rust
if plan.granularity == rocket_surgeon_protocol::types::TickGranularity::Layer {
    if step_driver::is_layer_boundary(tracking_layer, layer) {
        if drain_state == step_driver::DrainState::Draining {
            break;
        }
        ticks_consumed += 1;
        if ticks_consumed >= plan.ticks_to_drain {
            drain_state = step_driver::DrainState::Draining;
        }
    }
    tracking_layer = Some(layer);
} else {
    ticks_consumed += 1;
    if ticks_consumed >= plan.ticks_to_drain {
        break;
    }
}
```

Also add `let mut drain_state = step_driver::DrainState::Counting;` near line 661 where other loop variables are initialized (near `let mut tracking_layer`).

The logic: when the Nth boundary fires, we transition to `Draining`. We keep consuming ticks until the NEXT boundary fires, then break. This means the loop processes all components of the Nth layer before stopping.

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_layer_step.py 2>&1`

Expected: PASS — stepped 2 layers, stopped_at.layer = 1, inspect returns real tensor data at that layer.

- [ ] **Step 6: Run existing tests to check for regressions**

Run: `cargo test -p rocket-surgeon-worker 2>&1 | tail -5`
Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_phase1.py 2>&1 | tail -5`
Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_inspect.py 2>&1 | tail -5`

Expected: All pass. Existing e2e tests use component granularity, which is unaffected.

- [ ] **Step 7: Commit**

```bash
git add crates/rocket-surgeon-worker/src/step_driver.rs crates/rocket-surgeon-worker/src/dispatch.rs tests/test_e2e_layer_step.py
git commit -m "fix(step): drain layer-granularity steps to completion

'Step N layers' now completes all components of the Nth layer
before stopping, instead of breaking at the first component of
the next layer. This fixes inspect and checkpoint failures when
stepping by layer granularity."
```

---

## Task 2: Forward Pass Auto-Reset on Completion

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/dispatch.rs:641-645`

### Understanding the current code

In `run_step_loop`, line 641-645:
```rust
let plan = step_driver::plan_step(req.count, req.granularity);
let resuming = state.forward_pass.is_some();

ensure_forward_pass(py, state, handle, req.input_ids.as_deref())?;
```

`ensure_forward_pass` (line 471) checks `if state.forward_pass.is_some() { return Ok(()); }`. After a forward pass completes, `forward_pass` is still `Some` with `forward_complete = true`. So the next step call no-ops — it sees the forward pass exists and returns immediately, never starting a new one.

The IOI test needs to run 144+ forward passes with the same tokens. Without auto-reset, only the first one runs.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_e2e_layer_step.py`, after the existing test:

```python
def run_auto_reset_test() -> None:
    """After forward_complete, a new step with tokens starts a fresh forward pass."""
    proc = spawn_daemon()

    try:
        # Initialize + Attach
        send_message(proc, make_request("initialize", {
            "client_name": "auto-reset-test",
            "protocol_version": "0.3.0",
        }, req_id=1))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None

        send_message(proc, make_request("attach", {
            "model_path": MODEL_SOURCE,
            "model_family": MODEL_FAMILY,
            "device": "cpu",
            "num_ranks": 1,
        }, req_id=2))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 2)
        assert resp.get("error") is None

        # First forward pass — step to completion
        print("\n[test] First forward pass to completion")
        send_message(proc, make_request("rocket/step", {
            "direction": "forward",
            "count": 200,
            "granularity": "layer",
        }, req_id=3))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None
        data = resp["result"]["data"]
        first_ticks = data["ticks_executed"]
        assert data.get("forward_complete") is True or first_ticks > 0, (
            "first forward pass should complete"
        )
        print(f"  first pass: {first_ticks} ticks")

        # Second forward pass — step again, should auto-reset and run
        print("\n[test] Second forward pass (auto-reset)")
        send_message(proc, make_request("rocket/step", {
            "direction": "forward",
            "count": 200,
            "granularity": "layer",
        }, req_id=4))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None, f"second step error: {resp.get('error')}"
        data = resp["result"]["data"]
        second_ticks = data["ticks_executed"]
        assert second_ticks > 0, (
            f"Expected second forward pass to run, got ticks_executed={second_ticks}"
        )
        print(f"  second pass: {second_ticks} ticks")
        print("  PASS")

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()
```

Update `__main__`:

```python
if __name__ == "__main__":
    build_binaries()
    run_test()
    run_auto_reset_test()
    print("\nAll layer step tests passed!")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_layer_step.py 2>&1`

Expected: `run_auto_reset_test` FAILS — second step returns `ticks_executed=0` because `ensure_forward_pass` sees the existing (completed) forward pass and no-ops.

- [ ] **Step 3: Add auto-reset in run_step_loop**

In `dispatch.rs`, modify `run_step_loop` at lines 641-645. Replace:

```rust
    let plan = step_driver::plan_step(req.count, req.granularity);
    let resuming = state.forward_pass.is_some();

    ensure_forward_pass(py, state, handle, req.input_ids.as_deref())?;
```

With:

```rust
    let plan = step_driver::plan_step(req.count, req.granularity);

    if state
        .forward_pass
        .as_ref()
        .is_some_and(|fwd| fwd.forward_complete)
    {
        let fwd = state.forward_pass.take().unwrap();
        Python::with_gil(|py_inner| {
            let _ = crate::bridge::remove_hooks(
                py_inner,
                &fwd.sentinel_handles
                    .iter()
                    .chain(fwd.capture_handles.iter())
                    .chain(fwd.passive_handles.iter())
                    .cloned()
                    .collect::<Vec<_>>(),
            );
        });
    }

    let resuming = state.forward_pass.is_some();
    ensure_forward_pass(py, state, handle, req.input_ids.as_deref())?;
```

This tears down the completed forward pass (removing its hooks) before `ensure_forward_pass` creates a new one. The new forward pass uses `req.input_ids` (the tokens from the step request).

Note: check the exact types of `sentinel_handles`, `capture_handles`, `passive_handles` on `ForwardPassState`. They should be `Vec<pyo3::PyObject>`. The `remove_hooks` function takes `&[pyo3::PyObject]`. If the handles are stored differently, adapt the collection.

- [ ] **Step 4: Verify ForwardPassState handle types**

Read `dispatch.rs` around line 522-530 where `ForwardPassState` is constructed to confirm the handle field types. Adjust the teardown code if needed.

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_layer_step.py 2>&1`

Expected: Both `run_test` and `run_auto_reset_test` PASS.

- [ ] **Step 6: Run regression tests**

Run: `cargo test -p rocket-surgeon-worker 2>&1 | tail -5`
Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_phase1.py 2>&1 | tail -5`

Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add crates/rocket-surgeon-worker/src/dispatch.rs tests/test_e2e_layer_step.py
git commit -m "feat(step): auto-reset forward pass on completion

When forward_complete is true and a new step arrives, tear down the
completed forward pass and start fresh. This enables repeated forward
passes with the same or different tokens without detach/re-attach."
```

---

## Task 3: Head-Level Bracket Notation — Intervention Slicing

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/dispatch.rs` (function `try_apply_interventions`)
- Modify: `python/rocket_surgeon/host/interventions/matching.py`
- Create: `tests/test_e2e_head_ablation.py`

### Understanding the current code

Intervention flow:
1. Client sends `rocket/intervene` with target `gpt2:0:9:o_proj[7]:output`
2. Daemon stores the recipe, includes it in `HostStepRequest.interventions`
3. Worker's `try_apply_interventions` (dispatch.rs:596) serializes recipes to JSON, calls `bridge::apply_interventions_at_point` with `canonical="o_proj"`, `layer=9`, etc.
4. Python's `apply_interventions_at_point` (bridge.py:369) calls `apply_interventions` (engine.py:23)
5. Python's `filter_recipes` (composition.py:13) calls `target_matches` (matching.py:6)
6. `target_matches` compares recipe target segment `o_proj[7]` against actual `o_proj` → NO MATCH

Two changes needed:
- Python `target_matches`: strip bracket notation before comparing
- Rust `try_apply_interventions`: when a recipe has a bracket index AND the component is attention-path, slice the tensor before/after calling Python

- [ ] **Step 1: Write the Python matching test**

Create `python/tests/test_head_matching.py`:

```python
from __future__ import annotations

from rocket_surgeon.host.interventions.matching import target_matches


def test_exact_component_matches():
    assert target_matches(
        target="gpt2:0:9:o_proj:output",
        family="gpt2",
        rank=0,
        layer=9,
        component="o_proj",
        event="output",
    )


def test_bracket_component_matches_base():
    assert target_matches(
        target="gpt2:0:9:o_proj[7]:output",
        family="gpt2",
        rank=0,
        layer=9,
        component="o_proj",
        event="output",
    )


def test_bracket_different_component_no_match():
    assert not target_matches(
        target="gpt2:0:9:o_proj[7]:output",
        family="gpt2",
        rank=0,
        layer=9,
        component="down_proj",
        event="output",
    )


def test_bracket_different_layer_no_match():
    assert not target_matches(
        target="gpt2:0:9:o_proj[7]:output",
        family="gpt2",
        rank=0,
        layer=8,
        component="o_proj",
        event="output",
    )


def test_wildcard_with_bracket():
    assert target_matches(
        target="*:*:*:o_proj[3]:output",
        family="gpt2",
        rank=0,
        layer=5,
        component="o_proj",
        event="output",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_head_matching.py -v 2>&1`

Expected: `test_bracket_component_matches_base`, `test_wildcard_with_bracket` FAIL — `o_proj[7]` != `o_proj`.

- [ ] **Step 3: Fix Python target_matches to strip brackets**

Modify `python/rocket_surgeon/host/interventions/matching.py`:

```python
"""Probe-point target matching for intervention recipes."""

from __future__ import annotations

import re

_BRACKET_RE = re.compile(r"\[(\d+)\]$")


def strip_bracket(segment: str) -> str:
    return _BRACKET_RE.sub("", segment)


def extract_head_index(segment: str) -> int | None:
    m = _BRACKET_RE.search(segment)
    return int(m.group(1)) if m else None


def target_matches(
    *,
    target: str,
    family: str,
    rank: int,
    layer: int,
    component: str,
    event: str,
) -> bool:
    """Return True if a recipe target matches the current execution point.

    Target format: family:rank:layer:component:event
    Wildcards: '*' matches any single segment.
    Bracket notation on component (e.g., o_proj[7]) is stripped before
    matching — head slicing is handled by the caller.
    """
    segments = target.split(":")
    expected_segments = 5
    if len(segments) != expected_segments:
        return False

    actual = [family, str(rank), str(layer), component, event]
    for pattern_seg, actual_seg in zip(segments, actual, strict=True):
        if pattern_seg == "*":
            continue
        if strip_bracket(pattern_seg) != actual_seg:
            return False
    return True
```

- [ ] **Step 4: Run Python test to verify it passes**

Run: `PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_head_matching.py -v 2>&1`

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit matching fix**

```bash
git add python/rocket_surgeon/host/interventions/matching.py python/tests/test_head_matching.py
git commit -m "feat(interventions): strip bracket notation in target matching

target_matches now treats o_proj[7] as matching component o_proj.
Head index extraction is exposed for callers to use for tensor slicing."
```

- [ ] **Step 6: Write the e2e head ablation test**

Create `tests/test_e2e_head_ablation.py`:

```python
"""E2e test: head-level ablation via bracket notation.

Ablates a single attention head (o_proj[0]) and verifies:
1. The intervention fires
2. The ablated head's slice is zeroed
3. Other heads are NOT zeroed
"""

from __future__ import annotations

import base64
import struct

from e2e_harness import (
    MODEL_FAMILY,
    MODEL_SOURCE,
    assert_jsonrpc,
    build_binaries,
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

TIMEOUT = 60


def run_test() -> None:
    build_binaries()
    proc = spawn_daemon()

    try:
        # Initialize + Attach
        send_message(proc, make_request("initialize", {
            "client_name": "head-ablation-test",
            "protocol_version": "0.3.0",
        }, req_id=1))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None

        send_message(proc, make_request("attach", {
            "model_path": MODEL_SOURCE,
            "model_family": MODEL_FAMILY,
            "device": "cpu",
            "num_ranks": 1,
        }, req_id=2))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 2)
        assert resp.get("error") is None
        num_heads = resp["result"]["data"]["num_heads"]
        hidden_dim = resp["result"]["data"]["hidden_dim"]
        head_dim = hidden_dim // num_heads
        print(f"  num_heads={num_heads}, hidden_dim={hidden_dim}, head_dim={head_dim}")

        # Register head-level ablation: zero head 0 of o_proj at layer 0
        print("\n[test] Register ablation on *:0:0:o_proj[0]:output")
        send_message(proc, make_request("rocket/intervene", {
            "action": "set",
            "recipe": {
                "id": "ablate-head-0",
                "type": "ablate",
                "target": "*:0:0:o_proj[0]:output",
                "params": {"mode": "zero"},
                "priority": 0,
            },
        }, req_id=3))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None, f"intervene error: {resp.get('error')}"
        print("  PASS — intervention registered")

        # Step through layer 0 (need enough to complete layer 0)
        print("\n[test] Step 1 layer")
        send_message(proc, make_request("rocket/step", {
            "direction": "forward",
            "count": 1,
            "granularity": "layer",
        }, req_id=4))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None, f"step error: {resp.get('error')}"
        data = resp["result"]["data"]

        # Check intervention fired
        fired = data.get("fired_interventions", [])
        assert "ablate-head-0" in fired, (
            f"Expected ablate-head-0 to fire, got: {fired}"
        )
        print(f"  interventions fired: {fired}")
        print("  PASS — intervention fired")

        # Inspect full o_proj at layer 0
        print("\n[test] Inspect o_proj at layer 0")
        send_message(proc, make_request("rocket/inspect", {
            "target": "*:0:0:o_proj:output",
            "detail": "full",
        }, req_id=5))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 5)
        assert resp.get("error") is None, f"inspect error: {resp.get('error')}"
        tensors = resp["result"]["data"]["tensors"]
        assert len(tensors) >= 1, "expected at least 1 tensor"
        tensor = tensors[0]
        data_b64 = tensor.get("data_base64")
        assert data_b64 is not None, "expected base64 tensor data"

        raw = base64.b64decode(data_b64)
        shape = tensor["shape"]
        dtype = tensor["dtype"]
        print(f"  shape={shape}, dtype={dtype}")

        # Decode float32 values
        assert dtype == "float32", f"expected float32, got {dtype}"
        values = list(struct.unpack(f"<{len(raw)//4}f", raw))

        # The tensor is shape [batch, seq, hidden_dim].
        # Head 0 occupies indices [0:head_dim] along the last dimension.
        # Head 1 occupies [head_dim:2*head_dim], etc.
        total_per_row = shape[-1]
        rows = len(values) // total_per_row

        head0_all_zero = True
        head1_all_zero = True
        for row in range(rows):
            base = row * total_per_row
            for i in range(head_dim):
                if values[base + i] != 0.0:
                    head0_all_zero = False
                if values[base + head_dim + i] != 0.0:
                    head1_all_zero = False

        assert head0_all_zero, "Head 0 should be all zeros after ablation"
        assert not head1_all_zero, "Head 1 should NOT be all zeros (not ablated)"
        print("  Head 0: all zeros (ablated) ✓")
        print("  Head 1: non-zero (not ablated) ✓")
        print("  PASS")

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    run_test()
    print("\nPASS — head-level ablation e2e")
```

- [ ] **Step 7: Run e2e test to verify it fails**

Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_head_ablation.py 2>&1`

Expected: FAIL — either the intervention doesn't fire (matching was step 3-4, should be fixed now), or the entire tensor is zeroed instead of just head 0 (because the Rust side doesn't slice by head yet).

- [ ] **Step 8: Add head-slicing in try_apply_interventions**

In `dispatch.rs`, modify `try_apply_interventions` (around line 596). The function currently takes the full output tensor and passes it to Python. For head-level interventions, we need to:

1. Before calling Python: check if any recipe in `req.interventions` has a bracket index on an attention-path component
2. If so: slice the tensor, pass the slice to Python, then copy the modified slice back

Replace the function body. The current function is:

```rust
fn try_apply_interventions<'py>(
    py: Python<'py>,
    state: &WorkerState,
    req: &HostStepRequest,
    tuple: &pyo3::Bound<'py, PyTuple>,
    layer: u32,
    canonical: &str,
    handle: u64,
) -> anyhow::Result<Option<(Bound<'py, pyo3::PyAny>, Vec<String>)>> {
    if req.interventions.is_empty() || tuple.len() <= 2 {
        return Ok(None);
    }
    let output = tuple.get_item(2)?;
    if output.is_none() {
        return Ok(None);
    }
    let family = state
        .component_map
        .as_ref()
        .map_or("unknown", |m| m.model_family.as_str());
    let recipes_json = serde_json::to_string(&req.interventions)?;
    let (modified, fired) = crate::bridge::apply_interventions_at_point(
        py,
        &output,
        &recipes_json,
        family,
        state.rank,
        layer,
        canonical,
        "output",
        state.tick_state.tick_id(),
        handle,
    )?;
    if fired.is_empty() {
        Ok(None)
    } else {
        Ok(Some((modified, fired)))
    }
}
```

Replace with:

```rust
const ATTENTION_HEAD_COMPONENTS: &[&str] = &["o_proj", "q_proj", "k_proj", "v_proj"];

fn extract_head_index(target: &str) -> Option<(String, u32)> {
    let segments: Vec<&str> = target.split(':').collect();
    if segments.len() != 5 {
        return None;
    }
    let comp = segments[3];
    if let Some(bracket_start) = comp.find('[') {
        let base = &comp[..bracket_start];
        let idx_str = &comp[bracket_start + 1..comp.len() - 1];
        if let Ok(idx) = idx_str.parse::<u32>() {
            let stripped = format!(
                "{}:{}:{}:{}:{}",
                segments[0], segments[1], segments[2], base, segments[4]
            );
            return Some((stripped, idx));
        }
    }
    None
}

fn try_apply_interventions<'py>(
    py: Python<'py>,
    state: &WorkerState,
    req: &HostStepRequest,
    tuple: &pyo3::Bound<'py, PyTuple>,
    layer: u32,
    canonical: &str,
    handle: u64,
) -> anyhow::Result<Option<(Bound<'py, pyo3::PyAny>, Vec<String>)>> {
    if req.interventions.is_empty() || tuple.len() <= 2 {
        return Ok(None);
    }
    let output = tuple.get_item(2)?;
    if output.is_none() {
        return Ok(None);
    }
    let family = state
        .component_map
        .as_ref()
        .map_or("unknown", |m| m.model_family.as_str());

    let mut head_recipes = Vec::new();
    let mut regular_recipes = Vec::new();

    for recipe in &req.interventions {
        if let Some((stripped_target, head_idx)) = extract_head_index(&recipe.target) {
            let base_comp = stripped_target.split(':').nth(3).unwrap_or("");
            if ATTENTION_HEAD_COMPONENTS.contains(&base_comp) && base_comp == canonical {
                head_recipes.push((recipe.clone(), head_idx));
            }
        } else {
            regular_recipes.push(recipe.clone());
        }
    }

    let mut all_fired = Vec::new();
    let mut current_output = output.clone();

    for (mut recipe, head_idx) in head_recipes {
        let shape: Vec<i64> = current_output.getattr("shape")?.extract()?;
        let hidden = *shape.last().unwrap() as u32;
        let num_heads = state.num_heads;
        let head_dim = hidden / num_heads;
        let start = head_idx * head_dim;
        let end = start + head_dim;

        let builtins = py.import("builtins")?;
        let py_slice = builtins.getattr("slice")?.call1((start, end))?;
        let py_ellipsis = builtins.getattr("Ellipsis")?;
        let index = PyTuple::new(py, [py_ellipsis.into_any(), py_slice.into_any()])?;
        let head_slice = current_output.call_method1("__getitem__", (index.clone(),))?;
        let head_clone = head_slice.call_method0("clone")?;

        // Strip bracket from target for Python matching
        if let Some((stripped, _)) = extract_head_index(&recipe.target) {
            recipe.target = stripped;
        }
        let recipes_json = serde_json::to_string(&[&recipe])?;

        let (modified_slice, fired) = crate::bridge::apply_interventions_at_point(
            py,
            &head_clone,
            &recipes_json,
            family,
            state.rank,
            layer,
            canonical,
            "output",
            state.tick_state.tick_id(),
            handle,
        )?;
        all_fired.extend(fired);

        current_output.call_method1("__setitem__", (index, modified_slice))?;
    }

    // Apply regular (non-head) interventions
    if !regular_recipes.is_empty() {
        let recipes_json = serde_json::to_string(&regular_recipes)?;
        let (modified, fired) = crate::bridge::apply_interventions_at_point(
            py,
            &current_output,
            &recipes_json,
            family,
            state.rank,
            layer,
            canonical,
            "output",
            state.tick_state.tick_id(),
            handle,
        )?;
        all_fired.extend(fired);
        current_output = modified;
    }

    if all_fired.is_empty() {
        Ok(None)
    } else {
        Ok(Some((current_output, all_fired)))
    }
}
```

**Important:** This code references `state.num_heads` which doesn't exist yet. We need to add it to `WorkerState` and populate it during attach.

- [ ] **Step 9: Add num_heads to WorkerState**

In `dispatch.rs`, add `pub num_heads: u32` to `WorkerState` struct (around line 40):

```rust
pub struct WorkerState {
    pub component_map: Option<ComponentMap>,
    pub component_index: HashMap<(String, u32), usize>,
    pub module_paths: Vec<String>,
    pub container_paths: Vec<String>,
    pub model_handle: Option<u64>,
    pub rank: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub hidden_dim: u32,
    // ... rest unchanged
}
```

Initialize in `WorkerState::new()`:
```rust
num_heads: 0,
hidden_dim: 0,
```

Populate in `handle_host_attach` from the attach response data (search for where `state.num_layers` is set and add `num_heads` and `hidden_dim` there too). The attach response from the Python bridge includes `num_heads` and `hidden_dim`.

- [ ] **Step 10: Run e2e test to verify it passes**

Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_head_ablation.py 2>&1`

Expected: PASS — head 0 is zeroed, head 1 is not.

- [ ] **Step 11: Run regression tests**

Run: `cargo test -p rocket-surgeon-worker 2>&1 | tail -5`
Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_interventions.py 2>&1 | tail -5`
Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_inspect.py 2>&1 | tail -5`

Expected: All pass.

- [ ] **Step 12: Commit**

```bash
git add crates/rocket-surgeon-worker/src/dispatch.rs python/rocket_surgeon/host/interventions/matching.py python/tests/test_head_matching.py tests/test_e2e_head_ablation.py
git commit -m "feat(interventions): head-level ablation via bracket notation

o_proj[7] targets head 7 of the attention output projection.
The Rust worker slices the tensor to the head range before passing
to the Python intervention engine. Bracket notation is stripped
from the target for matching. Only attention-path components
(o_proj, q_proj, k_proj, v_proj) support head indexing."
```

---

## Task 4: Head-Level Bracket Notation — Inspect Slicing

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/dispatch.rs` (function `collect_tensors`)

### Understanding the current code

`collect_tensors` (dispatch.rs:847-913) iterates matched components, reads tensor data from `last_outputs`, computes BLAKE3 hash, and returns `CapturedTensor` with bytes. It currently returns the full tensor for the matched component.

For head-level inspect (e.g., target `gpt2:0:0:o_proj[7]:output`), we need to:
1. Match the component (strip bracket for matching — this happens in `probe_matches_target` in capture.rs)
2. Slice the tensor to head 7's range
3. Return stats/bytes for the slice only

- [ ] **Step 1: Write a test for head-level inspect**

Add to `tests/test_e2e_head_ablation.py`, a new function:

```python
def run_inspect_head_test() -> None:
    """Inspect a specific head returns only that head's data."""
    proc = spawn_daemon()

    try:
        # Initialize + Attach
        send_message(proc, make_request("initialize", {
            "client_name": "inspect-head-test",
            "protocol_version": "0.3.0",
        }, req_id=1))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 1)
        assert resp.get("error") is None

        send_message(proc, make_request("attach", {
            "model_path": MODEL_SOURCE,
            "model_family": MODEL_FAMILY,
            "device": "cpu",
            "num_ranks": 1,
        }, req_id=2))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 2)
        assert resp.get("error") is None
        num_heads = resp["result"]["data"]["num_heads"]
        hidden_dim = resp["result"]["data"]["hidden_dim"]
        head_dim = hidden_dim // num_heads

        # Step 1 layer to populate data
        send_message(proc, make_request("rocket/step", {
            "direction": "forward",
            "count": 1,
            "granularity": "layer",
        }, req_id=3))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 3)
        assert resp.get("error") is None

        # Inspect full o_proj
        print("\n[test] Inspect full o_proj at layer 0")
        send_message(proc, make_request("rocket/inspect", {
            "target": "*:0:0:o_proj:output",
            "detail": "summary",
        }, req_id=4))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 4)
        assert resp.get("error") is None
        full_shape = resp["result"]["data"]["tensors"][0]["shape"]
        print(f"  full shape: {full_shape}")

        # Inspect head 0 of o_proj
        print("\n[test] Inspect o_proj[0] at layer 0")
        send_message(proc, make_request("rocket/inspect", {
            "target": "*:0:0:o_proj[0]:output",
            "detail": "summary",
        }, req_id=5))
        resp = recv_message(proc)
        assert_jsonrpc(resp, 5)
        assert resp.get("error") is None, f"inspect head error: {resp.get('error')}"
        head_tensors = resp["result"]["data"]["tensors"]
        assert len(head_tensors) >= 1
        head_shape = head_tensors[0]["shape"]
        print(f"  head shape: {head_shape}")

        # Head shape should have head_dim as the last dimension, not hidden_dim
        assert head_shape[-1] == head_dim, (
            f"Expected last dim = head_dim ({head_dim}), got {head_shape[-1]}"
        )
        print("  PASS — head inspect returns correct shape")

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()
```

Update `__main__`:
```python
if __name__ == "__main__":
    build_binaries()
    run_test()
    run_inspect_head_test()
    print("\nAll head ablation tests passed!")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_head_ablation.py 2>&1`

Expected: `run_inspect_head_test` FAIL — either "no components match" (bracket not stripped in probe matching) or full tensor returned (no slicing).

- [ ] **Step 3: Update probe_matches_target for bracket stripping**

In `crates/rocket-surgeon-worker/src/capture.rs`, the `probe_matches_target` function handles 5-segment normalization. Add bracket stripping to the component segment before matching. Find where the target is parsed and strip brackets before the probe grammar match.

The simplest approach: strip `[N]` from the component segment of the target before parsing. The existing 5-to-6 segment normalization already manipulates the target string. Add bracket stripping there.

- [ ] **Step 4: Add head slicing in collect_tensors**

In `dispatch.rs`, modify `collect_tensors` (line 847). After retrieving the tensor from `last_outputs` (line 872), check if the inspect target has a bracket index. If so, slice the tensor before computing bytes/stats.

The `handle_host_inspect` function at line 790 has access to `req.target`. Pass the target (or extracted head index) through to `collect_tensors`.

Modify `collect_tensors` signature to accept an optional head index:

```rust
fn collect_tensors(
    state: &mut WorkerState,
    matched_components: &[crate::adapter::MappedComponent],
    head_index: Option<u32>,
) -> anyhow::Result<Vec<CapturedTensor>> {
```

In `handle_host_inspect`, extract the head index from `req.target` before calling `collect_tensors`:

```rust
let head_index = extract_head_index_from_target(&req.target);
let tensors = match collect_tensors(state, &matched_components, head_index) {
```

Inside `collect_tensors`, after line 872 (`if let Some(tensor_obj) = dict.get_item(&key)?`), add:

```rust
let tensor_to_use = if let Some(hi) = head_index {
    if ATTENTION_HEAD_COMPONENTS.contains(&comp.canonical.as_str()) {
        let shape: Vec<i64> = bridge::get_tensor_shape_i64(py, &tensor_obj)?;
        let hidden = *shape.last().unwrap() as u32;
        let head_dim = hidden / state.num_heads;
        let start = hi * head_dim;
        let end = start + head_dim;
        let builtins = py.import("builtins")?;
        let py_slice = builtins.getattr("slice")?.call1((start, end))?;
        let py_ellipsis = builtins.getattr("Ellipsis")?;
        let index = PyTuple::new(py, [py_ellipsis.into_any(), py_slice.into_any()])?;
        tensor_obj.call_method1("__getitem__", (index,))?.call_method0("contiguous")?
    } else {
        tensor_obj.clone()
    }
} else {
    tensor_obj.clone()
};
```

Then use `tensor_to_use` instead of `tensor_obj` for `tensor_to_bytes`, `get_tensor_shape`, etc.

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_head_ablation.py 2>&1`

Expected: All tests PASS — head inspect returns tensor with `head_dim` as last dimension.

- [ ] **Step 6: Run regression tests**

Run: `PYTHONPATH=python .venv/bin/python tests/test_e2e_inspect.py 2>&1 | tail -5`

Expected: PASS — existing tests don't use bracket notation.

- [ ] **Step 7: Commit**

```bash
git add crates/rocket-surgeon-worker/src/dispatch.rs crates/rocket-surgeon-worker/src/capture.rs tests/test_e2e_head_ablation.py
git commit -m "feat(inspect): head-level tensor slicing via bracket notation

rocket/inspect with target o_proj[7] returns only head 7's slice
of the tensor. Shape, stats, and bytes reflect the sliced data."
```

---

## Task 5: IOI Initial Reproduction

**Files:**
- Create: `tests/test_ioi_circuit.py`

### Prerequisites

Tasks 1-4 must be complete and passing.

- [ ] **Step 1: Write the IOI test**

Create `tests/test_ioi_circuit.py`:

```python
"""IOI initial reproduction on GPT-2 124M.

Wang et al. 2023 "Interpretability in the Wild" identifies a circuit of
~26 attention heads that drive indirect object identification. This test
reproduces the core finding: a sparse subset of heads drives the
logit_diff, with name mover heads concentrated in layers 9-11.

Single prompt, zero-ablation sweep across all 144 heads (12 layers x 12 heads).
Not a full replication — proof that rocket_surgeon can perform real
interpretability analysis.

Usage:
    PYTHONPATH=python .venv/bin/python tests/test_ioi_circuit.py
"""

from __future__ import annotations

import base64
import json
import struct
import time

from e2e_harness import (
    make_request,
    recv_message,
    send_message,
    spawn_daemon,
)

TIMEOUT = 120

# "When Mary and John went to the store, John gave a drink to"
# GPT-2 BPE token IDs (verified against tiktoken / HF tokenizer)
# The model should predict " Mary" (token 5335) over " John" (token 1757)
PROMPT_TOKENS = [2437, 5335, 290, 1757, 1816, 284, 262, 3650, 11, 1757, 2921, 257, 4144, 284]
MARY_TOKEN = 5335
JOHN_TOKEN = 1757


def run() -> None:
    proc = spawn_daemon()
    req_id = 0

    def rpc(method, params=None):
        nonlocal req_id
        req_id += 1
        send_message(proc, make_request(method, params, req_id=req_id))
        return recv_message(proc, timeout=TIMEOUT)

    def assert_ok(resp, label):
        if resp.get("error"):
            raise AssertionError(f"{label}: {resp['error']['message']}")

    try:
        # --- Initialize ---
        print("[1] Initialize")
        resp = rpc("initialize", {"client_name": "ioi-circuit", "protocol_version": "0.3.0"})
        assert_ok(resp, "initialize")

        # --- Attach GPT-2 ---
        print("[2] Attach GPT-2 124M")
        t0 = time.monotonic()
        resp = rpc("attach", {
            "model_path": "gpt2",
            "model_family": "gpt2",
            "device": "cpu",
            "num_ranks": 1,
        })
        assert_ok(resp, "attach")
        attach_data = resp["result"]["data"]
        num_layers = attach_data["num_layers"]
        num_heads = attach_data["num_heads"]
        hidden_dim = attach_data["hidden_dim"]
        head_dim = hidden_dim // num_heads
        print(f"    layers={num_layers} heads={num_heads} hidden={hidden_dim} head_dim={head_dim}")
        print(f"    attach took {time.monotonic() - t0:.1f}s")

        # --- Baseline forward pass ---
        print(f"\n[3] Baseline forward pass ({len(PROMPT_TOKENS)} tokens)")
        t0 = time.monotonic()
        resp = rpc("rocket/step", {
            "direction": "forward",
            "count": 500,
            "granularity": "layer",
            "tokens": PROMPT_TOKENS,
        })
        assert_ok(resp, "baseline step")
        print(f"    ticks={resp['result']['data']['ticks_executed']} ({time.monotonic() - t0:.3f}s)")

        # --- Read baseline logits ---
        print("\n[4] Read baseline logits from lm_head")
        resp = rpc("rocket/inspect", {
            "target": "gpt2:0:*:lm_head:output",
            "detail": "full",
        })
        assert_ok(resp, "inspect lm_head")
        tensors = resp["result"]["data"]["tensors"]
        assert len(tensors) >= 1, "expected lm_head tensor"
        tensor = tensors[0]

        raw = base64.b64decode(tensor["data_base64"])
        shape = tensor["shape"]
        vocab_size = shape[-1]
        seq_len = shape[-2] if len(shape) >= 2 else 1
        print(f"    shape={shape} dtype={tensor['dtype']}")

        # Read logits at last token position
        values = struct.unpack(f"<{len(raw)//4}f", raw)
        last_pos_offset = (seq_len - 1) * vocab_size
        mary_logit = values[last_pos_offset + MARY_TOKEN]
        john_logit = values[last_pos_offset + JOHN_TOKEN]
        baseline_logit_diff = mary_logit - john_logit
        print(f"    logit(Mary)={mary_logit:.4f} logit(John)={john_logit:.4f}")
        print(f"    baseline logit_diff = {baseline_logit_diff:.4f}")
        assert baseline_logit_diff > 0, (
            f"Model should prefer Mary over John, got logit_diff={baseline_logit_diff:.4f}"
        )

        # --- Ablation sweep ---
        print(f"\n[5] Ablation sweep: {num_layers}x{num_heads} = {num_layers * num_heads} heads")
        results = []
        t0 = time.monotonic()

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                target = f"gpt2:0:{layer_idx}:o_proj[{head_idx}]:output"

                # Set ablation
                resp = rpc("rocket/intervene", {
                    "action": "set",
                    "recipe": {
                        "id": "sweep-ablate",
                        "type": "ablate",
                        "target": target,
                        "params": {"mode": "zero"},
                        "priority": 0,
                    },
                })
                assert_ok(resp, f"intervene L{layer_idx}H{head_idx}")

                # Forward pass with ablation
                resp = rpc("rocket/step", {
                    "direction": "forward",
                    "count": 500,
                    "granularity": "layer",
                    "tokens": PROMPT_TOKENS,
                })
                assert_ok(resp, f"step L{layer_idx}H{head_idx}")

                # Read logits
                resp = rpc("rocket/inspect", {
                    "target": "gpt2:0:*:lm_head:output",
                    "detail": "full",
                })
                assert_ok(resp, f"inspect L{layer_idx}H{head_idx}")
                raw = base64.b64decode(resp["result"]["data"]["tensors"][0]["data_base64"])
                vals = struct.unpack(f"<{len(raw)//4}f", raw)
                ablated_mary = vals[last_pos_offset + MARY_TOKEN]
                ablated_john = vals[last_pos_offset + JOHN_TOKEN]
                ablated_diff = ablated_mary - ablated_john
                delta = baseline_logit_diff - ablated_diff

                results.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "ablated_logit_diff": ablated_diff,
                    "delta": delta,
                })

                # Clear intervention
                resp = rpc("rocket/intervene", {
                    "action": "clear",
                    "intervention_id": "sweep-ablate",
                })
                assert_ok(resp, f"clear L{layer_idx}H{head_idx}")

            # Progress
            elapsed = time.monotonic() - t0
            done = (layer_idx + 1) * num_heads
            total = num_layers * num_heads
            print(f"    layer {layer_idx} done ({done}/{total}, {elapsed:.1f}s)")

        sweep_time = time.monotonic() - t0
        print(f"    sweep complete in {sweep_time:.1f}s")

        # --- Analysis ---
        print("\n[6] Analysis")
        results.sort(key=lambda r: abs(r["delta"]), reverse=True)

        threshold = abs(baseline_logit_diff) * 0.1
        significant = [r for r in results if abs(r["delta"]) > threshold]

        print(f"\n    Baseline logit_diff: {baseline_logit_diff:.4f}")
        print(f"    Threshold (10%): {threshold:.4f}")
        print(f"    Significant heads: {len(significant)} / {num_layers * num_heads}")

        print("\n    Top 20 heads by |delta|:")
        print(f"    {'Layer':>5} {'Head':>4} {'Delta':>10} {'Ablated LD':>10}")
        print(f"    {'-'*5:>5} {'-'*4:>4} {'-'*10:>10} {'-'*10:>10}")
        for r in results[:20]:
            marker = " ← NAME MOVER?" if r["layer"] >= 9 and r["delta"] > threshold else ""
            print(
                f"    {r['layer']:>5} {r['head']:>4} "
                f"{r['delta']:>+10.4f} {r['ablated_logit_diff']:>10.4f}{marker}"
            )

        # --- Validation ---
        print("\n[7] Validation")

        # Check 1: sparse subset drives the effect
        assert len(significant) < 30, (
            f"Expected <30 significant heads, got {len(significant)}"
        )
        print(f"    ✓ Sparse: {len(significant)} significant heads (<30)")

        # Check 2: top heads should include some in layers 9-11
        top10_layers = {r["layer"] for r in results[:10]}
        has_late_layers = bool(top10_layers & {9, 10, 11})
        if has_late_layers:
            print(f"    ✓ Late-layer heads in top 10: layers {top10_layers & {9, 10, 11}}")
        else:
            print(f"    ⚠ No late-layer heads in top 10 (layers present: {top10_layers})")
            print("      (This may differ from Wang et al. with single prompt)")

        # Check 3: ablating top head should substantially reduce logit_diff
        top_delta = results[0]["delta"]
        print(f"    Top head: L{results[0]['layer']}H{results[0]['head']} delta={top_delta:+.4f}")
        assert abs(top_delta) > threshold, (
            f"Top head delta ({top_delta:.4f}) should exceed threshold ({threshold:.4f})"
        )
        print(f"    ✓ Top head effect exceeds threshold")

        # --- Detach ---
        print("\n[8] Detach")
        resp = rpc("detach")
        assert_ok(resp, "detach")

        print("\n" + "=" * 70)
        print("IOI INITIAL REPRODUCTION COMPLETE")
        print(f"  Model: GPT-2 124M ({num_layers} layers, {num_heads} heads)")
        print(f"  Baseline logit_diff: {baseline_logit_diff:.4f}")
        print(f"  Significant heads: {len(significant)} / {num_layers * num_heads}")
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
    run()
```

- [ ] **Step 2: Verify the prompt token IDs**

Before running, verify the token IDs are correct. Run:

```bash
.venv/bin/python -c "
from transformers import GPT2Tokenizer
t = GPT2Tokenizer.from_pretrained('gpt2')
prompt = 'When Mary and John went to the store, John gave a drink to'
ids = t.encode(prompt)
print(f'Token IDs: {ids}')
print(f'Decoded: {t.decode(ids)}')
mary_id = t.encode(' Mary')[0]
john_id = t.encode(' John')[0]
print(f'Mary token: {mary_id}')
print(f'John token: {john_id}')
"
```

Update `PROMPT_TOKENS`, `MARY_TOKEN`, `JOHN_TOKEN` in the test file if they differ from the hardcoded values.

- [ ] **Step 3: Run the IOI test**

Run: `PYTHONPATH=python .venv/bin/python tests/test_ioi_circuit.py 2>&1`

Expected: PASS — baseline logit_diff positive, ablation sweep identifies sparse subset of heads, at least some in layers 9-11.

If it fails, debug by looking at the specific error (step failure, inspect failure, logit_diff negative, etc.) and fix.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ioi_circuit.py
git commit -m "feat: IOI initial reproduction on GPT-2 124M

144-head zero-ablation sweep reproduces the core finding from
Wang et al. 2023: a sparse subset of attention heads drives
indirect object identification, with name mover heads concentrated
in late layers. Answers the 10th Dentist's condition #3."
```

- [ ] **Step 5: Push**

```bash
git push origin phase3/replay-reverse-divergence
```
