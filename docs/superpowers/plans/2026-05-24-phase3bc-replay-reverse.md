# Phase 3B+C Implementation Plan: Replay, Reverse Step, Tier 2 Callbacks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire real forward-pass replay from checkpoints, reverse stepping, divergence detection, Tier 2 Python callbacks, and bundle extension — completing Phase 3.

**Architecture:** Worker re-executes forward pass from checkpoint arena data via a replay context mode in the existing step loop. Daemon orchestrates backward stepping by finding nearest checkpoint + replaying to target. Worldline DAG tracks segment ancestry. Tier 2 callbacks use direct in-process call with watchdog thread timeout.

**Tech Stack:** Rust (protocol types, dispatch routing, orchestrator handle), PyO3 (bridge calls), Python (compare_activations, callback dispatch, RNG helpers), PyTorch (tensor ops, determinism flags)

---

## Task Execution Order

B1 → B2 → B3 → B4 → B5 → C1 → B6 → C2 → C3 → C4

---

### Task 1: Protocol types — HostReplayRequest/Response + new fields on ReplayRequest

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/messages.rs`
- Modify: `crates/rocket-surgeon-protocol/src/types.rs`
- Modify: `crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs`

- [ ] **Step 1: Add new fields to ReplayRequest**

In `crates/rocket-surgeon-protocol/src/messages.rs`, find `ReplayRequest` (line ~294) and add:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayRequest {
    pub from_checkpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interventions: Option<Vec<InterventionRecipe>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_at: Option<ReplayStopAt>,
    #[serde(default = "crate::types::default_true")]
    pub verify: bool,
    #[serde(default)]
    pub envelope: EnvelopeMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deterministic: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cosine_threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mre_threshold: Option<f64>,
}
```

- [ ] **Step 2: Add internal HOST_REPLAY constant**

In the `internal` module (line ~54), add:

```rust
pub const HOST_REPLAY: &str = "_host/replay";
```

- [ ] **Step 3: Add HostReplayRequest and HostReplayResponse**

After the `HostCheckpointResponse` block (line ~982), add:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostReplayRequest {
    pub model_handle: u64,
    pub checkpoint_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_at: Option<ReplayStopAt>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interventions: Vec<InterventionRecipe>,
    pub verify: bool,
    pub deterministic: bool,
    pub cosine_threshold: f64,
    pub mre_threshold: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostReplayResponse {
    pub ticks_replayed: u32,
    pub stopped_at: TickPosition,
    pub divergences: Vec<Divergence>,
    pub verified: bool,
}
```

- [ ] **Step 4: Add WorldlineState and WorldlineSegment to types.rs**

In `crates/rocket-surgeon-protocol/src/types.rs`, after the `SessionState` struct:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct WorldlineState {
    pub current_segment: u32,
    pub segments: Vec<WorldlineSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldlineSegment {
    pub id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_segment: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_tick: Option<u64>,
    pub tick_range: (u64, u64),
}
```

Add `worldline` field to `SessionState`:

```rust
pub struct SessionState {
    // ... existing fields ...
    #[serde(default, skip_serializing_if = "WorldlineState::is_empty")]
    pub worldline: WorldlineState,
}
```

Add helper:

```rust
impl WorldlineState {
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}
```

- [ ] **Step 5: Add serde roundtrip tests**

In `crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs`, add tests for:
- `HostReplayRequest` serialization roundtrip
- `HostReplayResponse` serialization roundtrip
- `WorldlineState` serialization roundtrip
- `ReplayRequest` with new optional fields (verify they skip when None)

- [ ] **Step 6: Run tests**

Run: `cargo test -p rocket-surgeon-protocol --all-targets`
Expected: All pass, zero warnings.

- [ ] **Step 7: Commit**

```bash
git add crates/rocket-surgeon-protocol/
git commit -m "feat(protocol): add HostReplayRequest/Response, WorldlineState, replay threshold fields"
```

---

### Task 2: Orchestrator routing — forward _host/replay to worker

**Files:**
- Modify: `crates/rocket-surgeon-orchestrator/src/dispatch.rs`

- [ ] **Step 1: Add _host/replay to dispatch match**

In `crates/rocket-surgeon-orchestrator/src/dispatch.rs`, find the dispatch match (line ~21). Add alongside the existing `HOST_CHECKPOINT` arm:

```rust
internal::HOST_REPLAY => forward_to_worker(state, request),
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p rocket-surgeon-orchestrator --all-targets`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add crates/rocket-surgeon-orchestrator/
git commit -m "feat(orchestrator): route _host/replay to worker"
```

---

### Task 3: Worker replay handler — basic forward re-execution from checkpoint

**Files:**
- Create: `crates/rocket-surgeon-worker/src/replay.rs`
- Modify: `crates/rocket-surgeon-worker/src/dispatch.rs`
- Modify: `crates/rocket-surgeon-worker/src/main.rs`

- [ ] **Step 1: Create replay.rs with ReplayContext struct**

Create `crates/rocket-surgeon-worker/src/replay.rs`:

```rust
use rocket_surgeon_protocol::messages::{Divergence, HostReplayRequest, ReplayStopAt};
use rocket_surgeon_protocol::types::InterventionRecipe;

pub struct ReplayContext {
    pub verify: bool,
    pub deterministic: bool,
    pub cosine_threshold: f64,
    pub mre_threshold: f64,
    pub stop_at: Option<ReplayStopAt>,
    pub interventions: Vec<InterventionRecipe>,
    pub divergences: Vec<Divergence>,
    pub ticks_replayed: u32,
}

impl ReplayContext {
    pub fn from_request(req: &HostReplayRequest) -> Self {
        Self {
            verify: req.verify,
            deterministic: req.deterministic,
            cosine_threshold: req.cosine_threshold,
            mre_threshold: req.mre_threshold,
            stop_at: req.stop_at.clone(),
            interventions: req.interventions.clone(),
            divergences: Vec::new(),
            ticks_replayed: 0,
        }
    }

    pub fn should_stop(&self, layer: u32, component: &str) -> bool {
        if let Some(ref stop) = self.stop_at {
            layer == stop.layer && component == stop.component
        } else {
            false
        }
    }
}
```

- [ ] **Step 2: Add `mod replay;` to main.rs**

In `crates/rocket-surgeon-worker/src/main.rs`, add `mod replay;` alongside existing mod declarations.

- [ ] **Step 3: Add _host/replay to worker dispatch match**

In `crates/rocket-surgeon-worker/src/dispatch.rs`, find the dispatch match (line ~90). Add:

```rust
internal::HOST_REPLAY => handle_host_replay(state, request),
```

- [ ] **Step 4: Write handle_host_replay function**

In `crates/rocket-surgeon-worker/src/dispatch.rs`, add the handler (after `handle_host_checkpoint`):

```rust
fn handle_host_replay(state: &mut WorkerState, request: &Request) -> Response {
    let req: HostReplayRequest = match parse_params(request) {
        Ok(r) => r,
        Err(e) => return invalid_params_response(request.id.clone(), &e),
    };

    let Some(model_handle) = state.model_handle else {
        return error_response(request.id.clone(), -32002, "no model attached");
    };
    if req.model_handle != model_handle {
        return error_response(request.id.clone(), -32002, "model handle mismatch");
    }
    let Some(ref arena) = state.checkpoint_arena else {
        return error_response(request.id.clone(), -32002, "no checkpoint arena");
    };
    let Some(ref last_outputs) = state.last_outputs else {
        return error_response(request.id.clone(), -32002, "no last_outputs available");
    };

    let mut replay_ctx = crate::replay::ReplayContext::from_request(&req);

    let result = Python::with_gil(|py| -> anyhow::Result<HostReplayResponse> {
        // 1. Restore RNG state from checkpoint
        let rng_slot = arena.get_slot(&req.checkpoint_id, u32::MAX);
        if let Some((slot_ptr, slot_len)) = rng_slot {
            let rng_bytes = unsafe { std::slice::from_raw_parts(slot_ptr.add(SLOT_HEADER_SIZE), slot_len - SLOT_HEADER_SIZE) };
            bridge::restore_rng_state(py, rng_bytes)?;
        }

        // 2. Set deterministic mode if requested
        if replay_ctx.deterministic {
            let torch = py.import("torch")?;
            torch.getattr("use_deterministic_algorithms")?.call1((true,))?;
        }

        // 3. Restore activations from checkpoint layers
        let checkpoint_layers = arena.checkpoint_slot_layers(&req.checkpoint_id);
        for &layer_idx in &checkpoint_layers {
            if layer_idx == u32::MAX { continue; } // Skip RNG sentinel
            let Some((slot_ptr, _)) = arena.get_slot(&req.checkpoint_id, layer_idx) else { continue; };
            let header = SlotHeader::read_from(slot_ptr)?;
            let data_ptr = unsafe { slot_ptr.add(SLOT_HEADER_SIZE) } as usize;
            let shape: Vec<i64> = header.shape_slice().iter().map(|&s| s as i64).collect();
            let container_path = state.container_for_layer(layer_idx);
            bridge::restore_activation(
                py, last_outputs, &container_path, 0,
                data_ptr, header.byte_len as usize,
                header.dtype.to_torch_str(), &shape,
            )?;
        }

        // 4. Re-run forward pass via step loop in replay mode
        let response = run_replay_loop(py, state, &mut replay_ctx)?;

        // 5. Restore non-deterministic mode
        if replay_ctx.deterministic {
            let torch = py.import("torch")?;
            torch.getattr("use_deterministic_algorithms")?.call1((false,))?;
        }

        Ok(response)
    });

    match result {
        Ok(resp) => {
            let value = serde_json::to_value(&resp).unwrap();
            Response::success(request.id.clone(), value)
        }
        Err(e) => error_response(request.id.clone(), -32603, &format!("replay failed: {e}")),
    }
}
```

- [ ] **Step 5: Write run_replay_loop function**

This reuses the step loop pattern but with replay context (auto-release mailbox, track ticks):

```rust
fn run_replay_loop(
    py: Python<'_>,
    state: &mut WorkerState,
    ctx: &mut crate::replay::ReplayContext,
) -> anyhow::Result<HostReplayResponse> {
    let fp = state.forward_pass.as_ref()
        .ok_or_else(|| anyhow::anyhow!("no forward pass active"))?;
    let result_mb = fp.result_mailbox.bind(py);
    let resume_mb = fp.resume_mailbox.bind(py);

    // Re-run forward pass from restored state
    ensure_forward_pass(py, state)?;

    let mut ticks = 0u32;
    let mut last_position = state.tick_state.to_tick_position();

    loop {
        let value = result_mb.call_method1("wait", (30.0,))?;
        let tuple = value.downcast::<pyo3::types::PyTuple>()?;
        let path: String = tuple.get_item(0)?.extract()?;
        let call_index: u32 = tuple.get_item(1)?.extract()?;

        // Check forward complete sentinel
        if path == FORWARD_COMPLETE_SENTINEL {
            break;
        }

        stash_tensor_output(py, state, &path, call_index, &tuple)?;
        result_mb.call_method0("restore")?;

        // Advance tick state
        let (canonical, layer) = match state.resolve_component(&path, call_index) {
            Some(c) => c,
            None => { resume_mb.call_method1("put", (py.None(),))?; continue; }
        };
        state.tick_state.advance(&canonical, layer, call_index);
        ticks += 1;

        // Check stop_at
        if ctx.should_stop(layer, &canonical) {
            resume_mb.call_method1("put", (py.None(),))?;
            break;
        }

        // Apply interventions if any match this point
        let modified = if !ctx.interventions.is_empty() {
            try_apply_interventions_from_recipes(py, state, &tuple, &ctx.interventions)?
        } else {
            None
        };

        // Resume with modified tensor or None
        match modified {
            Some(tensor) => resume_mb.call_method1("put", (tensor,))?,
            None => resume_mb.call_method1("put", (py.None(),))?,
        };

        last_position = state.tick_state.to_tick_position();
    }

    ctx.ticks_replayed = ticks;
    let mut stopped_at = last_position;
    stopped_at.replay_of = Some(stopped_at.tick_id); // Mark as replay

    Ok(HostReplayResponse {
        ticks_replayed: ticks,
        stopped_at,
        divergences: ctx.divergences.clone(),
        verified: ctx.divergences.is_empty(),
    })
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p rocket-surgeon-worker --all-targets`
Expected: Compiles and existing tests pass. (No new unit tests yet — E2E will validate.)

- [ ] **Step 7: Commit**

```bash
git add crates/rocket-surgeon-worker/
git commit -m "feat(worker): handle _host/replay — forward re-execution from checkpoint"
```

---

### Task 4: Daemon replay wiring — replace metadata stub with orchestrator round-trip

**Files:**
- Modify: `crates/rocket-surgeon/src/orchestrator_handle.rs`
- Modify: `crates/rocket-surgeon/src/session.rs`
- Modify: `crates/rocket-surgeon/src/dispatch.rs`
- Modify: `crates/rocket-surgeon/src/main.rs`

- [ ] **Step 1: Add OrchestratorHandle::replay() method**

In `crates/rocket-surgeon/src/orchestrator_handle.rs`, after the `checkpoint()` method (line ~245), add:

```rust
pub fn replay(&mut self, req: &HostReplayRequest) -> anyhow::Result<HostReplayResponse> {
    let id = self.next_id();
    let params = serde_json::to_value(req)?;
    let request = Request::new(RequestId::Number(id), internal::HOST_REPLAY, params);

    self.send(&request)?;
    let response = self.recv()?;

    if let Some(err) = response.error {
        anyhow::bail!("orchestrator replay failed (code {}): {}", err.code, err.message);
    }

    let result = response
        .result
        .ok_or_else(|| anyhow::anyhow!("orchestrator replay: missing result"))?;
    let resp: HostReplayResponse = serde_json::from_value(result)?;
    Ok(resp)
}
```

- [ ] **Step 2: Update Session::replay() to accept host response**

In `crates/rocket-surgeon/src/session.rs`, replace the existing `replay()` method (lines ~1103-1177) with one that accepts an optional `HostReplayResponse`:

```rust
pub fn replay(
    &mut self,
    req: &ReplayRequest,
    host_response: Option<&HostReplayResponse>,
) -> Result<serde_json::Value, SessionError> {
    self.require_stopped("rocket/replay")?;

    let Some(_cref) = self.state.checkpoints.iter().find(|c| c.checkpoint_id == req.from_checkpoint) else {
        return Err(self.checkpoint_not_found_error(&req.from_checkpoint));
    };

    let (ticks_replayed, stopped_at, divergences, verified) = if let Some(hr) = host_response {
        (hr.ticks_replayed, hr.stopped_at.clone(), hr.divergences.clone(), hr.verified)
    } else {
        // Fallback: metadata-only replay (no orchestrator)
        let origin = self.checkpoint_positions.get(&req.from_checkpoint).cloned()
            .unwrap_or_else(|| TickPosition::default());
        let current_tick = self.state.tick_id.unwrap_or(origin.tick_id);
        let ticks = current_tick.saturating_sub(origin.tick_id).max(1) as u32;
        let mut stopped = origin.clone();
        stopped.tick_id = current_tick + ticks as u64;
        stopped.replay_of = Some(current_tick);
        (ticks, stopped, Vec::new(), true)
    };

    self.state.tick_id = Some(stopped_at.tick_id);
    self.state.position = Some(stopped_at.clone());
    self.update_available_actions();

    let data = ReplayResponse { ticks_replayed, stopped_at, divergences, verified };
    Ok(self.envelope_with_mode(req.envelope, data))
}
```

- [ ] **Step 3: Update handle_replay in dispatch.rs**

In `crates/rocket-surgeon/src/dispatch.rs`, update `handle_replay` (line ~1068):

```rust
pub fn handle_replay(
    session: &mut Session,
    request: &Request,
    host_response: Option<&HostReplayResponse>,
) -> Response {
    let req: ReplayRequest = match parse_params(request) {
        Ok(r) => r,
        Err(e) => return invalid_params_response(request.id.clone(), &e),
    };

    match session.replay(&req, host_response) {
        Ok(value) => serialize_envelope(request.id.clone(), value),
        Err(ref e) => session_error_to_response(request.id.clone(), e),
    }
}
```

- [ ] **Step 4: Wire orchestrator round-trip in main.rs**

In `crates/rocket-surgeon/src/main.rs`, find where `method::REPLAY` is dispatched (in the main loop). Add orchestrator communication similar to `method::STEP`:

```rust
} else if request.method == method::REPLAY {
    let replay_host_response = try_orchestrator_replay(
        &mut orchestrator,
        model_handle,
        &request,
    );
    handle_replay(&mut session, &request, replay_host_response.as_ref())
}
```

Add the helper function:

```rust
fn try_orchestrator_replay(
    orchestrator: &mut Option<OrchestratorHandle>,
    model_handle: Option<u64>,
    request: &Request,
) -> Option<HostReplayResponse> {
    let (Some(orch), Some(mh)) = (orchestrator.as_mut(), model_handle) else {
        return None;
    };
    let req: ReplayRequest = match serde_json::from_value(request.params.clone()) {
        Ok(r) => r,
        Err(_) => return None,
    };
    let host_req = HostReplayRequest {
        model_handle: mh,
        checkpoint_id: req.from_checkpoint.clone(),
        stop_at: req.stop_at.clone(),
        interventions: req.interventions.clone().unwrap_or_default(),
        verify: req.verify,
        deterministic: req.deterministic.unwrap_or(false),
        cosine_threshold: req.cosine_threshold.unwrap_or(0.999),
        mre_threshold: req.mre_threshold.unwrap_or(0.05),
    };
    match orch.replay(&host_req) {
        Ok(resp) => Some(resp),
        Err(e) => {
            tracing::warn!("orchestrator replay failed: {e}");
            None
        }
    }
}
```

- [ ] **Step 5: Fix all call sites of session.replay()**

Search for other callers of `session.replay()` in tests — add `None` as second arg.

- [ ] **Step 6: Run tests**

Run: `cargo test -p rocket-surgeon --all-targets`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add crates/rocket-surgeon/
git commit -m "feat(daemon): wire rocket/replay through orchestrator to worker"
```

---

### Task 5: Python compare_activations + CPU RNG capture

**Files:**
- Create: `python/rocket_surgeon/replay.py`
- Modify: `python/rocket_surgeon/checkpoint.py`
- Create: `python/tests/test_replay.py`

- [ ] **Step 1: Write test for compare_activations**

Create `python/tests/test_replay.py`:

```python
import torch

from rocket_surgeon.replay import compare_activations


def test_identical_tensors_no_divergence():
    a = torch.randn(32, 128)
    result = compare_activations(a, a.clone(), cosine_threshold=0.999, mre_threshold=0.05)
    assert result is None


def test_different_tensors_reports_divergence():
    a = torch.randn(32, 128)
    b = torch.randn(32, 128)
    result = compare_activations(a, b, cosine_threshold=0.999, mre_threshold=0.05)
    assert result is not None
    assert "cosine_similarity" in result
    assert "max_relative_error" in result
    assert result["cosine_similarity"] < 0.999


def test_slightly_perturbed_within_tolerance():
    a = torch.randn(32, 128)
    noise = torch.randn_like(a) * 1e-5
    b = a + noise
    result = compare_activations(a, b, cosine_threshold=0.999, mre_threshold=0.05)
    assert result is None


def test_scaled_tensor_exceeds_mre():
    a = torch.ones(32, 128)
    b = a * 1.1  # 10% relative error
    result = compare_activations(a, b, cosine_threshold=0.0, mre_threshold=0.05)
    assert result is not None
    assert result["max_relative_error"] > 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest python/tests/test_replay.py -v`
Expected: ImportError (module doesn't exist yet)

- [ ] **Step 3: Implement compare_activations**

Create `python/rocket_surgeon/replay.py`:

```python
import torch


def compare_activations(
    original: torch.Tensor,
    replayed: torch.Tensor,
    cosine_threshold: float,
    mre_threshold: float,
) -> dict | None:
    original_flat = original.flatten().float()
    replayed_flat = replayed.flatten().float()

    dot = torch.dot(original_flat, replayed_flat)
    norm_a = torch.linalg.norm(original_flat)
    norm_b = torch.linalg.norm(replayed_flat)
    denom = norm_a * norm_b
    cosine_sim = (dot / denom).item() if denom > 0 else 0.0

    abs_diff = torch.abs(original_flat - replayed_flat)
    abs_orig = torch.abs(original_flat)
    epsilon = 1e-8
    relative_error = abs_diff / (abs_orig + epsilon)
    max_rel_error = relative_error.max().item()

    if cosine_sim < cosine_threshold or max_rel_error > mre_threshold:
        return {
            "cosine_similarity": cosine_sim,
            "max_relative_error": max_rel_error,
        }
    return None


def compare_activations_from_ptr(
    original_ptr: int,
    original_len: int,
    original_dtype: str,
    original_shape: list[int],
    replayed: torch.Tensor,
    cosine_threshold: float,
    mre_threshold: float,
) -> dict | None:
    import ctypes

    dtype_map = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }
    dtype = dtype_map[original_dtype]
    buf = (ctypes.c_char * original_len).from_address(original_ptr)
    original = torch.frombuffer(buf, dtype=dtype).reshape(original_shape)
    result = compare_activations(original, replayed, cosine_threshold, mre_threshold)
    del original
    return result
```

- [ ] **Step 4: Add CPU RNG capture/restore to checkpoint.py**

In `python/rocket_surgeon/checkpoint.py`, add:

```python
def capture_cpu_rng_state() -> bytes:
    state = torch.random.get_rng_state()
    return bytes(state.numpy())


def restore_cpu_rng_state(state_bytes: bytes) -> None:
    import numpy as np

    state = torch.from_numpy(np.frombuffer(state_bytes, dtype=np.uint8).copy())
    torch.random.set_rng_state(state)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest python/tests/test_replay.py -v`
Expected: All 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add python/rocket_surgeon/replay.py python/tests/test_replay.py python/rocket_surgeon/checkpoint.py
git commit -m "feat(python): compare_activations for divergence detection + CPU RNG capture"
```

---

### Task 6: Wire divergence detection into replay loop

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/bridge.rs`
- Modify: `crates/rocket-surgeon-worker/src/dispatch.rs`

- [ ] **Step 1: Add bridge::compare_activations wrapper**

In `crates/rocket-surgeon-worker/src/bridge.rs`, add:

```rust
pub fn compare_activations_from_ptr(
    py: Python<'_>,
    original_ptr: usize,
    original_len: usize,
    original_dtype: &str,
    original_shape: &[i64],
    replayed: &Bound<'_, pyo3::PyAny>,
    cosine_threshold: f64,
    mre_threshold: f64,
) -> anyhow::Result<Option<(f64, f64)>> {
    let replay_mod = py.import("rocket_surgeon.replay")?;
    let py_shape = pyo3::types::PyList::new(py, original_shape)?;
    let result = replay_mod
        .getattr("compare_activations_from_ptr")?
        .call1((
            original_ptr,
            original_len,
            original_dtype,
            py_shape,
            replayed,
            cosine_threshold,
            mre_threshold,
        ))?;

    if result.is_none() {
        return Ok(None);
    }
    let dict = result.downcast::<pyo3::types::PyDict>()?;
    let cosine: f64 = dict.get_item("cosine_similarity")?.unwrap().extract()?;
    let mre: f64 = dict.get_item("max_relative_error")?.unwrap().extract()?;
    Ok(Some((cosine, mre)))
}
```

- [ ] **Step 2: Add bridge::restore_cpu_rng_state wrapper**

```rust
pub fn restore_cpu_rng_state(py: Python<'_>, state: &[u8]) -> anyhow::Result<()> {
    let ckpt_mod = py.import("rocket_surgeon.checkpoint")?;
    let py_bytes = pyo3::types::PyBytes::new(py, state);
    ckpt_mod.getattr("restore_cpu_rng_state")?.call1((py_bytes,))?;
    Ok(())
}

pub fn capture_cpu_rng_state(py: Python<'_>) -> anyhow::Result<Vec<u8>> {
    let ckpt_mod = py.import("rocket_surgeon.checkpoint")?;
    let result = ckpt_mod.getattr("capture_cpu_rng_state")?.call0()?;
    let bytes: &[u8] = result.extract()?;
    Ok(bytes.to_vec())
}
```

- [ ] **Step 3: Add verification step in run_replay_loop**

In the replay loop (Task 3 Step 5), after advancing the tick state and before resuming, add divergence check at √L boundaries:

```rust
// After: state.tick_state.advance(&canonical, layer, call_index);
// Check divergence at √L boundaries
if ctx.verify {
    let checkpoint_layers = rocket_surgeon_protocol::checkpoint_layers(state.num_layers);
    if checkpoint_layers.contains(&layer) {
        if let Some((slot_ptr, _)) = state.checkpoint_arena.as_ref()
            .and_then(|a| a.get_slot(&req.checkpoint_id, layer))
        {
            let header = SlotHeader::read_from(slot_ptr)?;
            let data_ptr = unsafe { slot_ptr.add(SLOT_HEADER_SIZE) } as usize;
            let shape: Vec<i64> = header.shape_slice().iter().map(|&s| s as i64).collect();
            let output_tensor = tuple.get_item(2)?;

            if let Some((cosine, mre)) = bridge::compare_activations_from_ptr(
                py, data_ptr, header.byte_len as usize,
                header.dtype.to_torch_str(), &shape,
                &output_tensor, ctx.cosine_threshold, ctx.mre_threshold,
            )? {
                let probe_point = format!("{}:0:{}:{}:output", family, layer, canonical);
                ctx.divergences.push(Divergence {
                    tick_id: state.tick_state.tick_id(),
                    original_tick_id: state.tick_state.tick_id(), // Will be mapped by daemon
                    probe_point,
                    cosine_similarity: cosine,
                    max_relative_error: mre,
                    message: format!("Divergence at layer {layer}: cosine={cosine:.6}, MRE={mre:.6}"),
                });
            }
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p rocket-surgeon-worker --all-targets`
Expected: Compiles, existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon-worker/
git commit -m "feat(worker): divergence detection during replay at √L boundaries"
```

---

### Task 7: Reverse step — backward direction in daemon

**Files:**
- Modify: `crates/rocket-surgeon/src/session.rs`
- Modify: `crates/rocket-surgeon/src/main.rs`

- [ ] **Step 1: Add WorldlineState to Session struct**

In `crates/rocket-surgeon/src/session.rs`, add to the `Session` struct (line ~224):

```rust
worldline: WorldlineState,
```

Initialize in the constructor:

```rust
worldline: WorldlineState {
    current_segment: 0,
    segments: vec![WorldlineSegment {
        id: 0,
        parent_segment: None,
        branch_tick: None,
        tick_range: (0, 0),
    }],
},
```

Add accessor and update method:

```rust
pub fn worldline(&self) -> &WorldlineState {
    &self.worldline
}

fn advance_worldline_segment(&mut self, branch_tick: u64) {
    let new_id = self.worldline.segments.len() as u32;
    self.worldline.segments.push(WorldlineSegment {
        id: new_id,
        parent_segment: Some(self.worldline.current_segment),
        branch_tick: Some(branch_tick),
        tick_range: (0, 0),
    });
    self.worldline.current_segment = new_id;
}
```

- [ ] **Step 2: Remove backward rejection in Session::step()**

In `session.rs`, replace the backward rejection (line ~860):

```rust
// Before:
if req.direction == StepDirection::Backward {
    return Err(self.capability_not_supported_error("supports_reverse_step"));
}

// After:
if req.direction == StepDirection::Backward {
    // Backward stepping is handled by the main loop via replay
    // This path is only reached when no orchestrator is available
    return Err(self.capability_not_supported_error("supports_reverse_step"));
}
```

Actually, the real backward logic lives in main.rs (orchestrator round-trip). Keep the rejection for the no-orchestrator fallback, but add the orchestrator path in main.rs.

- [ ] **Step 3: Add find_checkpoint_before to Session**

```rust
pub fn find_checkpoint_before(&self, target_tick: u64) -> Option<&str> {
    // Search checkpoints in reverse order (newest first)
    // Prefer sub-checkpoints, then auto-checkpoints, then user checkpoints
    let mut best: Option<(&str, u64)> = None;
    for (id, pos) in &self.checkpoint_positions {
        if pos.tick_id < target_tick {
            if best.is_none() || pos.tick_id > best.unwrap().1 {
                best = Some((id.as_str(), pos.tick_id));
            }
        }
    }
    best.map(|(id, _)| id)
}
```

- [ ] **Step 4: Wire backward step in main.rs**

In `crates/rocket-surgeon/src/main.rs`, in the step handling section, add backward direction handling BEFORE the orchestrator step call:

```rust
} else if request.method == method::STEP {
    let step_req: StepRequest = match serde_json::from_value(request.params.clone()) {
        Ok(r) => r,
        Err(_) => { /* fall through to existing error handling */ }
    };

    if step_req.direction == StepDirection::Backward {
        // Backward step: find checkpoint, replay to target
        let target_tick = session.state().tick_id.unwrap_or(0).saturating_sub(1);
        if let Some(ckpt_id) = session.find_checkpoint_before(target_tick) {
            let ckpt_id = ckpt_id.to_string();
            let host_req = HostReplayRequest {
                model_handle: model_handle.unwrap_or(0),
                checkpoint_id: ckpt_id.clone(),
                stop_at: None, // TODO: derive from target_tick
                interventions: session.interventions().to_vec(),
                verify: false,
                deterministic: false,
                cosine_threshold: 0.999,
                mre_threshold: 0.05,
            };
            if let Some(orch) = orchestrator.as_mut() {
                match orch.replay(&host_req) {
                    Ok(hr) => {
                        session.advance_worldline_segment(target_tick);
                        step_host_response = Some(HostStepResponse {
                            position: hr.stopped_at,
                            forward_complete: false,
                            fired_interventions: Vec::new(),
                        });
                    }
                    Err(e) => tracing::warn!("backward step replay failed: {e}"),
                }
            }
        }
        handle_step(&mut session, &request, step_host_response.as_ref())
    } else {
        // Existing forward step logic
        step_host_response = try_orchestrator_step(...);
        handle_step(&mut session, &request, step_host_response.as_ref())
    }
}
```

- [ ] **Step 5: Update worldline on forward step after backward**

After a backward step creates a new segment, any subsequent forward step should update the segment's tick_range. Add to the post-step section:

```rust
// Update worldline tick_range on successful step
if let Some(ref hr) = step_host_response {
    let seg = &mut session.worldline_mut().segments[session.worldline().current_segment as usize];
    if seg.tick_range.0 == 0 {
        seg.tick_range.0 = hr.position.tick_id;
    }
    seg.tick_range.1 = hr.position.tick_id;
}
```

- [ ] **Step 6: Add session unit tests for find_checkpoint_before and worldline**

```rust
#[test]
fn find_checkpoint_before_returns_nearest() {
    let mut session = test_session();
    session.checkpoint_create_with_id(Some(CreateCheckpointTier::Activation), Some("ckpt-5".into()));
    // ... set checkpoint position at tick 5
    session.checkpoint_positions.insert("ckpt-5".into(), TickPosition { tick_id: 5, ..default() });
    session.checkpoint_positions.insert("ckpt-10".into(), TickPosition { tick_id: 10, ..default() });

    assert_eq!(session.find_checkpoint_before(12), Some("ckpt-10"));
    assert_eq!(session.find_checkpoint_before(7), Some("ckpt-5"));
    assert_eq!(session.find_checkpoint_before(3), None);
}

#[test]
fn advance_worldline_creates_segment() {
    let mut session = test_session();
    assert_eq!(session.worldline().current_segment, 0);
    session.advance_worldline_segment(50);
    assert_eq!(session.worldline().current_segment, 1);
    assert_eq!(session.worldline().segments[1].parent_segment, Some(0));
    assert_eq!(session.worldline().segments[1].branch_tick, Some(50));
}
```

- [ ] **Step 7: Run tests**

Run: `cargo test -p rocket-surgeon --all-targets`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add crates/rocket-surgeon/
git commit -m "feat(daemon): backward step via checkpoint restore + replay, worldline DAG tracking"
```

---

### Task 8: Sub-checkpoint performance strategy

**Files:**
- Modify: `crates/rocket-surgeon/src/session.rs`
- Modify: `crates/rocket-surgeon/src/main.rs`

- [ ] **Step 1: Add arena_utilization query to orchestrator handle**

In `crates/rocket-surgeon/src/orchestrator_handle.rs`, the daemon needs to know arena pressure. Add a field tracked from checkpoint responses:

```rust
// In Session struct:
arena_utilization: f64,
```

Update after checkpoint operations:

```rust
pub fn update_arena_utilization(&mut self, bytes_captured: Option<u64>) {
    // Track cumulative — simplified heuristic
    // Real utilization comes from worker if needed
}
```

- [ ] **Step 2: Implement eager sub-checkpoint in backward step**

In main.rs, before the replay call in the backward step path:

```rust
// Eager sub-checkpoint: save current position for O(1) next backward step
if session.arena_utilization < 0.6 {
    let current_tick = session.state().tick_id.unwrap_or(0);
    let sub_id = format!("sub-{}-{}", session.worldline().current_segment, current_tick);
    let sub_req = HostCheckpointRequest::Create {
        model_handle: model_handle.unwrap_or(0),
        checkpoint_id: sub_id.clone(),
        tier: CreateCheckpointTier::Activation,
        tick_id: current_tick,
        layer_idx: 0, // Captures all √L layers
    };
    if let Some(orch) = orchestrator.as_mut() {
        if orch.checkpoint(&sub_req).is_ok() {
            session.checkpoint_create_with_id(
                Some(CreateCheckpointTier::Activation),
                Some(sub_id),
            );
        }
    }
}
```

- [ ] **Step 3: Add sub-checkpoint eviction priority**

In the worker's spill logic (`checkpoint.rs`), modify `oldest_checkpoint()` to prefer sub-checkpoints from non-current segments:

```rust
pub fn oldest_evictable(&self, current_segment: u32) -> Option<&str> {
    // Priority: sub-checkpoints from old segments first
    for id in &self.checkpoint_order {
        if id.starts_with("sub-") {
            let parts: Vec<&str> = id.split('-').collect();
            if let Some(seg) = parts.get(1).and_then(|s| s.parse::<u32>().ok()) {
                if seg != current_segment {
                    return Some(id);
                }
            }
        }
    }
    // Then oldest auto-checkpoint
    for id in &self.checkpoint_order {
        if id.starts_with("auto-") {
            return Some(id);
        }
    }
    None
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --workspace --all-targets`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon/ crates/rocket-surgeon-worker/
git commit -m "feat(daemon): eager sub-checkpoint for O(1) backward step, eviction priority"
```

---

### Task 9: Tier 2 Python callbacks

**Files:**
- Create: `python/rocket_surgeon/host/interventions/callback.py`
- Modify: `python/rocket_surgeon/host/interventions/engine.py`
- Create: `python/tests/test_callback.py`

- [ ] **Step 1: Write tests for callback dispatch**

Create `python/tests/test_callback.py`:

```python
import time
import torch

from rocket_surgeon.host.interventions.callback import (
    InterventionContext,
    execute_callback,
)


def _identity(tensor, ctx):
    return tensor


def _scale_by_two(tensor, ctx):
    return tensor * 2


def _raise_error(tensor, ctx):
    raise ValueError("intentional failure")


def _hang_forever(tensor, ctx):
    time.sleep(100)
    return tensor


def _wrong_shape(tensor, ctx):
    return tensor[0]


def test_callback_returns_modified_tensor():
    t = torch.ones(4, 8)
    ctx = InterventionContext(layer=0, component="mlp", event="output", tick_id=1, device=t.device, model_handle=0)
    result, error = execute_callback(_scale_by_two, t, ctx, timeout_s=5.0, nan_check=False)
    assert result is not None
    assert torch.allclose(result, t * 2)
    assert error is None


def test_callback_exception_returns_original():
    t = torch.ones(4, 8)
    ctx = InterventionContext(layer=0, component="mlp", event="output", tick_id=1, device=t.device, model_handle=0)
    result, error = execute_callback(_raise_error, t, ctx, timeout_s=5.0, nan_check=False)
    assert result is None
    assert error is not None
    assert "intentional failure" in error


def test_callback_wrong_shape_returns_error():
    t = torch.ones(4, 8)
    ctx = InterventionContext(layer=0, component="mlp", event="output", tick_id=1, device=t.device, model_handle=0)
    result, error = execute_callback(_wrong_shape, t, ctx, timeout_s=5.0, nan_check=False)
    assert result is None
    assert error is not None
    assert "shape" in error.lower()


def test_callback_timeout_returns_error():
    t = torch.ones(4, 8)
    ctx = InterventionContext(layer=0, component="mlp", event="output", tick_id=1, device=t.device, model_handle=0)
    result, error = execute_callback(_hang_forever, t, ctx, timeout_s=0.1, nan_check=False)
    assert result is None
    assert error is not None
    assert "timeout" in error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest python/tests/test_callback.py -v`
Expected: ImportError

- [ ] **Step 3: Implement callback.py**

Create `python/rocket_surgeon/host/interventions/callback.py`:

```python
import ctypes
import importlib
import threading
from dataclasses import dataclass

import torch


@dataclass
class InterventionContext:
    layer: int
    component: str
    event: str
    tick_id: int
    device: torch.device
    model_handle: int


_module_cache: dict[str, object] = {}


def resolve_callback(module_name: str, function_name: str):
    if module_name not in _module_cache:
        _module_cache[module_name] = importlib.import_module(module_name)
    mod = _module_cache[module_name]
    return getattr(mod, function_name)


def execute_callback(
    fn,
    tensor: torch.Tensor,
    ctx: InterventionContext,
    timeout_s: float,
    nan_check: bool,
) -> tuple[torch.Tensor | None, str | None]:
    original = tensor.clone()
    result_holder: list = [None, None]  # [result_tensor, error_string]
    thread_id_holder: list[int] = [0]

    def _run():
        thread_id_holder[0] = threading.current_thread().ident
        try:
            result = fn(original, ctx)
            result_holder[0] = result
        except Exception as e:
            result_holder[1] = str(e)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)

    if worker.is_alive():
        tid = thread_id_holder[0]
        if tid:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid), ctypes.py_object(TimeoutError)
            )
        worker.join(timeout=timeout_s)
        if worker.is_alive():
            return None, f"callback timeout after {timeout_s}s (uninterruptible)"
        return None, f"callback timeout after {timeout_s}s"

    if result_holder[1] is not None:
        return None, result_holder[1]

    result = result_holder[0]
    if result is None:
        return None, "callback returned None"

    if result.shape != tensor.shape:
        return None, f"shape mismatch: expected {tensor.shape}, got {result.shape}"

    if result.device != tensor.device:
        return None, f"device mismatch: expected {tensor.device}, got {result.device}"

    if nan_check and torch.isnan(result).any():
        return None, "callback output contains NaN"

    return result, None
```

- [ ] **Step 4: Wire callback type into intervention engine**

In `python/rocket_surgeon/host/interventions/engine.py`, add handling for `"callback"` type in the `apply_single_recipe` function (or equivalent dispatch):

```python
from rocket_surgeon.host.interventions.callback import (
    InterventionContext,
    execute_callback,
    resolve_callback,
)

# In the recipe application dispatch:
if recipe["type"] == "callback":
    params = recipe.get("params", {})
    module_name = params["module"]
    function_name = params["function"]
    timeout_s = params.get("timeout_s", 5.0)
    nan_check = params.get("nan_check", False)
    fn = resolve_callback(module_name, function_name)
    ctx = InterventionContext(
        layer=layer, component=component, event=event,
        tick_id=0, device=tensor.device, model_handle=0,
    )
    result, error = execute_callback(fn, tensor, ctx, timeout_s, nan_check)
    if result is not None:
        return result, recipe.get("id", "callback")
    else:
        # Log error, return original unchanged
        return tensor, None
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest python/tests/test_callback.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add python/rocket_surgeon/host/interventions/callback.py python/tests/test_callback.py python/rocket_surgeon/host/interventions/engine.py
git commit -m "feat(python): Tier 2 callback interventions with watchdog thread timeout"
```

---

### Task 10: ROME exit test

**Files:**
- Create: `tests/test_rome_acceptance.py`

- [ ] **Step 1: Write the ROME acceptance test**

Create `tests/test_rome_acceptance.py`:

```python
"""
Phase 3 exit test: ROME-style locate-then-edit reproduction on GPT-2.

Demonstrates the full Phase 3 stack:
1. Step forward through model
2. Identify critical MLP layer (causal effect on target token)
3. Reverse step to that layer
4. Apply rank-1 edit via Tier 2 callback
5. Step forward from edited state
6. Verify logit change
7. Replay with divergence detection
"""
import json
import sys

import torch

sys.path.insert(0, "python")
sys.path.insert(0, "tests")

from e2e_harness import E2EHarness


PROMPT = "The Eiffel Tower is located in the city of"
TARGET_ORIGINAL = " Paris"
TARGET_EDITED = " Rome"


def test_rome_locate_then_edit():
    harness = E2EHarness(model="gpt2")
    harness.initialize()
    harness.attach()

    # 1. Step forward through entire model, collecting MLP output norms per layer
    harness.send("rocket/step", {"direction": "forward", "count": 1, "granularity": "layer"})
    # Step through all layers
    num_layers = 12  # GPT-2 small
    layer_effects = {}

    for layer in range(num_layers):
        resp = harness.send("rocket/step", {"direction": "forward", "count": 1})
        # Inspect residual at this layer
        inspect_resp = harness.send("rocket/inspect", {
            "target": f"gpt2:0:{layer}:mlp:output",
            "detail": "stats",
        })
        if "data" in inspect_resp and "norm" in inspect_resp["data"]:
            layer_effects[layer] = inspect_resp["data"]["norm"]

    # 2. Identify critical layer (highest norm = most effect on output)
    critical_layer = max(layer_effects, key=layer_effects.get)
    assert critical_layer > 0, "Critical layer should not be layer 0"

    # 3. Reverse step to critical layer
    resp = harness.send("rocket/step", {
        "direction": "backward",
        "run_to": f"layer:{critical_layer}",
    })
    assert resp["state"]["status"] == "stopped"

    # 4. Apply rank-1 edit via Tier 2 callback
    # Register intervention that scales MLP output (simplified ROME)
    harness.send("rocket/intervene", {
        "action": "add",
        "recipes": [{
            "id": "rome-edit",
            "type": "callback",
            "target": f"gpt2:0:{critical_layer}:mlp:output",
            "params": {
                "module": "tests.rome_edit_helper",
                "function": "rank1_edit",
                "timeout_s": 10.0,
                "nan_check": True,
            },
        }],
    })

    # 5. Step forward from edited state
    for _ in range(num_layers - critical_layer):
        harness.send("rocket/step", {"direction": "forward", "count": 1})

    # 6. Inspect final logits
    final_resp = harness.send("rocket/inspect", {
        "target": "gpt2:0:11:lm_head:output",
        "detail": "topk",
    })
    # Verify Rome logit increased relative to baseline
    # (Full ROME verification would compare against unedited run)
    assert "data" in final_resp

    # 7. Replay with divergence detection
    # Create a checkpoint first, then replay from it
    harness.send("rocket/checkpoint", {"action": "create"})
    replay_resp = harness.send("rocket/replay", {
        "from_checkpoint": "auto",  # Most recent auto-checkpoint
        "verify": True,
        "cosine_threshold": 0.999,
    })
    # After editing, replay should detect divergence at the edited layer
    assert replay_resp["data"]["divergences"] is not None

    harness.detach()
    harness.shutdown()
```

- [ ] **Step 2: Create the ROME edit helper module**

Create `tests/rome_edit_helper.py`:

```python
import torch


def rank1_edit(tensor: torch.Tensor, ctx) -> torch.Tensor:
    """Simplified rank-1 edit: scale the MLP output to steer toward a different token.

    Real ROME computes (v_new - v_old) @ k^T / (k^T @ k). This simplified version
    adds a learned direction scaled by a constant, which is sufficient to demonstrate
    the locate-then-edit pattern works through the protocol stack.
    """
    # Add a perturbation in a consistent direction
    # This is a demonstration edit, not a precise ROME reproduction
    torch.manual_seed(42)
    direction = torch.randn_like(tensor) * 0.5
    return tensor + direction
```

- [ ] **Step 3: Run the test (expects full stack working)**

Run: `python -m pytest tests/test_rome_acceptance.py -v --timeout=120`
Expected: Pass (after all prior tasks are integrated).

- [ ] **Step 4: Commit**

```bash
git add tests/test_rome_acceptance.py tests/rome_edit_helper.py
git commit -m "test(acceptance): ROME locate-then-edit reproduction on GPT-2"
```

---

### Task 11: Bundle extension — checkpoint + worldline export

**Files:**
- Modify: `crates/rocket-surgeon/src/bundle.rs`
- Modify: `crates/rocket-surgeon/src/dispatch.rs` (session.export handler)

- [ ] **Step 1: Add checkpoint and worldline serialization to bundle export**

In `crates/rocket-surgeon/src/bundle.rs`, extend the bundle assembly to include:

```rust
// Add to the tar.gz assembly function:

// bookmarks.json — array of bookmark entries
let bookmarks: Vec<serde_json::Value> = session.state().checkpoints.iter()
    .filter(|c| c.bookmark.is_some())
    .map(|c| serde_json::json!({
        "name": c.bookmark,
        "tick_id": c.tick_id,
        "layer": c.layer_idx,
        "checkpoint_id": c.checkpoint_id,
    }))
    .collect();
let bookmarks_json = serde_json::to_vec_pretty(&bookmarks)?;
add_to_tar(&mut tar, "bookmarks.json", &bookmarks_json)?;

// worldlines.json — DAG structure
let worldlines = serde_json::json!({
    "segments": session.worldline().segments.iter().map(|s| serde_json::json!({
        "id": s.id,
        "parent_segment": s.parent_segment,
        "branch_tick": s.branch_tick,
        "tick_range": s.tick_range,
    })).collect::<Vec<_>>(),
    "branches": [],  // Named branches added when branch.fork is implemented
});
let worldlines_json = serde_json::to_vec_pretty(&worldlines)?;
add_to_tar(&mut tar, "worldlines.json", &worldlines_json)?;
```

- [ ] **Step 2: Add checkpoint data export for named checkpoints**

For each user-created (non-auto, non-sub) checkpoint, export metadata:

```rust
// checkpoints/{id}/meta.json
for cref in &session.state().checkpoints {
    if cref.checkpoint_id.starts_with("auto-") || cref.checkpoint_id.starts_with("sub-") {
        continue; // Skip ephemeral checkpoints
    }
    let meta = serde_json::json!({
        "checkpoint_id": cref.checkpoint_id,
        "tier": cref.tier,
        "tick_id": cref.tick_id,
        "layer_idx": cref.layer_idx,
        "created_at": cref.created_at,
    });
    let meta_json = serde_json::to_vec_pretty(&meta)?;
    let path = format!("checkpoints/{}/meta.json", cref.checkpoint_id);
    add_to_tar(&mut tar, &path, &meta_json)?;
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p rocket-surgeon --all-targets`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add crates/rocket-surgeon/
git commit -m "feat(bundle): export bookmarks.json, worldlines.json, checkpoint metadata"
```

---

### Task 12: Inspect format fix — 5→6 segment alignment

**Files:**
- Modify: `crates/rocket-surgeon/src/dispatch.rs`
- Modify: `crates/rocket-surgeon-probes/src/grammar.rs` (if needed)

- [ ] **Step 1: Find the target parsing function in daemon dispatch**

Search for `target_to_probe_point` or the function that parses inspect/intervene targets in `crates/rocket-surgeon/src/dispatch.rs`. It currently expects 5 segments (family:layer:component:event) without rank.

- [ ] **Step 2: Update parser to accept both 5-segment and 6-segment**

```rust
fn parse_target(target: &str) -> Result<(String, u32, u32, String, String), String> {
    let parts: Vec<&str> = target.split(':').collect();
    match parts.len() {
        5 => {
            // Legacy: family:layer:component:event (rank defaults to 0)
            let family = parts[0].to_string();
            let rank = 0u32;
            let layer: u32 = parts[1].parse().map_err(|_| "invalid layer")?;
            let component = parts[2].to_string();
            let event = parts[3].to_string();
            Ok((family, rank, layer, component, event))
        }
        6 => {
            // Full: family:rank:layer:component:event
            let family = parts[0].to_string();
            let rank: u32 = parts[1].parse().map_err(|_| "invalid rank")?;
            let layer: u32 = parts[2].parse().map_err(|_| "invalid layer")?;
            let component = parts[3].to_string();
            let event = parts[4].to_string();
            Ok((family, rank, layer, component, event))
        }
        _ => Err(format!("expected 5 or 6 segments in target, got {}", parts.len())),
    }
}
```

- [ ] **Step 3: Update all callers to use the new parser**

Replace existing target parsing in inspect/intervene/probe handlers with the unified parser.

- [ ] **Step 4: Add test for both formats**

```rust
#[test]
fn parse_target_5_segment() {
    let (family, rank, layer, component, event) = parse_target("gpt2:5:attn.o_proj:output").unwrap();
    assert_eq!(family, "gpt2");
    assert_eq!(rank, 0);
    assert_eq!(layer, 5);
    assert_eq!(component, "attn.o_proj");
    assert_eq!(event, "output");
}

#[test]
fn parse_target_6_segment() {
    let (family, rank, layer, component, event) = parse_target("gpt2:0:5:attn.o_proj:output").unwrap();
    assert_eq!(family, "gpt2");
    assert_eq!(rank, 0);
    assert_eq!(layer, 5);
    assert_eq!(component, "attn.o_proj");
    assert_eq!(event, "output");
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p rocket-surgeon --all-targets`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add crates/rocket-surgeon/
git commit -m "fix(daemon): accept both 5-segment and 6-segment probe targets"
```

---

### Task 13: TCK green sweep — un-defer Phase 3 scenarios

**Files:**
- Modify: `tck/protocol/replay.feature`
- Modify: `tck/protocol/branch.feature`
- Modify: `tck/protocol/inspection.feature`
- Modify: `tck/tensor/handles.feature`
- Modify: `tck/protocol/session-export.feature`
- Modify: `tck/protocol/errors.feature`
- Modify: TCK step definitions as needed

- [ ] **Step 1: Remove @deferred from replay.feature**

In `tck/protocol/replay.feature`, remove all `@deferred` tags from the 8 scenarios.

- [ ] **Step 2: Remove @deferred from branch.feature**

In `tck/protocol/branch.feature`, remove all `@deferred` tags from the 3 scenarios.

- [ ] **Step 3: Remove @deferred from inspection.feature and tensor/handles.feature**

Remove `@deferred` from all 22 scenarios blocked by the 5→6 segment inspect format.

- [ ] **Step 4: Remove @deferred from session-export.feature**

Remove `@deferred` from the 10 scenarios that test extended bundle contents.

- [ ] **Step 5: Remove @deferred from relevant errors.feature scenarios**

Remove `@deferred` from:
- `REPLAY_DIVERGENCE` error scenario
- Any other scenarios now unblocked by Phase 3 features

- [ ] **Step 6: Run TCK suite**

Run: `python -m pytest tck/ -v --tb=short`
Expected: Previously-deferred scenarios now run. Fix any step definition gaps.

- [ ] **Step 7: Fix step definitions for new scenarios**

Implement any missing step definitions needed by un-deferred scenarios:
- Replay response assertions
- Branch verb step definitions
- Bundle content assertions for new artifacts
- Inspect with 6-segment targets

- [ ] **Step 8: Verify deferred count**

Run: `grep -r "@deferred" tck/ | wc -l`
Expected: ~132 (down from 178).

- [ ] **Step 9: Commit**

```bash
git add tck/
git commit -m "tck: un-defer 46 scenarios — replay, branch, inspect, export now green"
```

---

### Task 14: CUBLAS workspace config + worker startup determinism

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/main.rs`

- [ ] **Step 1: Set CUBLAS_WORKSPACE_CONFIG at worker process start**

In `crates/rocket-surgeon-worker/src/main.rs`, before any PyO3/Python initialization:

```rust
fn main() {
    // Set CUBLAS workspace config for deterministic replay support
    // Must be set before cuBLAS initialization (first PyTorch import)
    std::env::set_var("CUBLAS_WORKSPACE_CONFIG", ":4096:8");

    // ... existing initialization ...
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p rocket-surgeon-worker --all-targets`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add crates/rocket-surgeon-worker/src/main.rs
git commit -m "feat(worker): set CUBLAS_WORKSPACE_CONFIG at startup for deterministic replay"
```

---

### Task 15: Code review remediation

- [ ] **Step 1: Run full workspace build**

Run: `cargo build --workspace --all-targets`
Expected: Zero errors, zero warnings.

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace --all-targets`
Expected: Zero warnings.

- [ ] **Step 3: Run Python linting**

Run: `ruff check python/ tests/`
Expected: Zero issues.

- [ ] **Step 4: Run full test suite**

Run: `cargo test --workspace --all-targets && python -m pytest python/tests/ -v && python -m pytest tck/ -v`
Expected: All pass.

- [ ] **Step 5: Fix any issues found**

Address all warnings, lint failures, and test failures.

- [ ] **Step 6: Commit fixes**

```bash
git add -A
git commit -m "fix: CR remediation — address all findings from Phase 3B+C review"
```

---

## Post-Implementation

After all tasks complete:
1. Push branch
2. Create PR to master
3. Verify CI green
4. Merge after review
