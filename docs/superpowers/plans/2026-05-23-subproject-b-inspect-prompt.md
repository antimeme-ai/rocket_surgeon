# Sub-project B: Inspect Pipeline + Prompt Input — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable end-to-end logit capture: set input tokens via `rocket/step`, run a real forward pass, inspect `lm_head` output with raw tensor data.

**Architecture:** `StepRequest` gains `tokens: Option<Vec<u64>>`, forwarded through `HostStepRequest.input_ids` to the worker, which passes them to `bridge::run_forward` instead of creating dummy zeros. The daemon catalog gains `lm_head` at layer 0 so it appears in discover and is inspectable.

**Tech Stack:** Rust (protocol, daemon, worker), Python (bridge), PyO3, serde

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/rocket-surgeon-protocol/src/messages.rs` | Add `tokens` to `StepRequest`, `input_ids` to `HostStepRequest` |
| `crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs` | Update roundtrip tests for new fields |
| `crates/rocket-surgeon-worker/src/bridge.rs` | `run_forward` accepts `Option<&[u64]>` instead of hardcoding zeros |
| `crates/rocket-surgeon-worker/src/dispatch.rs` | `ensure_forward_pass` passes `input_ids` to bridge |
| `crates/rocket-surgeon/src/main.rs` | Forward `tokens` from `StepRequest` to `HostStepRequest.input_ids` |
| `crates/rocket-surgeon/src/session.rs` | Add `lm_head` to `default_catalog` |

---

### Task 1: Protocol — add tokens and input_ids fields

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/messages.rs:148-158` (StepRequest)
- Modify: `crates/rocket-surgeon-protocol/src/messages.rs:847-858` (HostStepRequest)

- [ ] **Step 1: Add `tokens` to StepRequest**

In `crates/rocket-surgeon-protocol/src/messages.rs`, find `StepRequest` (line 148). Add `tokens` field after `run_to`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRequest {
    pub direction: StepDirection,
    pub count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granularity: Option<TickGranularity>,
    #[serde(default)]
    pub envelope: EnvelopeMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_to: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u64>>,
}
```

- [ ] **Step 2: Add `input_ids` to HostStepRequest**

Find `HostStepRequest` (line 847). Add `input_ids` field after `interventions`:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostStepRequest {
    pub model_handle: u64,
    pub count: u32,
    #[serde(default)]
    pub direction: StepDirection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granularity: Option<TickGranularity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_events: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interventions: Vec<InterventionRecipe>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_ids: Option<Vec<u64>>,
}
```

- [ ] **Step 3: Fix all compile errors from new field**

Every place that constructs `StepRequest` or `HostStepRequest` needs the new field. Known sites:

In `crates/rocket-surgeon/src/main.rs:149` (default StepRequest):
```rust
    run_to: None,
    tokens: None,
```

In `crates/rocket-surgeon/src/main.rs:163` (HostStepRequest construction):
```rust
    interventions: interventions.to_vec(),
    input_ids: None,
```

In `crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs:1170` (HostStepRequest roundtrip):
```rust
    interventions: vec![],
    input_ids: None,
```

In `crates/rocket-surgeon/src/orchestrator_handle.rs:264-276` (step method test):
```rust
    interventions: vec![],
    input_ids: None,
```

Search for any other construction sites: `grep -rn "HostStepRequest {" --include='*.rs'` and `grep -rn "StepRequest {" --include='*.rs'`

- [ ] **Step 4: Verify it compiles**

Run: `cargo check --workspace`
Expected: Compiles cleanly

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(protocol): add tokens to StepRequest, input_ids to HostStepRequest"
```

---

### Task 2: Protocol roundtrip tests

**Files:**
- Modify: `crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs`

- [ ] **Step 1: Add roundtrip test for tokens field**

After the existing `host_step_request_round_trip` test (line ~1178), add:

```rust
#[test]
fn host_step_request_with_input_ids_round_trip() {
    let req = HostStepRequest {
        model_handle: 1,
        count: 1,
        direction: StepDirection::Forward,
        granularity: None,
        max_events: None,
        interventions: vec![],
        input_ids: Some(vec![50256, 464, 3797, 318]),
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: HostStepRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(req, parsed);
    assert!(json.contains("input_ids"));
}

#[test]
fn host_step_request_without_input_ids_omits_field() {
    let req = HostStepRequest {
        model_handle: 1,
        count: 1,
        direction: StepDirection::Forward,
        granularity: None,
        max_events: None,
        interventions: vec![],
        input_ids: None,
    };
    let json = serde_json::to_string(&req).unwrap();
    assert!(!json.contains("input_ids"));
    let parsed: HostStepRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.input_ids, None);
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p rocket-surgeon-protocol --quiet`
Expected: All pass including new tests

- [ ] **Step 3: Commit**

```bash
git add crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs
git commit -m "test(protocol): roundtrip tests for input_ids on HostStepRequest"
```

---

### Task 3: Rust bridge — accept input_ids parameter

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/bridge.rs:332-345`

- [ ] **Step 1: Change run_forward signature**

In `crates/rocket-surgeon-worker/src/bridge.rs`, replace the `run_forward` function (lines 332-345):

```rust
pub fn run_forward(
    py: Python<'_>,
    handle: u64,
    input_ids: Option<&[u64]>,
    done_callback: &Bound<'_, pyo3::PyAny>,
) -> anyhow::Result<()> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let torch = py.import("torch")?;

    let py_input = match input_ids {
        Some(ids) => {
            let py_list = PyList::new(py, ids.iter().map(|&id| id as i64))?;
            let tensor = torch.getattr("tensor")?.call1((vec![py_list],))?;
            tensor.call_method1("to", (torch.getattr("long")?,))?
        }
        None => {
            let zeros = torch.getattr("zeros")?.call1(((1, 2),))?;
            zeros.call_method1("to", (torch.getattr("long")?,))?
        }
    };

    bridge
        .getattr("run_forward")?
        .call1((handle, py_input, done_callback))?;
    Ok(())
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p rocket-surgeon-worker`
Expected: Compile error — call site in dispatch.rs needs updating (Task 4 fixes this)

- [ ] **Step 3: Commit**

```bash
git add crates/rocket-surgeon-worker/src/bridge.rs
git commit -m "feat(worker/bridge): run_forward accepts input_ids parameter"
```

---

### Task 4: Worker dispatch — pass input_ids through

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/dispatch.rs:395-457`

- [ ] **Step 1: Update ensure_forward_pass signature**

Change `ensure_forward_pass` (line 395) to accept `input_ids`:

```rust
fn ensure_forward_pass(
    py: Python<'_>,
    state: &mut WorkerState,
    handle: u64,
    input_ids: Option<&[u64]>,
) -> anyhow::Result<()> {
```

- [ ] **Step 2: Pass input_ids to bridge::run_forward**

Change line 445 from:

```rust
    bridge::run_forward(py, handle, done_callback.as_any())?;
```

to:

```rust
    bridge::run_forward(py, handle, input_ids, done_callback.as_any())?;
```

- [ ] **Step 3: Update call site in run_step_loop**

In `run_step_loop` (line 567), change:

```rust
    ensure_forward_pass(py, state, handle)?;
```

to:

```rust
    ensure_forward_pass(py, state, handle, req.input_ids.as_deref())?;
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p rocket-surgeon-worker`
Expected: Compiles cleanly

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon-worker/src/dispatch.rs
git commit -m "feat(worker): pass input_ids from step request to run_forward"
```

---

### Task 5: Daemon — forward tokens to input_ids

**Files:**
- Modify: `crates/rocket-surgeon/src/main.rs:163-170`

- [ ] **Step 1: Forward tokens from StepRequest to HostStepRequest**

In `try_orchestrator_step` (line 163), change the `HostStepRequest` construction:

```rust
    let host_req = rocket_surgeon_protocol::messages::HostStepRequest {
        model_handle: mh,
        count: step_req.count,
        direction: step_req.direction,
        granularity,
        max_events: None,
        interventions: interventions.to_vec(),
        input_ids: step_req.tokens,
    };
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p rocket-surgeon`
Expected: Compiles cleanly

- [ ] **Step 3: Run tests**

Run: `cargo test -p rocket-surgeon --quiet`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add crates/rocket-surgeon/src/main.rs
git commit -m "feat(daemon): forward step tokens to orchestrator as input_ids"
```

---

### Task 6: Session catalog — add lm_head

**Files:**
- Modify: `crates/rocket-surgeon/src/session.rs:169-214`

- [ ] **Step 1: Add lm_head to default_catalog**

In `default_catalog` (line 169), after the `for layer in 0..num_layers` loop that builds per-layer entries, add a single `lm_head` entry at layer 0 (matching adapter assignment — `lm_head` module path has no numeric segment, so `extract_layer_index` returns None, defaulting to 0):

After the closing `}` of the for loop (around line 211), before `catalog`:

```rust
    catalog.push(ProbePointEntry {
        family: family.to_owned(),
        layer: 0,
        canonical: "lm_head".to_owned(),
        event: "output".to_owned(),
        tensor_shape: vec![1, hidden],
        aliases: vec!["lm_head".to_owned()],
    });
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p rocket-surgeon`
Expected: Compiles cleanly

- [ ] **Step 3: Run tests**

Run: `cargo test -p rocket-surgeon --quiet`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add crates/rocket-surgeon/src/session.rs
git commit -m "feat(daemon): add lm_head to default discover catalog"
```

---

### Task 7: Full verification

- [ ] **Step 1: Run pre-commit checks**

Run: `cargo clippy --workspace -- -D warnings && ruff check && mypy python/rocket_surgeon/bridge.py`
Expected: All clean

- [ ] **Step 2: Run full test suite**

Run: `cargo test --workspace --quiet`
Expected: All pass

- [ ] **Step 3: Push**

```bash
git push -u origin HEAD
```
