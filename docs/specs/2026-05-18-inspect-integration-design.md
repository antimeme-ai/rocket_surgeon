# WU 1.11: Inspect Integration — Design Spec

**Status**: Approved
**Depends on**: WU 1.4 (tensor store), WU 1.10 (step integration)
**Blocks**: WU 1.12 (probe events), WU 1.13 (built-in views)

## Goal

Wire `rocket/inspect` end-to-end: client sends an inspect request while stopped at a
tick, daemon routes to the worker, worker returns captured tensor bytes inline via
JSON-RPC, daemon ingests into TensorStore (BLAKE3 dedup, Rust-side stats), and returns
`TensorSummary` + optional slice data in a `ResponseEnvelope<InspectResponse>`.

Near-term target: GPT-2 on CPU. Design must not preclude multi-GPU or large-model
scenarios — those get their own data-plane design later (WU 1.8 shared memory, tiered
eviction/spill).

## Data flow

```
Client                    Daemon                   Orchestrator             Worker (Rust+PyO3)
  │                         │                          │                        │
  │ rocket/inspect          │                          │                        │
  │ {target, detail}        │                          │                        │
  │────────────────────────►│                          │                        │
  │                         │ _host/inspect            │                        │
  │                         │ {model_handle, target,   │                        │
  │                         │  detail, slices}         │                        │
  │                         │─────────────────────────►│                        │
  │                         │                          │ forward to worker      │
  │                         │                          │───────────────────────►│
  │                         │                          │                        │
  │                         │                          │  Python GIL:           │
  │                         │                          │  - match target vs     │
  │                         │                          │    component_map       │
  │                         │                          │  - lookup tensor in    │
  │                         │                          │    last_outputs        │
  │                         │                          │  - tensor_to_bytes()   │
  │                         │                          │  - base64 encode       │
  │                         │                          │                        │
  │                         │                          │◄───────────────────────│
  │                         │                          │  HostInspectResponse   │
  │                         │                          │  {tensors: [...]}      │
  │                         │◄─────────────────────────│                        │
  │                         │                          │                        │
  │                         │  Daemon:                 │                        │
  │                         │  - base64 decode         │                        │
  │                         │  - tensor_store.insert() │                        │
  │                         │  - summarize() / slice() │                        │
  │                         │                          │                        │
  │◄────────────────────────│                          │                        │
  │ InspectResponse         │                          │                        │
  │ {tensors, slice_data}   │                          │                        │
```

## Tensor capture strategy

### Capture at barrier time

The capture hooks already pass the tensor output through the mailbox:
`result_mailbox.put((path, idx, output))` (bridge.py:247). Currently the worker's
step loop reads only `(path, call_index)` and ignores the tensor.

Change: the step loop stashes every barrier-fired tensor into a Python-side
`last_outputs` dict keyed by `(module_path, call_index)`. This dict accumulates
across ticks within a single forward pass. Cleared on detach or new forward pass
start.

### Memory implications

For GPT-2 (~124M params, small activations): negligible.

For large models (Llama-3-8B, 70B): holding one activation per component across
an entire forward pass can consume significant GPU/CPU memory. This is acceptable
for now — the tiered eviction/spill system is a separate design conversation that
must happen before production use on large models. The clearing semantics (detach /
new forward pass) bound the worst case to one full forward pass worth of activations.

### Future: tiered eviction/spill

Not in scope for this WU. The endgame is a tiered system: hot tensors in memory,
warm tensors in a fast backing store, cold tensors retrievable on demand. The inline
JSON-RPC path designed here is the "hot" tier and will be replaced by shared memory
(WU 1.8) for the Python→Rust handoff. The inspect protocol contract stays the same
regardless of backing store.

## Protocol additions

### New internal messages

Add to `crates/rocket-surgeon-protocol/src/messages.rs`:

```rust
// _host/inspect
pub struct HostInspectRequest {
    pub model_handle: u64,
    pub target: String,
    pub detail: InspectDetail,
    pub slices: Option<Vec<[u64; 2]>>,
}

pub struct HostInspectResponse {
    pub tensors: Vec<CapturedTensor>,
}

pub struct CapturedTensor {
    pub module_path: String,
    pub canonical: String,
    pub layer: u32,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub device: String,
    pub data_base64: String,
}
```

### Internal method constant

Add `HOST_INSPECT` to the `internal` module in `messages.rs`.

### Orchestrator routing

The orchestrator already has a `forward_to_worker` handler for `_host/step`,
`_host/configure_hooks`, `_host/update_probes`. Add `_host/inspect` to that
match arm — no new logic needed.

## Worker: handle_host_inspect

The worker receives `HostInspectRequest` and:

1. Validates `model_handle` matches current state.
2. Resolves `target` against `component_map` using the existing `capture.rs`
   probe-matching logic. Supports wildcards.
3. For each matched component, looks up `(module_path, call_index)` in
   `last_outputs`.
4. For each found tensor:
   - Calls `bridge::tensor_to_bytes()` via PyO3 to get raw bytes.
   - Reads shape, dtype, device from the Python tensor object.
   - Base64-encodes the bytes.
   - Builds a `CapturedTensor`.
5. Returns `HostInspectResponse` with all matched tensors.

If target matches valid components but none have captured tensors (haven't stepped
there yet), return an empty tensors list — not an error. The daemon decides whether
that's an error based on context.

### Step loop changes

The step loop in `dispatch.rs::handle_host_step` currently reads a 2-tuple from the
mailbox:

```python
let path: String = tuple.get_item(0)?.extract()?;
let call_index: u32 = tuple.get_item(1)?.extract()?;
```

The mailbox actually puts a 3-tuple: `(path, idx, output)`. Change the step loop to:

1. Extract all three elements.
2. Stash the output tensor into `last_outputs` (a `PyObject` stored on
   `ForwardPassState` or `WorkerState`).
3. Continue with step logic as before.

The `last_outputs` dict is a Python dict held as a `PyObject` on `WorkerState`.
Created fresh on each `ensure_forward_pass`. Cleared on detach.

## Daemon: session.inspect() and dispatch

### Session.inspect()

New method on `Session`:

```rust
pub fn inspect(
    &self,
    req: &InspectRequest,
    tensors: Vec<TensorSummary>,
    slice_data: Option<String>,
) -> Result<ResponseEnvelope<InspectResponse>, SessionError>
```

- Validates state: must be Stopped (otherwise `INVALID_STATE`).
- Validates model attached (otherwise `MODEL_NOT_ATTACHED`).
- Read-only: no state transitions, returns to Stopped.
- Wraps the provided tensors + slice_data in a `ResponseEnvelope`.

### Daemon dispatch: handle_inspect

New handler in `crates/rocket-surgeon/src/dispatch.rs`:

1. Parse `InspectRequest` params.
2. If orchestrator available: build `HostInspectRequest`, call
   `orchestrator.inspect()`, receive `HostInspectResponse`.
3. For each `CapturedTensor` in the response:
   - Base64-decode `data_base64` to bytes.
   - `tensor_store.insert(bytes, shape, dtype, device)` → `TensorHandle`.
   - `tensor_store.summarize(tensor_id)` → `TensorSummary`.
4. If `detail == Slice` and slices provided:
   - `tensor_store.slice(tensor_id, offset, len)` for the first tensor.
   - Base64-encode the result into `slice_data`.
5. If no tensors returned and target was valid: `TENSOR_NOT_FOUND`.
6. If target matches no components: `INVALID_TARGET`.
7. Call `session.inspect()` with the assembled data.

### OrchestratorHandle.inspect()

New method on `OrchestratorHandle`, same pattern as `step()`:

```rust
pub fn inspect(&mut self, req: &HostInspectRequest) -> anyhow::Result<HostInspectResponse>
```

### Daemon main loop

Same pattern as step: intercept `rocket/inspect` in the main loop, route through
orchestrator, pass result to `dispatch::handle_inspect`.

### TensorStore integration

The existing `TensorStore` (WU 1.4) handles:

- BLAKE3 content-addressable dedup on insert.
- LRU eviction by entry count and byte budget.
- Lazy summary computation (cached after first call).
- Byte-range slicing with bounds checking.

The daemon holds a `TensorStore` instance alongside the `Session`. The inspect
handler feeds tensor bytes into it and reads summaries back out. The store persists
across inspect calls within a session — repeated inspection of the same tensor
returns the cached summary with the same `tensor_id`.

## Error cases

| Condition | Error code | Severity |
|-----------|-----------|----------|
| Not in Stopped state | `INVALID_STATE` | recoverable |
| No model attached | `MODEL_NOT_ATTACHED` | recoverable |
| Target matches no components | `INVALID_TARGET` | recoverable |
| Target valid but no tensor captured | `TENSOR_NOT_FOUND` | recoverable |
| Slice indices out of bounds | `SLICE_OUT_OF_BOUNDS` | recoverable |
| Slice response exceeds 64 KB cap | `RESPONSE_TOO_LARGE` | recoverable |

## TCK coverage

The existing `tck/protocol/inspection.feature` has 10 scenarios. This WU should
make the following pass (removing xfail):

- Inspect with detail=summary returns TensorSummary
- Inspect defaults to summary when detail is omitted
- Inspect with target matching a component returns tensor for that component
- Inspect with wildcard target returns multiple tensors
- Inspect nonexistent target returns INVALID_TARGET error
- Inspect with detail=slice and valid slices returns slice_data
- Inspect with slice out of bounds returns SLICE_OUT_OF_BOUNDS error
- TensorSummary includes tensor_id as BLAKE3 hash (64 hex chars)
- Inspect response includes full SessionState in envelope

Deferred to WU 1.13:
- Inspect with built-in view "residual_stream_norm" returns view_result

Deferred to future:
- Same tensor content at two probe points yields same tensor_id
  (requires specific test harness setup — dedup logic is tested at the
  TensorStore unit level)

## Out of scope

- Built-in views (residual_stream_norm, attention_pattern) — WU 1.13
- Shared memory data plane — WU 1.8 (separate major design effort)
- Tiered eviction/spill — future design conversation
- DTensor/sharding-aware inspection — Phase 5 (multi-GPU)
- `rocket/tensor.slice` as a separate verb
- `rocket/tensor.gather` (all-gather for distributed tensors)
