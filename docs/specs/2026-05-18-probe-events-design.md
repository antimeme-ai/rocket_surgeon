# WU 1.12: Probe Events — Design Spec

## Goal

Wire the probe lifecycle end-to-end: client defines/enables probes via `rocket/probe`, daemon manages `ProbeRegistry`, propagates active probes to worker, step loop evaluates probes at each tick and produces `ProbeFiredEvent`s. Reconcile `capture.rs` ad-hoc matching with grammar-based `ProbePoint::matches`.

## Dependencies

- WU 1.10 (step integration): barrier-driven stepping, tick state — done
- WU 1.11 (inspect integration): tensor serialization, TensorStore ingest — done
- WU 1.14 (subscribe/events): NOT in scope — events are produced, not delivered

## TCK Contract

12 scenarios in `tck/protocol/probes.feature`:

1. Define probe with capture action returns probe_id
2. Define probe with six-level hierarchical point pattern
3. List probes returns all defined probes
4. Enable probe by ID
5. Disable probe by ID
6. Remove probe by ID
7. Enable nonexistent probe returns PROBE_NOT_FOUND
8. Wildcard probe matches all points
9. Two probes at same point fire in priority order
10. Probe with assert action pauses execution on predicate violation
11. set_granularity changes tick granularity for matching layers
12. Probe at same point as intervention both execute

---

## 1. Grammar Extension (5 → 6 segments)

### Format

```
model:rank:layer:component:call_index:event
```

| Position | Name | Type | Examples |
|----------|------|------|----------|
| 0 | model | NameOrWild | `llama`, `mixtral`, `*` |
| 1 | rank | NumOrWild | `0`, `3`, `*` |
| 2 | layer | NumOrWild | `12`, `*` |
| 3 | component | ComponentOrWild | `attn.o_proj`, `experts[3].gate_proj`, `*` |
| 4 | call_index | NumOrWild | `0`, `*` |
| 5 | event | NameOrWild | `output`, `input`, `fwd`, `*` |

Ordering: coarse-to-fine physical topology. An LLM writes `llama:*:*:attn.o_proj:*:output` — reads as "llama, any rank, any layer, attention output projection, any instance, output event."

### Changes

- `ProbePoint` struct: add `call_index: NumOrWild` field between `component` and `event`
- Parser (`grammar.rs`): insert `call_index` parsing between component and event
- `Display` impl: include call_index segment
- `matches()`: add `num_matches(&self.call_index, &other.call_index)`
- All grammar tests: 5-segment → 6-segment
- TCK `probes.feature`: update all probe point patterns to 6-segment
- Adapter (`adapter.rs`): already emits 6-segment — no change
- `capture.rs`: delete ad-hoc `probe_matches` function, replace callers with grammar-based matching

### Validation

LLM-ergonomics review: APPROVED. Each segment has exactly one semantic meaning. No inner-grammar in component field. Coarse-to-fine ordering supports natural partial-wildcard patterns.

---

## 2. Daemon-side Probe Management

### Architecture

The daemon owns the `ProbeRegistry`. All `rocket/probe` CRUD requests are handled daemon-side without worker RPC for the mutation itself.

### Request Flow

```
Client → rocket/probe {action: "define", probe: {...}}
  → Daemon: validate point via grammar parse
  → Daemon: ProbeRegistry.define(probe)
  → Daemon: propagate active probes to worker via _host/update_probes
  → Client ← ProbeResponse {probes: [...], probe_id: "p-cap-1"}
```

### CRUD Operations

| Action | Handler | Worker RPC? |
|--------|---------|-------------|
| `define` | grammar parse → registry.define → propagate | Yes (update_probes) |
| `list` | registry.list (sorted by priority, insertion) | No |
| `enable` | registry.enable → propagate | Yes (update_probes) |
| `disable` | registry.disable → propagate | Yes (update_probes) |
| `remove` | registry.remove → propagate | Yes (update_probes) |
| `set_granularity` | store scopes on session state | No |

### Propagation Protocol

After any mutation, daemon sends `_host/update_probes` with full enabled `ProbeDefinition`s (not just IDs). The worker needs definitions to evaluate actions and configs at step time.

```rust
// Updated:
pub struct HostUpdateProbesRequest {
    pub model_handle: u64,
    pub active_probes: Vec<ProbeDefinition>,  // was Vec<String>
}
```

### Error Mapping

| Registry error | error_code | severity |
|---|---|---|
| `DuplicateId` | `DUPLICATE_PROBE_ID` | `recoverable` |
| `NotFound` | `PROBE_NOT_FOUND` | `recoverable` |
| `InvalidPoint` | `INVALID_POINT` | `recoverable` |

All `recoverable` — the LLM can fix the request and retry.

### Session State

`SessionState.active_probes` updated to reflect enabled probe IDs after each mutation.

---

## 3. Worker-side Probe Evaluation

### Worker State

```rust
// New field on WorkerState:
active_probes: Vec<(ProbeDefinition, ProbePoint)>,
// definition + pre-parsed point for matching without re-parsing each tick
```

Updated by `handle_host_update_probes` — receives full definitions, parses each point once.

### Step Loop Integration

In `handle_host_step`, after each tick and before `resume_mb.put()`:

1. Construct a `ProbePoint` from current tick data (model family, rank, layer, canonical, call_index, event)
2. For each active probe: check `probe.parsed_point.matches(&current_point)`
3. For matching probes (priority order): execute action
4. Collect `ProbeFiredEvent`s
5. If any assert action fails predicate → stop stepping (breakpoint)

### Action Execution

| Action | Behavior |
|--------|----------|
| `capture` | Grab tensor from last_outputs → compute summary if config.summary → produce ProbeFiredEvent with tensor_summary |
| `assert` | Same as capture + evaluate predicate → if violated, produce event + stop stepping |
| `trace` | Produce ProbeFiredEvent with no tensor data (lightweight marker) |
| `checkpoint` | Stub: produce event only |
| `aggregate` | Stub: produce event only |
| `intervene` | Stub: produce event only |

### HostStepResponse Change

```rust
// Current:
pub struct HostStepResponse {
    pub position: TickPosition,
    pub capture: Option<TensorSummary>,  // singular
    pub forward_complete: bool,
}

// New:
pub struct HostStepResponse {
    pub position: TickPosition,
    pub events: Vec<ProbeFiredEvent>,    // all fired events
    pub forward_complete: bool,
    pub events_truncated: bool,          // pressure valve
}
```

### HostStepRequest Addition

```rust
// New optional field:
pub max_events: Option<u32>,  // default 256
```

When event count exceeds limit, stop collecting and set `events_truncated: true`. Protects LLM context windows when broad wildcard probes fire on every component.

### Assert Predicate Engine

Grammar: `<field> <op> <literal>`

- **Fields:** `mean`, `std`, `min`, `max`, `abs_max`, `sparsity`, `l2_norm`, `norm` (alias for `l2_norm`)
- **Operators:** `<`, `>`, `<=`, `>=`, `==`, `!=`
- **Literal:** floating-point number

Intentionally minimal. Complex predicates belong in the LLM's own logic.

Parser: winnow, same pattern as probe point grammar. Lives in `rocket-surgeon-probes` crate.

### Composition Ordering

When probes and interventions target the same point:
- Non-mutating (capture, trace, assert) execute BEFORE mutating (intervene)
- Within non-mutating: priority order (lower first), then insertion order
- Probes observe the original tensor before interventions modify it

---

## 4. Granularity Control

### Mechanism

`set_granularity` stores per-layer granularity overrides on daemon session state.

```json
{
  "action": "set_granularity",
  "scopes": [
    {"match": "layers[12]", "granularity": "component"},
    {"match": "layers[*]", "granularity": "layer"}
  ]
}
```

Scopes evaluated in order — first match wins.

### Resolution

When daemon builds `HostStepRequest`:
1. If step request has explicit `granularity` → use it (per-request override)
2. Else walk stored scopes, first match for current layer position
3. Else default to `Component`

### Match Pattern

- `layers[N]` — specific layer index
- `layers[*]` — all layers

Intentionally simpler than probe point grammar. Could expand to rank-specific scoping later.

### No Worker Propagation

Granularity scopes live on daemon. The daemon resolves granularity before sending `_host/step`. Worker just receives the resolved `granularity` field in `HostStepRequest`.

---

## 5. Files Changed

### Protocol crate (`rocket-surgeon-protocol`)
- `messages.rs`: HostUpdateProbesRequest (Vec<String> → Vec<ProbeDefinition>), HostStepResponse (capture → events), HostStepRequest (add max_events)
- `errors.rs`: add DUPLICATE_PROBE_ID, INVALID_POINT error codes
- `types.rs`: no changes (ProbeDefinition, ProbeAction, ProbeConfig already defined)

### Probes crate (`rocket-surgeon-probes`)
- `grammar.rs`: 5→6 segment, add call_index field + parsing + matching + display
- `registry.rs`: no changes (already complete)
- New: `assertion.rs` — predicate parser and evaluator

### Worker crate (`rocket-surgeon-worker`)
- `dispatch.rs`: implement `handle_host_update_probes` (store parsed probes), enhance `handle_host_step` (evaluate probes at each tick)
- `capture.rs`: delete `probe_matches` ad-hoc function, replace `handle_host_inspect` caller with grammar-based matching
- `adapter.rs`: no changes (already 6-segment)

### Daemon crate (`rocket-surgeon`)
- `dispatch.rs`: implement `handle_probe` (CRUD + propagation), unwire from stub group
- `main.rs`: add ProbeRegistry to session state, add granularity scopes, implement propagation helper

### TCK
- `tck/protocol/probes.feature`: update all probe points to 6-segment format

### E2E test
- `tests/test_e2e_probes.py`: new E2E exercising define → step → probe fires → events returned
