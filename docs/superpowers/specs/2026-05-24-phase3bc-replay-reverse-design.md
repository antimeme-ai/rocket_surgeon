# Phase 3B+C Design Spec: Forward Replay, Reverse Step, Tier 2 Callbacks

**Date:** 2026-05-24
**Branch:** TBD (will be created from master)
**Depends on:** Phase 3A (checkpoint state tier, merged PR #42)
**Exit criteria:** Reverse-step works. Replay divergence detected. ROME reproduction passes. TCK deferred reduced from 178 to ~130.

---

## Scope

Two sub-projects completing Phase 3:

- **Sub-project B:** Forward replay engine, divergence detection, reverse step,
  determinism enforcement, ROME exit test
- **Sub-project C:** Tier 2 Python callbacks, bundle extension, inspect format
  fix, TCK green sweep

---

## Sub-project B: Forward Replay + Reverse Step

### B1: Basic Replay (Worker Re-execution)

The worker receives `_host/replay` and re-executes the forward pass from a
checkpoint. This is REAL re-execution, not metadata synthesis.

**Flow:**

1. Restore CPU + CUDA RNG state from checkpoint sentinel slot
2. If `deterministic: true`: set `torch.use_deterministic_algorithms(True)`
3. Restore activations at checkpoint layer into model's `last_outputs`
4. Enter step loop in REPLAY CONTEXT mode:
   - Mailbox barriers auto-release (no daemon round-trip)
   - Auto-checkpoint behavior: suppressed unless sub-checkpoint policy triggers
   - Probes fire normally (re-fire during replay)
   - Events emit with `replay_of` set on every tick
5. If `stop_at` specified: halt replay at that position
6. If `interventions` present: apply at matching hook points
7. Return: `ticks_replayed`, `stopped_at` (fresh tick_id, replay_of), `divergences`, `verified`

**Protocol types:**

The client-facing `ReplayRequest` (already exists in messages.rs) gains new optional fields:
`deterministic`, `cosine_threshold`, `mre_threshold`. The daemon translates this into the
internal `_host/replay` wire type for the worker:

```rust
// New internal type (daemon → orchestrator → worker):
pub struct HostReplayRequest {
    pub model_handle: u64,
    pub checkpoint_id: String,
    pub stop_at: Option<ReplayStopAt>,
    pub interventions: Vec<InterventionRecipe>,
    pub verify: bool,
    pub deterministic: bool,
    pub cosine_threshold: f64,  // daemon applies default 0.999 if client omits
    pub mre_threshold: f64,     // daemon applies default 0.05 if client omits
}
```

The daemon owns the defaults. The worker receives resolved values (no Option).

**Replay context mode (step loop changes):**

The step loop gains a `ReplayContext` struct:

```rust
pub struct ReplayContext {
    pub active: bool,
    pub original_tick_ids: Vec<u64>,    // tick_ids being replayed
    pub verify: bool,
    pub thresholds: DivergenceThresholds,
    pub interventions: Vec<InterventionRecipe>,
    pub collected_divergences: Vec<Divergence>,
}
```

When `context.active`, the step loop:
- Auto-releases mailbox barriers with pre-loaded interventions
- Tags all produced ticks with `replay_of`
- Triggers verification comparison at √L boundaries (if `verify`)
- Does NOT trigger normal auto-checkpoint (but sub-checkpoint policy may fire)

### B2: Divergence Detection

At each √L boundary during verified replay:

1. Read the stored activation from the arena (checkpoint from the ORIGINAL pass)
2. Read the freshly-computed activation (from the replay pass)
3. Compute:
   - `cosine_similarity`: `dot(a, b) / (norm(a) * norm(b))`
   - `max_relative_error`: `max(|a - b| / max(|a|, epsilon))`
4. If `cosine_sim < cosine_threshold` OR `mre > mre_threshold`:
   - Create `Divergence { tick_id, original_tick_id, probe_point, cosine_similarity, max_relative_error, message }`
   - Emit `rocket/replay.divergence` notification to subscribers
5. Collect all divergences in response

**Tolerance defaults:** cosine 0.999 (3-nines), MRE 0.05 (5%). Configurable per-request.

**Comparison lives in Python bridge** (needs torch for efficient tensor math):

```python
def compare_activations(
    original_ptr: int, original_len: int, original_dtype: str, original_shape: list,
    replayed: torch.Tensor,
    cosine_threshold: float,
    mre_threshold: float,
) -> Optional[dict]:
    """Returns divergence dict if thresholds exceeded, None if within tolerance."""
```

### B3: Interventions During Replay

Replay applies interventions at hook points using the same `apply_interventions_at_point`
path as normal stepping. The interventions are pre-loaded from the replay request
(not from the daemon session registry, since this is a what-if exploration).

This enables the core use case: "what would have happened if I'd applied this
intervention during the original pass?"

### B4: Reverse Step

**User-facing API:** `rocket/step` with `direction: "backward"` and optional `run_to`.

**Implementation in daemon dispatch:**

```
handle_step(direction=backward, run_to=None):
    target_tick = current_tick - 1
    nearest_checkpoint = find_checkpoint_before(target_tick)
    
    # Sub-checkpoint policy
    if arena_utilization() < 0.6:
        # Eager: checkpoint current position for O(1) next backward step
        create_sub_checkpoint(current_position)
    
    # Replay from nearest checkpoint to target
    replay_request = HostReplayRequest {
        checkpoint_id: nearest_checkpoint.id,
        stop_at: tick_position_for(target_tick),
        verify: false,  // internal backward step, no verification needed
        ...
    }
    response = orchestrator.replay(replay_request)
    
    # Update session state
    session.position = response.stopped_at
    session.current_segment += 1  // new worldline segment
    
    return StepResponse { stopped_at: response.stopped_at, ... }
```

**With `run_to`:**

```
handle_step(direction=backward, run_to={layer: 5, component: "mlp"}):
    target = resolve_backward_target(run_to, current_position)
    nearest_checkpoint = find_checkpoint_before(target)
    # Same replay logic, just different stop_at
```

**Finding the nearest checkpoint:**

Search order (newest first):
1. Sub-checkpoints (created by eager backward-step policy)
2. Auto-checkpoints (created at √L boundaries during forward pass)
3. User checkpoints (explicit `rocket/checkpoint create`)
4. Spilled checkpoints (load from NVMe if needed — prefetch at replay start)

### B5: Performance (Sub-checkpoint Strategy)

**Dual strategy based on arena pressure:**

| Arena utilization | Strategy | Backward step cost |
|-------------------|----------|-------------------|
| < 60% | Eager sub-checkpoint | O(1) amortized — each backward step checkpoints current position |
| 60-80% | √L replay-internal | O(√L) — sub-checkpoints created within replay region only |
| > 80% | Spill + √L | O(√L) + spill oldest auto-checkpoint to NVMe |

**Sub-checkpoint naming:** `sub-{segment_id}-{tick_id}` — distinct from auto-checkpoints
(`auto-{uuid}`) and user checkpoints (user-chosen names).

**Eviction priority (lowest to highest):**
1. Sub-checkpoints from non-current worldline segments
2. Auto-checkpoints from early in the pass (far from current position)
3. Auto-checkpoints near current position
4. User checkpoints (never evicted, only spilled)

**Prefetch at replay start:**
When the worker's `handle_host_replay` receives the request, before entering the
step loop: scan checkpoint slots for the target range, identify any that are spilled
to NVMe, issue `readahead()` syscall for those files. By the time the step loop
reaches those layers for verification comparison, data is in page cache.

### B6: ROME Exit Test

End-to-end acceptance test against GPT-2-small:

1. **Attach + Step forward:** Run complete forward pass on prompt
   "The Eiffel Tower is located in the city of"
2. **Locate critical layer:** At each layer, inspect residual stream norm.
   Identify the MLP layer with highest causal effect on "Paris" logit
   (simplified: compare logits with and without that layer's MLP contribution).
3. **Reverse step:** `step(direction: backward, run_to: {layer: critical_layer})`
4. **Apply rank-1 edit:** Tier 2 callback that adds a rank-1 matrix
   `(v_new - v_old) @ k^T / (k^T @ k)` to the MLP output, steering toward "Rome"
5. **Step forward:** Continue from edited state to final layer
6. **Assert:** Top-1 logit changed from "Paris" to "Rome" (or at minimum:
   "Rome" logit increased significantly relative to baseline)
7. **Verify divergence:** Replay original path with `verify: true`.
   Assert divergence detected at the edited layer.

---

## Sub-project C: Tier 2 Callbacks + Bundle + TCK Sweep

### C1: Tier 2 Python Callbacks

**Recipe type:** `"callback"` added alongside existing `ablate/scale/patch/add/clamp`.

```json
{
  "id": "my-rank1-edit",
  "type": "callback",
  "target": "llama:0:5:mlp.down_proj:output",
  "params": {
    "module": "my_interventions",
    "function": "rank1_edit",
    "timeout_s": 10.0,
    "nan_check": true
  }
}
```

**Contract:**

```python
def rank1_edit(tensor: torch.Tensor, ctx: InterventionContext) -> torch.Tensor:
    """
    Args:
        tensor: The activation tensor at the intervention point (on device)
        ctx: InterventionContext with fields:
            - layer: int
            - component: str
            - event: str ("input" | "output")
            - tick_id: int
            - device: torch.device
            - model_handle: int
            - tensor_store: TensorStoreAccessor (read-only)
    Returns:
        Modified tensor (same shape, same device)
    Raises:
        Any exception → intervention marked failed, original preserved
    """
```

**Execution model:**

1. **Module resolution:** `importlib.import_module(params["module"])` → cache after first resolve
2. **Function resolution:** `getattr(module, params["function"])` → cache
3. **Watchdog thread:** Daemon thread with `threading.Timer(timeout_s, fire_timeout)`
4. **Call:** `result = fn(tensor.clone(), ctx)` — clone so original is always available
5. **On return:**
   - Validate `result.shape == tensor.shape`
   - Validate `result.device == tensor.device`
   - If `nan_check`: `assert not torch.isnan(result).any()`
   - If all pass: return result as the new activation
6. **On exception:** Log, report error in response `fired_interventions`, return original tensor
7. **On timeout:**
   - `ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, TimeoutError)` — soft interrupt
   - If still blocked after `2 * timeout_s`: set `worker_must_die` flag → clean process exit
   - Daemon detects worker death, reports to client

**Why no signal.alarm:** CUDA driver is not async-signal-safe. Signal delivery during
kernel launch corrupts driver state. PyO3 longjmp skips Rust destructors (UB).
Watchdog thread + PyThreadState_SetAsyncExc is the correct mechanism.

**Why no subprocess/fork:** The callback needs access to the model, other tensors,
custom libraries — all in the worker process. Fork + IPC for large tensors is
absurd overhead. This is a development tool, not a multi-tenant service.

### C2: Bundle Extension

Session bundles (`.rsb` tar.gz) gain new artifacts:

| Artifact | Content | When included |
|----------|---------|---------------|
| `checkpoints/{id}/meta.json` | `{tier, tick_position, created_at, layers}` | Per named/bookmarked checkpoint |
| `checkpoints/{id}/slots.bin` | Raw slot data (spill format: header + index + data with CRC32) | Per named/bookmarked checkpoint |
| `bookmarks.json` | `[{name, tick_id, layer, component, annotation}]` | Always (may be empty array) |
| `worldlines.json` | `{segments: [{id, parent_segment, branch_tick, ticks}], branches: [{id, segment, name}]}` | Always |

**Policy:** Only NAMED checkpoints exported (user-created + bookmarked). Auto-checkpoints
and sub-checkpoints are ephemeral — not in bundles. Keeps size bounded.

**worldlines.json schema:**

```json
{
  "segments": [
    {
      "id": 0,
      "parent_segment": null,
      "branch_tick": null,
      "tick_range": [1, 100]
    },
    {
      "id": 1,
      "parent_segment": 0,
      "branch_tick": 50,
      "tick_range": [101, 115]
    }
  ],
  "branches": [
    {
      "id": "branch-rome-edit",
      "segment_id": 1,
      "name": "rome-edit-layer-17",
      "created_at": "2026-05-24T12:00:00Z"
    }
  ]
}
```

### C3: Inspect Format Fix (5→6 Segment Alignment)

**Problem:** Daemon's `target_to_probe_point()` parses targets as 5-segment
(`family:layer:component:event` without rank). Worker's ProbePoint parser expects
6-segment (`family:rank:layer:component:event`). 22 TCK scenarios blocked.

**Fix:**
- Daemon resolver accepts BOTH 5-segment (backward compat, rank defaults to 0)
  and 6-segment (full form)
- Internal representation always 6-segment
- Worker already correct (6-segment)
- Surgical change in `crates/rocket-surgeon/src/dispatch.rs` target parsing

**Files affected:**
- `crates/rocket-surgeon/src/dispatch.rs` — target resolver
- Possibly `crates/rocket-surgeon-probes/src/grammar.rs` — if the grammar needs rank slot

### C4: TCK Green Sweep

**Scenarios to un-defer after B+C implementation:**

| Feature file | Count | Blocker removed by |
|---|---|---|
| `replay.feature` | 8 | B1-B4 (all replay scenarios) |
| `branch.feature` | 3 | B4 worldline model + branch verbs |
| `session-export.feature` | 10 | C2 bundle extension |
| `inspection.feature` | 11 | C3 inspect format fix |
| `tensor/handles.feature` | 11 | C3 inspect format fix |
| `errors.feature` (partial) | 3-4 | Various (REPLAY_DIVERGENCE, etc.) |
| **Total** | **~46** | |

**Target:** Deferred count drops from 178 to ~132.

---

## Determinism Enforcement

**Always (set at worker process start):**
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` in worker environment before PyTorch import
- CPU + CUDA RNG state captured in checkpoints (CPU capture added in 3B)

**Opt-in (per replay/step request with `deterministic: true`):**
- `torch.use_deterministic_algorithms(True)` set before replay, restored after
- If an op lacks a deterministic implementation → PyTorch raises → reported as error

**Why not always-strict:** Forces deterministic kernels that disable some ops and
can be slower. Many models (especially with scatter ops) fail entirely under strict mode.
The 3-nines tolerance handles FP non-determinism without breaking models.

---

## Worldline Model

**Concepts:**

- **Worldline segment:** Contiguous sequence of forward ticks from same starting conditions
- **Branching point:** Tick where a segment diverges from its parent (stored as `replay_of`)
- **Named branch:** Explicit snapshot via `branch.fork` for later comparison/persistence

**Rules:**
- Stepping backward = navigation (no implicit fork)
- Stepping forward after backward = implicit new worldline segment
- `branch.fork` = explicitly name current segment for persistence
- `branch.compare` = divergence metrics between named branches
- `branch.drop` = release arena resources for a branch

**Tick identity:**
- Fresh tick_ids are always monotonically increasing (never reuse)
- `replay_of` field on TickPosition points to the original tick being revisited
- A replayed tick and its original share the same POSITION (layer/component/event)
  but have different tick_ids

**Session state tracks:**

```rust
pub struct WorldlineState {
    pub current_segment: u32,
    pub segments: Vec<WorldlineSegment>,
}

pub struct WorldlineSegment {
    pub id: u32,
    pub parent_segment: Option<u32>,
    pub branch_tick: Option<u64>,  // tick_id where this diverged from parent
    pub tick_range: (u64, u64),    // first and last tick_id in this segment
}
```

---

## Safety Invariants

1. **DMA fence (Phase 3):** Single-GPU + single-thread + single default stream = stream ordering
   is sufficient. No explicit CUDA event fence needed. Debug assertion in eviction path.
   Real event fence deferred to Phase 5 (multi-GPU).

2. **Arena access:** Spill is synchronous (triggered only on dispatch thread during checkpoint
   write). RefCell is correct for Phase 3. Slot-level atomics deferred to Phase 5.

3. **Tier 2 timeout:** No signal.alarm. Watchdog thread + PyThreadState_SetAsyncExc.
   GPU-bound callbacks cannot be interrupted without killing the worker — documented,
   not worked around.

4. **frombuffer alias lifetime:** Every `torch.frombuffer(arena_ptr)` tensor must be
   `del`'d before the function returns. No reference escapes into Python closures or
   hook locals. Enforced by code pattern (local variable in narrow scope).

5. **Checkpoint identity:** Sub-checkpoints named `sub-{segment}-{tick}`. Auto-checkpoints
   named `auto-{uuid}`. User checkpoints user-named. Namespaces don't collide.

---

## Execution Order

**Sub-project B (replay + reverse-step):** 6 slices, serial.
1. B1: Basic replay (worker dispatch + step loop replay context)
2. B2: Divergence detection (compare_activations bridge + threshold config)
3. B3: Interventions during replay (pre-loaded from request)
4. B4: Reverse step (daemon dispatch for backward direction + worldline tracking)
5. B5: Performance (sub-checkpoint policy + prefetch)
6. B6: ROME exit test (acceptance test)

**Sub-project C (bundle + callbacks + TCK):** 4 slices, serial.
1. C1: Tier 2 Python callbacks (recipe type + watchdog + validation)
2. C2: Bundle extension (checkpoint/bookmark/worldline export)
3. C3: Inspect format fix (5→6 segment alignment)
4. C4: TCK sweep (un-defer + verify green)

**Ordering:** B before C, with one exception: **C1 (Tier 2 callbacks) ships before B6
(ROME test).** The ROME test requires a user-defined Python callback for the rank-1 edit.
The minimal callback dispatch (module import + function call + watchdog) is a prerequisite.

Revised order: B1 → B2 → B3 → B4 → B5 → C1 → B6 → C2 → C3 → C4.

This means C1 is implemented on the B branch. The C branch handles C2-C4.

---

## Protocol Changes

**New fields on existing types:**

| Type | New field | Purpose |
|------|-----------|---------|
| `StepRequest` | `direction: String` | "forward" (default) or "backward" |
| `StepRequest` | `deterministic: Option<bool>` | Opt-in strict mode |
| `ReplayRequest` | `deterministic: Option<bool>` | Opt-in strict mode |
| `ReplayRequest` | `cosine_threshold: Option<f64>` | Override default 0.999 |
| `ReplayRequest` | `mre_threshold: Option<f64>` | Override default 0.05 |
| `SessionState` | `worldline: WorldlineState` | DAG tracking in envelope |

**New internal types:**

| Type | Location | Purpose |
|------|----------|---------|
| `HostReplayRequest` | protocol crate | _host/replay wire type |
| `HostReplayResponse` | protocol crate | _host/replay response |
| `ReplayContext` | worker crate | Step loop mode flag |
| `WorldlineState` | protocol crate | Segment DAG |
| `WorldlineSegment` | protocol crate | Single segment |
| `InterventionContext` | python bridge | Tier 2 callback context |

**New verbs routed:**

| Verb | Handler | Status |
|------|---------|--------|
| `rocket/replay` | `handle_replay` | Stub exists → replace with real impl |
| `rocket/branch.fork` | `handle_branch_fork` | New |
| `rocket/branch.drop` | `handle_branch_drop` | New |
| `rocket/branch.compare` | `handle_branch_compare` | New |

---

## Files Inventory (Estimated)

### New files:
- `crates/rocket-surgeon-worker/src/replay.rs` — replay context + execution logic
- `python/rocket_surgeon/replay.py` — compare_activations, CPU RNG helpers
- `python/rocket_surgeon/host/interventions/callback.py` — Tier 2 dispatch
- `tests/test_e2e_replay.py` — replay E2E tests
- `tests/test_e2e_reverse_step.py` — reverse step E2E tests
- `tests/test_rome_acceptance.py` — ROME exit test

### Modified files:
- `crates/rocket-surgeon-protocol/src/messages.rs` — new fields, HostReplay types
- `crates/rocket-surgeon/src/dispatch.rs` — handle_replay, handle_branch_*, backward step
- `crates/rocket-surgeon/src/session.rs` — WorldlineState, sub-checkpoint policy
- `crates/rocket-surgeon/src/main.rs` — replay routing, CUBLAS env at start
- `crates/rocket-surgeon-orchestrator/src/dispatch.rs` — _host/replay routing
- `crates/rocket-surgeon-worker/src/dispatch.rs` — _host/replay handler, replay context
- `crates/rocket-surgeon-worker/src/bridge.rs` — compare_activations, Tier 2 call
- `crates/rocket-surgeon-worker/src/checkpoint.rs` — sub-checkpoint naming, prefetch
- `python/rocket_surgeon/checkpoint.py` — CPU RNG capture/restore
- `python/rocket_surgeon/host/interventions/engine.py` — callback recipe dispatch
- `crates/rocket-surgeon/src/bundle.rs` — checkpoint/worldline export
- `tck/protocol/replay.feature` — remove @deferred tags
- `tck/protocol/branch.feature` — remove @deferred tags
