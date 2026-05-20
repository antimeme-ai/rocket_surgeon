# TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the rocket_surgeon TUI per the design spec at `docs/superpowers/specs/2026-05-20-tui-design.md`, including all protocol backports required for v0.3.0, the librocket_viz C core, and the LLM ergonomics surface.

**Architecture:** Middle-out — protocol data model lands first (three-clock tick model, KV cache verbs, branching verbs, AttachResponse extensions), then the TUI foundation (Elm-style state reducer, input abstraction, tiling manager, rendering scaffold), then views as bridges between data and intent. librocket_viz C kernels and LLM surface verbs are parallel tracks.

**Tech Stack:** Rust (TUI, protocol, daemon), C (librocket_viz SIMD kernels), ratatui + crossterm (terminal rendering), Kitty graphics protocol / Sixel (graphical views), Gherkin/pytest-bdd (TCK).

**Spec:** `docs/superpowers/specs/2026-05-20-tui-design.md`

**Replaces:** `docs/specs/plan.md` Phase 4 (tasks 4.1–4.13) in its entirety.

---

## Changes to Existing Plan Phases

The TUI design spec identifies protocol work that must land in existing phases before
Phase 4 begins. These are additions, not replacements.

### Phase 3 additions (checkpoint + reverse step)

The following tasks are added to Phase 3. They are branching infrastructure that the
Worldline view depends on:

| New task | Description | Depends on |
|----------|-------------|------------|
| 3.13 | `branch.fork` verb: create branch from checkpoint, allocate VRAM, return branch_id | 3.3, 3.4 |
| 3.14 | `branch.drop` verb: release branch resources (live → spilled → dropped tier transition) | 3.13 |
| 3.15 | `branch.compare` verb: compute divergence metrics (cosine sim, max relative error, KL div, per-layer norm delta) between two branches | 3.13, 3.5 |
| 3.16 | Branch events: `branch.created`, `branch.tier_changed` | 3.13, 3.14 |
| 3.17 | Branch error codes: `E_BRANCH_NOT_FOUND` (-32022), `E_BRANCH_MERGE_REFUSED` (-32023), `E_VRAM_EXHAUSTED` (-32024) | 3.13 |
| 3.18 | KV cache read foundation: `kv.read` verb (layer range, position range, head range). Safe access via checkpoint snapshot. | 3.3 |
| 3.19 | KV cache events: `kv.update` (cache grew), `kv.evicted` (position dropped) | 3.18 |
| 3.20 | TCK green for all branching and KV read scenarios | 3.13–3.19 |

Phase 3 exit criteria gain: branch lifecycle works (fork, compare, drop). KV cache
readable at rest. `E_VRAM_EXHAUSTED` fires before OOM with per-branch memory accounting
in error details.

---

## Phase 4 — TUI (replaces plan.md Phase 4 entirely)

Organized into sub-phases by dependency. No calendar estimates — structural ordering
only. Each work unit follows the JSMNTL cycle: TCK red → implement → green → review.

### Dependency graph

```
4.0 Protocol v0.3.0 ──────────────────────────────────────────────┐
    │                                                              │
    ├── 4.1 TUI Foundation ──── 4.3 Core Views ──── 4.4 Extended  │
    │       │                                            │         │
    │       └── needs 4.2 for graphical widget ──────────┘         │
    │                                                              │
    ├── 4.2 librocket_viz (parallel) ───── 4.5 Sugiyama ──────────┤
    │                                         │                    │
    │                               4.4.3 Worldline (needs 3.13+) │
    │                                                              │
    └── 4.6 LLM Surface (parallel) ───────────────────────────────┘
                                                                   │
                                                        4.7 Integration
```

### Phase 4.0 — Protocol v0.3.0 Transition

**Goal:** Land all protocol type changes required by the TUI spec. This is the sync
point — nothing else in Phase 4 starts until 4.0 is complete and TCK green.

---

#### 4.0.1 — Three-clock tick model

Add `TickClock` to the protocol types and integrate it into `TickPosition`.

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/types.rs`
- Modify: `crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs`
- Modify: `crates/rocket-surgeon-worker/src/tick.rs`
- Create: `tck/protocol/tick-clock.feature`

- [ ] **Step 1: Write TCK scenarios**

```gherkin
# tck/protocol/tick-clock.feature
Feature: Three-clock tick model
  The tick model carries three incommensurable clocks:
  token (sequence position), operator (within-token traversal),
  and wall (nanosecond real time).

  Scenario: TickPosition carries all three clocks
    Given a session in Stopped state at layer 5 component "attn.q"
    Then the tick position has a "clock" field
    And clock.token is the current token position
    And clock.operator is the within-token traversal index
    And clock.wall_ns is a non-zero nanosecond timestamp

  Scenario: tick_id is alias for clock.operator
    Given a tick position with clock.operator = 42
    Then tick_id equals 42

  Scenario: clock.operator resets each token
    Given a session stepping through token 0
    When the session advances to token 1
    Then clock.token increments by 1
    And clock.operator resets to 0

  Scenario: Backward compatibility — tick_id still present
    Given a response from protocol version 0.3.0
    Then the tick position JSON contains both "tick_id" and "clock" fields
    And tick_id equals clock.operator
```

- [ ] **Step 2: Run TCK to verify red**

Run: `pytest tck/protocol/tick-clock.feature -v`
Expected: FAIL (TickClock type does not exist)

- [ ] **Step 3: Add TickClock type**

```rust
// crates/rocket-surgeon-protocol/src/types.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TickClock {
    pub token: u64,
    pub operator: u64,
    pub wall_ns: u64,
}
```

Add `clock: TickClock` field to `TickPosition`. Keep `tick_id` as a serialized alias for
backward compatibility (custom serde: serialize both `tick_id` and `clock.operator` to
the same value).

- [ ] **Step 4: Update TickState in worker**

`crates/rocket-surgeon-worker/src/tick.rs`: `to_tick_position()` populates all three
clock fields. `wall_ns` from `std::time::Instant` delta since session start.
`token` from `self.token_position`. `operator` from `self.tick_id`.

- [ ] **Step 5: Update serde roundtrip tests**

Add `TickClock` roundtrip test. Verify `tick_id` backward compat serialization.

- [ ] **Step 6: Run tests, verify green**

Run: `cargo test -p rocket-surgeon-protocol && cargo test -p rocket-surgeon-worker`

- [ ] **Step 7: Commit**

```bash
git add crates/rocket-surgeon-protocol/src/types.rs \
       crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs \
       crates/rocket-surgeon-worker/src/tick.rs \
       tck/protocol/tick-clock.feature
git commit -m "feat(protocol): add TickClock three-clock model to TickPosition"
```

---

#### 4.0.2 — AttachResponse extensions

Add component vocabulary, module tree, alias table, and tick map to `AttachResponse`.

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/messages.rs`
- Modify: `crates/rocket-surgeon-protocol/src/types.rs`
- Create: `tck/protocol/attach-discovery.feature`

- [ ] **Step 1: Write TCK scenarios**

```gherkin
# tck/protocol/attach-discovery.feature
Feature: Model discovery via attach response
  The attach response provides everything an LLM or TUI needs
  to construct valid probe points without trial and error.

  Scenario: Attach response includes component vocabulary
    Given an initialized session
    When the client sends attach with model_family "llama"
    Then the response includes "component_vocabulary" as an array
    And each entry has "canonical" (string), "event" (string), "tensor_shape" (array)

  Scenario: Attach response includes module tree
    Given an attached session with model_family "llama"
    Then the response includes "module_tree" as an array of strings
    And the tree contains at least one entry per layer

  Scenario: Attach response includes alias table
    Given an attached session
    Then the response includes "alias_table" as an array
    And each entry has "canonical" and "aliases" fields
    And "blocks.0.attn.hook_q" appears as an alias for the layer 0 attn.q component

  Scenario: Attach response includes tick map
    Given an attached session
    Then the response includes "tick_map" as an object
    And tick_map contains an entry for granularity "component"
    And each granularity entry lists ticks per layer with ordering
```

- [ ] **Step 2: Run TCK to verify red**

- [ ] **Step 3: Define discovery types**

```rust
// crates/rocket-surgeon-protocol/src/types.rs

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentEntry {
    pub canonical: String,
    pub event: String,
    pub tensor_shape: Vec<u64>,
    pub category: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasEntry {
    pub canonical: String,
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickMapEntry {
    pub granularity: TickGranularity,
    pub ticks_per_layer: Vec<TickLayerInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickLayerInfo {
    pub layer: u32,
    pub components: Vec<String>,
    pub tick_count: u32,
}
```

- [ ] **Step 4: Extend AttachResponse**

Add `component_vocabulary: Vec<ComponentEntry>`, `module_tree: Vec<String>`,
`alias_table: Vec<AliasEntry>`, `tick_map: Vec<TickMapEntry>` to `AttachResponse` in
`messages.rs`.

- [ ] **Step 5: Update serde roundtrip tests**

- [ ] **Step 6: Run tests, verify green**

- [ ] **Step 7: Commit**

---

#### 4.0.3 — Response envelope compactness

Add `envelope` field to request types for LLM context window management.

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/types.rs`
- Modify: `crates/rocket-surgeon-protocol/src/messages.rs`
- Create: `tck/protocol/envelope-compactness.feature`

- [ ] **Step 1: Write TCK scenarios**

```gherkin
Feature: Response envelope compactness
  Clients can negotiate response envelope verbosity to manage
  context window pressure.

  Scenario: Default envelope is full
    Given an attached session
    When the client sends step with no envelope field
    Then the response includes the complete SessionState

  Scenario: Position-only envelope
    Given an attached session
    When the client sends step with envelope "position"
    Then the response includes status and tick position
    And the response does not include active_probes or checkpoints

  Scenario: No envelope
    Given an attached session
    When the client sends step with envelope "none"
    Then the response includes only the data payload
    And no SessionState fields are present
```

- [ ] **Step 2: Run TCK to verify red**

- [ ] **Step 3: Define EnvelopeMode enum**

```rust
// crates/rocket-surgeon-protocol/src/types.rs

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvelopeMode {
    #[default]
    Full,
    Position,
    None,
}
```

- [ ] **Step 4: Add envelope field to request types**

Every request struct in `messages.rs` that returns a `ResponseEnvelope` gains:
`#[serde(default)] pub envelope: EnvelopeMode`

- [ ] **Step 5: Add PositionEnvelope response type**

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositionEnvelope {
    pub status: Status,
    pub position: Option<TickPosition>,
}
```

- [ ] **Step 6: Update serde tests, run green**

- [ ] **Step 7: Commit**

---

#### 4.0.4 — Step run_to extension

Extend `StepRequest` with `run_to` for LLM-friendly stepping.

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/messages.rs`
- Create: `tck/protocol/step-run-to.feature`

- [ ] **Step 1: Write TCK scenarios**

```gherkin
Feature: Step with run_to destination
  LLM clients can name a destination instead of counting ticks.

  Scenario: Step to a specific component
    Given an attached session at layer 0
    When the client sends step with run_to "llama:*:12:attn.o_proj:output"
    Then the session stops at layer 12 component "attn.o_proj"

  Scenario: Step to completion
    Given an attached session at layer 0
    When the client sends step with run_to "completion"
    Then the session stops at the final component of the final layer

  Scenario: run_to with invalid target
    Given an attached session
    When the client sends step with run_to "llama:*:99:nonexistent:output"
    Then the response is an error with code "INVALID_TARGET"
    And error details include "nearest_matches"
```

- [ ] **Step 2: Run TCK to verify red**

- [ ] **Step 3: Add run_to field**

```rust
// In StepRequest:
#[serde(skip_serializing_if = "Option::is_none")]
pub run_to: Option<String>,
```

When `run_to` is set, `count` and `granularity` are ignored. The daemon steps until the
probe-point target matches the current position, or until the forward pass completes
(for `"completion"`).

- [ ] **Step 4: Update serde tests, run green**

- [ ] **Step 5: Commit**

---

#### 4.0.5 — Subscribe filtering

Add filter parameter to `SubscribeRequest`.

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/messages.rs`
- Create: `tck/protocol/subscribe-filter.feature`

- [ ] **Step 1: Write TCK scenarios**

```gherkin
Feature: Subscribe with event filtering
  Clients can filter which events they receive to reduce
  notification volume.

  Scenario: Filter by event type
    Given an attached session
    When the client subscribes with filter events ["tick.stopped"]
    Then the client receives tick.stopped events
    And the client does not receive probe.fired events

  Scenario: Filter by layer range
    Given an attached session with probes on layers 0-31
    When the client subscribes with filter layers [10, 11, 12]
    Then probe.fired events only arrive for layers 10, 11, 12

  Scenario: Filter by component pattern
    Given an attached session
    When the client subscribes with filter components ["attn.*"]
    Then tick.stopped events only arrive for attention components
```

- [ ] **Step 2: Run TCK to verify red**

- [ ] **Step 3: Define SubscribeFilter type and update SubscribeRequest**

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubscribeFilter {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub events: Option<Vec<EventType>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub components: Option<Vec<String>>,
}

// Update SubscribeRequest:
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubscribeRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<SubscribeFilter>,
}
```

- [ ] **Step 4: Update serde tests, run green**

- [ ] **Step 5: Commit**

---

#### 4.0.6 — Error expressiveness retrofit

Add `recovery_hint` to `ErrorData`, add new TUI-driven error codes, retrofit
existing errors with structured `context` requirements.

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/errors.rs`
- Create: `tck/protocol/error-expressiveness.feature`

- [ ] **Step 1: Write TCK scenarios**

```gherkin
Feature: Error expressiveness
  Every error carries what happened, why, and what to do about it.

  Scenario: ErrorData includes recovery_hint
    Given any error response
    Then the error data has a "recovery_hint" field (string or null)

  Scenario: INVALID_TARGET includes nearest matches
    Given an attached session
    When the client inspects target "llama:*:12:attn.out_proj:output"
    Then the error code is "INVALID_TARGET"
    And error context includes "attempted" = "attn.out_proj"
    And error context includes "nearest_matches" as a non-empty array
    And error context includes "valid_components_at_layer" as an array

  Scenario: E_VRAM_EXHAUSTED includes memory accounting
    Given a session near VRAM capacity
    When an operation would exceed the VRAM headroom
    Then the error code is "VRAM_EXHAUSTED"
    And error context includes "used_mb", "total_mb", "headroom_mb"
    And error context includes "per_branch" array with id and size_mb per branch
    And error context includes "recommendation" string
```

- [ ] **Step 2: Run TCK to verify red**

- [ ] **Step 3: Add recovery_hint and new error codes**

```rust
// In ErrorData:
#[serde(skip_serializing_if = "Option::is_none")]
pub recovery_hint: Option<String>,

// New error codes (Phase 3.17 already landed -32022 through -32024
// for BranchNotFound, BranchMergeRefused, VramExhausted):
#[serde(rename = "CROSS_REQUEST_KV")]
CrossRequestKv,       // -32025
#[serde(rename = "KV_EVICTED")]
KvEvicted,            // -32026
```

- [ ] **Step 4: Update severity mapping for new codes**

`VramExhausted` → `Fatal`. Others → `Recoverable`.

- [ ] **Step 5: Update serde tests, run green**

- [ ] **Step 6: Commit**

---

#### 4.0.7 — Activation patching refinements

Extend `InterventionParams::Ablate` with mode, add new intervention types.

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/types.rs`
- Modify: `tck/protocol/intervention.feature`

- [ ] **Step 1: Write TCK scenarios**

```gherkin
Feature: Extended activation patching

  Scenario: Ablate with mode zero (default)
    Given an intervention recipe with type "ablate" and params {"mode": "zero"}
    Then the intervention deserializes successfully
    And mode is AblateMode::Zero

  Scenario: Ablate with mode mean
    Given an intervention recipe with type "ablate" and params {"mode": "mean", "reference_run": "ckpt-baseline"}
    Then the intervention deserializes successfully
    And mode is AblateMode::Mean

  Scenario: AttentionMask intervention
    Given an intervention recipe with type "attention_mask"
    And params {"source_positions": [0, 3], "target_positions": [5], "mask_value": -10000.0}
    Then the intervention deserializes successfully

  Scenario: EmbedSwap intervention
    Given an intervention recipe with type "embed_swap"
    And params {"position": 5, "new_token_id": 1234}
    Then the intervention deserializes successfully

  Scenario: EmbedNoise intervention
    Given an intervention recipe with type "embed_noise"
    And params {"position": 5, "std": 0.1, "seed": 42}
    Then the intervention deserializes successfully
```

- [ ] **Step 2: Run TCK to verify red**

- [ ] **Step 3: Update intervention types**

```rust
// Replace InterventionParams::Ablate:
Ablate {
    #[serde(default)]
    mode: AblateMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_run: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_tensor_id: Option<String>,
},

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AblateMode {
    #[default]
    Zero,
    Mean,
    Resample,
}

// New InterventionType variants:
AttentionMask,
EmbedSwap,
EmbedNoise,

// New InterventionParams variants:
AttentionMask {
    source_positions: Vec<u64>,
    target_positions: Vec<u64>,
    mask_value: f64,
},
EmbedSwap {
    position: u64,
    new_token_id: u64,
},
EmbedNoise {
    position: u64,
    std: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
},
```

- [ ] **Step 4: Update existing Ablate tests, add new type tests**

- [ ] **Step 5: Run tests, verify green**

- [ ] **Step 6: Commit**

---

#### 4.0.8 — New BuiltInView variants and KV/branch message types

Add KV cache, branch, and lens view types plus their request/response messages.

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/types.rs`
- Modify: `crates/rocket-surgeon-protocol/src/messages.rs`
- Create: `tck/protocol/kv-cache.feature`
- Create: `tck/protocol/branch.feature`

- [ ] **Step 1: Write TCK scenarios for KV read**

```gherkin
# tck/protocol/kv-cache.feature
Feature: KV cache protocol surface

  Scenario: kv.read returns cache slice
    Given an attached session with KV cache populated
    When the client sends kv.read with layers [0, 1], positions [0, 1, 2]
    Then the response includes cache entries with norms per layer and position

  Scenario: kv.read with evicted position
    Given a session where position 5 was evicted
    When the client sends kv.read for position 5
    Then the error code is "KV_EVICTED"
    And error context includes evicted_at_tick and nearest_checkpoint
```

- [ ] **Step 2: Write TCK scenarios for branch verbs**

```gherkin
# tck/protocol/branch.feature
Feature: Worldline branching

  Scenario: branch.fork creates a new branch
    Given a session with checkpoint "ckpt-1"
    When the client sends branch.fork from "ckpt-1"
    Then the response includes a branch_id
    And a branch.created event is emitted

  Scenario: branch.compare returns divergence metrics
    Given two branches "branch-a" and "branch-b" from the same checkpoint
    When the client sends branch.compare for "branch-a" and "branch-b"
    Then the response includes cosine_similarity, max_relative_error, kl_divergence
    And per_layer_norm_delta is an array with one entry per layer

  Scenario: branch.drop releases resources
    Given a live branch "branch-x"
    When the client sends branch.drop for "branch-x"
    Then a branch.tier_changed event is emitted with tier "dropped"
```

- [ ] **Step 3: Add BuiltInView variants**

```rust
// In BuiltInView enum:
TunedLens,
KvCacheRibbon,
KvCacheDetail,
WorldlineDag,
```

- [ ] **Step 4: Define KV cache message types**

```rust
// messages.rs

pub mod method {
    // ... existing ...
    pub const KV_READ: &str = "rocket/kv.read";
    pub const KV_INTERVENE: &str = "rocket/kv.intervene";
    pub const BRANCH_FORK: &str = "rocket/branch.fork";
    pub const BRANCH_DROP: &str = "rocket/branch.drop";
    pub const BRANCH_COMPARE: &str = "rocket/branch.compare";
    pub const DISCOVER: &str = "rocket/discover";
    pub const SWEEP: &str = "rocket/sweep";
    pub const VIEW_FOCUS: &str = "rocket/view.focus";
    pub const VIEW_DEFINE: &str = "rocket/view.define";
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvReadRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub positions: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heads: Option<Vec<u32>>,
    #[serde(default = "default_kv_slot")]
    pub slot: KvSlot,
    #[serde(default)]
    pub metric: KvMetric,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvSlot { K, V, #[default] Both }

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvMetric { #[default] L2Norm, Mean, AbsMax }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvReadResponse {
    pub entries: Vec<KvCacheEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCacheEntry {
    pub layer: u32,
    pub position: u64,
    pub head: u32,
    pub k_metric: Option<f64>,
    pub v_metric: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlay: Option<KvOverlay>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvOverlay { Sink, HeavyHitter, Evicted, Quantized, PageBoundary, SharedPrefix }
```

- [ ] **Step 5: Define branch message types**

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchForkRequest {
    pub from_checkpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchForkResponse {
    pub branch_id: String,
    pub tier: BranchTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BranchTier { Live, Spilled, Dropped }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchDropRequest {
    pub branch_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchDropResponse {
    pub branch_id: String,
    pub freed_mb: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchCompareRequest {
    pub branch_a: String,
    pub branch_b: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchCompareResponse {
    pub cosine_similarity: f64,
    pub max_relative_error: f64,
    pub kl_divergence: f64,
    pub per_layer_norm_delta: Vec<f64>,
}

// Events
pub mod event {
    // ... existing ...
    pub const KV_UPDATE: &str = "kv.update";
    pub const KV_EVICTED: &str = "kv.evicted";
    pub const BRANCH_CREATED: &str = "branch.created";
    pub const BRANCH_TIER_CHANGED: &str = "branch.tier_changed";
    pub const SPEC_STEP: &str = "spec.step";
    pub const SWEEP_TRIAL_COMPLETE: &str = "sweep.trial_complete";
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvUpdateEvent {
    pub layer: u32,
    pub new_positions: Vec<u64>,
    pub total_positions: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvEvictedEvent {
    pub layer: u32,
    pub evicted_positions: Vec<u64>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchCreatedEvent {
    pub branch_id: String,
    pub from_checkpoint: String,
    pub tier: BranchTier,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchTierChangedEvent {
    pub branch_id: String,
    pub old_tier: BranchTier,
    pub new_tier: BranchTier,
}
```

- [ ] **Step 6: Update serde roundtrip tests for all new types**

- [ ] **Step 7: Run tests, verify green**

- [ ] **Step 8: Commit**

---

#### 4.0.9 — LLM ergonomic verbs (discover, sweep, view.focus, view.define)

Define the message types for LLM-facing protocol verbs.

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/messages.rs`
- Create: `tck/protocol/discover.feature`
- Create: `tck/protocol/sweep.feature`
- Create: `tck/protocol/view-focus.feature`

- [ ] **Step 1: Write TCK scenarios**

```gherkin
# tck/protocol/discover.feature
Feature: Probe-point discovery

  Scenario: Discover with wildcard returns matching points
    Given an attached session with model_family "llama"
    When the client sends discover with pattern "llama:*:12:*:output"
    Then the response includes all layer 12 output components
    And each entry has canonical name, tensor_shape, and aliases

  Scenario: Discover with partial match suggests corrections
    Given an attached session
    When the client sends discover with pattern "llama:*:12:attn.out_proj:output"
    Then the response includes 0 exact matches
    And includes "suggestions" with nearest valid patterns
```

```gherkin
# tck/protocol/sweep.feature
Feature: Batch experiment sweep

  Scenario: Sweep runs multiple trials from a checkpoint
    Given a session with checkpoint "ckpt-clean"
    When the client sends sweep with 3 trial specs
    Then the response includes results keyed by trial index
    And each result includes collected tensor summaries

  Scenario: Sweep streams trial_complete events
    Given a subscribed session running a sweep
    Then a sweep.trial_complete event fires after each trial
```

```gherkin
# tck/protocol/view-focus.feature
Feature: View focus for LLM navigation

  Scenario: Focus by position
    Given an attached session with tokenized input
    When the client sends view.focus with selector by_position 5
    Then the response includes the token at position 5
    And per-layer summaries for that position

  Scenario: Focus by regex
    When the client sends view.focus with selector by_regex "defendant"
    Then the response includes the first matching token
```

- [ ] **Step 2: Run TCK to verify red**

- [ ] **Step 3: Define message types**

```rust
// Discover
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoverRequest {
    pub pattern: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoverResponse {
    pub matches: Vec<DiscoverMatch>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoverMatch {
    pub canonical: String,
    pub tensor_shape: Vec<u64>,
    pub aliases: Vec<String>,
}

// view.focus
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FocusSelector {
    ById { token_id: u64 },
    ByPosition { position: u64 },
    ByRegex { pattern: String },
    ByAnchor { anchor: FocusAnchor },
    ByRange { start: u64, end: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FocusAnchor { Bos, Eos, PadBoundary, Sink, MaxAttention }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewFocusRequest {
    pub selector: FocusSelector,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewFocusResponse {
    pub position: u64,
    pub token: serde_json::Value,
    pub per_layer_summaries: Vec<TensorSummary>,
}

// Sweep
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepRequest {
    pub baseline_checkpoint: String,
    pub trials: Vec<SweepTrial>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric: Option<SweepMetric>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepTrial {
    pub interventions: Vec<InterventionRecipe>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collect: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepMetric {
    #[serde(rename = "type")]
    pub metric_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepResponse {
    pub results: Vec<SweepTrialResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepTrialResult {
    pub trial_index: u32,
    pub stopped_at: TickPosition,
    pub collected: Vec<TensorSummary>,
    pub metric_value: Option<f64>,
}

// view.define
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewDefineRequest {
    pub name: String,
    pub spec: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewDefineResponse {
    pub name: String,
    pub registered: bool,
}
```

- [ ] **Step 4: Add InterventionRecipe id as optional**

```rust
// In InterventionRecipe, change:
pub id: String,
// to:
#[serde(skip_serializing_if = "Option::is_none")]
pub id: Option<String>,
```

- [ ] **Step 5: Update serde roundtrip tests for all new types**

- [ ] **Step 6: Run tests, verify green**

- [ ] **Step 7: Commit**

---

#### 4.0.10 — Protocol version bump and schema freeze

Bump protocol version to 0.3.0, update `Capabilities::phase1_defaults()`, verify all
new TCK scenarios are wired up.

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/types.rs`
- Modify: `tck/protocol/capabilities.feature`

- [ ] **Step 1: Update protocol version**

In `Capabilities::phase1_defaults()`, change `protocol_version` from `"0.2.0"` to
`"0.3.0"`. Add new `BuiltInView` variants to `built_in_views`. Add `Websocket` to
`transports`.

- [ ] **Step 2: Run full TCK suite**

Run: `cargo xtask test && cargo xtask e2e`
Expected: All green.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(protocol): bump to v0.3.0 — three clocks, KV/branch verbs, LLM surface"
```

---

### Phase 4.1 — TUI Foundation

**Goal:** Build the reactive core that all views depend on. At the end of 4.1, the TUI
binary launches, connects to a daemon, receives events, and renders an empty frame with
mode indicator and status bar.

**All work in:** `crates/rocket-surgeon-tui/src/`

---

#### 4.1.1 — Protocol client

JSON-RPC client that connects to the daemon, sends requests, and ingests the event
stream.

**Files:**
- Create: `crates/rocket-surgeon-tui/src/client.rs`
- Create: `crates/rocket-surgeon-tui/src/client/connection.rs`
- Create: `crates/rocket-surgeon-tui/src/client/subscription.rs`
- Test: `crates/rocket-surgeon-tui/src/client.rs` (unit tests with mock transport)

- [ ] **Step 1: Write failing test** — client sends initialize, receives capabilities
- [ ] **Step 2: Implement** — Unix socket connection, JSON-RPC framing (reuse `rocket-surgeon-transport` crate framing), async request/response with `tokio::sync::oneshot`, event stream via `tokio::sync::broadcast`
- [ ] **Step 3: Write failing test** — client reconnects after disconnect
- [ ] **Step 4: Implement** — reconnection with exponential backoff, state rehydration on reconnect via `rocket/status`
- [ ] **Step 5: Write failing test** — subscription management (subscribe/unsubscribe based on active views)
- [ ] **Step 6: Implement** — `SubscriptionManager` tracks which events are needed, issues subscribe/unsubscribe when view config changes
- [ ] **Step 7: Run tests, verify green**
- [ ] **Step 8: Commit**

---

#### 4.1.2 — Input abstraction

`InputSource` trait, abstract event types, terminal input decoder, mode state machine.

**Files:**
- Create: `crates/rocket-surgeon-tui/src/input.rs`
- Create: `crates/rocket-surgeon-tui/src/input/events.rs`
- Create: `crates/rocket-surgeon-tui/src/input/terminal.rs`
- Create: `crates/rocket-surgeon-tui/src/input/mode.rs`

- [ ] **Step 1: Define abstract event types**

```rust
// input/events.rs

pub enum NavigationEvent {
    Up, Down, Left, Right,
    JumpTo(JumpTarget),
    ZoomIn, ZoomOut,
    PageUp, PageDown,
    Home, End,
    ContinuousAdjust { axis: Axis, value: f32 },
}

pub enum CommandEvent {
    Char(char),
    Execute,
    Cancel,
    TabComplete,
    HistoryPrev,
    HistoryNext,
}

pub enum ModeEvent {
    EnterCommand,
    EnterInspect,
    EnterIntervene,
    ExitToNormal,
}

pub enum Axis { Layer, TokenPosition, Head, Custom(String) }
pub enum JumpTarget { Layer(u32), Token(u64), Component(String) }
```

- [ ] **Step 2: Define InputSource trait**

```rust
pub trait InputSource: Send + 'static {
    fn poll(&mut self) -> Option<RawEvent>;
}
```

- [ ] **Step 3: Implement terminal input decoder** — maps crossterm `KeyEvent` to abstract events based on current mode
- [ ] **Step 4: Implement mode state machine** — Normal ↔ Command (`:`) ↔ Inspect (`i`) ↔ Intervene (`I`), tracks current mode, validates transitions
- [ ] **Step 5: Write tests** — mode transitions, key mappings per mode
- [ ] **Step 6: Run tests, verify green**
- [ ] **Step 7: Commit**

---

#### 4.1.3 — State reducer and diff engine

Elm-style `(State, Event) → State` with dirty tracking.

**Files:**
- Create: `crates/rocket-surgeon-tui/src/state.rs`
- Create: `crates/rocket-surgeon-tui/src/state/reducer.rs`
- Create: `crates/rocket-surgeon-tui/src/state/diff.rs`
- Create: `crates/rocket-surgeon-tui/src/state/cache.rs`

- [ ] **Step 1: Define UiState**

```rust
// state.rs

pub struct UiState {
    pub session: SessionState,
    pub cursor: CursorState,
    pub mode: Mode,
    pub views: ViewConfig,
    pub tensor_cache: TensorCache,
    pub kv_cache_meta: Option<KvCacheMeta>,
    pub branch_graph: BranchGraph,
    pub resource_usage: ResourceUsage,
    pub pending_requests: PendingRequests,
}

pub struct CursorState {
    pub layer: u32,
    pub component: String,
    pub token_position: u64,
    pub focused_view: ViewId,
}
```

- [ ] **Step 2: Implement reducer** — pure function, match on event type, return new state. Events: `DaemonEvent` (from protocol), `InputEvent` (from input decoder), `InternalEvent` (timers, animations)
- [ ] **Step 3: Implement diff engine** — `DirtySet` that tracks which `ViewId`s need re-render. Views register `data_deps: &[DataDep]`. Diff compares old/new state for each `DataDep` variant.
- [ ] **Step 4: Implement tensor cache** — LRU with configurable memory budget, `TensorSummary` keyed by `(tick_id, probe_point)`
- [ ] **Step 5: Implement prefetch heuristics** — on cursor move, issue inspect requests for adjacent data (cursor at layer N → prefetch layers N-1, N+1; cursor at token T → prefetch token T+1). Prefetched results land in tensor cache. Budget: one-hop adjacency in layer and token dimensions.
- [ ] **Step 6: Write tests** — reducer state transitions, dirty propagation, cache eviction, prefetch fires on cursor move
- [ ] **Step 7: Run tests, verify green**
- [ ] **Step 8: Commit**

---

#### 4.1.4 — Tiling manager

Dynamic split system with context-reactive layout proposals.

**Files:**
- Create: `crates/rocket-surgeon-tui/src/tiling.rs`

- [ ] **Step 1: Define tiling model**

```rust
pub enum Layout {
    Single(ViewId),
    HSplit { left: Box<Layout>, right: Box<Layout>, ratio: f32 },
    VSplit { top: Box<Layout>, bottom: Box<Layout>, ratio: f32 },
}
```

- [ ] **Step 2: Implement split/unsplit operations** — from Single to HSplit/VSplit and back. Ratio adjustment via `NavigationEvent::ContinuousAdjust` or keyboard shortcuts.
- [ ] **Step 3: Implement context-reactive proposals** — `propose_layout(old_state, new_state) -> Option<Layout>`. Rules: attention component focused → suggest Inspector alongside Tower; `branch.fork` → suggest Worldline; `:kv` command → suggest Ribbon.
- [ ] **Step 4: Implement user preference persistence** — context → layout override map, serialized to a config file.
- [ ] **Step 5: Write tests** — split operations, proposal logic, preference override
- [ ] **Step 6: Run tests, verify green**
- [ ] **Step 7: Commit**

---

#### 4.1.5 — Rendering scaffold

ratatui frame loop with terminal capability detection and the graphical widget
placeholder for librocket_viz integration.

**Files:**
- Modify: `crates/rocket-surgeon-tui/src/main.rs`
- Create: `crates/rocket-surgeon-tui/src/render.rs`
- Create: `crates/rocket-surgeon-tui/src/render/capability.rs`
- Create: `crates/rocket-surgeon-tui/src/render/compositor.rs`

- [ ] **Step 1: Implement terminal capability detection** — query Kitty graphics support, Sixel support, color depth. Determine degradation tier (1–4).
- [ ] **Step 2: Implement frame loop** — 60fps target. `crossterm` raw mode, alternate screen. Event poll with 16ms timeout. Reducer → diff → render → flush.
- [ ] **Step 3: Implement compositor** — renders `Layout` tree into ratatui `Rect` allocations. Each `ViewId` gets a `Rect`. Calls `view.render(state, rect)`.
- [ ] **Step 4: Wire main.rs** — CLI args (daemon socket path, etc.), tokio runtime, spawn client + input + render tasks, channel plumbing between them.
- [ ] **Step 5: Implement animated transitions** — `InternalEvent::AnimationTick` fires at frame rate. Reducer interpolates cursor-driven state changes over ~150ms (configurable). Views see intermediate states during transition, not snapping. The transition itself carries information (§3.3): residual norm grows visually as you descend layers, attention patterns shift as you move across positions.
- [ ] **Step 6: Implement graphical widget placeholder** — a ratatui widget that accepts a pixel buffer and renders it via Kitty/Sixel/half-block based on detected tier. The buffer source will be librocket_viz (4.2), but the widget works now with a test pattern.
- [ ] **Step 7: Manual test** — launch TUI, verify it connects to daemon (or shows connection error gracefully), renders empty frame with mode indicator and degradation tier. Verify smooth interpolation on cursor movement.
- [ ] **Step 8: Commit**

---

### Phase 4.2 — librocket_viz (parallel track)

**Goal:** Build the C rendering library. Can proceed in parallel with 4.1 since the FFI
boundary is well-defined.

**All work in:** `librocket_viz/` (new top-level directory, added to workspace as a
`-sys` style crate at `crates/rocket-surgeon-viz-sys/`).

---

#### 4.2.1 — C library skeleton and Rust bindings

**Files:**
- Create: `librocket_viz/CMakeLists.txt` (or Makefile — minimal build system)
- Create: `librocket_viz/include/rocket_viz.h`
- Create: `librocket_viz/src/palette.c`
- Create: `crates/rocket-surgeon-viz-sys/Cargo.toml`
- Create: `crates/rocket-surgeon-viz-sys/build.rs`
- Create: `crates/rocket-surgeon-viz-sys/src/lib.rs`

- [ ] **Step 1: Define C header with initial functions**

```c
// librocket_viz/include/rocket_viz.h
#ifndef ROCKET_VIZ_H
#define ROCKET_VIZ_H

#include <stdint.h>
#include <stddef.h>

// Palette
void rsviz_palette_init(uint8_t palette[256][3]);
void rsviz_colormap_scalar(const float *values, uint32_t n,
                           uint8_t *out_palette_indices,
                           float vmin, float vmax,
                           uint8_t palette_start, uint8_t palette_end);

#endif
```

- [ ] **Step 2: Implement palette initialization** — fill the 256-color LUT per spec §5.3 (viridis 16-127, RdBu_r 128-207, Okabe-Ito 208-215, chrome 216-255)
- [ ] **Step 3: Create Rust -sys crate** — `build.rs` compiles the C library via `cc` crate, `lib.rs` has `extern "C"` declarations
- [ ] **Step 4: Write Rust test** — call `rsviz_palette_init`, verify viridis endpoints
- [ ] **Step 5: Run tests, verify green**
- [ ] **Step 6: Commit**

---

#### 4.2.2 — Colormap kernels (scalar reference + SIMD)

**Files:**
- Create: `librocket_viz/src/colormap.c`
- Create: `librocket_viz/src/colormap_sse2.c`
- Create: `librocket_viz/src/colormap_avx2.c`
- Create: `librocket_viz/src/colormap_neon.c`
- Create: `librocket_viz/tests/test_colormap.c`

- [ ] **Step 1: Implement scalar reference** — `rsviz_colormap_scalar`: linear interpolation of value into palette range, clamp to [vmin, vmax]
- [ ] **Step 2: Write C tests** — boundary values, NaN handling, empty input
- [ ] **Step 3: Implement SSE2 variant** — process 4 floats at a time
- [ ] **Step 4: Implement AVX2 variant** — process 8 floats at a time
- [ ] **Step 5: Implement NEON variant** — process 4 floats at a time (ARM64)
- [ ] **Step 6: Add runtime dispatch** — `rsviz_colormap` function selects best available SIMD at init time
- [ ] **Step 7: Write Rust integration test** — call from Rust, verify output matches scalar reference
- [ ] **Step 8: Commit**

---

#### 4.2.3 — Downsampling kernels (M4, LTTB)

**Files:**
- Create: `librocket_viz/src/downsample.c`
- Create: `librocket_viz/tests/test_downsample.c`

- [ ] **Step 1: Implement M4** — min-max downsampling for time series. Input: float array + length. Output: per-bucket (min, max, first, last). Bucket count = target column width.
- [ ] **Step 2: Implement LTTB** — Largest-Triangle-Three-Buckets for visually-preserving downsampling. Reference: Sveinn Steinarsson's 2013 paper.
- [ ] **Step 3: Write C tests** — identity (n < target), halving, known inputs
- [ ] **Step 4: Write Rust integration test**
- [ ] **Step 5: Commit**

---

#### 4.2.4 — Kitty and Sixel encoding kernels

**Files:**
- Create: `librocket_viz/src/kitty_encode.c`
- Create: `librocket_viz/src/sixel_encode.c`
- Create: `librocket_viz/tests/test_encode.c`

- [ ] **Step 1: Implement Kitty Unicode placeholder encoding** — takes RGBA pixel buffer + dimensions, outputs base64-encoded payload with Kitty protocol escape sequences. Chunked transmission for large images.
- [ ] **Step 2: Implement Sixel encoding** — RGBA → sixel band encoding. Reference: DEC documentation + VT340 test suite.
- [ ] **Step 3: Write C tests** — 1x1 pixel, known 4x4 pattern, encoding roundtrip properties
- [ ] **Step 4: Write Rust integration test** — encode a test pattern, verify output starts with correct escape sequence
- [ ] **Step 5: Commit**

---

### Phase 4.3 — Core Views

**Goal:** Build the first usable TUI. Tower + Inspector prove the reactive dataflow.
Token Axis and Command Bar complete the navigation surface. Distribution adds the
cross-layer aggregate view.

**Depends on:** 4.1 (foundation), 4.0 (protocol types)

---

#### 4.3.1 — View trait and framework

**Files:**
- Create: `crates/rocket-surgeon-tui/src/view.rs`
- Create: `crates/rocket-surgeon-tui/src/view/registry.rs`

- [ ] **Step 1: Define View trait**

```rust
pub trait View {
    fn data_deps(&self) -> &[DataDep];
    fn dirty(&self, prev: &UiState, next: &UiState) -> bool;
    fn render(&self, state: &UiState, area: ratatui::layout::Rect,
              buf: &mut ratatui::buffer::Buffer);
    fn handle(&mut self, event: &InputEvent, state: &UiState) -> ViewAction;
    fn nav_geometry(&self) -> NavGeometry;
}

pub enum NavGeometry { List, Grid { cols: u32 }, Tree, Graph }

pub enum ViewAction {
    None,
    StateUpdate(Box<dyn FnOnce(&mut UiState)>),
    ProtocolRequest(serde_json::Value),
    ModeSwitch(Mode),
}
```

- [ ] **Step 2: Implement view registry** — maps `ViewId` to `Box<dyn View>`, handles creation/destruction
- [ ] **Step 3: Write tests** — registry add/remove, trait object dispatch
- [ ] **Step 4: Commit**

---

#### 4.3.2 — Tower view

**Files:**
- Create: `crates/rocket-surgeon-tui/src/view/tower.rs`

- [ ] **Step 1: Write failing test** — tower renders layer list with residual norm sparklines
- [ ] **Step 2: Implement** — one line per layer (collapsed), expandable to show components. Cursor navigation (j/k moves between components, `zo`/`zc` expands/collapses). Highlighted cursor line. Color: residual norm mapped to viridis palette.
- [ ] **Step 3: Write failing test** — cursor movement updates `CursorState` in `ViewAction`
- [ ] **Step 4: Implement** — `handle()` maps navigation events to cursor state changes
- [ ] **Step 5: Write failing test** — tower dirty when tick position or layer changes
- [ ] **Step 6: Implement** — `data_deps` returns `[DataDep::TickPosition, DataDep::TensorCache]`
- [ ] **Step 7: Run tests, verify green**
- [ ] **Step 8: Commit**

---

#### 4.3.3 — Inspector view

**Files:**
- Create: `crates/rocket-surgeon-tui/src/view/inspector.rs`

- [ ] **Step 1: Write failing test** — inspector renders tensor stats for current cursor target
- [ ] **Step 2: Implement** — displays: mean, std, min, max, abs_max, sparsity, l2_norm. Histogram rendered as inline sparkline. Top-K values listed. Updates when Tower cursor moves.
- [ ] **Step 3: Write failing test** — inspector triggers `inspect` request when cache miss
- [ ] **Step 4: Implement** — `handle()` checks tensor cache, emits `ProtocolRequest` for `rocket/inspect` if needed
- [ ] **Step 5: Write failing test** — attention pattern sub-view when target is attention output
- [ ] **Step 6: Implement** — conditional rendering: if component is `attn.*`, show per-head attention summary
- [ ] **Step 7: Run tests, verify green**
- [ ] **Step 8: Commit**

---

#### 4.3.4 — Token Axis strip

**Files:**
- Create: `crates/rocket-surgeon-tui/src/view/token_axis.rs`

- [ ] **Step 1: Write failing test** — token axis renders at correct LOD for available width
- [ ] **Step 2: Implement** — 5-level LOD selection based on `Rect` width and token count. L0: full text_repr. L4: color-only columns. Scrollable viewport centered on cursor token.
- [ ] **Step 3: Write failing test** — special characters render with escape glyphs
- [ ] **Step 4: Implement** — whitespace → visible glyphs (·, →, ↵). Control chars → caret notation. Byte-fallback → hex.
- [ ] **Step 5: Write failing test** — navigation selects token position
- [ ] **Step 6: Implement** — left/right moves token cursor, updates `CursorState.token_position`
- [ ] **Step 7: Run tests, verify green**
- [ ] **Step 8: Commit**

---

#### 4.3.5 — Command Bar

**Files:**
- Create: `crates/rocket-surgeon-tui/src/view/command_bar.rs`
- Create: `crates/rocket-surgeon-tui/src/command/grammar.rs`
- Create: `crates/rocket-surgeon-tui/src/command/completion.rs`

- [ ] **Step 1: Write failing test** — grammar parses `L 12 AT` into layer 12 attention
- [ ] **Step 2: Implement grammar parser** — recursive descent or nom for the EBNF in spec §3.4. `context_sel? function arg* flag*`
- [ ] **Step 3: Write failing test** — tab completion suggests valid completions from component vocabulary
- [ ] **Step 4: Implement completion** — prefix match against component vocabulary + command names. Uses `AttachResponse.component_vocabulary` (loaded at connect time).
- [ ] **Step 5: Write failing test** — command bar renders context in Normal mode, input line in Command mode
- [ ] **Step 6: Implement** — dual-mode rendering. Normal: status line showing position, model, branch, VRAM. Command: text input with cursor, inline completions, error display.
- [ ] **Step 7: Write failing test** — `:t /pattern/` command maps to `view.focus` with `by_regex`
- [ ] **Step 8: Implement** — command → protocol verb mapping for all grammar productions
- [ ] **Step 9: Run tests, verify green**
- [ ] **Step 10: Commit**

---

#### 4.3.6 — Distribution view

**Files:**
- Create: `crates/rocket-surgeon-tui/src/view/distribution.rs`

- [ ] **Step 1: Write failing test** — distribution renders residual norm per layer as bar chart
- [ ] **Step 2: Implement** — horizontal bars, viridis-colored, one per layer. Current cursor layer highlighted. Click/navigate to jump Tower cursor.
- [ ] **Step 3: Write failing test** — logit lens predictions displayed when available
- [ ] **Step 4: Implement** — if `LogitLens` view data available, show top-1 prediction per layer alongside norm bars.
- [ ] **Step 5: Run tests, verify green**
- [ ] **Step 6: Commit**

---

### Phase 4.4 — Extended Views

**Depends on:** 4.3 (core views), 4.2 (librocket_viz for graphical rendering)

---

#### 4.4.1 — Timeline view

**Files:**
- Create: `crates/rocket-surgeon-tui/src/view/timeline.rs`

- [ ] **Step 1: Write failing test** — timeline renders per-token sparkline of residual norms
- [ ] **Step 2: Implement** — horizontal axis = token positions. Downsampled via LTTB (from librocket_viz) when token count exceeds columns. Probe firing markers as colored dots. Current token highlighted.
- [ ] **Step 3: Write failing test** — horizontal navigation moves token cursor
- [ ] **Step 4: Implement** — left/right moves token_position in CursorState
- [ ] **Step 5: Write failing test** — speculative decoding overlay renders candidate tokens
- [ ] **Step 6: Implement** — subscribe to `spec.step` events. Render speculative candidate tokens as ghost glyphs (dimmed, palette index 240) ahead of the committed position. Accepted candidates solidify; rejected candidates flash and disappear.
- [ ] **Step 7: Run tests, verify green**
- [ ] **Step 8: Commit**

---

#### 4.4.2 — KV Ribbon view

**Files:**
- Create: `crates/rocket-surgeon-tui/src/view/kv_ribbon.rs`

- [ ] **Step 1: Write failing test** — ribbon renders position × layer heatmap
- [ ] **Step 2: Implement** — graphical rendering via librocket_viz colormap. Pixel buffer → Kitty/Sixel widget. Default metric: L2 norm of K vectors. Downsampled when positions exceed pixels.
- [ ] **Step 3: Write failing test** — overlays render with correct priority stacking
- [ ] **Step 4: Implement** — overlay glyphs (★, ◆, ·, ≈, │, ⌐) rendered on top of heatmap based on `KvOverlay` data from `kv.read` responses.
- [ ] **Step 5: Write failing test** — K/V toggle and physical/logical head toggle
- [ ] **Step 6: Implement** — keyboard shortcuts to toggle `KvSlot` and GQA view mode. Re-requests data with updated params.
- [ ] **Step 7: Write failing test** — subscribes to `kv.update` / `kv.evicted` events
- [ ] **Step 8: Implement** — subscription management, state updates on events
- [ ] **Step 9: Run tests, verify green**
- [ ] **Step 10: Commit**

---

#### 4.4.3 — Worldline view

**Depends on:** Phase 3.13+ (branch verbs), 4.5 (Sugiyama layout)

**Files:**
- Create: `crates/rocket-surgeon-tui/src/view/worldline.rs`

- [ ] **Step 1: Write failing test** — worldline renders branch DAG with nodes and edges
- [ ] **Step 2: Implement** — build `rsviz_dag_t` from branch graph state. Run Sugiyama layout via FFI. Render pixel buffer via Kitty/Sixel widget. Nodes show checkpoint ID + tier badge. Edges annotated with intervention summaries.
- [ ] **Step 3: Write failing test** — divergence sparklines on edges
- [ ] **Step 4: Implement** — `branch.compare` data rendered as sparklines along edges
- [ ] **Step 5: Write failing test** — resource badges show tier and VRAM cost
- [ ] **Step 6: Implement** — color-coded tier badges (green=live, yellow=spilled, gray=dropped) with size in MB
- [ ] **Step 7: Write failing test** — navigation: fork, drop, compare, restore from worldline view
- [ ] **Step 8: Implement** — `handle()` maps navigation events to `branch.fork`, `branch.drop`, `branch.compare`, checkpoint restore protocol requests
- [ ] **Step 9: Run tests, verify green**
- [ ] **Step 10: Commit**

---

### Phase 4.5 — Sugiyama Layout (C, parallel after 4.2.1)

**Goal:** Implement the four-phase Sugiyama algorithm in C. Required by Worldline view.

**Reference:** Sky-Claude Volume IV §B (full pseudocode).

---

#### 4.5.1 — DAG data structure and cycle removal

**Files:**
- Create: `librocket_viz/src/dag.c`
- Create: `librocket_viz/src/sugiyama.c`
- Create: `librocket_viz/tests/test_sugiyama.c`

- [ ] **Step 1: Implement `rsviz_dag_t`** — CSR adjacency. `rsviz_dag_new(n_nodes)`, `rsviz_dag_add_edge(dag, from, to)`, `rsviz_dag_finalize(dag)`, `rsviz_dag_free(dag)`.
- [ ] **Step 2: Implement cycle removal** — guarded DFS. Reverse back-edges to make DAG acyclic. `rsviz_sugiyama_remove_cycles(dag)`.
- [ ] **Step 3: Write C tests** — acyclic graph unchanged, single cycle broken, multi-cycle
- [ ] **Step 4: Commit**

---

#### 4.5.2 — Layer assignment

**Files:**
- Modify: `librocket_viz/src/sugiyama.c`

- [ ] **Step 1: Implement layer assignment** — `rsviz_sugiyama_assign_layers(dag, method)`. Methods: `RSVIZ_LAYER_NATURAL` (topological order), `RSVIZ_LAYER_LONGEST_PATH`, `RSVIZ_LAYER_COFFMAN_GRAHAM`.
- [ ] **Step 2: Implement virtual node insertion** — edges spanning >1 layer get virtual nodes for proper routing.
- [ ] **Step 3: Write C tests** — linear chain, diamond, known layer counts
- [ ] **Step 4: Commit**

---

#### 4.5.3 — Crossing minimization

**Files:**
- Modify: `librocket_viz/src/sugiyama.c`

- [ ] **Step 1: Implement barycenter + median heuristic** — `rsviz_sugiyama_minimize_crossings(dag, max_sweeps)`. 24 sweeps default. Early termination at <1% improvement per sweep.
- [ ] **Step 2: Write C tests** — known crossing count for small graphs
- [ ] **Step 3: Commit**

---

#### 4.5.4 — Coordinate assignment (Brandes-Kopf)

**Files:**
- Modify: `librocket_viz/src/sugiyama.c`
- Create: `librocket_viz/src/brandes_kopf.c`

- [ ] **Step 1: Implement Brandes-Kopf 2001 algorithm** with the 2020 erratum corrections (Algorithm 3b, both flaw corrections from Gronemann & Jünger).
- [ ] **Step 2: Implement templated transformer layout** — solve one ~30-node block (embed → ln → attn → mlp → resid per layer), replicate ×N_layers. Only compute the template once.
- [ ] **Step 3: Write C tests** — coordinate output for known small graphs, template replication
- [ ] **Step 4: Write Rust integration test** — end-to-end Sugiyama on a 4-layer model DAG
- [ ] **Step 5: Commit**

---

#### 4.5.5 — Reingold-Tilford tree layout

**Files:**
- Create: `librocket_viz/src/tree_layout.c`

- [ ] **Step 1: Implement Reingold-Tilford** — for tree-shaped branching subgraphs (simpler than Sugiyama).
- [ ] **Step 2: Write C tests** — balanced binary tree, deep linear chain, single node
- [ ] **Step 3: Commit**

---

### Phase 4.6 — LLM Surface (parallel track)

**Goal:** Implement daemon-side handlers for LLM-facing verbs. Can proceed in parallel
with TUI work since these are protocol handlers, not UI.

**Depends on:** 4.0 (types exist), relevant daemon infrastructure (attach, step, inspect)

---

#### 4.6.1 — rocket/discover handler

**Files:**
- Modify: `crates/rocket-surgeon/src/dispatch.rs`
- Create: `crates/rocket-surgeon/src/discover.rs`

- [ ] **Step 1: Write TCK test** — discover with wildcard returns matches
- [ ] **Step 2: Implement** — pattern matching against component vocabulary (already available from `HostAttachResponse.component_vocabulary`). Edit-distance suggestions for near-misses using a simple Levenshtein function.
- [ ] **Step 3: Run TCK, verify green**
- [ ] **Step 4: Commit**

---

#### 4.6.2 — view.focus handler

**Files:**
- Modify: `crates/rocket-surgeon/src/dispatch.rs`

- [ ] **Step 1: Write TCK test** — focus by_position returns token + per-layer summaries
- [ ] **Step 2: Implement** — resolve selector to position, gather token data + run inspect at that position for all layers with `detail: Summary`.
- [ ] **Step 3: Write TCK test** — focus by_regex finds first matching token
- [ ] **Step 4: Implement** — regex match against token text_repr. Requires token data from worker.
- [ ] **Step 5: Run TCK, verify green**
- [ ] **Step 6: Commit**

---

#### 4.6.3 — Structural observations on tick.stopped

**Files:**
- Modify: `crates/rocket-surgeon/src/notifications.rs`
- Create: `crates/rocket-surgeon/src/observations.rs`

- [ ] **Step 1: Write TCK test** — tick.stopped event carries observations array
- [ ] **Step 2: Define Observation type**

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub kind: ObservationKind,
    pub layer: u32,
    pub component: String,
    pub message: String,
    pub value: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationKind {
    NormAnomaly,
    AttentionConcentration,
    LogitLensDelta,
    SparsityShift,
    SinkDetection,
}
```

- [ ] **Step 3: Implement** — after each tick, compute cheap metrics from captured tensor summaries. Flag anomalies > 2σ from running statistics. Append to `TickStoppedEvent`.
- [ ] **Step 4: Run TCK, verify green**
- [ ] **Step 5: Commit**

---

#### 4.6.4 — rocket/sweep handler

**Files:**
- Modify: `crates/rocket-surgeon/src/dispatch.rs`
- Create: `crates/rocket-surgeon/src/sweep.rs`

- [ ] **Step 1: Write TCK test** — sweep runs 2 trials and returns results
- [ ] **Step 2: Implement** — iterate trials: restore checkpoint, apply interventions, step to run_to/completion, collect inspections at collect points, compute metric. Emit `sweep.trial_complete` event per trial.
- [ ] **Step 3: Write TCK test** — sweep with invalid checkpoint returns error
- [ ] **Step 4: Implement** — error handling with recovery hints
- [ ] **Step 5: Run TCK, verify green**
- [ ] **Step 6: Commit**

---

#### 4.6.5 — rocket/view.define handler

**Files:**
- Modify: `crates/rocket-surgeon/src/dispatch.rs`

- [ ] **Step 1: Write TCK test** — define a view, then request it via rocket/view
- [ ] **Step 2: Implement** — store named view definitions in session state. When `rocket/view` is called with a user-defined view name, execute the spec (composition of inspect calls + metric computation) and return results.
- [ ] **Step 3: Run TCK, verify green**
- [ ] **Step 4: Commit**

---

### Phase 4.7 — Integration and Dogfood

**Depends on:** 4.3 (core views), 4.4 (extended views at least partially), 4.6 (LLM surface)

---

#### 4.7.1 — End-to-end integration

**Files:**
- Modify: `crates/rocket-surgeon-tui/src/main.rs`
- Create: `tests/test_e2e_tui.py`

- [ ] **Step 1: Wire all views into the registry** — Tower, Inspector, Token Axis, Command Bar, Distribution, Timeline.
- [ ] **Step 2: Wire tiling manager defaults** — default layout: Tower + Inspector (HSplit). Command Bar always at bottom. Token Axis always as strip above Command Bar.
- [ ] **Step 3: Write e2e test** — launch daemon + TUI, attach model, step, verify TUI renders without crash.
- [ ] **Step 4: Manual test** — full debug session against Llama-3-8B (or GPT-2-small for CI). Navigate Tower, inspect tensors, switch modes, use command bar.
- [ ] **Step 5: Commit**

---

#### 4.7.2 — MIDI controller support (proof of concept)

**Files:**
- Create: `crates/rocket-surgeon-tui/src/input/midi.rs`
- Create: `crates/rocket-surgeon-tui/src/output/midi.rs`

- [ ] **Step 1: Implement MidiInput** — `InputSource` trait impl using `midir` crate. Maps MIDI note-on to `NavigationEvent::JumpTo`, CC to `ContinuousAdjust`, pad velocity to mode events.
- [ ] **Step 2: Define default mapping file format** — TOML mapping `(channel, cc/note) → abstract_event`. Load at startup.
- [ ] **Step 3: Implement MidiOutput** — state-reactive MIDI output. TUI state changes drive note/CC messages back to the controller: mode indicator → pad LED colors, cursor layer → CC value, active view → button LED state. Uses the same TOML mapping in reverse. Architecture must not preclude haptic feedback output in a future iteration.
- [ ] **Step 4: Manual test** — if MIDI controller available, verify bidirectional: navigation from pads, LED feedback on state changes. If not, verify graceful fallback (both MidiInput and MidiOutput return None when no device).
- [ ] **Step 5: Commit**

---

#### 4.7.3 — Dogfood session

Not a code task — a structured test session. Use the TUI to drive a real mechanistic
interpretability investigation against Llama-3-8B. Document findings as protocol
refinements in a bead.

- [ ] **Step 1: Run IOI experiment** — identify attention heads responsible for indirect object identification using Tower + Inspector + interventions.
- [ ] **Step 2: File BEAD** — document any protocol gaps, UX issues, or missing features discovered during dogfooding.
- [ ] **Step 3: Triage findings** — which go into Phase 4 patches vs. later phases.

---

### Phase 4 exit criteria

- [ ] TUI launches and connects to daemon via Unix socket
- [ ] Tower + Inspector render correctly and react to cursor movement
- [ ] Command Bar parses Bloomberg grammar and executes protocol commands
- [ ] Token Axis renders at 5 LOD levels
- [ ] Distribution view shows per-layer residual norms and logit lens
- [ ] Timeline view renders per-token sparklines with speculative decoding overlay
- [ ] KV Ribbon renders position × layer heatmap with overlays (requires Phase 3 KV work)
- [ ] Worldline renders branch DAG via Sugiyama (requires Phase 3 branching work)
- [ ] librocket_viz colormap, downsampling, and encoding kernels pass SIMD tests
- [ ] Sugiyama layout produces correct coordinates for transformer model DAGs
- [ ] LLM surface: `rocket/discover`, `view.focus`, `rocket/sweep`, `rocket/view.define` all TCK green
- [ ] Structural observations appear on `tick.stopped` events
- [ ] Mode switching (Normal/Command/Inspect/Intervene) works
- [ ] Animated transitions interpolate on cursor movement (not snapping)
- [ ] Prefetch heuristics keep one-hop adjacency in cache
- [ ] Degradation ladder: Tier 1 (Kitty) and Tier 3 (half-block) both render
- [ ] MIDI proof of concept: bidirectional (input + LED/state output) with at least one controller
- [ ] Dogfood session completed, findings filed as bead
- [ ] Protocol v0.3.0 — all new types, verbs, events, errors TCK green
- [ ] All Phase 4 TCK scenarios green
