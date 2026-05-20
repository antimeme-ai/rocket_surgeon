# TickPosition Correction — Plan Document

**Date:** 2026-05-20
**Source:** sky-claude Volume III §D, Volume II three-clock model
**Prerequisite for:** Token-axis rendering, KV cache views, worldline branching
**Protocol version:** 0.1.0 → 0.2.0

## Problem

The current `TickPosition` tracks only operator-time (`tick_id` monotonically
counts hook firings). It has no concept of:

1. **Which token** in the sequence is being processed — `tick_id` advances
   identically whether we're in prefill (processing token 0..N in parallel)
   or decode (generating token N+1).
2. **What phase** the forward pass is in — prefill vs decode are fundamentally
   different execution regimes. Sarathi-Serve chunked prefill introduces a
   third regime where a "step" advances by k < seq_len positions.

Without these fields, the token axis (Volume III §C), the KV cache ribbon
(Volume IV §A), and worldline branching (Volume IV §C) have no coordinate
to attach to. This is a non-negotiable correction that ships before any
L>1 rendering work.

## Current State

```rust
// crates/rocket-surgeon-protocol/src/types.rs:53-63
pub struct TickPosition {
    pub tick_id: u64,
    pub direction: StepDirection,
    pub rank: Option<u32>,
    pub layer: u32,
    pub component: String,
    pub event: TickEvent,
    pub replay_of: Option<u64>,
}
```

Protocol version is `"0.1.0"` (types.rs:343, session.rs:45).

### Construction sites (all must be updated)

| File | Function/context | Count |
|------|-----------------|-------|
| `protocol/src/types.rs` | struct definition | 1 |
| `protocol/tests/serde_roundtrip.rs` | `sample_tick_position()` + `tick_position_with_replay` | 2 |
| `worker/src/tick.rs` | `TickState::to_tick_position()` | 1 |
| `rocket-surgeon/src/main.rs` | `default_position()` | 1 |
| `rocket-surgeon/src/session.rs` | 9 test instances | 9 |
| `rocket-surgeon/src/perfetto_sink.rs` | `make_position()` test helper | 1 |

**Total: 15 construction sites.**

## Design

### 1. New `Phase` enum

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Phase {
    Prefill,
    Decode,
    PrefillChunked {
        chunk_size: u32,
        chunk_index: u32,
        total_chunks: u32,
    },
}
```

Internally-tagged (`"type": "prefill"` / `"type": "decode"` /
`"type": "prefill_chunked"`) so LLM clients can pattern-match on it without
positional ambiguity. `PrefillChunked` carries the Sarathi-Serve metadata.

### 2. Updated `TickPosition`

```rust
pub struct TickPosition {
    pub tick_id: u64,
    pub direction: StepDirection,
    pub rank: Option<u32>,
    pub layer: u32,
    pub component: String,
    pub event: TickEvent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_of: Option<u64>,
    // --- new fields ---
    pub phase: Phase,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_position: Option<u64>,
}
```

- `phase` is required (not Optional) — every tick happens in a phase. Phase 1
  code that doesn't know the phase uses `Phase::Decode` as default since the
  existing test harness simulates single-token decode steps.
- `token_position` is `Option<u64>` — in prefill, this is the sequence length
  being processed; in decode, the position of the token being generated; `None`
  only at tick 0 (before any forward pass).

### 3. `TickState` update (worker)

`TickState` in `worker/src/tick.rs` gains `phase: Phase` and
`token_position: Option<u64>` fields. `advance()` takes these as parameters.
`to_tick_position()` threads them through.

### 4. Protocol version bump

`PROTOCOL_VERSION` in session.rs and `phase1_defaults()` in types.rs both
change from `"0.1.0"` to `"0.2.0"`. The initialize handshake already validates
version match — clients sending `"0.1.0"` will get a clear error.

### 5. Forward-compatible deserialization

Add `#[serde(default)]` on `phase` (defaulting to `Phase::Decode`) and
`token_position` (defaulting to `None`) so that JSON from a 0.1.0 client
that omits these fields still deserializes. This is the forward-compat path
for existing test fixtures and any stored JSON.

### 6. Serde roundtrip test additions

New tests in `serde_roundtrip.rs`:
- `tick_position_has_phase_field` — verify `phase` appears in JSON
- `tick_position_phase_prefill_chunked` — verify tagged enum serialization
- `tick_position_token_position_present` — verify token_position serialization
- `tick_position_forward_compat` — deserialize JSON without phase/token_position

## Blast radius

### Files modified

1. `crates/rocket-surgeon-protocol/src/types.rs` — Phase enum, TickPosition fields
2. `crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs` — sample_tick_position + new tests
3. `crates/rocket-surgeon-worker/src/tick.rs` — TickState fields + advance() + to_tick_position()
4. `crates/rocket-surgeon/src/main.rs` — default_position()
5. `crates/rocket-surgeon/src/session.rs` — PROTOCOL_VERSION + 9 test instances
6. `crates/rocket-surgeon/src/perfetto_sink.rs` — make_position() test helper

### Files NOT modified

- `messages.rs` — uses TickPosition by reference, no construction
- `dispatch.rs` — imports only
- `tck/*.feature` — existing Gherkin scenarios don't assert on phase/token_position; they continue to pass

### Risk

Low. This is an additive schema change with serde defaults for backward compat.
No behavioral changes to stepping logic — just richer position metadata.
The protocol version bump is the most visible external change but the version
validation already exists and will correctly reject stale clients.

## Execution order

1. Write TCK `.feature` file for the new fields
2. Add `Phase` enum to types.rs
3. Add `phase` and `token_position` to `TickPosition` with serde defaults
4. Bump protocol version to 0.2.0
5. Update all 15 construction sites
6. Add serde roundtrip tests
7. Run full test suite → green
8. Code review
