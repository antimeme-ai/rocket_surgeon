# Sub-plan: rocket/replay — daemon orchestration tier

**Bead:** BEAD-0018
**Date:** 2026-05-21

## Problem

`rocket/replay` routes to `handle_stub_requires_stopped`. The protocol types
are frozen in the v0.3.0 schema (`ReplayRequest`, `ReplayResponse`,
`ReplayStopAt`, `Divergence`, `ReplayDivergenceEvent`) — only the daemon
handler is missing. This WU ships the orchestration tier: re-seat from a
checkpoint and synthesize a replay result. Worker re-execution is a later tier.

## Design

Mirror `Session::checkpoint_restore` (checkpoint lookup) and `Session::step`
(envelope-aware response, `state` mutation).

### `Session::replay`

```rust
pub fn replay(&mut self, req: &ReplayRequest)
    -> Result<serde_json::Value, SessionError>
```

1. `require_stopped("rocket/replay")`.
2. Look up `req.from_checkpoint` in `state.checkpoints`; on miss return
   `checkpoint_not_found_error` (`CHECKPOINT_NOT_FOUND`).
3. `origin` = the checkpoint's retained `TickPosition` (`checkpoint_positions`),
   with the same forward-only fallback `checkpoint_restore` uses.
4. `current` = `state.position` / `state.tick_id` — the original run's endpoint.
5. `ticks_replayed = max(1, current_tick - origin.tick_id)`.
6. Build `stopped_at`:
   - `tick_id` = `current_tick + ticks_replayed` (fresh, monotonic, `>` current).
   - `replay_of` = `Some(current_tick)` (the original run being replayed).
   - `layer` / `component` = `req.stop_at` if present, else `current`'s.
   - `direction = Forward`, `event = Output`, other fields from `current` or
     defaults.
7. Update `state.tick_id` / `state.position` to `stopped_at`.
8. `ReplayResponse { ticks_replayed, stopped_at, divergences: vec![],
   verified: divergences.is_empty() }`.
9. Return `self.envelope_with_mode(req.envelope, data)`.

Tier 1 synthesizes the result from checkpoint metadata + current position —
the same approach `handle_step` uses when no host is attached. No model is
re-run, so `divergences` is empty and `verified` is vacuously `true`
(`verified` ⟺ no divergences). `req.interventions` is accepted and ignored —
the worker tier applies them.

### `handle_replay` (replaces the stub in `dispatch.rs`)

`parse_params` → `invalid_params_response`; then `session.replay(&req)` →
`serialize_envelope` / `session_error_to_response`. Route `method::REPLAY` to
it. Mirrors `handle_step`.

## Files

- `crates/rocket-surgeon/src/session.rs` — `Session::replay`; unit tests.
- `crates/rocket-surgeon/src/dispatch.rs` — `handle_replay`; route
  `method::REPLAY`; handler tests.

## TCK

`tck/protocol/replay.feature` already exists. Tier-1 scenarios that go green:
happy path (`ticks_replayed`, `stopped_at`), `stop_at`, `verify=true`,
tick-identity (`replay_of`), and `CHECKPOINT_NOT_FOUND`. Mirrored as Rust
handler tests in `dispatch.rs` and `Session::replay` tests in `session.rs`.

## Out of scope

- Worker re-execution, intervention application during replay, real divergence
  detection, the `rocket/replay.divergence` event — the two divergence
  scenarios in `replay.feature`.
