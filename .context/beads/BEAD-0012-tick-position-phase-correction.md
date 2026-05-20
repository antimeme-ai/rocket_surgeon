# BEAD-0012: TickPosition Phase Correction

**Status:** closed
**Created:** 2026-05-20
**Closed:** 2026-05-20
**Branch:** feat/tick-position-phase-correction

## Problem

TickPosition modeled tick_id (the operator clock) but had no concept of
inference phase or token position. The sky-claude Volume III analysis
identified three distinct clocks: tick_token (token generation),
tick_operator (hook firings), and tick_wall (wall clock). Without phase
and token_position, the protocol cannot distinguish prefill from decode,
cannot track chunked prefill regimes (Sarathi-Serve), and cannot anchor
ticks to their position in the sequence.

## Solution

Added two fields to TickPosition:
- `phase: Phase` — enum with Prefill, Decode, PrefillChunked{chunk_size, chunk_index, total_chunks}
- `token_position: Option<u64>` — position in the token sequence

Phase enum uses internally-tagged serde (`#[serde(tag = "type")]`) for
LLM client ergonomics. Derives Copy + Default (Decode).

Forward-compatible: old 0.1.0 JSON without phase/token_position
deserializes correctly (phase defaults to Decode, token_position to None).

Protocol version bumped 0.1.0 → 0.2.0.

## Files changed

- `crates/rocket-surgeon-protocol/src/types.rs` — Phase enum, TickPosition fields
- `crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs` — 8 new tests
- `crates/rocket-surgeon-protocol/src/messages.rs` — test update
- `crates/rocket-surgeon-worker/src/tick.rs` — TickState gains phase/token_position
- `crates/rocket-surgeon/src/dispatch.rs` — construction site + version bump
- `crates/rocket-surgeon/src/main.rs` — default_position update
- `crates/rocket-surgeon/src/perfetto_sink.rs` — test helper update
- `crates/rocket-surgeon/src/session.rs` — PROTOCOL_VERSION const + all test sites

## Artifacts

- Plan: `docs/specs/2026-05-20-tick-position-correction-plan.md`
- TCK: `tck/protocol/tick-position-phase.feature`
