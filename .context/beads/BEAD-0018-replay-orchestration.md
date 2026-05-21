---
id: BEAD-0018
title: rocket/replay — daemon orchestration tier (replay from checkpoint)
status: open
priority: high
created: 2026-05-21
---

## Description

`rocket/replay` is a stub (`handle_stub_requires_stopped`). This WU implements
the **daemon orchestration tier**: the verb re-seats from a checkpoint and
synthesizes a replay result — `ticks_replayed`, a `stopped_at` position with a
fresh tick_id and `replay_of` set, honoring `stop_at` and the response
envelope. Actually re-executing the forward pass and detecting divergence is a
deferred worker tier.

Mirrors WU-C checkpoint (metadata tier) and BEAD-0017 intervene (registry
tier): ship the verb fully per-protocol; heavy execution as a later tier.
Unblocked by BEAD-0017 — a `ReplayRequest` carries `interventions`.

## Scope — Tier 1 (this WU, daemon orchestration)

- `handle_replay` replacing the stub; `Session::replay`.
- Validate `from_checkpoint` exists → `CHECKPOINT_NOT_FOUND` (reuse
  `checkpoint_not_found_error`).
- Re-seat from the checkpoint's retained `TickPosition`; mint a `stopped_at`
  with a fresh tick_id (`>` current) and `replay_of` referencing the original
  run; update `state.tick_id` / `state.position`.
- Honor `stop_at` (drives `stopped_at.layer` / `component`).
- Honor `envelope` via `Session::envelope_with_mode` — this closes BEAD-0013
  acceptance criterion 2 ("`rocket/replay` honors envelope once implemented").
- `ReplayResponse { ticks_replayed, stopped_at, divergences: [], verified }`,
  with `verified = divergences.is_empty()` (vacuously `true` in Tier 1).
- TCK: `replay.feature` happy-path, `stop_at`, `verify`, tick-identity
  (`replay_of`), and `CHECKPOINT_NOT_FOUND` scenarios green.

## Out of scope — Tier 2 (worker)

- Worker restores the checkpoint and re-runs the forward pass.
- Applying the request's `interventions` during replay.
- Real divergence detection (cosine similarity / max relative error) and the
  `rocket/replay.divergence` notification event.
- `replay.feature`'s two divergence scenarios.
