---
id: BEAD-0013
title: EnvelopeMode is only honored by rocket/step
status: closed
priority: low
created: 2026-05-20
closed: 2026-05-21
branch: feat/envelope-inspect-view
---

## Description

WU-A wired `EnvelopeMode` (Full / Position / None response compactness) into
`rocket/step` only. `InspectRequest`, `ViewRequest`, and `ReplayRequest` all
carry an `envelope: EnvelopeMode` field in the frozen v0.3.0 schema
(`crates/rocket-surgeon-protocol/src/messages.rs` ~186 / ~293 / ~394), but
their daemon handlers ignore it and always emit the full `SessionState`
envelope.

Consequence: a client sending `rocket/inspect` with `envelope: "none"` (or
`"position"`) silently receives the full envelope anyway.

Not a TCK failure — `tck/protocol/envelope-compactness.feature` only scripts
`step` scenarios — so WU-A satisfies the spec. This is a protocol-completeness
gap, deferred deliberately to keep WU-A focused.

## Why deferred

`handle_inspect` shares a response path (`ingest_and_respond`) with a caller
that WU-A's agent did not own; routing `inspect`/`view` responses through
`Session::envelope_with_mode` needs that path untangled first. Small, but its
own change.

## Acceptance criteria

- `rocket/inspect` and `rocket/view` responses honor the request's `envelope`
  field via `Session::envelope_with_mode`.
- `rocket/replay` honors it once replay is implemented (WU-D — replay is
  currently a stub).
- Extend `envelope-compactness.feature` with `inspect`/`view` scenarios.

## Resolution (2026-05-21)

`inspect` and `view` now honor `EnvelopeMode`, mirroring the existing
`rocket/step` path:

- `Session::inspect` gains a `mode: EnvelopeMode` parameter and returns
  `serde_json::Value` via `Session::envelope_with_mode` — structurally
  identical to `Session::step`.
- `ingest_and_respond` passes `req.envelope` into `Session::inspect`.
- `handle_view` routes its `ViewResponse` through
  `Session::envelope_with_mode(req.envelope, resp)` (previously discarded the
  parsed request as `_req`).
- `envelope-compactness.feature` extended with 6 scenarios (inspect/view ×
  full/position/none); mirrored by 7 `dispatch.rs` handler tests.

Error responses still ignore `EnvelopeMode` — they are JSON-RPC error objects,
not `SessionState` envelopes, so compaction is structurally inapplicable. This
matches `rocket/step`.

`rocket/replay` is unchanged: still a stub, its `envelope` field remains
deferred to WU-D (acceptance criterion 2 — explicitly WU-D scope, not a
blocker for closing this bead).

Plan: `docs/specs/2026-05-21-envelope-inspect-view-plan.md`
