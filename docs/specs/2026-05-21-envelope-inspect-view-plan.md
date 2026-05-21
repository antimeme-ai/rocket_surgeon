# Sub-plan: EnvelopeMode for rocket/inspect and rocket/view

**Bead:** BEAD-0013
**Date:** 2026-05-21

## Problem

WU-A wired `EnvelopeMode` (Full / Position / None response compactness) into
`rocket/step` only. `InspectRequest` and `ViewRequest` both carry an
`envelope: EnvelopeMode` field in the frozen v0.3.0 schema, but their daemon
handlers ignore it and always emit the full `SessionState` envelope. A client
sending `rocket/inspect` with `envelope: "none"` silently receives the full
envelope anyway.

`rocket/replay` also carries the field, but replay is a stub deferred to WU-D;
honoring its envelope is out of scope here (BEAD-0013 acceptance criterion 3).

## Design

Mirror the existing `rocket/step` path exactly. `Session::step` returns
`serde_json::Value` and routes its response through
`Session::envelope_with_mode(req.envelope, data)`. `inspect` becomes the same
shape; `view` (which has no `Session` method — its response is built in
`dispatch.rs`) calls `envelope_with_mode` directly.

### Changes

1. `Session::inspect` (`crates/rocket-surgeon/src/session.rs`)
   - Add a `mode: EnvelopeMode` parameter.
   - Return `Result<serde_json::Value, SessionError>` instead of
     `Result<ResponseEnvelope<InspectResponse>, SessionError>`.
   - Body builds `InspectResponse` then returns
     `Ok(self.envelope_with_mode(mode, data))`.

2. `ingest_and_respond` (`crates/rocket-surgeon/src/dispatch.rs`)
   - Pass `req.envelope` into `session.inspect(...)`. The `Ok` arm already
     flows through `serialize_envelope`, which is generic over `Serialize`,
     so a `Value` payload needs no further change.

3. `handle_view` (`crates/rocket-surgeon/src/dispatch.rs`)
   - Stop discarding the parsed request (`let _req` -> `let req`).
   - Replace `session.envelope(resp)` with
     `session.envelope_with_mode(req.envelope, resp)`.

### Tests broken by the signature change

`session.rs` unit tests calling `Session::inspect` use the typed envelope
(`.state`, `.data`). Convert them to `serde_json::Value` indexing and pass an
explicit `EnvelopeMode`, exactly as the existing `Session::step` tests already
do: `inspect_from_stopped_succeeds`, `inspect_from_initialized_returns_error`,
`inspect_with_slice_data`, `inspect_does_not_change_session_state`.

## TCK

Extend `tck/protocol/envelope-compactness.feature` with inspect and view
scenarios for default (full), position, and none envelopes — mirroring the
three existing `step` scenarios.

New Rust tests live at the handler level in `dispatch.rs` (that is where
`req.envelope` is read): `handle_inspect` and `handle_view` for each of the
three modes.

## Out of scope

- `rocket/replay` envelope honoring — deferred to WU-D (replay is a stub).
