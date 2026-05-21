# Sub-plan: rocket/intervene — daemon intervention registry

**Bead:** BEAD-0017
**Date:** 2026-05-21

## Problem

`rocket/intervene` routes to `handle_stub_requires_stopped`. The protocol types
are already frozen in the v0.3.0 schema (`InterveneRequest::{Set,Clear,List}`,
`InterveneResponse`, `InterventionRecipe`, `InterventionParams`) — only the
daemon handler is missing. This WU ships the registry tier: set/clear/list a
validated recipe registry. Worker-side application is a later tier.

## Design

Mirror the existing daemon-side registry verbs (`handle_subscribe` storing a
filter, the probe registry, the checkpoint registry).

### Intervention registry in `Session`

`Session` gains an ordered intervention registry keyed by recipe `id`:
- `set_intervention(recipe)` — insert or replace by `id`.
- `clear_intervention(&id)` — remove; returns whether it existed.
- `interventions()` — list in insertion order, for the response and for a
  future worker tier to consult.

Recipes persist across `rocket/step` (the step path does not touch the
registry). Cleared on `detach`, like other session-scoped state.

### `handle_intervene` (replaces the stub in `dispatch.rs`)

1. Parse `InterveneRequest` (`#[serde(tag = "action")]` → Set/Clear/List).
2. `require_stopped("rocket/intervene")` → `INVALID_STATE` on failure.
3. `Set`: validate the recipe, then `set_intervention`.
   - `Clear`: `clear_intervention`.
   - `List`: no mutation.
4. Respond `InterveneResponse { active_interventions, applied }` —
   `applied = Some(true)` for `Set`, `None` for `Clear`/`List`.

### Validation (the `Set` path)

- **`INVALID_TARGET`** — reuse `handle_inspect`'s component check:
  `target_component` / `component_leaf` / `CANONICAL_COMPONENTS`. A non-canonical
  leaf is a caller mistake.
- **`INVALID_RECIPE`** — `InterventionParams` is `#[serde(untagged)]`, so a
  type/params mismatch deserializes silently (e.g. `type: scale`, `params: {}`
  → `InterventionParams::Ablate`). A `recipe_type_matches_params(recipe)` check
  rejects the mismatch with `INVALID_RECIPE`. Also rejects a missing/empty `id`.

## Files

- `crates/rocket-surgeon/src/session.rs` — intervention registry + methods,
  cleared on detach; unit tests.
- `crates/rocket-surgeon/src/dispatch.rs` — `handle_intervene`; route
  `method::INTERVENE` to it (leave `REPLAY` on the stub); handler tests.
- `crates/rocket-surgeon-protocol/tests/` (or `messages.rs` tests) —
  recipe-type deserialization coverage.

## TCK

`tck/protocol/intervention.feature` already exists. Tier-1 scenarios that go
green: the five `set` types, `clear`, `list`, persistence-across-steps, and the
three error cases (`INVALID_TARGET`, `INVALID_RECIPE`, `INVALID_STATE`). The
"Extended activation patching" scenarios are pure deserialization checks.

Mirror these as Rust tests (handler tests in `dispatch.rs`, registry tests in
`session.rs`), per the project's TCK-as-Rust-tests practice.

## Out of scope

- Worker application of interventions during the forward pass.
- Composition / priority-order execution and `CompositionMode` semantics —
  the `intervention.feature` scenarios that assert via `rocket/step`.
