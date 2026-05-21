---
id: BEAD-0017
title: rocket/intervene — daemon intervention registry (set/clear/list)
status: open
priority: high
created: 2026-05-21
---

## Description

`rocket/intervene` is currently a stub (`handle_stub_requires_stopped`). This
WU implements the **daemon registry tier**: the verb becomes fully functional
per the protocol — clients can set, clear, and list declarative intervention
recipes, which the daemon validates and retains in session state. Actually
applying interventions during the forward pass is a later tier (worker).

Mirrors the WU-C checkpoint tiering: ship the verb fully per-protocol, with the
heavy execution path as a defined later tier. Intervention is the keystone
surgical capability and unblocks WU-D replay (replay applies interventions).

## Scope — Tier 1 (this WU, daemon registry)

- `handle_intervene` replacing the stub: `Set` / `Clear` / `List` actions.
- Session-side intervention registry (insert-or-replace by `id`, clear-by-id,
  list). Persists across steps — `rocket/step` does not clear it.
- Validation:
  - `INVALID_STATE` — require stopped.
  - `INVALID_TARGET` — the recipe `target`'s component leaf must be canonical
    (reuse `handle_inspect`'s component validation).
  - `INVALID_RECIPE` — `intervention_type` must match the `params` variant
    (`InterventionParams` is `#[serde(untagged)]`, so e.g. `type: scale` with
    `params: {}` deserializes as `Ablate` params — the type/params agreement
    must be checked explicitly).
- `InterveneResponse { active_interventions, applied }`.
- Recipe-type deserialization coverage (ablate/scale/add/patch/clamp +
  attention_mask/embed_swap/embed_noise, `AblateMode`) as protocol unit tests.
- TCK: `intervention.feature` set/clear/list/persistence/error scenarios green.

## Out of scope — Tier 2 (later WU, worker application)

- Worker applies interventions during the forward pass via hooks.
- Composition / priority-order execution; `CompositionMode` replace vs additive.
- The `intervention.feature` scenarios that assert via `rocket/step`
  ("executes before", "takes effect at point").
- Target validation beyond the component leaf. The daemon checks only that the
  `target`'s component is canonical (mirroring `handle_inspect`); layer-index
  bounds and rank validity are the worker tier's responsibility — so e.g.
  `llama:0:999:o_proj:output` is accepted by the registry.

## Unblocks

- WU-D replay — `rocket/replay` applies interventions during re-execution.
