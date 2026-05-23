# TUI slice 5a — LayerStack component

**Bead:** BEAD-0015 slice 5 (first of six per-panel components) · **Crate:** `rocket-surgeon-tui`

## Goal

Replace the `View 0` placeholder — the screen's largest panel — with the
real `LayerStack` component: a bordered list of transformer layers with the
current cursor layer highlighted. The first per-panel slice of slice 5; the
five other `ViewKind`s (`TensorDetail`, `ProbeWatch`, `Timeline`, `KvCache`,
`Worldline`) each get their own slice.

## Design

- `LayerStack` — unit struct, stateless. Reads `state.cursor.layer` and
  `state.session.capabilities.as_ref().and_then(|c| c.num_layers)`.
- `draw` builds a ratatui `List` of `0..num_layers` rows; a per-frame
  `ListState` with `selected = cursor.layer` keeps the highlight in view
  (ratatui handles scroll). Without `num_layers` a `Paragraph` hint sits
  in the same bordered block — visual consistency, no flicker on attach.
- Bordered `Block` titled "Layers" in both states.

## Files

| File | Change |
|---|---|
| `components/layer_stack.rs` (new) | `LayerStack` + `impl Component` + tests |
| `components/mod.rs` | declare the new submodule |
| `app.rs` | own `layer_stack: LayerStack`; dispatch `ViewKind::LayerStack`; rename `draw_renders_placeholder_for_layerstack` → `draw_renders_placeholder_for_unmapped_kind` (LayerStack has a component now; the placeholder path is exercised by swapping in `TensorDetail`) |
| `.context/beads/BEAD-0015-...md` | sub-checklist under slice 5 with `LayerStack` ticked |

## Tests (TestBackend)

- `layer_stack_no_caps_shows_hint` — no `capabilities`, render, grep
  "awaiting capabilities".
- `layer_stack_with_caps_lists_layers` — `capabilities.num_layers = 8`,
  render, grep `Layer   0` and `Layer   7`.
- `layer_stack_highlights_cursor_layer` — `num_layers = 8`, `cursor.layer = 3`,
  render, assert the row containing `Layer   3` has the highlight bg style.

## `Component::update` — still deferred

LayerStack is stateless. `ListState` is per-frame, recomputed from
`cursor.layer`. `Component::update` lands when the first component genuinely
needs persistent view-local state (likely `TensorDetail` with selection or
`ProbeWatch` with filters).

## Out of scope

- Capturing the full `Capabilities` from the initialize response. Slice 2
  captures only `protocol_version`, so `num_layers` is `None` at runtime —
  meaning LayerStack today shows the hint even when the link is live. Real
  layer data requires both that capture and a model attached upstream.
- Per-layer component breakdown (mlp/attn rows under each layer).
- Pin / select / zoom interactions.
