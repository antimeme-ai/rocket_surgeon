# Sub-plan — BEAD-0015 slice 3: Component trait + existing panels

**Date:** 2026-05-21
**Branch:** `feat/tui-component-trait`
**ADR:** ADR-0009 · **Spec:** `docs/specs/2026-05-21-tui-architecture-design.md`

## Goal

Introduce a `Component` trait, migrate `StatusBar` and `CommandLine` behind
it, have `App` walk the `Layout` and dispatch `draw`, and dissolve
`render::compositor`. Scope is strictly slice 3 — no `update` method, no
per-panel components beyond the two existing render arms.

## Change set

1. NEW `components/mod.rs` — `Component` trait with `draw` ONLY. Declares
   `status_bar` and `command_line` submodules.
2. NEW `components/status_bar.rs` — `pub struct StatusBar;` (unit, stateless);
   `impl Component` whose `draw` is the old `render_status_bar` body.
3. NEW `components/command_line.rs` — `pub struct CommandLine;` (unit,
   stateless); `impl Component` whose `draw` is the old `render_command_line`.
4. EDIT `app.rs`:
   - `App` owns `status_bar: StatusBar`, `command_line: CommandLine`.
   - `App::draw` walks `self.layout.resolve(frame.area())`; per `(ViewId,
     Rect)` finds the `ViewSlot`, matches `ViewKind`, dispatches to the owning
     component; unmapped kinds -> private `draw_placeholder` (moved from
     `render_placeholder`).
   - Add `CommandLine` slot `ViewId(2)` to `default_views()`.
   - `default_layout()` -> `vsplit(single(0), vsplit(single(1), single(2),
     0.5), 0.92)`.
5. EDIT `render.rs` — drop `pub mod compositor;` (keep `capability`).
6. DELETE `render/compositor.rs`.
7. EDIT `main.rs` — add `mod components;`.
8. EDIT `state.rs` — drop `#[allow(dead_code)]` from `ViewKind::CommandLine`
   (now reachable via the new view slot); fix the scaffolding comment.
   Re-check other allows in touched files only.

## Tests — written first (red)

- `components/status_bar.rs`: `status_bar_shows_mode` (TestBackend, asserts
  "Normal" in the buffer).
- `components/command_line.rs`: draw-without-panic test.
- `app.rs`: `draw_renders_without_panic`; `draw_dispatches_all_default_views`
  (3 default views resolve + render); `draw_renders_placeholder_for_layerstack`.
- All existing `app.rs` tests stay green.

## JSMNTL cycle

1. Plan (this file).
2. Write tests, confirm red.
3. Implement change set.
4. `cargo test -p rocket-surgeon-tui` green; `cargo fmt --all`;
   `cargo clippy -p rocket-surgeon-tui --all-targets -- -D warnings` clean.
5. Code-review subagent; fix all findings; repeat until clean.
6. Atomic commit on `feat/tui-component-trait`.
7. Mark slice 3 DONE in BEAD-0015.
8. Push; `gh pr create` base `master`. Do not merge.
