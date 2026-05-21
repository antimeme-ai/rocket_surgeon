---
id: BEAD-0015
title: Rebuild rocket-surgeon-tui on the Component template architecture (ADR-0009)
status: open
priority: medium
created: 2026-05-21
---

## Description

`rocket-surgeon-tui` was built without anchoring on a reference pattern: a
synchronous `main.rs` loop that only handles terminal input, sitting beside a
fully async (tokio) protocol client that nothing connects to, plus a bespoke
per-view dirty-tracking system built twice. ADR-0009 chose the target: the
Ratatui **Component template** pattern — a TEA spine (single `UiState` store,
one `Action` enum, unidirectional flow) with `Component` decomposition for the
view panels, on an owned `tokio` event loop.

Full blueprint: `docs/specs/2026-05-21-tui-architecture-design.md`.

## Why deferred

The decision (ADR-0009) and blueprint are recorded; the rework itself is its
own work. It must be delivered as incremental vertical slices, not a big-bang
rewrite — each slice compiles, tests, and leaves the crate runnable.

## Acceptance criteria

Deliver as separate slices (each a candidate sub-bead):

1. **Tokio loop, terminal only** — `action.rs`, `tui.rs`, `app.rs`; `main` runs
   a tokio runtime; existing terminal behavior runs through the new loop; the
   dirty system (`state::diff`, `mark_dep_dirty`/`mark_all_dirty`, `DataDep`,
   `UiState::dirty`) is deleted. **— DONE 2026-05-21 (PR #25).**
2. **Daemon wire-up** — `daemon.rs` connects `client/` as the second event
   source; `DaemonConnected`/`Disconnected` and `TickStopped` actions flow; the
   client is no longer dead code. **— DONE 2026-05-21 (PR #26).**
3. **Component trait** — `StatusBar` and `CommandLine` migrate behind a
   `Component` trait; App walks the `Layout` and calls `draw`.
4. **Effect channel** — effect `Action`s (`RequestStep`, `RequestInspect`, …)
   from `update()` route through `daemon.rs` into `rocket/*` requests.
5. **Per-panel components** — remaining `ViewKind`s (`LayerStack`,
   `TensorDetail`, `ProbeWatch`, `Timeline`, `KvCache`, `Worldline`) built one
   slice each; these are feature work the architecture unblocks.

Slices 1–4 are the architecture rework; slice 5 is the feature build-out.

## Notes

- The crate's deep leaf modules are kept: `tiling::Layout`, `input::mode::Mode`,
  `input::terminal::decode`, `state::cache::TensorCache`, `render::capability`,
  `client::connection`/`subscription`. See the disposition table in the design
  doc.
- Supersedes the unwired scaffold that BEAD-0014 documented; the `dead_code`
  allowances added in BEAD-0014 are removed as each module becomes reachable.
- Once slice 1 lands, the workspace clippy gate should be green again — closing
  the master-red situation noted in BEAD-0014.
