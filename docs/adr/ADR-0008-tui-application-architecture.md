# ADR-0008: TUI Application Architecture — Component Template on Ratatui

## Status
Accepted

## Context

ADR-0004 fixed `rs-tui` as Process C: a pure-Rust process doing Ratatui
rendering, acting as a pure protocol client, with no PyTorch awareness. It
deliberately did not specify the crate's *internal* architecture.

`rocket-surgeon-tui` was then built without anchoring on a reference pattern,
and drifted:

- `main.rs::run_loop` is **synchronous** and only ever produces terminal-input
  events. It sits beside a fully **async (tokio)** protocol client in `client/`
  that nothing connects to — the client, and every daemon-event path in the
  reducer, are unreachable (the bulk of the crate's clippy dead-code, BEAD-0014).
- A per-view dirty-tracking system was built **twice** — imperatively in
  `reducer::mark_dep_dirty`, and derivationally in `state/diff.rs` — the latter
  entirely unused.
- The crate is otherwise sound: leaf modules (`tiling::Layout`,
  `input::mode::Mode`, `input::terminal::decode`, `state::cache::TensorCache`,
  `render::capability`) have small interfaces, are well-tested, and are worth
  keeping.

The crate's real requirements:

1. Render multiple tiled view kinds (`LayerStack`, `TensorDetail`,
   `ProbeWatch`, `Timeline`, `KvCache`, `Worldline`, `CommandLine`,
   `StatusBar`) — eight independently-complex panels, some rendering tensors
   via terminal graphics protocols (Kitty/Sixel, already detected by
   `render::capability`).
2. Maintain one coherent session state — the debugging session is a single
   logical domain.
3. Consume the daemon's async JSON-RPC notification stream (`tick.stopped`,
   `probe.fired`, …) as a **first-class event source**, co-equal with terminal
   input.

Options considered:

- **The three Ratatui application patterns** — *Elm (TEA)*, *Component*, *Flux*.
- **tui-realm** — a TEA framework layered on ratatui.
- **FrankenTUI** — a ground-up, non-ratatui TUI kernel.
- **The Ratatui Component template** — `cargo generate ratatui/templates`: a
  `Component` trait + an `Action` enum + an owned tokio event loop.

## Decision

**Build `rocket-surgeon-tui` on the Ratatui Component template pattern**, with
no external TUI framework:

- **TEA spine** — a single `UiState` store, one `Action` enum, unidirectional
  flow, a pure-ish `update`.
- **Component decomposition for panels** — each view kind is a `Component`
  co-locating its event handling, update, and rendering.
- **An owned `tokio` event loop** — `main` runs a tokio runtime; one task reads
  terminal input, one task owns the daemon client and turns notifications into
  `Action`s; both feed a single action channel the loop drains.

This is the standard TEA-spine + Component-decomposition hybrid that the
Ratatui Component template ships, expressed as code the project owns. The
detailed module blueprint is `docs/specs/2026-05-21-tui-architecture-design.md`.

### Why each alternative was rejected

- **Flux** — its defining feature is *multiple stores*; the debugging session
  is one coherent domain, so multiple stores would fragment naturally-atomic
  state. Single-store Flux degenerates into TEA.
- **Pure TEA** — correct for data flow, but its centralized `update`/`view`
  does not co-locate per-panel logic; with eight substantial panels this
  becomes a dumping ground. It also has no built-in effect/async story.
- **Pure Component** — correct for decomposition, but says nothing about
  app-wide data flow; the session state still needs a single owner.
- **tui-realm** — a framework whose poll/Port-based loop is a poor fit for an
  async tokio daemon stream as a first-class event source; it also inverts
  control over a load-bearing surface and would mean discarding the crate's
  existing ratatui-based deep modules.
- **FrankenTUI** — early-stage with no stable public API (disqualifying for
  load-bearing code under JSMNTL), and not ratatui-based — adopting it would
  itself reverse ADR-0004's "Ratatui rendering".

The decisive factors are this project's fixed constraints — already on
ratatui, ADR-0004, JSMNTL rigor (own load-bearing code), and the async daemon
as a first-class event source — not the relative cleverness of any framework.

## Consequences

- **Good** — the async client becomes a native event source instead of
  orphaned code; the crate's existing deep leaf modules are retained; the
  bespoke dirty-tracking system is deleted in favour of immediate-mode redraw;
  the `App` becomes testable by feeding it `Action`s with no terminal or socket.
- **Good** — no external framework dependency on a load-bearing surface; the
  event loop is code the project owns and can put behind this ADR.
- **Bad** — the project owns the loop, the `Component` trait, and the event
  plumbing — standard code, but maintained rather than imported.
- **Bad** — the current scaffold must be reworked: `main.rs` rewritten around
  tokio, `reduce` reshaped into `App::update`, the dirty system removed,
  `compositor` render arms migrated to `Component`s.
- The rework is tracked as **BEAD-0015** and should be delivered as incremental
  vertical slices (terminal loop first, then daemon wire-up, then per-panel
  components), not a big-bang rewrite.
