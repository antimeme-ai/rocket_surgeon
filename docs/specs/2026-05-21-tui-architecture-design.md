# TUI Architecture Design — Component Template on Ratatui

**Date:** 2026-05-21
**ADR:** ADR-0009
**Bead:** BEAD-0015
**Crate:** `rocket-surgeon-tui`

This is the target-architecture blueprint for `rocket-surgeon-tui`. ADR-0009
records *why* the Component template was chosen; this document records *what*
to build.

## Reference

The Ratatui **Component template** (`cargo generate ratatui/templates`,
component variant): a `Component` trait + an `Action` enum + an owned `tokio`
event loop with separate terminal-event and async-task sources feeding one
action channel. We adopt the pattern, not the generated code — the crate keeps
its own naming and its existing deep modules.

## Target module layout

```
src/
  main.rs        CLI parse; tokio runtime; terminal enter/leave; run App
  app.rs         App: owns UiState + Layout + components; the action loop; update()
  action.rs      Action — the single unified message type (replaces UiEvent)
  tui.rs         Terminal wrapper + terminal-input task + tick task -> Event channel
  daemon.rs      Daemon task: owns the protocol client; notifications <-> Actions
  components/
    mod.rs       Component trait
    status_bar.rs, command_line.rs            (exist today as compositor arms)
    layer_stack.rs, tensor_detail.rs, ...     (one per ViewKind, built incrementally)
  client/        KEPT — connection.rs, subscription.rs (transport, under daemon.rs)
  input/         KEPT — terminal::decode (key -> Action), mode.rs
  render/        KEPT — capability.rs; compositor.rs dissolved into App + components
  state.rs       UiState, SessionSnapshot, CursorState — the Model
  tiling.rs      KEPT — Layout
```

## The Action enum

`Action` replaces `UiEvent`. One enum, three origins:

```rust
// action.rs — intended shape
pub enum Action {
    // 1. Terminal-derived (from input::terminal::decode)
    Navigate(NavigationEvent),
    Mode(ModeEvent),
    Command(CommandEvent),
    Resize { width: u16, height: u16 },
    Quit,

    // 2. Daemon-derived (from daemon.rs, mapped from JSON-RPC notifications)
    DaemonConnected { protocol_version: String },
    DaemonDisconnected,
    TickStopped(TickPosition),
    ProbeFired(/* ... */),
    SessionUpdated(SessionSnapshot),

    // 3. Effects — emitted by update(), consumed by daemon.rs
    RequestStep { count: u32 },
    RequestInspect { target: String },
    // ...
}
```

The third group is the effect/command channel that pure TEA lacks: `update()`
may *return* an `Action` that means "issue this RPC". The App routes effect
actions to the daemon task; it does not call the daemon directly.

## The Component trait

```rust
// components/mod.rs — intended shape
pub trait Component {
    /// React to an action; optionally emit a follow-up action.
    fn update(&mut self, action: &Action, state: &UiState) -> Option<Action> { None }

    /// Draw into the allocated rect. Immediate-mode; called every frame.
    fn draw(&self, frame: &mut Frame<'_>, area: Rect, state: &UiState);
}
```

One `Component` per `ViewKind`. The component co-locates its own update +
render. App-wide state (`UiState`) stays centralized and is passed in by
reference — components hold only view-local state (scroll offset, selection).

## Event loop & async model

```
main (tokio runtime)
  ├── tui.rs        task: crossterm EventStream -> Event channel
  ├── daemon.rs     task: owns ReconnectingClient
  │                       notification broadcast -> Action channel
  │                       effect Actions -> JSON-RPC requests
  └── app.rs        loop {
                      action = recv from (terminal Events mapped via decode)
                               + (daemon Action channel);
                      follow_up = app.update(action);
                      if let Some(a) = follow_up { dispatch(a); }
                      terminal.draw(|f| app.draw(f));   // every iteration
                    }
```

- `main` builds a `tokio` runtime (the protocol client is already tokio).
- `tui.rs` uses crossterm's async `EventStream` (the `event-stream` feature)
  for terminal input; a tick task bounds the redraw rate.
- `daemon.rs` owns the `ReconnectingClient`, subscribes to the notification
  stream, maps each notification to an `Action`, and accepts effect `Action`s
  to turn back into `rocket/*` requests.
- The App loop merges both sources into one stream of `Action`s
  (`tokio::select!`), runs `update`, then redraws.

## Rendering: immediate-mode, no dirty tracking

Redraw the whole frame each loop iteration (bounded by the tick task). A
debugger TUI is not redraw-bound; ratatui immediate-mode is cheap. The bespoke
dirty system is **deleted**: `state.dirty`, `reducer::mark_dep_dirty`,
`reducer::mark_all_dirty`, `state::diff`, and `DataDep` all go. `ViewSlot`
loses `data_deps`.

## Disposition of current modules

| Module | Disposition |
|---|---|
| `tiling::Layout` | **Keep** — deep, tested; App walks it to allocate rects. |
| `input::mode::Mode` | **Keep** — mode state machine is correct. |
| `input::terminal::decode` | **Keep** — retype to emit `Action` instead of `InputEvent`. |
| `input::events` | **Fold** into `action.rs` (Navigation/Command/Mode sub-enums survive). |
| `state::cache::TensorCache` | **Keep** — used by `TensorDetail` component. |
| `render::capability` | **Keep** — graphics-tier detection for tensor rendering. |
| `client::connection`, `client::subscription` | **Keep** — transport, owned by `daemon.rs`; seal the leaky `connection()` accessor. |
| `state::UiState`, `SessionSnapshot`, `CursorState` | **Keep** — the Model; drop the `dirty` field. |
| `render::compositor` | **Dissolve** — render arms become `Component`s; rect-walking moves to App. |
| `state::reducer::reduce` | **Reshape** into `App::update(&mut self, Action)`. |
| `state::diff`, `mark_dep_dirty`, `mark_all_dirty`, `DataDep`, `dirty` | **Delete** — superseded by immediate-mode redraw. |
| `main::run_loop` | **Replace** — becomes the tokio loop in `app.rs` + `main.rs`. |

## Migration path — vertical slices

Not a big-bang rewrite. Each slice compiles, tests, and leaves the crate
runnable.

1. **Tokio loop, terminal only.** Add `action.rs`, `tui.rs`, `app.rs`; `main`
   gets a tokio runtime. Existing terminal-only behavior runs through the new
   loop. Daemon not yet wired. Delete the dirty system here.
2. **Daemon wire-up.** Add `daemon.rs`; connect `client/` as the second event
   source. First real daemon actions: `DaemonConnected`/`Disconnected`,
   `TickStopped`. The client stops being dead code.
3. **Component trait + existing panels.** Introduce `Component`; migrate
   `StatusBar` and `CommandLine` (already implemented) behind it. App walks the
   `Layout` and calls `draw`.
4. **Effect channel.** Wire effect `Action`s (`RequestStep`, `RequestInspect`)
   from `update()` through `daemon.rs` to real `rocket/*` requests.
5. **Per-panel components.** Build the remaining `ViewKind`s one slice each
   (`LayerStack`, `TensorDetail`, …), each its own bead.

Slices 1–4 are the architecture; slice 5 is feature work that the architecture
unblocks.

## Out of scope

- The individual view designs (what `LayerStack`/`TensorDetail`/etc. actually
  show) — separate design work, per-panel.
- Multi-rank / multi-GPU view layout — follows the daemon's multi-GPU work.
