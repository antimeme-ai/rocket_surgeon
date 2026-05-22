# TUI Slice 4 — Effect channel

**Bead:** BEAD-0015 slice 4 · **ADR:** ADR-0009 · **Crate:** `rocket-surgeon-tui`
**Design:** `docs/specs/2026-05-21-tui-architecture-design.md` (migration path item 4)

## Goal

Close the command loop: a `:`-command typed in the TUI becomes a real
`rocket/*` request on the daemon link. `App::update()` emits an effect; the
loop routes it to the daemon task; the task issues the JSON-RPC request. The
daemon's `tick.stopped` notification (already wired in slice 2) flows back —
so `:step 3` is an end-to-end vertical slice.

## Design

- **`Effect`** (new, `action.rs`) — the app→daemon command type, the mirror of
  `DaemonEvent` (daemon→app). Slice 4 ships one variant: `RequestStep { count }`.
  Kept a distinct type, not an `Action` variant: effects travel app→daemon on a
  dedicated channel and are never loop input.
- **`App::update(&Action) -> Outcome`** — replaces `handle_terminal` /
  `handle_daemon`, matching the design's unified `update`. `Outcome { flow,
  effect }`. Terminal command-execute is the only effect source in slice 4.
- **`apply_input` returns `Option<Effect>`** — `reduce_command`'s `Execute` arm
  parses the buffer (`parse_command`) and yields the effect; all other input
  reductions yield `None`.
- **`daemon::spawn` returns `mpsc::Sender<Effect>`** — the task `select!`s the
  notification stream and the effect receiver; effects become `conn.request`
  calls. The `Connection` is now kept alive for the task's lifetime (slice 2
  dropped it for close-detection); transport death is detected on the request
  path. RPC-level errors are logged, not fatal.
- **`Component::update`** is *not* added here — no panel has view-local
  behaviour to co-locate yet; it lands with the first stateful panel (slice 5).
  Correct the slice-3 comment in `components/mod.rs` that anticipated slice 4.

## Files

| File | Change |
|---|---|
| `action.rs` | add `Effect` enum |
| `state/reducer.rs` | `apply_input -> Option<Effect>`; `reduce_command` Execute parses; add `parse_command` |
| `app.rs` | `update(&Action) -> Outcome`; fold in `handle_terminal`/`handle_daemon` |
| `daemon.rs` | `spawn -> Sender<Effect>`; `select!` notifications + effects; `dispatch_effect` |
| `main.rs` | wire the effect sender; loop calls `update`, routes the effect |
| `components/mod.rs` | correct the stale `update`-lands-in-slice-4 comment |

## Tests (TDD — red first)

- `reducer`: `parse_command` — `step`→1, `step 5`→5, `step bad`→1, `bogus`→None;
  `reduce_command` Execute returns the effect and still clears the buffer.
- `app`: `update(Terminal …)` for a `:s t e p ⏎` sequence yields `RequestStep`;
  `update(Daemon …)` / `update(Tick)` yield no effect; quit still returns `Quit`.
- `daemon`: an `Effect` sent to the task produces a `rocket/step` frame on a
  scripted server (extend the existing `drive`-handshake harness).

## Out of scope

- `pending_requests` request-lifecycle counter (needs response→Action plumbing).
- `RequestInspect` and other effects — the channel generalises trivially.
- Reconnection / silent-idle-link-death detection (its own slice).
- `dispatch_effect` awaits each request inline in the `select!` loop, so a
  slow response head-of-line-blocks the notification stream. Harmless at
  single-in-flight `:step` scale; revisit in the reconnection/lifecycle slice.
