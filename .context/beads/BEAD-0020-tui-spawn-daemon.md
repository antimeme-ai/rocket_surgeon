---
id: BEAD-0020
title: TUI daemon link — spawn `rocket-surgeon` as subprocess over stdio
status: open
priority: high
created: 2026-05-22
---

## Description

The `rocket-surgeon-tui` daemon link (BEAD-0015 slice 2) connects to a Unix
socket `/tmp/rocket-surgeon.sock`. The `rocket-surgeon` daemon, however, serves
JSON-RPC over **stdio** (no `UnixListener` anywhere in its crate) and is built
to be spawned as a child process — its CLI takes `--orchestrator-bin` /
`--worker-bin`, expecting a launching parent.

Nothing bridges the socket↔stdio gap. The TUI cannot reach the real daemon;
its link logic (`daemon.rs::run`) has only ever been tested against an
in-process `duplex` pipe (unit tests of `drive`). Discovered the moment slices
1–4 were finished and the TUI was actually launched — status stayed
`Uninitialized` because `UnixStream::connect("/tmp/rocket-surgeon.sock")`
errored `io: No such file or directory`.

## Scope

- `daemon.rs::run` spawns `rocket-surgeon` via `tokio::process::Command` with
  piped stdin/stdout and `Stdio::inherit()` stderr (daemon logs follow the
  TUI's stderr — user redirects them). Hand the piped stdio to
  `Connection::spawn`, which is already generic over `AsyncRead`/`AsyncWrite`.
- `main.rs` CLI: drop `--socket`; add `--daemon-bin <path>` defaulting to a
  sibling-of-current-exe lookup, mirroring the daemon's own
  `--orchestrator-bin` default.
- Kill the child on `run` exit (best-effort) plus `kill_on_drop(true)` as a
  backstop, so a TUI quit doesn't leak a daemon process.
- Existing `drive` / `map_notification` unit tests are unaffected (they use a
  `duplex` pipe). Verification is a manual end-to-end run: launch the TUI,
  observe the status bar flip to `Initialized` and the daemon's reported
  protocol version flow through.

## Out of scope

- A daemon `--listen <socket>` mode — an alternative architecture not chosen.
- Attach / model-host wiring. `rocket/step` against a freshly-spawned daemon
  still errors `INVALID_STATE` (no attach); a live step loop additionally
  needs the orchestrator + worker + a model and is its own work.
