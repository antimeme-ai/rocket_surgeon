# TUI daemon link — spawn `rocket-surgeon` over stdio

**Bead:** BEAD-0020 · **Crate:** `rocket-surgeon-tui`

## Goal

Make the TUI actually reach the daemon. Today's `UnixStream::connect` to
`/tmp/rocket-surgeon.sock` always fails because the daemon doesn't listen on a
socket — it serves JSON-RPC over stdio. Replace the socket connect with a
subprocess spawn that pipes the daemon's stdio into the existing `Connection`
infrastructure.

## Design

- `Connection::spawn` is already generic over `AsyncRead + AsyncWrite + Unpin
  + Send + 'static` (slice 2 was built that way for the `duplex`-pipe tests).
  `ChildStdout` / `ChildStdin` satisfy those bounds; this slice swaps the
  transport with zero change to the protocol logic.
- The daemon's tracing goes to stderr; `Stdio::inherit()` lets it land
  wherever the TUI's stderr is redirected. Same redirect, same log file —
  one log per session.
- Lifecycle: `kill_on_drop(true)` covers the panic path; an explicit
  `child.kill().await` on the normal exit path guarantees the daemon doesn't
  outlive a clean TUI quit.

## Files

| File | Change |
|---|---|
| `daemon.rs` | `spawn(daemon_bin: PathBuf, ...)`; `run` does `Command::spawn` + `Connection::spawn(stdout, stdin, ...)` + kill on exit |
| `main.rs` | drop `--socket`; add `--daemon-bin <path>` with sibling-of-exe default |

## Tests

`drive` and `map_notification` unit tests stay — they exercise the protocol
logic against `duplex` pipes, transport-agnostic. The subprocess wiring is
verified end-to-end manually: launch the TUI against the real binary, observe
the status bar flip to `Initialized` and the handshake's `protocol_version`
appear (a tmux `capture-pane` proof attaches to the PR).

## Out of scope

- Better UX for "daemon binary not found" (currently shows `Uninitialized`
  and logs the io error; a foreground error + exit would be nicer).
- Attach + step plumbing against a real model — needs the orchestrator,
  worker, and a host. Separate work.
