# Dev Loop Infrastructure — Implementation Plan

**Date:** 2026-05-25
**Goal:** Provide a long-lived dev cockpit so humans and agents can rapidly iterate on rocket_surgeon with a real attached model and live test feedback.

---

## Motivation

The architecture already supports the workflow — daemon is meant to be long-lived, clients are crash-isolated, the protocol is the same internally and externally. What's missing is the harness around it:

- Every e2e test spawns its own daemon from scratch (~5-10s build + attach overhead per cycle).
- No way to keep a daemon attached to a model while iterating on driver code, intervention recipes, or probe patterns.
- No automated test-on-change feedback during development.
- No canonical "get me to interesting state" entry point an agent can invoke.

Phase 3B/C (replay, reverse step, divergence) is exactly the territory where iterating on state machines is painful without persistent state, which makes this the right time to land it.

---

## Architecture

Five components, all glue:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         scripts/dev-cockpit.sh                        │
│                       (tmux 2x2 pane layout)                          │
│                                                                       │
│  ┌───────────────────────┐    ┌─────────────────────────────────┐   │
│  │ xtask watch           │    │ scripts/dev-session.py          │   │
│  │ rebuild on file save  │    │ owns daemon · canonical setup   │   │
│  │                       │    │ JSON-RPC REPL                   │   │
│  └───────────────────────┘    └─────────────────────────────────┘   │
│  ┌───────────────────────┐    ┌─────────────────────────────────┐   │
│  │ xtask test-watch      │    │ scratch shell                   │   │
│  │ rerun e2e on change   │    │ git, logs, TUI, ad-hoc          │   │
│  └───────────────────────┘    └─────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

The driver is the lynchpin. It owns the daemon process (spawned via the existing `tests/e2e_harness.py` helpers), runs canonical setup, then drops to a JSON-RPC REPL. When the daemon dies (binary swap, crash, intentional kill), the driver respawns and replays setup.

---

## Components

### 1. `scripts/dev-session.py` — canned-state driver

**Owns the daemon as a child process.** Built on `tests/e2e_harness.py` for framing + spawn.

**Startup sequence:**
1. `build_binaries()` — match e2e harness
2. `spawn_daemon()` — same helper as tests
3. `initialize` → record `session_id`
4. `attach` tiny-llama (`hf-internal-testing/tiny-random-LlamaForCausalLM`)
5. `rocket/probe define` — single capture-all probe at `*:*:*:*:*:*` with summary-only
6. `rocket/step` count=1, granularity=component — get us off the start tick
7. Print banner to stderr listing session_id, current state, available commands

**REPL loop:**
- Read line-delimited JSON objects from stdin. Each is the `params` portion of a JSON-RPC request, with a `_method` key naming the verb (e.g. `{"_method": "rocket/inspect", "target": "..."}`)
- Auto-assign request IDs
- Forward to daemon, pretty-print response to stdout
- Special commands (lines starting with `:`):
  - `:help` — list commands and example requests
  - `:state` — `rocket/status` shortcut
  - `:reset` — kill daemon, respawn, re-run setup
  - `:setup` — re-run setup against existing daemon (idempotent for some verbs, not all)
  - `:script <path>` — run a file of newline-delimited commands
  - `:quit` — clean shutdown

**Crash recovery:** if `recv_message` raises `EOFError` (daemon died), print warning, respawn, replay setup, prompt user that state was lost.

**Why a separate file from `tests/e2e_harness.py`:** the harness is library code (no `__main__`), this is the executable. They share the framing/spawn helpers.

### 2. `cargo xtask watch` — file watcher for rebuilds

- Watches `crates/**/*.rs` and `python/**/*.py`
- On change: two-phase rebuild (workspace excluding PyO3 crates, then worker with PyO3 env vars) matching `e2e_harness::build_binaries`
- Prints pass/fail banners with timestamps
- Does NOT manage daemon lifecycle — the driver owns that

### 3. `cargo xtask test-watch [<pattern>]` — test runner on save

- Watches same files + `tests/**/*.py`, `tck/**/*.feature`
- With pattern: runs matching `tests/test_e2e_<pattern>.py` scripts
- Without: runs full `cargo test` (sans PyO3 crates) + all e2e scripts
- Pass/fail banners with timestamps

### 4. `scripts/dev-cockpit.sh` — tmux launcher

- Idempotent (`tmux has-session -t rs-dev`)
- 2x2 grid: xtask watch · dev-session · xtask test-watch · scratch
- Pane titles via `tmux select-pane -T`
- Focuses the driver pane
- Companion `scripts/dev-cockpit-kill.sh`

### 5. `tests/test_e2e_dev_session.py` — coverage on the cockpit itself

- Verifies dev-session.py reaches canonical state (sees a `stopped` state with non-null `tick_id`)
- Sends an inspect command, asserts a valid response
- Triggers `:reset`, asserts state returns
- Gates the dev loop against regressions — this is the "coverage on us" piece

---

## Execution order

1. Driver (`scripts/dev-session.py`) — foreground, owner: human
2. `xtask watch` + `xtask test-watch` — parallel, owner: subagent
3. `dev-cockpit.sh` — parallel, owner: subagent
4. Driver smoke test (`test_e2e_dev_session.py`) — after driver lands
5. Docs section in `CONTRIBUTING.md` — after everything green

---

## Non-goals

- Hot Python reload inside the daemon (no `rocket/dev/reload` verb today; defer)
- Multi-rank dev loop (single-GPU/CPU only for now)
- TUI integration into the cockpit (the driver owns the daemon, the TUI would need a separate one; punt to a later "dev-cockpit-tui" variant if the user wants it)
- Recording/replaying REPL sessions (interesting, but a separate feature)

---

## Acceptance

- `bash scripts/dev-cockpit.sh` opens a tmux session, all four panes run their commands without errors.
- Driver pane reaches `stopped` state with tiny llama attached in under 30 seconds (cold) or 5 seconds (warm).
- Editing `crates/rocket-surgeon-protocol/src/types.rs` triggers a rebuild in the watch pane and re-runs tests in the test-watch pane.
- `python tests/test_e2e_dev_session.py` passes from a clean checkout.
- `CONTRIBUTING.md` documents the three loops (researcher / TUI / daemon-iteration) and the cockpit entry point.
