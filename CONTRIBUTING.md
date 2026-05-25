# Contributing

## How to contribute

1. Fork the repo
2. Create a feature branch off `master`
3. Make your changes
4. Run the checks: `cargo xtask ci`
5. Open a PR

## Code standards

- Zero warnings: `cargo clippy --workspace --all-targets -- -D warnings`
- Format: `cargo fmt --all`
- Python lint: `ruff check python/ && ruff format --check python/`
- Tests must pass: `cargo test --workspace --all-targets`

Git hooks are managed by [lefthook](https://lefthook.dev/) (`lefthook.yml`).
`cargo xtask setup` installs them. Pre-commit runs fmt + clippy + ruff + mypy
in parallel with file-glob scoping; pre-push runs the test suites. If a hook
fails, the action fails — fix the issue and try again.

## Commits

- Sign your commits (`git commit -s`)
- Atomic commits — one logical change per commit
- Conventional commit messages: `feat(crate): what`, `fix(crate): what`, `test: what`, `docs: what`

## Architecture

Read `docs/specs/architecture.md` and the ADRs in `docs/adr/` before proposing structural changes. If your change is load-bearing, write an ADR.

## Dev loop

Three iteration shapes, pick the one that matches what you're working on.

### Loop 1 — researcher loop (no extra infra)

Daemon stays up across all your edits. Iterate on driver code, intervention
recipes, probe patterns, analysis scripts.

```bash
PYTHONPATH=python python scripts/dev-session.py
```

Builds binaries, spawns a daemon, attaches a tiny llama, defines a capture-all
probe, steps once. Drops to a JSON-RPC REPL — one JSON object per line where
`_method` names the verb. `:help` lists commands.

### Loop 2 — full cockpit (recommended for Phase 3+ work)

A tmux session with the rebuild watcher, the driver, the test watcher, and a
scratch shell side-by-side. Editing Rust or Python triggers rebuilds; the
driver's `:reset` respawns the daemon against the new binaries; tests rerun
automatically.

```bash
scripts/dev-cockpit.sh        # opens tmux session 'rs-dev'
scripts/dev-cockpit-kill.sh   # tears it down
```

Layout:

| Pane | Command | Purpose |
|------|---------|---------|
| top-left | `cargo xtask watch` | rebuild on save (workspace + worker two-phase) |
| top-right | `scripts/dev-session.py` | driver — owns the daemon, holds canonical state |
| bottom-left | `cargo xtask test-watch` | rerun all e2e + cargo test on save |
| bottom-right | scratch | git, logs, ad-hoc tools |

After a rebuild lands, type `:reset` in the driver pane to pick up the new
binaries against fresh state.

### Loop 3 — targeted test watch

For grinding a single e2e flow without the full cockpit:

```bash
cargo xtask test-watch stepping    # reruns tests/test_e2e_stepping.py on change
cargo xtask test-watch             # reruns full cargo test + all e2e on change
```

### Coverage on the dev loop itself

The dev loop is gated by `tests/test_e2e_dev_session.py` — runs the driver,
checks it reaches canonical state, exercises `:reset`, and exits cleanly. If
this fails, the dev loop is broken; fix it before merging.

```bash
PYTHONPATH=python python tests/test_e2e_dev_session.py
```

## What we don't want

- New runtime dependencies. We reimplement; prior art is reference, never a dependency.
- OOP. No class hierarchies. Data flows through functions.
- Python logic. Python is for PyTorch FFI only. All logic, state, and IPC live in Rust.

## Issues

File bugs and feature requests in GitHub Issues. Include reproduction steps for bugs.
