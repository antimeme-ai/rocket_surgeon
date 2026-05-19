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

## What we don't want

- New runtime dependencies. We reimplement; prior art is reference, never a dependency.
- OOP. No class hierarchies. Data flows through functions.
- Python logic. Python is for PyTorch FFI only. All logic, state, and IPC live in Rust.

## Issues

File bugs and feature requests in GitHub Issues. Include reproduction steps for bugs.
