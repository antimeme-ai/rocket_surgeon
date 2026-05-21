---
id: BEAD-0016
title: No remote CI — branch protection requires a check nothing produces
status: closed
priority: high
created: 2026-05-21
closed: 2026-05-21
---

## Description

`master` branch protection requires a status check, but the repo has no
GitHub Actions workflow (or other remote CI) to produce it. Every PR therefore
sits permanently at `mergeStateStatus: BLOCKED` and can only be merged with
`gh pr merge --admin`, bypassing branch protection entirely.

Consequence: every merge bypasses all gates. The local lefthook hooks
(fmt / clippy / test) run only on a developer's machine and are skipped by any
GitHub merge. Combined with the required-but-absent check, `master` has no
automated protection at all — this is the engine behind the recurring
"master goes red" pattern (see BEAD-0009, PR #16 / #18, BEAD-0014).

Observed 2026-05-21 merging PRs #18–#22: every one required `--admin`.

## Resolution

Preferred: add a GitHub Actions workflow that runs `cargo xtask ci`
(fmt, clippy, cargo test, e2e, pytest) on pull requests against `master`, and
name that job as the required status check in branch protection. PRs then have
a real, satisfiable gate and no longer need `--admin`.

Alternative (weaker): if a required check is not wanted, drop the
unsatisfiable rule from `master` branch protection so PRs merge on review
alone. This does not close the gate-bypass hole and is not recommended.

Note: the workflow needs the Python toolchain (`uv`) and a CPU-only model for
the e2e/pytest stages — confirm the e2e suite runs in a GitHub runner, or gate
the heavy stages behind a separate job.

## Acceptance criteria

- A PR against `master` triggers `cargo xtask ci` automatically.
- A green run satisfies branch protection; PRs merge without `--admin`.
- A red run blocks the merge.

## Resolution (2026-05-21)

Added `.github/workflows/ci.yml` (branch `ci/remote-github-actions`, PR to
`master`). One job, id and `name:` both `ci` — that string is the status
check `master` branch protection must require.

The job provisions the toolchain (`actions/checkout@v6`,
`dtolnay/rust-toolchain@1.85.0` with rustfmt+clippy, `astral-sh/setup-uv@v8`
for `uv` + Python 3.11, `lefthook` via `uv tool install`), caches the cargo
registry/target (`Swatinem/rust-cache@v2`) and the uv environment, then runs
`cargo xtask setup` followed by `cargo xtask ci` verbatim — fmt + clippy +
ruff + mypy + cargo test + pytest + e2e. No divergence from the local
`cargo xtask ci` gate, so remote and local gates are byte-identical.
Triggers: `pull_request` → `master` and `push` → `master`. Permissions are
least-privilege (`contents: read`).

Sub-plan: `.context/plans/2026-05-21-remote-ci.md`.

Remaining one-time admin action (not code, cannot be done by the workflow):
once this workflow's first run is green, set the required status check on
`master` branch protection to `ci`. PRs then merge on a real gate without
`--admin`.
