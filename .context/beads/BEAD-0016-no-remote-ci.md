---
id: BEAD-0016
title: No remote CI — branch protection requires a check nothing produces
status: open
priority: high
created: 2026-05-21
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
