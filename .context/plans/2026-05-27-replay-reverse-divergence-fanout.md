---
title: Fan-out plan for landing phase3/replay-reverse-divergence
date: 2026-05-27
status: ready-to-execute
related:
  - BEAD-0018 (replay orchestration)
  - .context/session-notes/2026-05-24-phase3a-checkpoint-state-tier.md
  - docs/adr/ADR-0010-perfetto-multi-rank-tracing.md (just-landed master change)
---

# Replay / reverse-step / divergence: branch landing fan-out

## TL;DR

There is a stale-but-substantial 21-commit feature branch
`origin/phase3/replay-reverse-divergence` (last commit 2026-05-25) that
implements the bulk of **Phase 3B (replay, backward step, divergence
detection, Tier 2 callbacks, worldline DAG)**. It is 25 commits behind
master and **conflicts on its first commit** because master independently
shipped the Phase 3B Task 1 protocol types (`7fd20bf`, `384a2da`). A naive
rebase fails immediately.

This document is the plan to land it as a series of small, mergeable PRs
that can be partially fanned-out across sessions. Each PR is independently
shippable, gates on `cargo xtask ci`, and is small enough to review.

## What's actually on the branch

21 commits, 20 files (1353 LOC added, 115 removed). Six logical work units,
grouped by the layer they touch. Listed bottom-up (protocol first, then
each layer that depends on it).

### WU-A. Protocol reconciliation
- `9446274 feat(protocol): add HostReplayRequest/Response, WorldlineState, replay threshold fields`

Conflicts with master's `7fd20bf` (which already shipped Phase 3B Task 1
types) and `384a2da` (`WorldlineState::is_empty()` fix). Need to diff the
branch's protocol commit against master's current types and produce a
**single reconciled commit** that adds only what master is missing.

The branch adds:
- `internal::HOST_REPLAY` const
- `ReplayRequest` fields: `deterministic`, `cosine_threshold`, `mre_threshold`
- `HostReplayRequest` / `HostReplayResponse` (internal three-process types)
- `Divergence` record type
- Whatever `types.rs` additions go with the above

Master's `7fd20bf` already added the top-level `WorldlineState` /
`ReplayThresholds` / replay envelope changes. **Do not re-add those.**

### WU-B. Replay request routing through the three processes
- `60236ce feat(orchestrator): route _host/replay to worker`
- `01d7632 feat(worker): handle _host/replay — forward re-execution from checkpoint`
- `7661ba0 feat(daemon): wire rocket/replay through orchestrator to worker`

End-to-end plumbing of the `rocket/replay` verb. Self-contained once WU-A
is in. Touches `orchestrator/dispatch.rs`, `worker/dispatch.rs`,
`worker/replay.rs` (new file), `rocket-surgeon/main.rs`,
`rocket-surgeon/orchestrator_handle.rs`.

### WU-C. Divergence detection
- `163abfb feat(python): compare_activations for divergence detection + CPU RNG capture`
- `08e7aa0 feat(worker): divergence detection during replay at √L boundaries`
- `7b4356e fix(replay): handle zero tensors and unknown dtypes in divergence detection`

The divergence story: Python compares captured-vs-replayed activations,
worker checks at √L boundaries, edge-case fixes for zero tensors (cosine
sim should be 1.0, not 0.0) and unknown dtype strings (descriptive
ValueError, not bare KeyError). Depends on WU-A + WU-B.

### WU-D. Backward step
- `f8d0bae feat(daemon): backward step via checkpoint restore + replay, worldline DAG tracking`
- `ce806f6 feat(daemon): eager sub-checkpoint for O(1) backward step, eviction priority`

Backward stepping = restore from checkpoint + replay forward. Eager
sub-checkpoint makes it O(1) for the common case. Touches
`rocket-surgeon/dispatch.rs`, `rocket-surgeon/session.rs`,
`worker/checkpoint.rs`. Depends on WU-A + WU-B (replay route is the
mechanism backward step calls).

### WU-E. Tier 2 callbacks
- `aeab244 feat(python): Tier 2 callback interventions with watchdog thread timeout`
- `61644b3 feat(callbacks): add module cache invalidation and hot-reload support`
- `113c916 fix(worker): thread tick_id and model_handle through callback interventions`

User-defined Python callbacks as a Tier 2 intervention type. Watchdog
thread enforces timeout. Hot-reload via `invalidate_callback_cache()` +
`resolve_callback(reload=)` kwarg. The fix commit threads real
`tick_id`/`model_handle` from Rust through `bridge.rs` → `bridge.py` →
`engine.py` → `callback.py` (was hardcoded to 0/0). New files:
`python/rocket_surgeon/host/interventions/callback.py`,
`python/tests/test_callback.py`.

**Mostly orthogonal to A–D** — Python-only with a small Rust shim for
threading the context. Can be a parallel PR once the small bridge.rs change
is rebased against master.

### WU-F. Independent quality fixes (can land standalone)
- `942b642 feat(inspect): support 4-part legacy and 5+-part target formats`
- `6b3a34a feat(bundle): export bookmarks.json, worldlines.json, checkpoint metadata`
- `88895c2 feat(worker): set CUBLAS_WORKSPACE_CONFIG at startup for deterministic replay`
- `c0f1e70 fix(daemon): return CAPABILITY_NOT_SUPPORTED when backward step has no checkpoint`
- `5674de9 fix(session): track worldline segment tick_range on step, replay, and branch`
- `44388ad fix(worker): correct replay_of and original_tick_id semantics in replay`
- `7c85c3c fix(test): stabilize replay tolerance test against near-zero values`

These are all small, orthogonal, and **mostly independent**. Some depend on
A–D existing (e.g. `replay_of` semantics fix), but the bundle-export and
inspect-target-format changes are pure independent quality fixes.

### WU-G. TCK un-defer
- `89565b6 tck: un-defer 45 scenarios — replay, branch, inspect, export, divergence now active`
- `477a29b revert: re-defer TCK scenarios pending pytest-bdd step definitions`

89565b6 un-defers, 477a29b re-defers because pytest-bdd step defs don't
exist. Net effect on this branch: zero. **Skip these two commits during
rebase** — they cancel out, and un-deferring is gated on actual step-def
work that lives elsewhere (BEAD-0011-adjacent).

## Fan-out strategy

PRs land in dependency order. A and B are sequential; C, D, E, F can fan
out once A+B are in.

```
PR-1: WU-A (protocol reconciliation)                    [BLOCKING all others]
   │
   ▼
PR-2: WU-B (replay routing through three processes)     [BLOCKING C, D]
   │
   ├──────────┬──────────┬──────────┐
   ▼          ▼          ▼          ▼
PR-3       PR-4       PR-5       PR-6
WU-C       WU-D       WU-E       WU-F
(diverg.)  (back.)    (callbacks)(quality)
(+ G skip)
```

**WU-E (Tier 2 callbacks) does not depend on WU-B in principle** —
the small Rust shim in `worker/bridge.rs` could be split out and landed
even before WU-B. But the test scaffolding lives in
`python/tests/test_callback.py` which expects a working `rocket/intervene`
flow, so practically it's safer to gate on PR-2 too.

**WU-F can be fanned out further into 2-3 standalone PRs** — e.g. the
`bundle export` commit, the `inspect target format` commit, and the
`CUBLAS_WORKSPACE_CONFIG` commit are all single-commit independent fixes
that any contributor (or LLM session) could pick up.

## Per-PR target shape

| PR | Files (approx) | LOC | Risk | Notes |
|----|----------------|-----|------|-------|
| PR-1 WU-A | 3 | ~120 | High | Conflict reconciliation requires careful diff vs master |
| PR-2 WU-B | 6 | ~250 | Med | Wire-correct routing; existing protocol types are ground truth |
| PR-3 WU-C | 5 | ~180 | Med | Numeric edge cases need property tests |
| PR-4 WU-D | 4 | ~280 | High | DAG state machine touches `session.rs` — coordinate with anything else editing it |
| PR-5 WU-E | 5 | ~250 | Low | Python-mostly, watchdog-timeout testing is the tricky bit |
| PR-6 WU-F | 7 | ~130 | Low | 7 commits, each could be its own PR if reviewers prefer |

## Quality gates per PR

Every PR must satisfy, before opening:

1. `cargo xtask ci` green locally (fmt + clippy + ruff + mypy + workspace tests + deny)
2. `pytest python/tests/tck/test_<relevant>.py` green
3. Each commit message follows the conventional-commit + body pattern seen
   in recent master (`feat(scope): …` + 1-2 paragraph body explaining
   _why_ + `Co-Authored-By:` trailer if used).
4. New behavior either has a TCK scenario or a Rust test asserting the
   contract — not just smoke-coverage.

## Operational gotchas (read these first)

- **Git identity**: This repo resets `user.{name,email}`. Before each
  commit run:
  ```
  git config user.name "antimemeai"
  git config user.email "hiya@antimeme.ai"
  ```
- **GitHub auth**: Use `gh-alt`, NOT plain `gh`, for any PR / API
  operation on `antimeme-ai/*`. The default `gh` returns
  `must be a collaborator (createPullRequest)`. The repo ships an
  `.envrc` (gitignored) that auto-exports `GH_CONFIG_DIR=~/.config/gh-alt`,
  so direnv-loaded shells get this for free.
- **Branch protection**: `master` requires a PR (force-push and deletion
  blocked). Approving-review requirement was removed 2026-05-26 because
  GitHub doesn't allow self-approval and this is a solo project. PR-then-
  merge is fine; never `git push origin master` directly.
- **`cargo-deny`**: Local install must be ≥ 0.18 for CVSS 4.0 support
  (master upgraded to 0.19.7). If `cargo xtask deny` fails with
  `unsupported CVSS version: 4.0`, run
  `cargo install --locked cargo-deny --version 0.19.7`.
- **Pre-push hook is slow (~3.5 min)**: Runs `cargo test` + full `pytest`.
  Plan for it. Don't `--no-verify` to skip; the PR's CI is the real gate.
- **Don't merge your own PR without explicit "merge it" from the user.**
  Open the PR, push commits, wait for CI green, then ask. The recent
  PR #45 / #46 set the precedent that "merge it" is a valid explicit
  authorization, but the default is wait-for-orchestrator.

## Risks called out

- **WU-A reconciliation may introduce semantic drift.** The branch wrote
  the protocol types with the LLM in one head-state; master's `7fd20bf`
  shipped them with possibly different field names, defaults, or
  envelope conventions. Diff carefully. If they differ in ways that matter,
  prefer master's version (it's the canonical one) and rewrite the branch's
  worker/orchestrator/daemon code to match.
- **WU-D touches `session.rs` heavily.** Any concurrent work elsewhere on
  the session state machine (BEAD-0017 intervene-registry would qualify)
  needs coordination. Land WU-D first or last in a coordinated window.
- **TCK step definitions don't exist for most of these scenarios.** The
  commits that un-defer them are net no-ops because they're reverted on
  the branch. Don't get distracted writing step defs as part of this work
  — that's BEAD-0011-adjacent scope.

## Suggested execution order for a single agent

If one agent is doing all of this sequentially: **WU-A → WU-B → WU-F →
WU-C → WU-E → WU-D**. F goes early because it's safe quick wins that
build momentum; D goes last because it's the riskiest and touches the
most state.

For fanned-out execution (multiple sessions / agents): synchronize on
WU-A and WU-B landing first, then peel C/D/E/F off in parallel.

---

# Kickoff prompt for the next session

Paste this verbatim into a new Claude Code session in
`~/workspace/antimeme/rocket_surgeon` to start landing one of these PRs.
Replace `WU-X` with the target work unit.

> I'm picking up Phase 3B replay work on `rocket_surgeon`. The plan and
> per-WU breakdown are in
> `.context/plans/2026-05-27-replay-reverse-divergence-fanout.md` — read
> that first. The source branch is `origin/phase3/replay-reverse-divergence`
> (do not rebase the whole thing onto master; it conflicts on commit 1).
>
> My target this session is **WU-X**. Land it as a standalone PR against
> master, gated by `cargo xtask ci`. Follow the operational gotchas in
> the plan doc — especially git identity, `gh-alt`, and the merge-policy
> rules in CLAUDE.md. If WU-X has unfinished dependencies (WU-A and WU-B
> must be merged first for C/D), check master and stop if they're missing.
>
> Read `.context/beads/BEAD-0018-replay-orchestration.md` for the original
> intent. Don't un-defer any TCK scenarios as part of this work — that's
> out of scope.
>
> Memory has the `gh-alt` persona reference and the merge-policy
> clarification — use them. When you're done, append a session-note under
> `.context/session-notes/` summarizing what landed and what didn't.
