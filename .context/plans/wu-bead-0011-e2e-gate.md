# Sub-plan: BEAD-0011 — gate the e2e suite

## Problem

`tests/test_e2e_*.py` (9 scripts that spawn the real daemon/orchestrator/worker
and drive the JSON-RPC protocol) is run by no gate. No `.github/workflows/`
exists; `lefthook` pre-push runs only `cargo xtask test` and
`pytest python/tests/`. This is how PR #7 shipped four broken e2e calls.

## Decision

Option 1 from BEAD-0011: add a `cargo xtask e2e` recipe, wire it into
`cargo xtask ci`, and add an `e2e` command to `lefthook` pre-push. Fully local
enforcement, closes the gap today. GitHub Actions (the no-CI gap) is left as a
separate, larger piece of infra — not in scope here.

Cost accepted: every `git push` runs the full e2e suite (~30-60s after the
first HF tiny-llama download; cargo build is incremental so the 9 per-script
rebuilds are near-free after the first).

## TCK note

This is build tooling, not protocol behavior — Gherkin/TCK does not apply. The
recipe's own correctness is verified by running it (see Verification).

## Changes

### 1. `xtask/src/main.rs`
- Add `E2e` variant to the `Xtask` enum (doc: "Run end-to-end tests").
- Add `Xtask::E2e => e2e()?` to the match.
- Add `e2e()?` to the `Ci` branch, after `pytest()`.
- New `fn e2e()`:
  - Resolve `tests/` under the current dir (xtask runs from repo root).
  - `read_dir`, keep files matching `test_e2e_*.py`, sort for determinism.
  - `bail!` if none found (guards against a silently-empty gate).
  - Run each as `python3 -u <script>`; collect failures, run all scripts
    regardless, then `bail!` with the full failure list if any failed.
  - Each script self-builds binaries and sets its own child env (DYLD etc.),
    so the recipe needs no special environment.

### 2. `lefthook.yml`
- Add an `e2e` command to `pre-push.commands` with the same `PATH` override
  as `cargo-test` so `python3` resolves to the venv interpreter.
- Switch `pre-push` from `parallel: true` to `piped: true`. Code review
  finding: `cargo-test` and `e2e` both build the workspace and share
  `target/`; run concurrently they serialize on Cargo's build lock and
  thrash the PyO3 feature-set rebuild. `piped` runs the commands
  sequentially in definition order and fails fast.

### 3. `.context/beads/BEAD-0011-e2e-suite-not-gated.md`
- Mark `status: closed`, add `resolved: 2026-05-20`.

## Verification

- `cargo xtask e2e` → all 9 scripts pass.
- `cargo xtask e2e` with a deliberately broken script → non-zero exit, failure
  named (manual spot-check, not committed).
- `cargo build -p xtask` clean; `cargo clippy` clean.
- Code-reviewer subagent over the diff; fix all findings.
- `git push` exercises the new pre-push `e2e` command end-to-end.
