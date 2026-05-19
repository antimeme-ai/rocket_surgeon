---
id: BEAD-0011
title: E2E test suite is not run by any gate
status: open
priority: high
created: 2026-05-19
---

## Description

The `tests/test_e2e_*.py` suite (9 scripts that spawn the real daemon,
orchestrator, and worker and exercise the full JSON-RPC protocol) is **not
executed by any automated gate**:

- `.github/workflows/` does not exist — there is no CI at all.
- `lefthook.yml` pre-push runs only `cargo xtask test` (Rust unit + doc tests)
  and `pytest python/tests/` (Python unit/TCK tests). Neither command picks up
  the top-level `tests/` directory.
- `cargo xtask ci` likewise does not invoke the e2e scripts.

This is how `test_e2e_perfetto.py` landed broken in PR #7 (WU 1.15) with
four independent defects — wrong method names (`rocket/attach`, `rocket/detach`
vs. the canonical `attach`, `detach`), a missing `device: "cpu"` /
`num_ranks: 1` in the attach payload (triggers a real `BACKEND_ATTACH_FAILED`
asking for `accelerate`), and a missing `direction` / `count` / `granularity`
in the step payload. A latent bug in `tests/e2e_harness.py` (asserting
`tick_id: int` instead of `Option<u64>`) was uncovered by the same test.

The e2e suite is the only test layer that actually proves the three-process
architecture works end-to-end. Not running it on every commit means we ship
broken integration paths and discover them only when a human happens to run a
script by hand.

## Acceptance criteria

- E2E suite runs on every push to a PR branch (or, at minimum, on `pre-push`).
- The runner builds the workspace binaries (or relies on `cargo xtask setup`)
  before invoking the scripts.
- Test discovery picks up `tests/test_e2e_*.py` automatically — no per-test
  registration.
- First-run cost (HF tiny-llama download, ~few MB) is acceptable; subsequent
  runs use the HF cache.
- The build the gate executes against is the debug build that the scripts
  expect at `target/debug/`.

## Options to explore

1. **Add `tests/test_e2e_*.py` to lefthook pre-push.** Cheapest. Cost: every
   push pays the build + 9-script runtime (~30-60s after first download).
2. **Add a GitHub Actions workflow.** Better separation of concerns —
   pre-push stays fast, CI catches regressions before merge. Needs HF
   credentials only if rate-limited; tiny-llama is public.
3. **Add a `cargo xtask e2e` recipe and invoke it from both `cargo xtask ci`
   and a pre-push hook.** Matches the existing xtask convention.

Option 3 is probably the right shape, but the trade-off vs option 2 (which
also covers the no-CI gap) deserves its own decision.

## Related

- PR #7 (WU 1.15 Perfetto trace sink) — landed with four broken e2e calls.
- BEAD-0009 — closed lint perimeter gap; same class of defect (gate didn't
  cover the directory the bug lived in).
