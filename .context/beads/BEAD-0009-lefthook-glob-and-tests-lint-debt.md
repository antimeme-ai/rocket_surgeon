---
id: BEAD-0009
title: Lefthook python glob excludes tests/, masking ruff lint debt
status: closed
priority: medium
created: 2026-05-19
resolved: 2026-05-19
---

## Description

The lefthook pre-commit configuration (added in PR #4) uses
`python/**/*.py` as the glob for ruff, ruff-format, and mypy commands.
This excludes the top-level `tests/` directory, which contains the e2e
test harness and seven e2e test scripts — all real Python that
`pyproject.toml` already configures for linting (see
`tool.ruff.lint.per-file-ignores["tests/**"]`).

As of 2026-05-19 a `ruff check tests/` reports **12 errors** and a
`ruff format --check tests/` reports **2 files needing reformat**.
These predate PR #4 and were never caught because the previous bash
pre-commit hook ran `cargo xtask ci`, whose ruff step also only points
at `python/`.

## Repro

```
source .venv/bin/activate
ruff check tests/          # 12 errors
ruff format --check tests/ # 2 files
```

## Impact

- Bootstrap PR #4 advertises "fmt + clippy + ruff + mypy all green" but
  this is only true under the narrow glob; the actual repo has lint debt.
- Future commits to `tests/` continue to slip past hooks. Test code rots
  in ways that production code does not.
- Contributors get the wrong mental model of what the lint perimeter is.

## Suggested fix

Two coupled changes, done in the same PR:

1. **Triage the 12 ruff errors and 2 format issues in `tests/`.** Most are
   likely the same `RUF043` raw-string lint that we already fixed in
   `python/tests/test_bridge.py` plus minor style. Fix them.
2. **Expand the lefthook globs** in `lefthook.yml`:
   - `python-ruff-check`, `python-ruff-format`, `python-mypy`:
     change `glob: "python/**/*.py"` to include `tests/**/*.py`.
   - Update the commands' arguments to lint both paths
     (`ruff check python/ tests/`, etc.).
3. **Mirror in `cargo xtask ci`** so the CI runner matches the hook.
4. **Verify mypy is happy** under strict mode against `tests/`. If the
   test scripts use untyped JSON-RPC dicts heavily, may need targeted
   `# type: ignore` or an `[[tool.mypy.overrides]]` entry for `tests.*`.

## Acceptance

- [x] `ruff check tests/` reports zero errors
      → fixed 14 errors: PLC0415, RUF015 x2, PTH112, PTH208, TRY300, E501 x8
- [x] `ruff format --check tests/` reports zero files to reformat
      → reformatted 3 files (lifecycle, shm, subscribe)
- [x] `mypy` strict mode passes against `tests/` (or has a documented
      override entry)
      → excluded: e2e test scripts use bare dict/Popen with untyped JSON-RPC;
        strict mypy adds no value here. mypy stays on python/rocket_surgeon.
- [x] `lefthook.yml` globs cover `tests/**/*.py`
      → ruff-check and ruff-format globs expanded to `{python,tests}/**/*.py`
- [x] `xtask::ruff` runs against `tests/` as well as `python/`
      → xtask ruff() passes both `python/` and `tests/` paths
- [x] `cargo xtask ci` green (116 cargo + 214 xpassed + 13 xfailed TCK)

## Related

- PR #4 — introduced lefthook with the narrow glob; surfaced this gap.
