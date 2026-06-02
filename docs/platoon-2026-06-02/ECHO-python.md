# PLATOON-FINDINGS — ECHO (Python lane)

**Brief:** B002 — bring RS's test suite up to MATERIA oracle standards.
**Lane:** `python/rocket_surgeon` — hypothesis properties over replay divergence
math, checkpoint capture/restore, and intervention target matching.
**Branch:** `platoon/test-python`. **Date:** 2026-06-02.

## Summary

The Python suite had **0** hypothesis tests (189 example-based functions). This
lane adds **29 property / metamorphic / model-based / exception-raising tests**
across three modules, lifting their oracles from tiers 2–3 to tiers 4–6. The
property-based runs surfaced **one genuine latent bug** (NaN divergence silently
swallowed), **one doc/impl divergence** (bracket-strip scope), **one weak oracle**
(shape-mismatch raises a generic `RuntimeError`), and **refuted one plausible
metamorphic relation** (cosine is not monotone under tick-append).

Dev-dep added: `hypothesis>=6.0` (in `pyproject.toml [project.optional-dependencies].dev`;
installed into the main-checkout `.venv` via `uv pip`).

Run command (per brief — tests THIS worktree's code, not the editable install):
```
PYTHONPATH=$PWD/python /Users/patrickbeam/projects/rocket_surgeon/.venv/bin/pytest \
    python/tests/test_replay_properties.py \
    python/tests/test_checkpoint_properties.py \
    python/tests/test_matching_properties.py
```

## Techniques applied (oracle tier in parens)

### `replay.py` — divergence math (`test_replay_properties.py`, 12 tests)
- **Metamorphic (4):** cosine scale-invariance under positive scaling (t4);
  cosine sign-flip under negation (t4); cosine symmetry `cos(a,b)==cos(b,a)` (t4);
  `max_relative_error` non-increasing when an identical block is appended to both
  sides (t4 — the per-tensor analogue of "appending matching ticks never increases
  divergence").
- **Property (2):** identical tensors report no divergence (t5); threshold
  monotonicity — tightening `cosine_threshold` can only add divergence, never
  remove it (t5).
- **Equivalence/roundtrip (1):** `compare_activations_from_ptr` over a tensor's
  raw bytes equals `compare_activations` on the same data (t4/t6 — the FFI byte
  path must match the in-memory path).
- **Exception-raising (5):** unsupported dtype string → `ValueError` (t2, 113×
  class); shape mismatch → raises (t2); zero/zero → no divergence, zero/nonzero →
  divergence (t2 boundary); NaN behaviour pinned (see R1).

To read raw metrics regardless of tolerance, tests use a **probe configuration**:
`cosine_threshold=1.1` (always < so the dict is returned) with `mre_threshold=inf`.

### `checkpoint.py` — capture/restore bridge (`test_checkpoint_properties.py`, 10 tests)
- **Roundtrip (1, crown jewel):** `capture` → arena bytes → `restore` is the
  identity, bit-for-bit, across float16/float32/float64 and 1–3-D shapes (t4).
- **Model (2):** capture's returned `(dtype, shape)` equals the abstract
  `(str(dtype), list(shape))` of the source (t6); a tuple-valued activation is
  handled identically to its first element (t6).
- **Model (1):** `activation_available` agrees with key-set membership for all
  inputs — the dict of keys *is* the abstract model (t6).
- **Exception-raising (4):** missing key → `KeyError` (t2); undersized capture
  slot → `ValueError "exceeds slot capacity"` (t2 — guards a buffer overrun);
  undersized restore slot → `ValueError` (t2); restore into empty dict →
  `KeyError "cannot restore"` (t2).
- **Metamorphic (1):** CPU RNG `restore(capture())` reproduces the exact random
  stream that followed capture (t4); CUDA path no-op pinned on the CPU-only host.

### `matching.py` — intervention target matching (`test_matching_properties.py`, 7 tests)
- **Model (1, flagship):** `target_matches` agrees with an **independent**
  reference matcher (built with `rpartition`, not the production anchored regex)
  across 600 generated `(target, point)` pairs spanning exact / wildcard / bracket
  / noise / wrong-segment-count cases (t6).
- **Roundtrip (1):** `strip_bracket(f"{base}[{i}]")==base` and
  `extract_head_index==i` for arbitrary base + index (t4).
- **Property/edge (2):** no-bracket segment is identity + `None`; malformed
  brackets (`[]`, `[-1]`, `[x]`, unbalanced) extract `None` (t5).
- **Exception-raising (2):** wrong segment count → `False`, never raises (t2);
  arbitrary text → returns a bool (t2 implicit oracle + postcondition).
- **Regression pin (1):** the doc/impl bracket-strip divergence (Finding M1).

## Generator distribution evidence

Measured with `hypothesis.event` + `--hypothesis-show-statistics`. Inputs are
**not** trivial-dominated:

- `target_matches` model property (600 ex): **65.6% match / 28.1% no-match**;
  segment counts spread 1,2,3,4,5(70%),6,8 — both decision branches and all
  malformed-arity cases are exercised.
- `capture_restore_roundtrip` (400 ex): dtypes ~27–30% each (f16/f32/f64);
  `ndim` 1/2/3 each ~25–31%; `nelem` buckets spread 2→256.
- `activation_available` (200 ex): **27.4% hit / 54.3% miss** — both membership
  branches well represented (note ~37% retried draws from the unique-key filter;
  acceptable, < the >10% *discard* concern since these are retries that still
  yield valid examples, not precondition rejections of generated tests).
- `unsupported_dtype` (200 ex): 0 invalid, all-unique diverse junk strings.

## Bugs / weak oracles / refutations

### R1 — KNOWN BUG: NaN in the replayed tensor is reported as *no divergence*
**Severity: high for a debugger.** `compare_activations` computes `cosine_sim`
and `max_relative_error`, then reports divergence iff
`cosine_sim < cosine_threshold or max_relative_error > mre_threshold`. When the
replayed tensor contains a NaN, both metrics are `NaN`, and **both comparisons are
`False` for NaN** — so the function returns `None` ("within tolerance, no
divergence"). A replay that blew up into NaN — arguably the single most important
divergence to surface — is silently dropped.

Minimal failing case:
```python
a = torch.ones(8); b = a.clone(); b[0] = float("nan")
compare_activations(a, b, cosine_threshold=0.999, mre_threshold=0.05)  # -> None  (BUG)
```
Root cause: `replay.py:30` — `cosine_sim < cosine_threshold or max_rel_error > mre_threshold`
has no `NaN` guard. **Fix (deferred — not in scope to change prod behaviour):**
treat `not isfinite(cosine_sim)` or `not isfinite(max_rel_error)` as divergence,
e.g. return the dict when either metric is non-finite. Pinned by
`test_nan_in_replay_is_silently_swallowed_known_bug` so a deliberate fix flips it.
(For `+inf`/`-inf` the bug is partially masked: `max_relative_error` becomes
`+inf` which trips `> mre_threshold` for finite thresholds — but the reported
`cosine_similarity` is still `NaN`, and an infinite `mre_threshold` would swallow
it too.)

### M1 — Doc/impl divergence: bracket strip is applied to *every* segment
`matching.py` docstring states bracket notation is stripped "on component", but
`target_matches` calls `strip_bracket` on **every** pattern segment (`matching.py:43`).
So `gpt2[1]:0:9:o_proj:output` matches family `gpt2` — a bracket on the family
segment is silently accepted. Low real-world impact (rank/layer are ints; family
rarely carries a bracket), but the spec and code disagree. Pinned by
`test_bracket_is_stripped_on_every_segment_not_just_component_known`. **Fix
options:** (a) narrow the strip to the component index only; (b) update the
docstring to state all-segment stripping is intentional. Left for the owner.

### W1 — Weak oracle: shape mismatch raises a generic `RuntimeError`
`compare_activations(torch.ones(8), torch.ones(9), ...)` flattens both and calls
`torch.dot`, which raises a low-level `RuntimeError "inconsistent tensor size"`.
There is no shape/element-count validation, so the error message doesn't name the
divergence-comparison context. Pinned (current behaviour) by
`test_shape_mismatch_raises_runtimeerror`. **Stronger contract (deferred):** an
explicit `ValueError` like `"cannot compare activations of differing element
counts: {n} vs {m}"`.

### Refuted relation — cosine divergence is NOT monotone under tick-append
The brief's suggested invariant "appending matching ticks never increases
divergence" is **false for the cosine metric**. Counterexample found by the suite:
`a = ones(48)`, `b = 2*ones(48)` have `cosine == 1.0` (parallel), but appending an
identical block `ones(48)` to both yields `cosine ≈ 0.949` — because cosine is
scale-invariant, the shared block injects absolute scale and *lowers* similarity.
The relation holds only for `max_relative_error` (which the appended block leaves
unchanged). Recorded so a higher-level replay-divergence invariant doesn't lean on
cosine monotonicity. (This is an honest refutation, not a code bug — it's a
property of cosine similarity.)

## Gaps left for follow-up

- **GPU paths uncovered.** `register_cuda_pinned` / `unregister_cuda_pinned`, the
  CUDA RNG capture/restore loop, and `is_cuda` `torch.cuda.synchronize()` branches
  in `checkpoint.py` are exercised only in their no-CUDA fallback (host has no
  CUDA). A CUDA-equipped runner should add the device-count > 0 RNG roundtrip and
  pinned-memory register/unregister roundtrip.
- **bfloat16 capture/restore.** Excluded from the roundtrip strategy because
  numpy has no native bfloat16 to generate finite elements cleanly; `restore`
  supports it (`_ELEMENT_SIZES`). A torch-native bfloat16 generator would close
  this.
- **`compare_activations_from_ptr` reshape mismatch.** Not yet tested: passing a
  `original_shape` whose product ≠ `original_len/elemsize` (should error, not
  silently mis-reshape). Candidate exception-raising property for a follow-up.
- **R1/M1/W1 fixes** are intentionally NOT applied (prod-behaviour changes beyond
  "add tests + record"); the pins will flip when an owner addresses them.

## Stop condition

New property/metamorphic/model tests green (29 added, 49 total in lane incl.
baseline). `ruff check` + `ruff format --check` + `mypy` clean on all three new
files. This findings file committed on `platoon/test-python`.
