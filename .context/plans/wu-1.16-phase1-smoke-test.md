# Sub-plan: WU 1.16 — Phase 1 end-to-end smoke test

Branch: `wu-1.16-thing-works`, stacked on `fix/bead-0011-e2e-gate` (PR #10).

## Goal

A scripted end-to-end test that proves the Phase 1 stack is not just *running*
but *correct*: attach a real transformer, step the full forward pass, and check
the debugger's reported tensor summaries against a direct PyTorch computation.

## Scope decision

`docs/specs/plan.md §1.16` bundles three deliverables:

1. E2E smoke test, summaries verified vs PyTorch — **this PR**.
2. Zero/active-probe overhead benchmark (5% / 15% budgets) — **deferred**.
   Measuring sub-5% overhead on a laptop CPU is noise-dominated, and isolating
   hook overhead from JSON-RPC round-trip time needs its own measurement
   design. A flaky benchmark in the push gate is worse than no benchmark.
   Filed as a follow-up; not stacked here.
3. Remove `xfail` from Phase 0/1 TCK scenarios that now pass (the "214 xpassed"
   noted earlier) — **deferred**, mechanical and independent.

This PR delivers (1). It is the piece that answers "does the thing actually
work" — correctness, not performance.

## Production fix carried by this PR (discovered while planning)

`python/rocket_surgeon/views.py::_residual_stream_norm` discovers decoder
layers by hardcoding the path filter `parts[0] == "model" and
parts[1] == "layers"` — **Llama module naming**. GPT-2's decoder blocks are
`transformer.h.{N}`, so the view returns `VIEW_DATA_UNAVAILABLE` for any
GPT-2 model. The block outputs *are* captured (`gpt2_declaration` in
`adapter.rs` declares `GPT2Block` as `Container` → passive hook); the view
simply looks under the wrong name.

Fix: replace the hardcoded name filter with architecture-agnostic discovery —
find the `nn.ModuleList` whose length equals `model.config.num_hidden_layers`
(`model.layers` for Llama, `transformer.h` for GPT-2) and enumerate its
children. This makes `residual_stream_norm` work for both families.

So WU 1.16 here is a JSMNTL red→green: the GPT-2 smoke test is red against the
current view, the view fix turns it green.

## Why GPT-2 (not the tiny-random Llama)

- All 9 existing e2e tests use `hf-internal-testing/tiny-random-LlamaForCausalLM`.
  The worker ships a real GPT-2 adapter (`crates/rocket-surgeon-worker/src/adapter.rs`,
  `gpt2_declaration`) that has **zero e2e coverage**. This test closes that gap.
- "Prove it works" means a real model with real weights, not another toy.
- `gpt2` (124M) is the model `plan.md §1.16` names for CI. ~500 MB download,
  HF-cached after the first run; adds ~10-25s to a push that runs the suite.
  Accepted — the e2e suite already takes ~37s and this is the milestone test.

## Key facts established (research)

- Daemon forward-pass input is **deterministic**: `bridge.rs:run_forward` passes
  `torch.zeros((1, 2)).long()` → `input_ids = [[0, 0]]`. The test replicates
  this exactly — no randomness to control for.
- Tensor summary stats are computed Rust-side in `crates/rocket-surgeon/src/
  tensor_stats.rs::compute_summary`: `mean` (Welford), `std` (population,
  `sqrt(m2/n)`), `l2_norm` (LAPACK scaled), histogram, top-k, sparsity.
  PyTorch equivalents: `t.mean()`, `t.std(correction=0)`, `t.norm(2)`.
- `model_family` for GPT-2 is `"gpt2"`; orchestrator `SUPPORTED_FAMILIES`
  includes it.
- `residual_stream_norm` view returns one norm per layer (seen in
  `test_e2e_view.py`). Exact formula to be confirmed from the view impl when
  wiring the comparison.

## Test design — `tests/test_e2e_phase1.py`

Standalone script in the established e2e style (`run_test()`, `build_binaries()`
in `__main__`), so `cargo xtask e2e` auto-discovers and gates it.

Steps:

1. **initialize** — protocol handshake.
2. **attach** `model_path="gpt2"`, `model_family="gpt2"`, `device="cpu"`,
   `num_ranks=1`. Assert `status == stopped`, capture `num_layers` (expect 12),
   `num_heads` (12), `hidden_dim` (768).
3. **step** the full forward pass at `granularity="component"` with a `count`
   large enough to drain it; assert ticks executed and monotonic `tick_id`.
4. **residual_stream_norm view** for every layer → daemon norms.
5. **inspect** `model:0:*:*:*:fwd` → at least one concrete tensor with
   `stats` (mean/std/l2_norm) and `shape`.
6. **Reference computation** in-process: `AutoModelForCausalLM.from_pretrained(
   "gpt2")` in float32, `model(torch.zeros((1,2), dtype=torch.long),
   output_hidden_states=True)`. `hidden_states` is the residual stream.
7. **Assertions**:
   - Per-layer residual norm: daemon vs `torch.norm` of the matching
     `hidden_states` entry — within relative tolerance (start `1e-3`, tighten
     to whatever passes; f32 Welford/scaled-L2 vs torch pairwise summation
     will not match to `1e-5` on 768-wide reductions, so `plan.md`'s `1e-5`
     is treated as aspirational and the achieved tolerance is reported).
   - Spot-check one inspected tensor's `mean`/`std`/`l2_norm` against torch on
     the same tensor, same tolerance — exercises the stats engine directly.
8. **detach** → assert `status == initialized`.
9. Clean daemon shutdown (the `finally` pattern from the other e2e scripts).

Parseable stdout: print each daemon-vs-torch pair as
`[verify] layer N: daemon=… torch=… rel_err=…` so drift is greppable across
commits (satisfies the `plan.md` "parseable format" criterion).

## Resolved during planning

- `residual_stream_norm` formula: full-tensor `torch.norm(t.float(), p=2)` of
  each decoder-block output. The block output for GPT-2 is a tuple; the view
  takes element 0. Reference: `torch.norm(out.hidden_states[i+1].float(),
  p=2)` — `hidden_states[i+1]` is the output of block `i`
  (`hidden_states[0]` is the embedding output). Tolerance can be tight
  (`rel=1e-4`): both sides are `torch.norm` of the same deterministic forward
  pass; read-only hooks do not perturb the math.
- `step`: use a large `count` (the `test_e2e_view.py` precedent uses 1000)
  to drain the whole forward pass; the response reports `ticks_executed`.
- Inspect spot-check: kept structural (tensors returned, `stats` present and
  finite, plausible `shape`) — this adds GPT-2 `inspect` coverage, which is
  currently absent. The deep numeric proof is the per-layer view comparison;
  the Rust stats engine already has a torch-comparison unit test
  (`python/tests/test_bridge_stats.py`).

## TCK note

WU 1.16 writes no new `.feature` files — `plan.md` says its TCK target is
"all Phase 1 TCK scenarios green," which is deferred item (3). This script is
itself the behavioral check for the integrated stack.

## Verification

- `python tests/test_e2e_phase1.py` passes.
- `cargo xtask e2e` — full suite still green, new script included (10 scripts).
- Code-reviewer subagent over the diff; fix all findings.
- `git push` exercises the new script through the `e2e` pre-push gate.
