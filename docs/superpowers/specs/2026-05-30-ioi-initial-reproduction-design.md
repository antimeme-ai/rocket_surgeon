# IOI Initial Reproduction — Design Spec

**Date:** 2026-05-30
**Branch:** phase3/replay-reverse-divergence (or new feature branch)
**Goal:** Answer the 10th Dentist's condition #3 — demonstrate that rocket_surgeon can reproduce a published interpretability result (Wang et al. 2023 IOI circuit) on GPT-2 124M.

---

## Motivation

The advisory's 10th Dentist identified five conditions that would validate the project. Condition #3: "A successful IOI circuit reproduction using the tool." This is the nearest achievable condition — we just ran a prompted forward pass through GPT-2 124M, but inspect returned no tensors and the architecture has never performed a real interpretability analysis.

Three changes and one test script prove the stepping debugger architecture can do real mechanistic interpretability work.

---

## Changes

### 1. Fix Layer Step Boundary Semantics

**File:** `crates/rocket-surgeon-worker/src/dispatch.rs`, `run_step_loop`

**Problem:** When `granularity == Layer` and `ticks_consumed >= ticks_to_drain`, the loop breaks immediately. This fires at the first component of the Nth new layer boundary — only `ln1` of the boundary layer has executed. The `stopped_at` reports this new layer, but none of its components (down_proj, o_proj, etc.) have data in `last_outputs`. Inspect and checkpoint both fail.

**Fix:** Add a `draining` flag. When `ticks_consumed >= ticks_to_drain`, set `draining = true`. While draining, keep consuming ticks. Break only when the next layer boundary is detected (layer index changes again) or `forward_complete` fires.

After the fix, "step 3 layers" means "advance through 3 complete layers." The `stopped_at.layer` is the last fully completed layer. All components at that layer have data available for inspect and checkpoint.

**Impact on existing tests:** All e2e tests use component granularity, so unaffected. `try_gpt2.py` uses layer granularity and would now stop at the correct position.

### 2. Head-Level Bracket Notation Semantics

**File:** `crates/rocket-surgeon-worker/src/dispatch.rs` (intervention + inspect paths), `crates/rocket-surgeon-worker/src/adapter.rs` (component type info)

**Grammar:** Already supports bracket notation. `o_proj[7]` parses as `ComponentSeg::Indexed { name: "o_proj", index: 7 }`. No grammar changes.

**Semantic rule:** Head slicing applies when a bracket index appears on a `Direct` component whose canonical name is one of `{o_proj, q_proj, k_proj, v_proj}` — the attention-path components whose output dimension equals `hidden_dim`:

- Compute `head_dim = hidden_dim / num_heads` (both values known from attach)
- Slice: `tensor[..., idx * head_dim : (idx + 1) * head_dim]`

When a bracket index appears on an indexed module group (like MoE `experts[3]`), it remains a module selector as before. A bracket index on any other Direct component (e.g., `lm_head[3]`) is an error — the canonical name check rejects it.

**How the worker distinguishes:** The adapter's resolved component tree already marks components as `Direct` vs `Indexed` children. A `Direct` component with a bracket index AND a canonical name in the attention set = head slice. An `Indexed` component group with a bracket index = module selector. Otherwise = error.

**For interventions:** Slice the target tensor to the head range, apply the intervention (ablate, scale, patch) to the slice, write back. One branch in `apply_interventions_at_point`.

**For inspect:** Return stats computed on the head slice only. One branch in `collect_tensors`.

**Not in scope:** `o_proj[*]` wildcard head iteration. The client generates individual targets per head.

### 3. IOI Initial Reproduction Test

**File:** `tests/test_ioi_circuit.py`

**What it does:** Runs a single-prompt zero-ablation sweep across all 144 attention heads of GPT-2 124M, measuring the causal effect of each head on the IOI task.

**Structure:**

1. **Setup:** Initialize, attach GPT-2 124M (CPU). Hardcoded IOI prompt token IDs for "When Mary and John went to the store, John gave a drink to". Hardcoded token IDs for " Mary" and " John".

2. **Baseline run:** Step full forward pass with prompt tokens. Inspect `gpt2:0:*:lm_head:0:output` (detail=full or slice at last position). Compute `baseline_logit_diff = logits[Mary_token] - logits[John_token]`. Assert positive (model predicts Mary). Create checkpoint.

3. **Ablation sweep:** For each (layer, head) in 0..12 × 0..12:
   - Restore from baseline checkpoint
   - Set intervention: `{ type: "ablate", mode: "zero", target: "gpt2:0:{layer}:o_proj[{head}]:0:output" }`
   - Step full forward pass (checkpoint has the input state, no tokens needed)
   - Inspect `lm_head` output, compute ablated logit_diff
   - Record `delta = baseline_logit_diff - ablated_logit_diff`
   - Clear intervention

4. **Validation:** Sort heads by `|delta|`. The top heads should include known name mover heads from Wang et al. 2023 (heads at layers 9-11 with large positive deltas). We validate that a sparse subset of heads drives the logit_diff — not exact numerical reproduction, but structural agreement with the published circuit.

5. **Output:** Print a ranked table of heads by causal effect. Flag heads where `|delta| > 0.1 * baseline_logit_diff`.

**What this is NOT:**
- Not a full Wang et al. reproduction (they used hundreds of prompts, counterfactual patching, functional group assignment)
- Not a unit test — manual smoke test like `try_gpt2.py` (requires model download, real compute time)
- Not proof that stepping is superior to batch execution — proof that the architecture can perform real interpretability analysis

**Runtime estimate:** 144 forward passes × ~0.012s each ≈ 1.7s compute on CPU, plus IPC overhead. Under 30 seconds total.

---

## Execution Phases

Each phase is a TDD execution cycle (TCK red → green → refactor).

| Phase | Deliverable | Depends on |
|-------|-------------|------------|
| A | Layer step boundary fix + tests | — |
| B | Head-level bracket semantics + tests | — |
| C | test_ioi_circuit.py (initial reproduction) | A, B |

Phases A and B are independent and can be developed in any order. Phase C requires both.

---

## Not in Scope

- `rocket/sweep` server-side handler (deferred until client loop validated)
- Patch intervention wiring (`previous_outputs`, `source_tensor_id` resolution) — needed for full counterfactual patching, not for zero-ablation
- Head-level tick granularity (`TickGranularity::Head`)
- `o_proj[*]` wildcard head iteration
- Full Wang et al. multi-prompt reproduction
- GPU testing
- Counterfactual activation patching (clean→corrupted prompt cross-run patching)
- `SweepTrial.collect` (tensor capture during sweep trials)

---

## Success Criteria

1. `try_gpt2.py` inspect returns real tensor data with non-zero stats after stepping through layers
2. Zero-ablation on `o_proj[7]` only affects head 7's slice (other heads unchanged)
3. IOI baseline logit_diff is positive (model predicts Mary over John)
4. Ablation sweep identifies a sparse subset of heads (<30 of 144) driving >80% of the logit_diff
5. Top-effect heads include at least some heads in layers 9-11 (consistent with Wang et al. name movers)
