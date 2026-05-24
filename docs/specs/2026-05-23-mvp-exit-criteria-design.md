# MVP Exit Criteria Gap Closure — Design Spec

Date: 2026-05-23

## Goal

Close all remaining Phase 2 exit criteria gaps so the MVP gate
(`docs/specs/plan.md:806-817`) passes without caveats. Six sub-projects,
executed in dependency order, each with its own JSMNTL cycle.

## Scope

Six gaps remain against the stated exit criteria:

| # | Gap | Sub-project |
|---|-----|-------------|
| 1 | Bundle missing 5 of 9 artifacts | A — Bundle Completion |
| 2 | Inspect can't capture logits for IOI measurement | B — Inspect Pipeline + Prompt Input |
| 3 | IOI test doesn't measure logit difference | C — IOI Logit Measurement |
| 4 | Conformance validates vocabulary only, no firing order, no Llama | D — Conformance + Probe Ordering |
| 5 | No overhead benchmarking | E — Overhead Benchmarking |
| 6 | TCK not fully green (38 features, most xfail/stub) | F — TCK Green |

## Dependency Graph and Execution Order

```
A (bundle)          E (overhead)
                         |
B (inspect) ──► C (IOI logits)
                          |
D (conformance)           |
                          |
                    F (all TCK) ◄── depends on A,B,C,D,E
```

D (conformance) uses `stopped_at` from step responses to validate firing
order — it does not depend on B. It can run in parallel with B/C.

Execution order: A → B → C → D → E → F

Each sub-project gets its own branch, plan doc, TCK-first execution cycle,
code review, and PR. Sub-project F is the largest — it pulls Phase 3-7
features forward to green all 38 TCK feature files.

## Non-goals

- MCP adapter (punted indefinitely)
- Multi-GPU intervention dispatch (Phase 5 scope, not exit criterion)
- Head-level tick granularity (Phase 7)

---

## Sub-project A — Bundle Completion

### Goal

Add the 5 missing artifacts to session bundle export: `model-info.json`,
`env.json`, `prompt.json`, `trace.perfetto-trace`, `bookmarks.json`.

### Current state

`handle_export` in `dispatch.rs:1133-1231` assembles 4 artifacts: manifest,
interventions, protocol-trace, and tensors. The protocol defines
`HostExportEnvRequest`/`HostExportEnvResponse` (`messages.rs:1061-1075`)
for collecting env/model/prompt data from the worker, but no handler exists.
The `PerfettoSink` accumulates trace data during the session and exposes
`path()`, but the export handler doesn't read it.

### Design

**Worker-side collection**: Implement `handle_host_export_env` in the worker
dispatch (`crates/rocket-surgeon-worker/src/dispatch.rs`). It calls a new
Python bridge function `collect_export_env()` that queries:

- `env.json`: `torch.__version__`, `torch.version.cuda`,
  `torch.cuda.get_device_name()`, `platform.platform()`,
  `sys.version`, `rocket_surgeon.__version__`
- `model-info.json`: model family, path, num_layers, num_heads, hidden_dim,
  parameter count (`sum(p.numel() for p in model.parameters())`), dtype
- `prompt.json`: `None` for MVP (no prompt.set verb yet — see Sub-project B)

**Daemon-side assembly**: Extend `handle_export` to:
1. Call `_host/export_env` via the orchestrator to get the three JSON blobs
2. Read the Perfetto trace file via `perfetto_path` (passed from `main.rs`)
3. Add `bookmarks.json` as an empty array (`[]`)

**Signature change**: `handle_export` gains two new parameters:
- `orchestrator: &mut OrchestratorHandle` — to send `_host/export_env`
- `perfetto_path: Option<&Path>` — from `perfetto.as_ref().map(|p| p.path())`

### Files changed

| File | Change |
|------|--------|
| `python/rocket_surgeon/bridge.py` | Add `collect_export_env(handle)` |
| `crates/rocket-surgeon-worker/src/bridge.rs` | Add `collect_export_env` PyO3 call |
| `crates/rocket-surgeon-worker/src/dispatch.rs` | Add `handle_host_export_env` |
| `crates/rocket-surgeon/src/dispatch.rs` | Extend `handle_export` signature and body |
| `crates/rocket-surgeon/src/main.rs` | Pass orchestrator + perfetto_path to `handle_export` |
| `tests/test_e2e_bundle.py` | Validate all 9 artifacts present |
| `tck/protocol/session-export.feature` | Add scenario for 9-artifact completeness |

### Artifact schemas

**env.json**:
```json
{
  "torch_version": "2.3.0",
  "cuda_version": "12.1",
  "cuda_available": true,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "nccl_version": null,
  "python_version": "3.11.9",
  "os": "Linux 5.15.0",
  "rocket_surgeon_version": "0.1.0"
}
```

**model-info.json**:
```json
{
  "model_family": "gpt2",
  "model_path": "gpt2",
  "num_layers": 12,
  "num_heads": 12,
  "hidden_dim": 768,
  "num_params": 124439808,
  "dtype": "float32"
}
```

**prompt.json**: `null` for MVP. Becomes populated once Sub-project B lands
the `input_ids` field on `HostStepRequest`.

**bookmarks.json**: `[]` — empty array, placeholder for Phase 3.

**trace.perfetto-trace**: Raw bytes from `PerfettoSink.path()`, included only
if the sink was active during the session.

---

## Sub-project B — Inspect Pipeline + Prompt Input

### Goal

Enable end-to-end logit capture: set input tokens, run a forward pass,
inspect `lm_head` output, retrieve raw tensor data.

### Current state

- `run_forward` in `bridge.rs:318-331` hardcodes `torch.zeros((1, 2), dtype=long)`
  as input — no mechanism to pass real tokens
- `HostStepRequest` has no `input_ids` field
- `StepRequest` has no `tokens` field
- Inspect works for summary stats + byte-level slicing (`InspectDetail::Slice`)
  but `lm_head` is not in the daemon's `default_catalog` (session.rs:180-198)

### Design

**Prompt input via StepRequest**: Add `tokens: Option<Vec<u64>>` to
`StepRequest` and `input_ids: Option<Vec<u64>>` to `HostStepRequest`. On the
first step of a session (or after a checkpoint restore), if `tokens` is
provided, pass them as `torch.tensor([tokens], dtype=long)` to
`bridge.run_forward()`. On subsequent steps the forward pass is already
running; `tokens` is ignored (or errors if provided after the first step).

**Bridge change**: `run_forward` in both `bridge.rs` and `bridge.py` accepts
an `input_ids` parameter instead of creating dummy zeros. The Python side
receives a list of ints and converts to `torch.tensor([ids], dtype=torch.long)`.

**Catalog expansion**: Add `lm_head` to `default_catalog` in `session.rs`
so `rocket/discover` returns it and `rocket/inspect` can target it.

**No new verb needed**: Token input piggybacks on the first `rocket/step`
call. This avoids protocol expansion — the step verb already carries the
execution context.

### Files changed

| File | Change |
|------|--------|
| `crates/rocket-surgeon-protocol/src/messages.rs` | Add `tokens` to `StepRequest`, `input_ids` to `HostStepRequest` |
| `crates/rocket-surgeon/src/session.rs` | Store `input_ids` in session, pass through step, add `lm_head` to catalog |
| `crates/rocket-surgeon/src/dispatch.rs` | Forward tokens from `StepRequest` to `HostStepRequest` |
| `crates/rocket-surgeon-worker/src/dispatch.rs` | Pass `input_ids` from `HostStepRequest` to bridge |
| `crates/rocket-surgeon-worker/src/bridge.rs` | Accept `input_ids: Option<Vec<u64>>`, pass to Python |
| `python/rocket_surgeon/bridge.py` | `run_forward` accepts `input_ids` parameter |
| `crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs` | Update roundtrip tests |

---

## Sub-project C — IOI Logit Measurement

### Goal

Extend the IOI acceptance test to measure logit difference and validate
≥50% reduction after name-mover head ablation. Per the exit criterion
(`plan.md:790-803`).

### Design

The test runs two forward passes over the same IOI prompt:

1. **Clean baseline**: Set IOI prompt tokens via `tokens` field in
   `rocket/step`. Step through the entire forward pass. Inspect
   `gpt2:0:*:lm_head:output` to capture the full logit tensor. Decode
   the base64 tensor bytes, reshape to `[1, seq_len, vocab_size]`. Extract
   `logit_IO = logits[0, -1, io_token_id]` and
   `logit_S = logits[0, -1, s_token_id]`. Compute
   `clean_diff = logit_IO - logit_S`.

2. **Ablated run**: Detach and re-attach to reset the forward pass.
   Register ablate interventions on name-mover heads
   (layers 9.9, 9.6, 10.0). Step with the same tokens. Inspect lm_head
   again. Compute `ablated_diff = logit_IO - logit_S`.

3. **Assert**: `(clean_diff - ablated_diff) / clean_diff >= 0.50`

4. **Side effect**: Export session bundle (validates Sub-project A).

The test runs on GPT-2-small (CPU), marked `@slow`. Uses IOI prompt from
`fixtures/ioi_prompts.json`. Token IDs for IO and S positions are derived
from the prompt template.

### Files changed

| File | Change |
|------|--------|
| `python/tests/test_ioi_acceptance.py` | Rewrite with logit measurement |
| `python/tests/fixtures/ioi_prompts.json` | Add token IDs for IO/S positions |

---

## Sub-project D — Conformance + Probe Ordering

### Goal

Validate that components fire in the expected canonical order during a
forward pass. Add Llama conformance test.

### Current state

`test_gpt2.py` validates component vocabulary via `rocket/discover` but
does not verify firing order. No Llama test exists.

### Design

**Probe firing order test**: Step through the forward pass one tick at a
time (`count: 1`), collecting `stopped_at.component` and
`stopped_at.layer` from each step response. Build an ordered list of
`(layer, component)` pairs. Assert this matches the expected canonical
order:

```
(0, "ln1"), (0, "attn.q_proj"), (0, "attn.k_proj"), (0, "attn.v_proj"),
(0, "attn.o_proj"), (0, "attn.scores"), (0, "ln2"), (0, "mlp"),
(0, "residual_post"),
(1, "ln1"), (1, "attn.q_proj"), ...
```

The exact ordering depends on what the daemon's tick model produces.
The test should first step through entirely and record the order, then
validate properties: monotonically increasing layers, all expected
components present per layer, no duplicates.

**Llama conformance**: Same structure as GPT-2 test but for
`meta-llama/Llama-3.2-1B` (smallest available). Marked `@slow`/`@nightly`.
Expected components differ (standard `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, `down_proj` plus `ln1`, `ln2`,
`residual_post`).

### Files changed

| File | Change |
|------|--------|
| `python/tests/conformance/test_gpt2.py` | Add `test_probe_firing_order` |
| `python/tests/conformance/test_llama.py` | New: Llama firing-order + vocabulary test |

---

## Sub-project E — Overhead Benchmarking

### Goal

Measure intervention overhead and assert ≤2% regression versus baseline
stepping. Per exit criterion (`plan.md:815`).

### Design

**Benchmark harness**: Use `criterion` crate for Rust-side microbenchmarks.
Create `crates/rocket-surgeon/benches/step_overhead.rs`. The benchmark:

1. Initializes a session, attaches GPT-2-small (CPU)
2. Measures step latency with 0 active interventions (baseline)
3. Measures step latency with 1 intervention (scale on layer 0 attn)
4. Measures step latency with 5 interventions (scale on layers 0-4)
5. Reports overhead percentage

Since the forward pass runs in the Python worker, Rust-side criterion
benchmarks can only measure the daemon dispatch overhead, not end-to-end.
For end-to-end measurement, add a Python benchmark script
`tests/bench_intervention_overhead.py` that:

1. Spawns daemon, attaches, steps N times (baseline wall time)
2. Registers interventions, steps N times (intervention wall time)
3. Computes and asserts `(intervention_time - baseline_time) / baseline_time <= 0.02`

Add `xtask bench` subcommand that runs both.

### Files changed

| File | Change |
|------|--------|
| `crates/rocket-surgeon/Cargo.toml` | Add `criterion` dev-dependency |
| `crates/rocket-surgeon/benches/step_overhead.rs` | Daemon dispatch benchmark |
| `tests/bench_intervention_overhead.py` | E2e Python benchmark |
| `xtask/src/main.rs` | Add `bench` subcommand |

---

## Sub-project F — TCK Green (All 38 Features)

### Goal

Every scenario in every `.feature` file under `tck/` passes. No xfail,
no stubs, no skips.

### Current state

- 38 feature files across `tck/protocol/`, `tck/model/`, `tck/perfetto/`,
  `tck/session/`, `tck/tensor/`, `tck/moe/`
- 14 have test stubs (all xfail)
- 24 have no test files
- Step definitions in `steps/common.py` are stubs
- `conftest.py` uses a `_StubRpcClient` — no real daemon communication

### Design

This sub-project has three tiers:

#### Tier 1: Real RPC client + green the implemented features

Replace `_StubRpcClient` in `conftest.py` with a real daemon subprocess
using Content-Length framed JSON-RPC over stdin/stdout (same pattern as
`tests/e2e_harness.py`). Then wire up step definitions in
`steps/common.py` to send real requests and assert on real responses.

Features that should go green immediately (verbs fully implemented):

- `lifecycle.feature`, `stepping.feature`, `inspection.feature`,
  `discover.feature`, `capabilities.feature`, `state-envelope.feature`,
  `error-expressiveness.feature`, `attach-discovery.feature`,
  `envelope-compactness.feature`, `probes.feature`, `adapter.feature`,
  `hooks.feature`, `handles.feature`, `shm.feature`

14 features.

#### Tier 2: Complete partial implementations

Features where the verb handler exists but edge cases or sub-features
are missing:

- `errors.feature` — green remaining error codes (KvEvicted,
  ReplayDivergence require Tier 3 infrastructure)
- `intervention.feature` — ensure execution-during-step scenarios pass
  (interventions fire and report in `fired_interventions`)
- `subscribe.feature` + `subscribe-filter.feature` — ensure event
  fan-out and filter matching work
- `session-export.feature` — depends on Sub-project A for 9-artifact
  completeness
- `tick-clock.feature` — populate operator/wall clocks in TickPosition
- `tick-position-phase.feature` — ensure all phases uniformly covered
- `view.feature` — ensure built-in views resolve
- `kv-cache.feature` — complete eviction handling
- `checkpoint.feature` — already implemented in session.rs; wire step
  definitions
- `bridge_discovery.feature`, `hook_lifecycle.feature`,
  `mailbox_barrier.feature` — model-tier scenarios
- `daemon-lifecycle.feature` — Perfetto lifecycle (create/close trace)

15 features.

#### Tier 3: Implement new verbs and features

Features requiring new protocol verbs or major new infrastructure:

| Feature | Required verb/infrastructure |
|---------|------------------------------|
| `replay.feature` | `rocket/replay` — replay from checkpoint with divergence detection. Session.replay() exists as stub; needs worker-side re-execution. |
| `branch.feature` | `rocket/branch.fork`, `branch.drop`, `branch.compare` — worldline branching. Protocol types exist (`messages.rs:519-560`); need session-tier state management + fork/compare logic. |
| `sweep.feature` | `rocket/sweep` — batch experiment trials from checkpoint. Protocol types exist (`messages.rs:632-680`); needs trial orchestration loop. |
| `step-run-to.feature` | `run_to` parameter on `StepRequest` — step until target reached. Field exists (`messages.rs:157`); needs matching logic in step handler. |
| `view-focus.feature` | `rocket/view.focus` — LLM navigation by position/regex. Protocol types exist; needs selector parsing + position indexing. |
| `bundle.feature` (session/) | Bundle restore — reload session from tar.gz. New handler; reverse of export. |
| `tick-granularity.feature` (moe/) | MoE tick granularities — router/expert-level stepping. Phase 6 scope; needs MoE adapter + 4-level tick model. |
| `track-hierarchy.feature` | Perfetto track tree builder. PerfettoSink exists; needs hierarchical track declaration validation. |
| `wire-format.feature` | Perfetto protobuf validation. Needs external protobuf parser or self-validation. |

9 features.

### Estimated scope

| Tier | Features | Effort |
|------|----------|--------|
| 1 — Real RPC + green implemented | 14 | Medium (harness rewrite + step definitions) |
| 2 — Complete partials | 15 | Medium-large (edge cases + sub-features) |
| 3 — New verbs | 9 | Large (new protocol verbs, session state, worker integration) |

### Files changed

| File | Change |
|------|--------|
| `python/tests/tck/conftest.py` | Replace stub with real daemon RPC client |
| `python/tests/tck/steps/common.py` | Implement all step definitions |
| `python/tests/tck/steps/*.py` | Domain-specific step files as needed |
| `python/tests/tck/test_*.py` | Remove xfail markers, add missing test files |
| `crates/rocket-surgeon/src/session.rs` | Branch, sweep, view_focus, bundle restore |
| `crates/rocket-surgeon/src/dispatch.rs` | New handlers for branch, sweep, view_focus, step run_to, bundle restore |
| `crates/rocket-surgeon-worker/src/dispatch.rs` | Worker-side replay execution |
| `python/rocket_surgeon/bridge.py` | Replay re-execution support |

---

## Exit Criteria (restated from plan.md:806-817)

All of these must pass after the six sub-projects complete:

- [ ] Five interventions work (ablate, scale, add, patch, clamp)
- [ ] Intervention composition (priority, additive, replace) works
- [ ] Session bundle export produces valid artifact with all 9 required contents
- [ ] Session bundle includes Perfetto trace that opens in Perfetto UI
- [ ] IOI reproduction acceptance test passes (logit difference reduced ≥50%)
- [ ] Model conformance test suite passes for Llama (nightly) and GPT-2 (CI)
- [ ] MVP documentation exists: quickstart, IOI tutorial, protocol reference
- [ ] No overhead regression from Phase 1 baseline (interventions add ≤2%)
- [ ] All Phase 0/1/2 TCK scenarios green
- [ ] Protocol schema frozen at v0.1.0
