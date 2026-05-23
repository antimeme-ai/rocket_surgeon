# Phase 2 Design: Interventions + MVP Completion

Date: 2026-05-21

## Goal

Deliver end-to-end intervention execution through the full stack — protocol,
daemon, worker, Python engine — validated by an IOI reproduction acceptance
test against GPT-2-small. Ship a complete 9-artifact session bundle for
reproducibility. MCP adapter is punted indefinitely.

## Scope

Six work units. Critical path: 2.1 → 2.2 → 2.7. Bundle (2.3) branches off
once 2.2 lands. Conformance (2.4) runs in parallel from the start. Docs (2.5)
written last.

```
2.4 (conformance) ─────────────────────────────────────────┐
2.1 (engine) → 2.2 (worker integration) → 2.7 (IOI test) ─┤→ Phase 2 DONE
                         └→ 2.3 (bundle export) ───────────┤
                                             2.5 (docs) ───┘
```

## Non-goals

- MCP adapter (punted indefinitely)
- Multi-GPU intervention dispatch (Phase 5)
- MoE routing interventions (Phase 6)
- Real replay divergence detection (deferred; daemon tier 1 returns empty
  divergences and vacuous verified=true)
- Intervention types beyond the core 5 (RouteOverride, AttentionMask,
  EmbedSwap, EmbedNoise are protocol-defined but not Phase 2 MVP)

---

## WU 2.1 — Python Intervention Engine

### Purpose

Standalone Python module that takes intervention recipes and applies them to
tensors. No daemon, no worker, no hooks — pure tensor operations testable
against CPU mock data.

### Location

```
python/rocket_surgeon/host/interventions/
├── __init__.py          — public API: apply_interventions()
├── engine.py            — InterventionEngine: filter, sort, apply
├── recipes.py           — recipe deserialization, validation
└── composition.py       — priority ordering, additive/replace semantics
```

### Interface

```python
def apply_interventions(
    tensor: torch.Tensor,
    recipes: list[dict],
    component: str,
    layer: int,
    tensor_store: Callable[[str], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Apply matching intervention recipes to a tensor.

    Args:
        tensor: the activation tensor to modify (mutated in-place)
        recipes: list of InterventionRecipe dicts (from protocol JSON)
        component: current component name (e.g. "attn.o_proj")
        layer: current layer index
        tensor_store: callback to resolve tensor_id → Tensor (for patch/add)

    Returns:
        (modified_tensor, list of recipe IDs that fired)
    """
```

### Recipe Types

| Type | Params | Operation |
|------|--------|-----------|
| `ablate` | `mode`: zero/mean/resample | zero: `tensor.zero_()`. mean: `tensor.fill_(tensor.mean())`. resample: `tensor.normal_(tensor.mean(), tensor.std())` |
| `scale` | `factor`: f64 | `tensor.mul_(factor)` |
| `add` | `vector`: inline `list[float]` or `tensor_id` str | `tensor.add_(vector_tensor)` |
| `patch` | `source_tensor_id`: str | `tensor.copy_(tensor_store(source_tensor_id))` |
| `clamp` | `min`: f64, `max`: f64 | `tensor.clamp_(min, max)` |

### Composition

1. Filter recipes whose `target` matches the current `layer:component`
2. Sort by `priority` ascending (lower = first)
3. Apply sequentially:
   - `mode: "additive"` (default): apply on top of prior modifications
   - `mode: "replace"`: discard all prior modifications, start from the
     original tensor (engine snapshots the original before the loop)
4. `condition` field: reserved for Phase 3. In Phase 2, if present, the recipe
   is skipped with a warning log.

### Target Matching

A recipe's `target` is a probe-point string:
`family:rank:layer:component:event`. The engine matches against the current
execution context. Wildcards (`*`) match any segment. Examples:

- `gpt2:0:11:attn.o_proj:output` — exact match
- `gpt2:*:*:attn.o_proj:output` — all ranks, all layers
- `*:*:11:*:output` — all components at layer 11

Matching reuses the probe grammar already implemented in
`crates/rocket-surgeon-probes/src/grammar.rs`. The Python engine uses a
simplified matcher (string split + wildcard comparison) that is
spec-compatible but does not import the Rust grammar.

### Testing

- Unit tests in `python/tests/test_interventions.py`
- Mock CPU tensors, no model loading
- Cover: all 5 types, composition (priority, additive, replace), target
  matching (exact, wildcard), edge cases (empty recipe list, no matching
  recipes, replace-then-additive ordering)
- TCK step definitions in `python/tests/tck/steps/` validate against
  `tck/protocol/intervention.feature` scenarios (registry operations only —
  execution scenarios deferred to 2.2 integration tests)

---

## WU 2.2 — Worker Integration

### Purpose

Wire the intervention engine into the worker's forward-pass hook system so
that `rocket/step` applies registered interventions at each hook barrier.

### Architecture

```
Daemon                       Worker (Rust)              Python Host
  │                            │                          │
  │─ _host/step ──────────────►│                          │
  │  (includes interventions   │                          │
  │   from session registry)   │                          │
  │                            │─ call engine ───────────►│
  │                            │  (component, layer,      │
  │                            │   tensor, recipes)       │
  │                            │                          │
  │                            │◄─ (modified tensor, ─────│
  │                            │    fired recipe IDs)     │
  │                            │                          │
  │◄─ step response ──────────│                          │
  │  (includes fired_interventions)                       │
```

### Dispatch Path

1. Daemon calls `_host/step` on the orchestrator/worker. The message includes
   the current `interventions` list from `session.interventions()`.

2. Worker stores the intervention list for the duration of the step.

3. At each forward hook callback (output capture point), the worker:
   a. Captures the output tensor (existing behavior)
   b. Calls `apply_interventions(tensor, recipes, component, layer)`
   c. If any recipes fired, replaces the hook output with the modified tensor
   d. Collects fired recipe IDs

4. After the step completes, worker includes `fired_interventions: Vec<String>`
   in the step response.

### Hook Integration

The existing hook system in `python/rocket_surgeon/hooks/` registers
`register_forward_hook` callbacks on each module. Currently these callbacks
capture the output tensor. The intervention integration extends the callback:

```python
def hook_fn(module, input, output):
    captured = capture_output(output)       # existing
    modified, fired = apply_interventions(  # new
        tensor=output,
        recipes=current_recipes,
        component=component_name,
        layer=layer_idx,
    )
    if fired:
        return modified  # PyTorch replaces the output
    # return None = no modification (existing behavior)
```

Returning a value from a forward hook replaces the module's output in the
computation graph. This is PyTorch's standard intervention mechanism.

### Protocol Changes

The `_host/step` internal message gains an `interventions` field:

```json
{
  "method": "_host/step",
  "params": {
    "direction": "forward",
    "count": 1,
    "interventions": [ /* InterventionRecipe[] */ ]
  }
}
```

The step response gains `fired_interventions`:

```json
{
  "result": {
    "ticks_executed": 1,
    "stopped_at": { /* TickPosition */ },
    "fired_interventions": ["ablate-head-9.1", "ablate-head-10.0"]
  }
}
```

### Testing

- E2E test: attach GPT-2-small, register a `scale` intervention on layer 0
  attention output, step, inspect the tensor, verify it differs from a
  baseline run without intervention.
- E2E test: register `ablate` (zero) on a specific component, step, verify
  the output tensor is all zeros at that component.
- E2E test: register two interventions with different priorities, verify
  application order via tensor value.

---

## WU 2.3 — Session Bundle Export

### Purpose

Assemble a self-contained reproducibility archive capturing everything needed
to understand and reproduce a debugging session.

### Verb

`rocket/session.export` — new protocol method.

```json
{
  "method": "rocket/session.export",
  "params": {
    "path": "/tmp/session-abc123.tar.gz",
    "include_tensors": true
  }
}
```

Response:

```json
{
  "result": {
    "path": "/tmp/session-abc123.tar.gz",
    "size_bytes": 12345678,
    "artifact_count": 9
  }
}
```

### Bundle Contents

```
session-bundle-<session_id>.tar.gz
├── manifest.json             ← protocol_version, session_id, timestamps,
│                               rocket_surgeon version, bundle schema version
├── model-info.json           ← model_family, model_id, num_layers, num_heads,
│                               hidden_dim, num_ranks, architecture hash
├── env.json                  ← GPU model, driver version, CUDA version,
│                               torch version, transformers version,
│                               python version, OS, hostname
├── protocol-trace.jsonl      ← every JSON-RPC request/response/notification
│                               logged during the session (from trace_log.rs)
├── prompt.json               ← input_ids, tokenizer name, raw text (if available)
├── tensors/                  ← captured tensors referenced in the session
│   ├── <tensor_id>.safetensors
│   └── ...
├── interventions.json        ← all intervention recipes (current + historical)
├── trace.perfetto-trace      ← Perfetto timeline (binary protobuf)
└── bookmarks.json            ← named tick bookmarks (may be empty)
```

### Assembly

1. Daemon gathers: manifest, protocol-trace (already logged by `trace_log.rs`),
   interventions (from session registry), bookmarks (from checkpoint bookmarks)
2. Daemon requests from worker via `_host/export_env`: env.json, model-info,
   prompt data
3. Tensors: daemon iterates tensor store, writes each to safetensors format
4. Perfetto: daemon reads the trace file (already written by `perfetto_sink.rs`)
5. Assemble tar.gz via Rust `tar` + `flate2` crates (or `std::process::Command`
   to `tar` — prefer the crate for portability)
6. Atomic write: write to `<path>.tmp`, rename to `<path>`

### Testing

- E2E test: run a session (attach, step, inspect, intervene), export bundle,
  verify all 9 artifacts present in the tar.gz
- Verify manifest.json is valid JSON with required fields
- Verify protocol-trace.jsonl has at least the attach + step + export requests
- Verify tensors/ contains at least one safetensors file if inspect was called

---

## WU 2.4 — Model Conformance Suite

### Purpose

Validate that rocket_surgeon's hook installation correctly observes all
canonical components of a model family, in the expected order.

### Location

```
python/tests/conformance/
├── conftest.py              — shared fixtures (model loading, daemon spawn)
├── test_gpt2.py             — GPT-2 family conformance
└── test_llama.py            — Llama family conformance (marked @nightly)
```

### Test Pattern

```python
def test_gpt2_component_ordering():
    # 1. Spawn daemon, attach GPT-2-small
    # 2. Subscribe to probe.fired events
    # 3. Step through complete forward pass
    # 4. Collect all probe events
    # 5. Assert canonical components present at every layer:
    #    - ln_1, attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
    #      ln_2, mlp.gate_proj (or mlp.c_fc for GPT-2), mlp.down_proj (or mlp.c_proj)
    # 6. Assert ordering: layer 0 before layer 1, attn before mlp within layer
    # 7. Assert no unexpected gaps or duplicates
```

### Canonical Components

GPT-2 uses different module names than Llama. The conformance test maps them:

| Canonical | GPT-2 | Llama |
|-----------|-------|-------|
| `attn.q_proj` | `attn.c_attn` (fused QKV) | `self_attn.q_proj` |
| `attn.k_proj` | (fused in c_attn) | `self_attn.k_proj` |
| `attn.v_proj` | (fused in c_attn) | `self_attn.v_proj` |
| `attn.o_proj` | `attn.c_proj` | `self_attn.o_proj` |
| `mlp.up_proj` | `mlp.c_fc` | `mlp.up_proj` |
| `mlp.down_proj` | `mlp.c_proj` | `mlp.down_proj` |

GPT-2's fused QKV (`c_attn`) means the probe fires once for the fused
projection, not three times. The conformance test must account for this.

### CI Integration

- GPT-2-small: runs in CI (CPU, ~500MB download, cached)
- Llama: `@nightly` marker, requires GPU + model access
- Add `cargo xtask conformance` subcommand

---

## WU 2.5 — MVP Documentation

### Location

```
docs/
├── tutorial/
│   ├── quickstart.md        — install, build, start daemon, attach GPT-2
│   └── ioi.md               — IOI reproduction walkthrough
└── protocol/
    └── examples.md          — usage examples for core verbs
```

### Quickstart Coverage

1. Prerequisites (Rust, Python 3.11, PyTorch, GPU optional)
2. Build (`cargo xtask setup`)
3. Start daemon (`rocket-surgeon --model gpt2`)
4. Attach and step (show JSON-RPC commands)
5. Inspect a tensor (show response)
6. Register an intervention (show ablate recipe)
7. Export session bundle

### IOI Tutorial Coverage

Step-by-step reproduction of the Indirect Object Identification circuit
(Wang et al. 2023) using only protocol commands:

1. Attach GPT-2-small
2. Run IOI prompt through forward pass
3. Identify name-mover heads via attention pattern inspection
4. Ablate candidate heads
5. Measure logit difference
6. Interpret results

All commands are copy-pasteable JSON-RPC.

### Written Last

Docs are written after 2.1, 2.2, 2.3, and 2.7 are complete. Every code
example is validated against the running system.

---

## WU 2.7 — IOI Acceptance Test

### Purpose

End-to-end validation that the full stack works: daemon, worker, intervention
engine, probe capture, bundle export. The test reproduces a published
mechanistic interpretability result (Indirect Object Identification) using
only protocol commands.

### Location

```
python/tests/test_ioi_acceptance.py
python/tests/fixtures/ioi_prompts.json
```

### Test Design

```python
@pytest.mark.slow
def test_ioi_ablation_reduces_logit_diff():
    """Reproduce Wang et al. 2023 IOI circuit identification on GPT-2-small.

    Steps:
    1. Attach GPT-2-small
    2. Run IOI prompt: "When Mary and John went to the store, John gave a drink to"
    3. Step through full forward pass
    4. Inspect attention patterns at name-mover head candidates
       (GPT-2-small: heads 9.9, 9.6, 10.0 per literature)
    5. Register ablate interventions on identified heads
    6. Re-run forward pass with interventions active
    7. Inspect final logits
    8. Compute logit_diff = logit["Mary"] - logit["John"]
    9. Assert ablation reduces logit_diff by >= 50% vs baseline
    10. Export session bundle (validates WU 2.3 as side effect)
    """
```

### IOI Prompts

```json
[
  {
    "text": "When Mary and John went to the store, John gave a drink to",
    "io": "Mary",
    "s": "John",
    "template": "ABB"
  },
  {
    "text": "When Alice and Bob went to the park, Bob gave a ball to",
    "io": "Alice",
    "s": "Bob",
    "template": "ABB"
  }
]
```

### Acceptance Criteria

- Logit diff reduction ≥ 50% after ablating name-mover heads
- Entire test uses only protocol verbs (no direct `model.forward()`)
- Session bundle exported and validated (all 9 artifacts present)
- Runs on CPU with GPT-2-small (no GPU requirement)
- Test duration < 60 seconds

### Known Risks

- GPT-2-small's name-mover heads may not exactly match published head indices
  (literature varies). The test should identify heads dynamically via attention
  pattern inspection, not hardcode indices.
- Logit diff threshold (50%) is conservative. Published results show ~100%
  reduction. If GPT-2-small differs, adjust threshold with a documented
  rationale.

---

## Dependencies

### New Workspace Dependencies

- `tar` crate — tar archive assembly (bundle export)
- `flate2` crate — gzip compression (bundle export)
- `safetensors` — already in workspace, used for tensor serialization in bundle

### Python Dependencies

- No new dependencies. `torch`, `transformers`, `safetensors` already present.
- GPT-2-small weights: downloaded on first test run, cached by `transformers`

---

## What This Design Does Not Cover

- **Worker-side checkpoint save/restore** — daemon tier 1 manages checkpoint
  metadata; actual tensor checkpoint I/O is Phase 3
- **Replay with real re-execution** — daemon tier 1 synthesizes replay
  responses; actual re-execution through the model is Phase 3
- **Divergence detection** — requires real replay; Phase 3
- **Conditional interventions** — `condition` field is reserved, skipped with
  warning in Phase 2
- **Extended intervention types** — RouteOverride, AttentionMask, EmbedSwap,
  EmbedNoise are protocol-defined but not implemented in Phase 2
- **Multi-GPU intervention dispatch** — Phase 5
- **MCP adapter** — punted indefinitely
