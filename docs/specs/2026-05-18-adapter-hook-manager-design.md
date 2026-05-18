# Phase 1 Chunk A: Model Adapter + Hook Manager

Design spec for WU 1.6 (model adapter) and WU 1.7 (hook manager). These are the
foundation for tick stepping, tensor inspection, and probe-driven capture.

## Goal

Given a loaded HuggingFace model, map its module tree to canonical RS probe-point names,
register PyTorch forward hooks on every mapped component, and implement a barrier gate
that pauses the forward pass at each tick boundary so the daemon can inspect and intervene.

## Architecture

Two layers:

1. **Core adapter framework** (Rust, model-family-agnostic) — understands module trees,
   mapping types (1:1 direct, 1:N fused), execution order discovery, canonical name
   construction, and fused output splitting.

2. **Per-family mapping declarations** (static data in Rust) — which module types map to
   which canonical names, which are fused and how they split. No logic, just declarations.

The Python bridge stays thin: it holds Python object references and calls PyTorch APIs.
All decision logic (probe matching, capture policy, adapter resolution, tick tracking)
lives in Rust.

**Boundary principle**: code goes in Rust when it gives real compile-time guarantees.
Code stays in Python when its correctness depends on PyTorch runtime behavior (hook
firing order, CUDA stream timing, GIL interactions). We don't port things into Rust
for false confidence.

---

## 1. Model Adapter

### Core Concepts (Rust)

A `ModuleMapping` describes how an HF module maps to RS canonical names:

- `Direct { canonical: &str }` — 1:1. Module output maps to one canonical component.
  The common case.
- `Fused { components: Vec<FusedComponent> }` — 1:N. One module's output maps to multiple
  canonical components, with split metadata (dimension, sizes derived from model config).
  Example: GPT-NeoX's `query_key_value` → `[q_proj, k_proj, v_proj]`.
- `Container` — not hookable directly (e.g., `nn.ModuleList`, `LlamaAttention` parent).
  Provides layer-grouping structure. Children are mapped individually.

A `FusedComponent` is:
```
{ canonical: &str, split_dim: i64, split_size: usize }
```

The 1:1 case is just the common case of 1:N where N=1. The core framework handles
fused module splitting generically — it is not per-family adapter logic.

### Per-Family Mapping Declarations (Static Data)

Each family is a table of `(module_type_name, ModuleMapping)`:

```
llama:
  "LlamaAttention"      → Container
  "LlamaMLP"            → Container
  "LlamaRMSNorm"        → Direct("ln")    # position within layer disambiguates ln1/ln2
  Linear "q_proj"       → Direct("q_proj")
  Linear "k_proj"       → Direct("k_proj")
  Linear "v_proj"       → Direct("v_proj")
  Linear "o_proj"       → Direct("o_proj")
  Linear "gate_proj"    → Direct("gate_proj")
  Linear "up_proj"      → Direct("up_proj")
  Linear "down_proj"    → Direct("down_proj")
  "LlamaRotaryEmbedding" → skip (internal)
  "Embedding"           → Direct("embed")
  Linear "lm_head"      → Direct("lm_head")

gpt2:
  (CI model, similar pattern)
```

When GPT-NeoX ships:
```
gpt_neox:
  Linear "query_key_value" → Fused([
    {canonical: "q_proj", split_dim: -1, split_size: num_heads * head_dim},
    {canonical: "k_proj", split_dim: -1, split_size: num_heads * head_dim},
    {canonical: "v_proj", split_dim: -1, split_size: num_heads * head_dim},
  ])
```

### Adapter Resolution Pipeline

1. Python bridge: `discover_modules(model_handle)` → raw module inventory
   `[{path, type_name, has_children}]`
2. Python bridge: `model_config(model_handle)` → `{model_type, num_layers, ...}`
3. Rust: select family declaration by `model_type`
4. Rust: match each module against the declaration by type name + path position
5. Rust: assign canonical names, detect layer structure (repeating `layers.N.*` pattern),
   group components by layer index
6. Rust: unknown modules → `_raw.<original.path>` fallback
7. Python bridge: `discover_execution_order(model_handle, sample_input)` → ordered list
   of module paths (ground truth from actual forward pass)
8. Rust: reorder the resolved components to match execution order
9. Rust: produce final `ComponentMap` — ordered list of hookable components with
   canonical probe-point paths and capture metadata

### Execution Order Cache

Discovery forward pass result stored in Rust, keyed by `(model_type, num_layers,
hidden_size)`. On subsequent attaches with matching key, the cached order is used but
re-validated with a verification forward pass. If order differs: warn, invalidate cache,
use fresh discovery result.

### Probe-Point Construction

Each component's canonical name is assembled into a full probe-point:
```
model:{rank}:{layer}:{canonical}:{event}
```

For Phase 1, rank is always 0. Layer is the integer index. Event is `fwd` (forward hook
fires after module.forward()). Example: `model:0:3:attn:fwd` = rank 0, layer 3,
attention output, forward event.

### Phase 1 Scope

Supported families: `llama` (Llama-2/3, Mistral, CodeLlama), `gpt2` (CI model).
The core framework supports fused mappings from day one. Additional family declarations
added as needed without framework changes.

---

## 2. Hook Manager

### Two Hook Layers

Both installed at attach time, removed at detach.

**Sentinel hooks** — no-op `lambda _, __, out: out` on every module in the tree. Defeats
PyTorch's fast-path optimization in `nn.Module._call_impl`. Without these, dynamically-
added hooks may silently fail because the module entered the fast path before hook
registration. Installed first, held for lifetime of attachment.

**Capture hooks** — registered with `prepend=True` on every mapped component from the
`ComponentMap`. Each hook's callback:
1. Fast-exit check: reads `active_probes` dict (written by Rust, read by Python).
   If no active probe matches this component → return None immediately. Zero overhead.
   (No concurrent access concern: Rust only updates `active_probes` while the forward
   thread is blocked on the barrier, so reads and writes never overlap.)
2. If probe matches: puts `(module_path, tensor_ref)` on the `result_queue`.
3. Blocks on `barrier_event.wait()`.
4. On resume: checks intervention slot. If Rust posted a modified tensor, returns it
   (replaces module output downstream). If empty, returns None (output unchanged).

The capture hook callback is Python. It holds tensor references, calls `event.wait()`,
reads from the intervention slot. This is PyTorch-runtime-dependent behavior — it stays
in Python where the dragons live.

### Barrier Mechanics

Two Python synchronization objects, created by Rust through PyO3:

- `barrier_event` (`threading.Event`) — forward thread hooks call `event.wait()` to block.
  Rust calls `event.set()` then `event.clear()` to advance one tick.
- `result_queue` (`queue.Queue`) — hooks put captured data on this queue. Rust reads from
  it after releasing the barrier and the next hook fires.

`Event.wait()` releases the GIL, so the Rust IPC thread can operate freely while the
forward thread is blocked.

### Tick Cycle

```
Rust IPC thread                     Python forward thread
───────────────                     ─────────────────────
recv "_host/step"
barrier_event.set()
barrier_event.clear()
                                    hook at component N resumes
                                    forward continues...
                                    hook at component N+1 fires
                                    active_probes match? → yes
                                    put (path, tensor_ref) on queue
                                    barrier_event.wait() [BLOCKED]
read result_queue
compute stats via bridge
  (torch ops on GPU — Python calls)
package stats + tick position
send response to orchestrator
```

### Capture Policy

Three modes, configured per-probe:

1. **None** (no active probe) — hook fast-exits. Zero overhead.
2. **Summary** (default for active probes) — bridge computes summary stats on GPU via
   PyTorch reduction ops (mean, std, min, max, abs_max, l2_norm, sparsity, shape, dtype).
   Tensor stays on GPU. Slice available on demand via follow-up request.
3. **Full** (explicit opt-in) — bridge calls `tensor.detach().cpu().numpy().tobytes()`.
   Full tensor transferred. More expensive but gives daemon raw bytes for content-
   addressable storage (BLAKE3 hashing in Rust).

For fused modules: the hook captures the full output tensor. Rust applies the split
metadata from the `ComponentMap` (via bridge's `split_fused_output`) to produce
per-component tensors before stats computation.

### Forward Pass Lifecycle

- `run_forward(model_handle, input_ids, done_callback)` spawns the Python forward thread.
- Thread calls `model(input_ids)` — hooks fire inside this call.
- On completion (all layers done), `done_callback` notifies Rust.
- On exception, `done_callback` carries the error. Rust reports `HOST_ERROR`.

---

## 3. Worker Crate Changes

### New Internal Protocol Commands

- `_host/configure_hooks` — sent after attach succeeds. Carries the `ComponentMap`.
  Worker installs sentinels + capture hooks, spawns forward thread.
- `_host/step` — advance one tick. Worker releases barrier, waits for next capture,
  returns stats + tick position.
- `_host/update_probes` — daemon sends updated active probe set. Worker updates the
  shared `active_probes` dict. Takes effect at next tick.
- `_host/intervene` — daemon sends intervention recipe for a specific component.
  Worker places it in intervention slot. Takes effect when that component's hook
  next resumes from the barrier.

### New Rust Modules

- `adapter.rs` — core adapter framework: `ModuleMapping`, `ComponentMap`, family
  declarations, execution order cache, canonical name resolution.
- `tick.rs` — tick state: current position (layer, component, event), tick ID generation
  (monotonic, unique), step counting.
- `capture.rs` — capture policy logic: probe matching against `ComponentMap`, stats
  packaging, intervention slot management.

### Bridge Growth

`bridge.py` (renamed from `skin.py`) gets these new functions:

- `discover_modules(model_handle) -> list[dict]`
- `discover_execution_order(model_handle, sample_input) -> list[str]`
- `install_sentinel_hooks(model_handle, module_paths) -> list`
- `install_capture_hooks(model_handle, module_paths, barrier_event, result_queue) -> list`
- `run_forward(model_handle, input_ids, done_callback)`
- `remove_hooks(handles)`
- `compute_tensor_stats(tensor) -> dict`
- `tensor_to_bytes(tensor) -> bytes`
- `split_fused_output(tensor, dim, sizes) -> list`

`bridge.rs` (renamed from `skin.rs`) grows PyO3 bindings for each.

### Existing Functions (renamed)

- `load_model` → stays, now in `bridge.py`
- `unload_model` → stays
- `model_metadata` → subsumed by `model_config` + `discover_modules`

---

## 4. Data Flow

### Attach Sequence

```
Daemon → Orchestrator → Worker:
  _host/attach {model_source, device, ...}

Worker (Rust):
  bridge.load_model(source, device, dtype)      → Python: AutoModel.from_pretrained()
  bridge.model_config(handle)                   → Python: config attrs
  bridge.discover_modules(handle)               → Python: named_modules()
  adapter::resolve(modules, config, family)     → Rust: build ComponentMap
  bridge.discover_execution_order(handle, input) → Python: tracing forward pass
  adapter::apply_execution_order(component_map)  → Rust: reorder components
  bridge.install_sentinel_hooks(handle, all_paths)
  bridge.install_capture_hooks(handle, mapped_paths, event, queue)
  bridge.run_forward(handle, sample_input, cb)  → Python: spawns forward thread

Worker → Orchestrator → Daemon:
  attach response with ComponentMap, model info
```

### Step Sequence

```
Daemon → Orchestrator → Worker:
  _host/step {count: 1}

Worker (Rust):
  barrier_event.set() / clear()                 → Python: hook resumes, next fires
  read result_queue                             → Python: (path, tensor_ref)
  bridge.compute_tensor_stats(tensor_ref)       → Python: torch reduction ops
  tick::advance()                               → Rust: update position, increment tick_id
  package response

Worker → Orchestrator → Daemon:
  step response with {tick_position, tick_id, stats (if captured)}
```

---

## 5. Error Handling

- **Hook exception**: PyTorch propagates up through `model()`. Forward thread catches it,
  notifies Rust via `done_callback` with error. Rust reports `HOST_ERROR` with traceback.
- **Forward thread dies**: Rust detects via done_callback or thread join. Reports
  `HOST_ERROR`. Daemon transitions to error state.
- **Worker crash**: Orchestrator detects child exit (existing `WorkerHandle`). Daemon
  transitions to error state.
- **Timeout**: No new timeout logic in hooks. If a hook blocks indefinitely (CUDA hang,
  deadlock), the orchestrator's existing timeout catches it. We don't mask inscrutable
  runtime behavior with Rust timeouts.
- **OOM during capture**: `tensor.cpu()` or stats computation can OOM. Python raises,
  hook exception path handles it. Failed capture reports error but doesn't kill session.

---

## 6. Multi-GPU Considerations

Designed for multi-GPU, implemented for single-rank in Phase 1:

- `ComponentMap` is per-rank. Each worker builds its own.
- `active_probes` dict includes rank in probe-point matching. `model:*:3:attn:fwd`
  matches all ranks; `model:0:3:attn:fwd` matches rank 0 only.
- Tick position carries rank. No single-rank assumptions in data structures.
- When Phase 5 adds multi-GPU, the framework doesn't change — more workers spawn,
  each builds a rank-specific `ComponentMap`, orchestrator coordinates barriers
  across ranks.

---

## 7. Testing Strategy

JSMNTL: TCK red first, then implementation.

**TCK scenarios to turn green** (already exist as xfail):
- `tck/model/adapter.feature` — 10 scenarios (canonical name resolution, unknown fallback)
- `tck/model/hooks.feature` — 8 scenarios (tier detection, hook registration order)

**New Rust unit tests**:
- Adapter resolution with llama family declaration
- Adapter resolution with fused mapping (gpt_neox declaration, even if not Phase 1)
- Unknown module fallback
- Execution order reordering
- ComponentMap construction
- Tick position tracking
- Probe matching against component map

**New Python tests**:
- `discover_modules` returns expected structure for tiny-random-Llama
- `discover_execution_order` returns consistent order across calls
- `install_capture_hooks` + barrier cycle captures expected tensor
- `compute_tensor_stats` returns correct values verified against direct torch computation
- `split_fused_output` splits correctly

**Integration test**:
- Extend existing e2e test: attach → configure hooks → step 1 tick → verify captured
  stats → step to end of forward pass → detach
