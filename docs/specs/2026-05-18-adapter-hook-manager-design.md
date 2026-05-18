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

### Key Design Decisions from Deep Research (2026-05-18)

Three revisions from the deep research pass on this spec:

1. **Lock-based mailboxes, not threading.Event** — `Event.set()` then `Event.clear()` has
   a race window where the forward thread misses the signal. nnsight solved this with
   `_thread.allocate_lock()`-based single-slot mailboxes. We adopt the same pattern.
   See §2 Barrier Mechanics.

2. **Autocast-aware stats** — hook outputs reflect autocast dtype (fp16/bf16), not model
   nominal dtype. Reduction ops run in that dtype unless explicitly cast. `compute_tensor_stats`
   casts to fp32 before computing to avoid precision loss. See §2 Capture Policy.

3. **Execution order is per-call** — MoE routing, gradient checkpointing, KV-cache, and
   weight-tying mean execution order can change between forward passes. Discovery runs
   on each forward pass. Components keyed by `(module_path, call_index)` not just path.
   See §1 Execution Order.

Corrections to prior assumptions:
- Mixtral experts ARE individually hookable in HF eager path (`nn.ModuleList`, not 3D tensors).
- GPT-2's `c_attn` is fused QKV (`Conv1D`, not `nn.Linear`) — CI model needs fused decomposition.
- Phi-3 has unequal QKV splits: `(n_heads*hd, n_kv_heads*hd, n_kv_heads*hd)`.
- Llama `rotary_emb` moved from per-attention to model-level in transformers 4.43+.

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
  "GPT2Attention"       → Container
  "GPT2MLP"             → Container
  "LayerNorm" "ln_1"    → Direct("ln1")
  "LayerNorm" "ln_2"    → Direct("ln2")
  Conv1D "c_attn"       → Fused([
    {canonical: "q_proj", split_dim: -1, split_size: n_embd},
    {canonical: "k_proj", split_dim: -1, split_size: n_embd},
    {canonical: "v_proj", split_dim: -1, split_size: n_embd},
  ])
  Conv1D "c_proj"       → Direct("o_proj")    # attn output projection
  Conv1D "c_fc"         → Direct("up_proj")    # MLP up
  Conv1D "c_proj"       → Direct("down_proj")  # MLP down (disambiguated by parent)
  "Embedding" "wte"     → Direct("embed")
  "Embedding" "wpe"     → Direct("pos_embed")
  Linear "lm_head"      → Direct("lm_head")   # weight-tied to wte
  "Dropout"             → skip (internal)
  "NewGELUActivation"   → skip (internal)
```

Note: GPT-2 uses HuggingFace's `Conv1D` (functionally equivalent to `nn.Linear` with
transposed weight) for all projections. The fused `c_attn` outputs `(batch, seq, 3*n_embd)`
and is split at runtime via `.split(self.split_size, dim=2)` — three equal chunks.
GPT-2's `lm_head.weight` is weight-tied to `wte.weight` (not in `named_parameters`).
This means `lm_head` fires a hook but shares parameters with `embed` — the adapter
must handle this gracefully (both get canonical names; the weight-tie is a parameter
property, not a hook property).

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
   of `(module_path, call_index)` pairs (ground truth from actual forward pass).
   Modules that fire multiple times (weight-tying, shared layers) get distinct entries.
8. Rust: reorder the resolved components to match execution order. Components keyed
   by `(module_path, call_index)`.
9. Rust: produce final `ComponentMap` — ordered list of hookable components with
   canonical probe-point paths, call indices, and capture metadata

### Execution Order Discovery

Execution order is a **per-call** property, not a per-model property. MoE routing,
gradient checkpointing, KV-cache prefill vs. decode, and weight-tying all mean the set
and order of hook firings can change between forward passes. Discovery therefore runs
on each forward pass being debugged.

Components are keyed by `(module_path, call_index)` — not just `module_path`. If a
module fires twice in one forward pass (weight-tying, shared layers), each call gets
a distinct component entry. `call_index` is 0-based, assigned in hook firing order.

The discovery pass itself is a lightweight pre-hook trace that records
`(module_path, call_index)` for every hook that fires. This runs once per forward pass,
before capture hooks block. The result is an ordered list of `(module_path, call_index)`
pairs that defines tick sequencing for that pass.

**Caching**: the discovery result may be cached as a hint, keyed by
`(model_type, num_layers, hidden_size)`. On subsequent passes, the cached order is
used to pre-assign tick positions, but the actual trace validates it. If the trace
differs (e.g., different MoE expert routing): warn, use the fresh trace, invalidate
the cache. Users can pin the cache to skip re-validation (opt-in, not default).

### Probe-Point Construction

Each component's canonical name is assembled into a full probe-point:
```
model:{rank}:{layer}:{canonical}:{call_index}:{event}
```

For Phase 1, rank is always 0. Layer is the integer index. `call_index` is 0 for the
first (and usually only) call to that module in a forward pass; it increments for
weight-tied or shared modules that fire multiple times. Event is `fwd` (forward hook
fires after module.forward()). Example: `model:0:3:q_proj:0:fwd` = rank 0, layer 3,
Q projection, first call, forward event. For a weight-tied module called twice:
`model:0:0:embed:0:fwd` and `model:0:0:embed:1:fwd`.

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
2. If probe matches: writes `(module_path, call_index, tensor_ref)` to the `result_mailbox` (single-slot,
   lock-based — see Barrier Mechanics below). This wakes the Rust IPC thread.
3. Blocks on `resume_mailbox.wait()` (acquires an already-held lock — blocks until Rust
   releases it).
4. On resume: reads intervention slot from `resume_mailbox`. If Rust posted a modified
   tensor, returns it (replaces module output downstream). If empty, returns None
   (output unchanged).
5. After consumption: `resume_mailbox.restore()` drops the tensor reference to prevent
   activation memory accumulation.

The capture hook callback is Python. It holds tensor references, blocks on lock
primitives, reads from the intervention slot. This is PyTorch-runtime-dependent
behavior — it stays in Python where the dragons live.

### Barrier Mechanics

Lock-based single-slot mailboxes, mirroring nnsight's `Mediator.Value` pattern.
Each mailbox is built on `_thread.allocate_lock()` — a thin C wrapper around a
pthread mutex with no Python-level bookkeeping.

**Why not `threading.Event`**: `Event.set()` then `Event.clear()` has a race window
where the forward thread never observes the set state (the flag transitions
True→False between GIL releases). nnsight's v0.6 design uses raw locks specifically
to avoid this. See deep research Q1(d) for details.

Two mailboxes per barrier, one per direction:

- `result_mailbox` (forward thread → Rust) — hook writes `(module_path, call_index,
  tensor_ref)` via `put()`, which stores the value and releases the lock. Rust's
  `wait()` blocks on `lock.acquire()` until the hook puts a value.
- `resume_mailbox` (Rust → forward thread) — Rust writes an intervention payload
  (or empty sentinel) via `put()`. Hook's `wait()` blocks on `lock.acquire()` until
  Rust puts a value.

Each mailbox exposes four methods:
- `put(value)` — store value, release lock (wakes consumer)
- `wait()` → value — acquire lock (blocks until producer puts), return stored value
- `get()` → value — non-blocking read of stored value (for inspection)
- `restore()` — clear stored value, drop references (prevents activation memory leak)

`lock.acquire()` releases the GIL (it goes through `PyThread_acquire_lock_timed` in
CPython), so the Rust IPC thread can operate freely while the forward thread is blocked.

### Tick Cycle

```
Rust IPC thread                     Python forward thread
───────────────                     ─────────────────────
recv "_host/step"
resume_mailbox.put(empty)           ← releases lock
                                    hook at component N:
                                      resume_mailbox.wait() returns
                                      resume_mailbox.restore()
                                      check intervention → apply if present
                                      return (output or replacement)
                                    forward continues...
                                    hook at component N+1 fires
                                    active_probes match? → yes
                                    result_mailbox.put(path, idx, tensor)
                                      ← releases lock, wakes Rust
                                    resume_mailbox.wait() [BLOCKED]
                                      ← lock.acquire(), GIL released
result_mailbox.wait() returns
result_mailbox.restore()
compute stats via bridge
  (torch ops on GPU — Python calls)
tick::advance()
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

**Autocast contract**: hook outputs reflect the autocast dtype (fp16/bf16), not the
model's nominal dtype. Reduction ops like `.mean()` and `.std()` run in the tensor's
dtype — if the tensor is fp16, reductions run in fp16, which loses precision near
range limits (fp16 overflows at 65504; bf16 has poor mantissa precision).
`compute_tensor_stats` therefore casts to fp32 before computing:
`tensor.float().mean()`, not `tensor.mean()`. The original dtype is reported alongside
stats so consumers know what they're looking at. `.detach().cpu()` is dtype-preserving
— an fp16 tensor copied to CPU stays fp16.

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
- `discover_execution_order(model_handle, sample_input) -> list[tuple[str, int]]`  # (path, call_index)
- `install_sentinel_hooks(model_handle, module_paths) -> list`
- `install_capture_hooks(model_handle, module_paths, result_mailbox, resume_mailbox) -> list`
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
  bridge.install_capture_hooks(handle, mapped_paths, result_mailbox, resume_mailbox)
  bridge.run_forward(handle, sample_input, cb)  → Python: spawns forward thread

Worker → Orchestrator → Daemon:
  attach response with ComponentMap, model info
```

### Step Sequence

```
Daemon → Orchestrator → Worker:
  _host/step {count: 1}

Worker (Rust):
  resume_mailbox.put(intervention_or_empty)     → Python: hook resumes, next hook fires
  result_mailbox.wait()                         → Python: (path, call_index, tensor_ref)
  result_mailbox.restore()                      → drop tensor ref after consumption
  bridge.compute_tensor_stats(tensor_ref)       → Python: torch reduction ops (fp32 cast)
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
- `active_probes` dict includes rank in probe-point matching. `model:*:3:q_proj:0:fwd`
  matches all ranks; `model:0:3:q_proj:0:fwd` matches rank 0 only.
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
- `discover_execution_order` returns `(path, call_index)` pairs, consistent across calls
  for deterministic models; detects call_index > 0 for weight-tied modules (GPT-2 lm_head)
- Mailbox `put`/`wait`/`restore` cycle works correctly across two threads
- `install_capture_hooks` + mailbox barrier cycle captures expected tensor
- `compute_tensor_stats` returns correct values verified against direct torch computation;
  specifically: fp16 input produces same stats as manual `.float().mean()` etc.
- `split_fused_output` splits correctly (equal chunks for GPT-2 c_attn, unequal for Phi-3)

**Integration test**:
- Extend existing e2e test: attach → configure hooks → step 1 tick → verify captured
  stats → step to end of forward pass → detach
