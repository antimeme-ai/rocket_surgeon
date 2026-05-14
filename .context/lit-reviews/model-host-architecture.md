---
topic: Model Host Architecture — hook registration, tick stepping, tensor capture, adapter naming, and multi-GPU coordination for instrumenting PyTorch forward passes
status: draft
created: 2026-05-14
sources: PyTorch hook API, baukit (Bau 2022), TransformerLens (Nanda 2022), nnsight (NDIF/Fiotto-Kaufman 2024), pyvene (Wu et al. 2024), transformer-debugger (OpenAI 2024), vllm-hook, nnterp, ADR-0004, ADR-0005, ADR-0006
---

# Model Host Architecture: Lit Review + Reference Implementation Study

How to build a Python-side model host that instruments arbitrary PyTorch transformer forward passes with tick-by-tick stepping, tensor capture, surgical intervention, and multi-GPU support.

## 1. Hook Registration Strategies

Five distinct approaches in the wild. Each trades generality for structure.

### 1.1 PyTorch Native Hooks on Existing Modules (baukit, pyvene, vllm-hook)

Register `register_forward_hook` / `register_forward_pre_hook` on modules found via `model.named_modules()`. No model modification required.

**baukit pattern:**
```python
module = get_module(model, "model.layers.12.self_attn.q_proj")
handle = module.register_forward_hook(lambda m, inp, out: capture(out))
```

Module resolution: walk `named_modules()` matching dotted names. Any `nn.Module` in the tree is hookable by its PyTorch name string.

**pyvene pattern:** Abstract component names (`"block_output"`, `"head_attention_value_output"`) mapped to concrete module paths via per-model-type dictionaries:
```python
type_to_module_mapping["llama"]["block_output"] = ("model.layers.%s", CONST_OUTPUT_HOOK)
```

**vllm-hook pattern:** Regex matching on module names across architectures:
```python
LAYER_PATTERNS = [
    re.compile(r"^model\.layers\.(\d+)$"),           # LLaMA/Qwen
    re.compile(r"^transformer\.h\.(\d+)$"),           # GPT-2
    re.compile(r"^model\.decoder\.layers\.(\d+)$"),   # OPT
]
```

**Strengths:** Works with any `nn.Module` without modification. Zero model awareness needed for basic hooking.

**Weaknesses:** Module granularity only — cannot hook mid-forward (e.g., between Q computation and K computation within a single `nn.Linear`). PyTorch hooks are synchronous and block the forward pass. Hook ordering is FIFO per module.

### 1.2 Custom Identity Modules as Hook Points (TransformerLens)

Insert custom `HookPoint(nn.Module)` at every interesting location in a re-implemented model. HookPoint's forward is the identity function; hooks fire via PyTorch's normal forward hook mechanism.

```python
class HookPoint(nn.Module):
    def forward(self, x): return x  # identity — hooks fire via PyTorch mechanism

# In model forward():
resid = self.hook_resid_pre(resid)  # fires any registered hooks
attn_out = self.attn(resid)
attn_out = self.hook_attn_out(attn_out)  # another hook point
```

**Strengths:** Hook granularity is unlimited — place HookPoints wherever you want, including mid-computation. Standardized naming across 50+ model families. Rich ActivationCache with interpretability helpers.

**Weaknesses:** Requires re-implementing every model architecture. Model parity concerns — re-implemented model may diverge from HuggingFace. Cannot instrument arbitrary models. No multi-GPU.

### 1.3 Inline Hook Call-Sites in Custom Forward (transformer-debugger)

Hooks are explicit function calls baked into the model's `forward()`. A `TransformerHooks` collection is passed as parameter:

```python
def forward(self, X, hooks=TransformerHooks()):
    Q = self.q_proj(X)
    Q = hooks.attn.q(Q)  # hook call-site
    K = self.k_proj(X)
    K = hooks.attn.k(K)  # hook call-site
```

Hierarchical hook tree: `TransformerHooks > AttentionHooks > FwdBwdHooks > Hooks`. Each FwdBwdHooks has three phases: fwd, bwd (via custom autograd.Function), fwd2.

**Strengths:** Explicit, typed hook points. Three-phase (fwd/bwd/fwd2) enables observe→intervene→observe-intervention workflows. Composable subgraphs (inject autoencoder hooks after ablation hooks).

**Weaknesses:** Requires completely custom model code. Cannot instrument arbitrary models. No multi-GPU.

### 1.4 One-Shot Self-Removing Hooks with Thread Sync (nnsight)

Lazy registration: hooks are registered on-demand when user code accesses `.output`/`.input` during a trace. Each hook fires once, self-removes, and wakes the blocked worker thread.

```python
def output_hook(mediator, module, path):
    def hook(module, args, output):
        if mediator.iteration_tracker[path] != target_iteration:
            return
        handle.remove()  # one-shot: self-remove after firing
        mediator.handle(path, output)  # wake worker thread
        return output  # potentially modified
    handle = module.register_forward_hook(hook)
```

Critical trick: a **sentinel hook** (`module.register_forward_hook(lambda _, __, output: output)`) is installed on every module to prevent PyTorch from fast-pathing when zero real hooks exist. Without it, dynamically-added hooks may not fire.

**Strengths:** Zero overhead when no intervention is active. Works with arbitrary models. Real Python semantics in intervention code (not graph tracing). Thread-based stepping closest to true debugger semantics.

**Weaknesses:** Must access modules in execution order (accessing layer 5 before layer 2 causes deadlock). AST rewriting for operation-level tracing is complex. Multi-GPU delegated entirely to vLLM.

### 1.5 Recommendation for rocket_surgeon

**PyTorch native hooks on existing modules** (baukit/pyvene approach) for generality — we must work with arbitrary models, not re-implemented ones.

**Adapter-based name mapping** (pyvene-style) for the canonical vocabulary: per-model-family dictionaries translating `attn.q_proj` → `model.layers.%s.self_attn.q_proj`.

**Not** one-shot hooks (nnsight) — rocket_surgeon's tick model requires hooks to remain registered across multiple steps. Hooks fire on every tick, controlled by a barrier gate.

**Not** re-implementation (TransformerLens/transformer-debugger) — violates "works with arbitrary models" requirement.

## 2. Tick Stepping Mechanisms

The key differentiator for rocket_surgeon. No existing tool implements true tick-by-tick stepping with bidirectional control.

### 2.1 Partial Execution (baukit)

`StopForward` exception raised from a hook, caught by context manager. Executes up to a named layer, then stops.

```python
class StopForward(Exception): pass

def retain_hook(m, inputs, output):
    # ... capture ...
    if stop:
        raise StopForward()

def __exit__(self, type, value, traceback):
    if self.stop and issubclass(type, StopForward):
        return True  # swallow
```

**Limitation:** One-shot. Cannot resume after stopping. Not a debugger — it's "run up to X."

### 2.2 Thread-Based Synchronization (nnsight)

Worker thread runs user intervention code, main thread runs forward pass. Synchronization via C-level locks (`_thread.allocate_lock()`).

```
Worker thread                    Main thread (forward pass)
─────────────                    ──────────────────────────
mediator.request(requester)  →   hook fires → mediator.handle(provider, value)
  ↓ blocks on response_queue      ↓ puts value on response_queue
  ← wakes with tensor value      ← continues forward pass
```

6 event types: VALUE, SWAP, SKIP, BARRIER, END, EXCEPTION.

CUDA stream propagation: worker threads capture caller's CUDA stream at start and set it in worker, because PyTorch worker threads default to stream 0.

**Strengths:** Real Python semantics. Real tensors (not proxies). Clean synchronization.

**Weaknesses:** Execution-order constraint (deadlock if out of order). No true pause-and-inspect — the sync is for read/modify, not for external debugger commands.

### 2.3 Barrier Gate Pattern (rocket_surgeon's approach)

What rocket_surgeon needs is fundamentally different from all reference implementations. The requirement from ADR-0005:

1. Hook fires at a component boundary
2. Hook captures tensor, sends to daemon (via shared memory + notification)
3. Hook **blocks on a barrier** — waits for daemon command
4. Daemon inspects, client issues commands (inspect, intervene, step)
5. Daemon sends "continue" (or "modify + continue") to host
6. Hook releases barrier, forward pass resumes

This is a **per-component barrier gate** controlled by an external process (the daemon). None of the reference implementations do this — they all have the control logic in-process.

**Implementation approaches:**

**Threading.Event per hook point:**
```python
gate = threading.Event()
def hook(module, args, output):
    capture_and_send(output)
    gate.clear()
    gate.wait()  # blocks until daemon says continue
    return modified_output if intervention else output
```

**Unix domain socket blocking read:**
Hook writes captured tensor info to daemon, then does a blocking read on the control socket waiting for the next command. Simpler than threading.Event but couples I/O to the hook.

**Condition variable with command queue:**
```python
cv = threading.Condition()
command_queue = queue.Queue()

def hook(module, args, output):
    capture_and_send(output)
    with cv:
        cv.wait()  # blocks until daemon command arrives
    cmd = command_queue.get()
    if cmd.type == "intervene":
        return apply_intervention(output, cmd)
    return output
```

**Recommendation:** `threading.Event` per tick, with a dedicated control thread that receives daemon commands via Unix domain socket and sets/clears events. The hook blocks on the event. This separates I/O from the hook's blocking, keeping the hook body clean. The control thread also handles intervention payloads.

### 2.4 Granularity Control

ADR-0005 defines 7 granularity levels. The hook registration strategy must support dynamic granularity switching:

- **layer**: one hook per decoder block
- **component** (default): one hook per sub-module (q_proj, k_proj, etc.)
- **head**: one hook per attention head (requires unfused execution)

For MoE (Phase 6): router_pre_topk, router_post_topk, expert, moe_layer.

**Approach:** Register hooks at the finest granularity the user has ever requested in this session, but only pause (hit the barrier gate) at the currently active granularity. Hooks at finer granularities become pass-through when coarser granularity is active.

```python
def hook(module, args, output):
    if self.granularity_filter(self.current_tick_position):
        # This tick matches current granularity — pause
        capture_and_send(output)
        wait_for_command()
    # Otherwise pass-through
    return output
```

## 3. Adapter Layer (Name Mapping)

### 3.1 The Naming Problem

Every model architecture uses different parameter names for equivalent components:

| Canonical (rocket_surgeon) | LLaMA (HF) | GPT-2 (HF) | Mixtral (HF) |
|---------------------------|-------------|-------------|---------------|
| `attn.q_proj` | `self_attn.q_proj` | `attn.c_attn` (fused) | `self_attn.q_proj` |
| `attn.k_proj` | `self_attn.k_proj` | (fused in c_attn) | `self_attn.k_proj` |
| `mlp.gate_proj` | `mlp.gate_proj` | `mlp.c_fc` | `block_sparse_moe.gate` |
| `ln1` | `input_layernorm` | `ln_1` | `input_layernorm` |
| `ln2` | `post_attention_layernorm` | `ln_2` | `post_attention_layernorm` |

### 3.2 Reference Approaches

**pyvene:** Static `type_to_module_mapping` dicts per model type. Explicit, complete, but requires a new dict for every architecture.

**nnterp:** Maps diverse HuggingFace conventions to standardized names via nnsight. Module renaming: GPT-2's `transformer.h` → `layers`; LLaMA's `model.layers` stays `layers`. I/O accessors handle tensor/tuple differences. Works across 50+ models.

**vllm-hook:** Regex patterns, but layer-level only. No sub-component mapping.

**TransformerLens:** Re-implements models with standardized names. Complete control but requires maintaining parity.

### 3.3 Recommendation for rocket_surgeon

**Adapter trait** with per-model-family implementations. Each adapter provides:

1. **Component vocabulary** — canonical names this model family supports
2. **Name mapping** — `canonical_name(layer_idx) → native_module_path`
3. **Reverse mapping** — `native_module_path → canonical_name`
4. **Model metadata** — num_layers, num_heads, hidden_dim, num_experts (if MoE)
5. **Fused component handling** — GPT-2's fused QKV needs special split logic

The TCK already specifies this contract (tck/model/adapter.feature, 10 scenarios).

Start with LLaMA adapter (most common research target), add GPT-2 and Mixtral adapters later.

## 4. Tensor Capture Pipeline

### 4.1 Reference Approaches

**baukit:** `recursive_copy` with optional clone/detach. Default is direct reference (zero cost, but tensor can be mutated by later in-place ops).

**TransformerLens:** `save_hook` closures: `tensor.detach().to(device)`. Stores in dict, wraps in ActivationCache.

**vllm-hook:** Accumulates GPU tensors during forward, bulk GPU→CPU after forward in `execute_model()`. Handles vLLM's packed sequence format.

**nnsight:** Real tensors delivered via lock-based sync. No explicit capture pipeline — the tensor is handed directly to the worker thread.

### 4.2 rocket_surgeon's Pipeline (from ADR-0006)

```
Hook fires → tensor.detach().contiguous().cpu()
    → CUDA event sync (not full device sync)
    → BLAKE3 hash (via PyO3 bridge, GIL-released)
    → memcpy into shared-memory ring buffer
    → write ProbeFrame header (128 bytes, already implemented)
    → notify daemon via Unix domain socket (1 byte)
```

The PyO3 bridge already implements BLAKE3 hashing and ProbeFrame header serialization. The shared-memory ring buffer is WU 1.8 (not this WU).

**For WU 1.5 (skeleton):** Implement the capture pipeline up to BLAKE3 + summary stats, communicating with the daemon via stdio JSON-RPC (not shared memory yet). Shared memory is a later optimization.

### 4.3 CPU Transfer Timing

Critical decision: when to move tensors to CPU.

**Eager (in-hook):** `tensor.detach().cpu()` inside the hook. Blocks the forward pass on GPU→CPU transfer. Simple but adds latency to every tick.

**Deferred (post-step):** Record tensor reference in-hook (with CUDA event for sync), do CPU transfer after the step completes. Requires keeping GPU tensors alive longer.

**Recommendation for skeleton:** Eager transfer. Simplicity over performance. The tick is already paused (barrier gate), so the GPU→CPU transfer overlaps with the daemon's inspection time. Optimize to deferred in a later WU if profiling shows it matters.

## 5. Intervention Engine

### 5.1 Reference Approaches

**baukit:** `edit_output` callback receives output tensor, returns modified version. Clean and general.

**TransformerLens:** Hook returns non-None to replace activation. Simple.

**pyvene:** Two-phase getter/setter. Getter captures source activations, setter applies intervention. Rich intervention hierarchy: vanilla swap, addition, subtraction, rotation, autoencoder, learned mask.

**transformer-debugger:** `ablating_hook_fn` clones tensor then modifies in-place at specific indices. Three-phase ordering: ablate → save → autoencoder.

**nnsight:** SWAP event replaces the tensor in the forward pass. SKIP event bypasses a module entirely.

### 5.2 Recommendation for rocket_surgeon

**Intervention-as-command pattern.** The daemon sends an intervention command to the host; the host applies it and resumes the forward pass.

```python
# Daemon sends via JSON-RPC:
{"method": "host/intervene", "params": {
    "tensor_id": "abc123...",
    "operation": "replace",
    "data": "<base64-encoded replacement bytes>",
    "slices": [[0, 10], [0, 768]]  # optional sub-tensor targeting
}}
```

The host applies the intervention in the hook before releasing the barrier gate:

```python
def hook(module, args, output):
    capture_and_send(output)
    cmd = wait_for_command()
    if cmd.type == "intervene":
        output = apply_intervention(output, cmd)
    return output
```

Start with two operations: `replace` (full tensor swap) and `patch` (sub-tensor modification). Add pyvene-style rich interventions (addition, rotation, autoencoder) in later phases.

## 6. Multi-GPU Coordination

### 6.1 Reference Approaches

**baukit, TransformerLens, pyvene, transformer-debugger:** No multi-GPU support.

**nnsight:** Delegates entirely to vLLM. Mediators serialized and shipped to GPU workers. Hooks fire on shard-local tensors.

**vllm-hook:** Per-rank workers save TP shards separately. Main process merges shards along hidden dimension on load.

### 6.2 rocket_surgeon's Approach (from ADR-0004)

One `rs-host` process per GPU rank. Each host process:
- Owns its model shard
- Registers hooks on its local modules
- Captures shard-local tensors
- Communicates independently with the daemon

The daemon coordinates across ranks:
- Receives tensor captures from all ranks
- Merges stats via Chan/Golub/LeVeque (already implemented in tensor_stats.rs)
- Broadcasts step/intervene commands to all ranks

**For WU 1.5 (skeleton):** Single-rank only. Multi-rank coordination is a later WU. But the architecture must not preclude it — the host must be designed as "one of N" from the start.

## 7. Lifecycle and Error Handling

### 7.1 Hook Lifecycle Patterns

**baukit:** Context manager. `__enter__` registers hooks, `__exit__` removes them. Clean, but ties hook lifetime to a `with` block.

**TransformerLens:** `hooks()` context manager with `context_level` counter for nested hooks. `reset_hooks()` removes by level. `run_with_hooks()` wraps forward in context manager.

**nnsight:** One-shot self-removing hooks. Zero residual state.

**vllm-hook:** Hooks installed permanently in `load_model()`, gated by `_hook_active` flag. Zero overhead when flag is off.

### 7.2 Recommendation for rocket_surgeon

**Persistent hooks with gate flag.** Install hooks during `attach`, remove during `detach`. Each hook checks a gate flag before doing any work. Gate flag controlled by session state machine (only active in stepping/inspecting/modifying states).

This matches the session lifecycle from ADR-0004: hooks exist for the lifetime of an attached model, but only fire when the session is in an appropriate state.

### 7.3 Error Handling in Hooks

Hooks that raise exceptions can corrupt the forward pass. Reference implementations handle this differently:

- **baukit:** StopForward is intentional; other exceptions propagate
- **nnsight:** EXCEPTION event type in mediator protocol
- **vllm-hook:** Catches exceptions in hook, logs, continues

**Recommendation:** Catch all exceptions in hooks, send error notification to daemon, release barrier gate (never deadlock the forward pass). The daemon reports the error to the client. The forward pass continues with unmodified output.

## 8. PyTorch Compatibility Constraints

From the pytorch-hooks-internals lit review — critical constraints that affect the host design:

1. **torch.compile:** Forward hooks registered AFTER first compilation are silently ignored. All hooks must be registered before any torch.compile call. **rocket_surgeon must refuse to attach to compiled models** (TCK scenario: hooks.feature line 188-198).

2. **DDP:** Pre-registered hooks silently ignored unless registered inside forward(). **rocket_surgeon hooks must be registered after DDP wrapping**, or use the DDP communication hook API.

3. **FSDP:** Custom hooks interfere with FSDP's internal communication hooks. Must be careful about hook registration order.

4. **Hook ordering:** PyTorch fires hooks in FIFO order per module. rocket_surgeon's probe priority ordering must map to hook registration order.

5. **GIL interaction:** Hooks run with GIL held. BLAKE3 hashing via PyO3 releases GIL. CPU transfer (`tensor.cpu()`) releases GIL during the CUDA memcpy.

6. **Memory:** Hooks that retain tensor references prevent garbage collection. Must explicitly `del` or use weak references for long-running sessions.

## 9. Architectural Synthesis

Combining all reference learnings with ADR-0004/0005/0006 constraints:

### Host Process Structure

```
┌─────────────────────────────────────────────────────────┐
│  rs-host (Python process, one per GPU rank)             │
│                                                         │
│  ┌───────────────┐  ┌─────────────────────────────────┐ │
│  │ Control Thread │  │ Model Thread (main)             │ │
│  │                │  │                                 │ │
│  │ Unix socket ←──│──│─→ PyTorch forward pass          │ │
│  │ JSON-RPC recv  │  │   with barrier-gated hooks      │ │
│  │ Command queue  │  │                                 │ │
│  │ Event signals  │  │ Hook fires:                     │ │
│  │                │  │   1. Check gate flag             │ │
│  │                │  │   2. Capture tensor              │ │
│  │                │  │   3. BLAKE3 hash                 │ │
│  │                │  │   4. Send to daemon              │ │
│  │                │  │   5. Wait on barrier event       │ │
│  │                │  │   6. Apply intervention (if any) │ │
│  │                │  │   7. Resume forward pass         │ │
│  └───────────────┘  └─────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Adapter (per model family)                          │ │
│  │  - Canonical ↔ native name mapping                  │ │
│  │  - Component vocabulary                             │ │
│  │  - Model metadata extraction                        │ │
│  │  - Hook point enumeration                           │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Bridge (PyO3, already implemented)                  │ │
│  │  - BLAKE3 hash (GIL-released)                       │ │
│  │  - ProbeFrame header (128 bytes)                    │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Hook mechanism | PyTorch native `register_forward_hook` | Must work with arbitrary models |
| Hook lifetime | Persistent with gate flag | Match session lifecycle (attach→detach) |
| Stepping | Barrier gate (threading.Event) | External daemon control, not in-process |
| Name mapping | Per-model adapter with canonical vocabulary | TCK-specified contract |
| Tensor capture | Eager detach+cpu in-hook | Simple, latency hidden by barrier wait |
| Daemon comms | JSON-RPC over Unix domain socket (skeleton) | Shared memory in later WU |
| Intervention | Command-based (daemon → host) | Dual-interface (TUI + LLM) requirement |
| Multi-GPU | Single-rank skeleton, multi-rank later | Architecture must not preclude it |
| Compiled models | Refuse to attach | PyTorch silently ignores post-compile hooks |

### WU 1.5 Skeleton Scope

The skeleton should establish:
1. Host process entry point and lifecycle (start, attach, detach, shutdown)
2. Adapter trait and LLaMA adapter implementation
3. Hook registration on model components via adapter
4. Barrier gate mechanism (threading.Event + control thread)
5. Tensor capture pipeline (detach → cpu → BLAKE3 → send to daemon)
6. JSON-RPC communication with daemon (stdio for skeleton, Unix socket later)
7. Intervention application (replace operation)

**Not** in skeleton scope: shared memory ring buffer (WU 1.8), multi-rank coordination, MoE-specific hooks, backward pass hooks, head-level granularity.

## Key References

- Bau, D. (2022). baukit: Tools for microanalysis of neural networks. GitHub.
- Fiotto-Kaufman, J. et al. (2024). NNsight and NDIF: Democratizing Access to Foundation Model Internals. arXiv:2407.14561.
- Nanda, N. (2022). TransformerLens. GitHub.
- Wu, Z. et al. (2024). pyvene: A Library for Understanding and Improving PyTorch Models via Interventions. NeurIPS 2024.
- OpenAI (2024). Transformer Debugger. GitHub.
- vllm-hook contributors (2024). vllm-hook-plugins. GitHub.
