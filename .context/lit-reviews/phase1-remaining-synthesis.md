# Phase 1 Remaining Work Units — Lit Review Synthesis

Consolidated findings from four parallel research threads for WU 1.6–1.16.
Focused on design-load-bearing findings that affect architectural decisions.

## 1. PyTorch Hook Internals — What Matters for RS

### Hook API Surface

`register_forward_hook(hook, *, prepend=False, with_kwargs=False, always_call=False)`
- Fires AFTER module.forward() returns
- If hook returns non-None, that value REPLACES the module output downstream
- `prepend=True` uses `OrderedDict.move_to_end(last=False)` — deterministic ordering
- `always_call=True` fires even if forward() or an earlier hook raised — essential for cleanup

`register_forward_pre_hook(hook, *, prepend=False, with_kwargs=False)`
- Fires BEFORE module.forward()
- Can modify inputs (return new args tuple)

### The Fast Path — Critical

```python
# Inside nn.Module._call_impl:
if not (self._backward_hooks or self._backward_pre_hooks
        or self._forward_hooks or self._forward_pre_hooks
        or _global_backward_pre_hooks or _global_backward_hooks
        or _global_forward_hooks or _global_forward_pre_hooks):
    return forward_call(*args, **kwargs)  # FAST PATH — skips all hook logic
```

If NO hooks exist on a module AND no global hooks exist, PyTorch skips the entire hook
dispatch machinery. This means dynamically-added hooks may not fire if the module
entered the fast path before registration.

**Design consequence**: RS must install sentinel no-op hooks on every module at attach
time to disable the fast path. nnsight does exactly this: `lambda _, __, output: output`
on every module. Without this, hooks added later (e.g., dynamic probe definitions) may
silently fail.

### named_modules() Order != Execution Order

`named_modules()` returns depth-first, pre-order traversal in **constructor insertion order**.
If `__init__` defines `self.mlp` before `self.attn` but `forward()` calls attn first,
named_modules gives the wrong order for stepping.

**Design consequence**: RS cannot rely on named_modules() order for tick sequencing.
Execution order must be discovered dynamically — either via a tracing forward pass or
by recording hook firing order during the first forward pass.

### Hook Safety

- `RemovableHandle.remove()` is safe to call from within a hook (PyTorch copies the hook
  dict items before iterating, so mutation during iteration doesn't raise)
- Exceptions in hooks propagate normally (not swallowed)
- Hooks work under `torch.no_grad()` and `torch.inference_mode()` (they're call machinery,
  not autograd)
- GPU tensors accessible in hooks after module.forward() completes (output data is ready)

### torch.compile Interaction

Hooks registered AFTER first compilation are silently ignored — the compiled graph bakes
in hook presence/absence and doesn't re-check. `torch._dynamo.config.skip_nnmodule_hook_guards`
controls this (default True = no guards = hooks frozen at compile time).

**Design consequence**: RS Tier A (eager mode) is the only supported tier for Phase 1.
Compiled models must be rejected at attach time. The session state machine already does this
(`execution_mode: compiled` check in session.rs).

## 2. Reference Implementation Patterns

### TransformerLens — Re-implementation Approach
- Re-implements models from scratch with `HookPoint(nn.Module)` identity modules at
  every interesting site
- No stepping/pausing — single-shot `run_with_hooks` / `run_with_cache`
- Not applicable to RS's hook-existing-models approach, but the canonical name vocabulary
  (resid_pre, resid_post, attn_out, mlp_out, etc.) is a useful reference

### nnsight — Closest Architectural Match
- Envoy proxy wraps every module, intercepts attribute access
- Deferred execution: captures AST inside `with model.trace()`, replays during forward
- **Ping-pong threading**: two threads, C-level locks, only one runs at a time
  - Forward pass thread blocks in hook, worker thread runs intervention code
  - Worker thread blocks, forward pass thread resumes
  - 6 event types: VALUE, SWAP, SKIP, BARRIER, END, EXCEPTION
- CUDA stream propagation: explicitly copies caller's stream into worker threads
  (PyTorch defaults worker threads to stream 0 — data race if not handled)
- Sentinel hooks on every module to defeat fast path

**Key difference from RS**: nnsight's barrier is Python-thread-to-Python-thread within
one process. RS's barrier is Rust-daemon-to-Python-worker across processes. The
synchronization mechanism is different (IPC vs threading) but the pattern is analogous.

### baukit/nethook — Context Manager Pattern
- `Trace` class wraps hook lifecycle in `__enter__`/`__exit__`
- `StopForward` exception for partial execution (one-shot, NOT resumable)
- `recursive_copy(detach=True)` for independent tensor copies
- `TraceDict` for multi-layer monitoring

### pyvene — Two-Pass Getter/Setter
- Getter phase: forward pass captures source activations
- Setter phase: forward pass applies interventions using captured sources
- 14+ intervention types in class hierarchy
- Per-model-family `type_to_module_mapping` dicts — closest to RS adapter pattern:
  ```python
  type_to_module_mapping["llama"]["block_output"] = ("model.layers.%s", CONST_OUTPUT_HOOK)
  ```

### OpenAI Transformer Debugger — Custom Model Code
- Hooks baked into custom forward() implementations, not registered dynamically
- No stepping/pausing
- Three-phase: fwd (observe) → bwd (gradient) → fwd2 (post-intervention observe)

## 3. Barrier/Pause Mechanism for RS

### The Pattern

```python
gate = threading.Event()

def hook(module, args, output):
    captured = output.detach().contiguous()
    cpu_tensor = captured.cpu()  # implicit CUDA sync
    send_to_daemon(cpu_tensor)
    gate.clear()
    gate.wait()  # blocks until daemon sends continue
    return modified_output if intervention_pending else output
```

### CUDA Considerations While Paused

- `gate.wait()` releases the GIL — other Python threads can run
- GPU continues processing already-queued work but no NEW ops are queued
- Must synchronize CUDA stream before reading tensor data:
  - `.cpu()` implicitly synchronizes (safest)
  - Alternatively: `torch.cuda.Event().record()` + `.wait()` (more targeted)
  - Avoid `torch.cuda.synchronize()` (synchronizes ALL streams, overkill)

### Memory While Paused

While paused at layer N of L:
- All intermediate activations from layers 0..N are alive (stack frames hold refs)
- Autograd graph holds additional references if requires_grad=True
- For 7B model paused at layer 16/32: ~half the activation memory is pinned

**Mitigation**: `detach()` captured tensors. Copy to CPU and delete GPU refs if memory
pressure is a concern. For Phase 1 (baby models on CPU), this isn't critical.

### RS-Specific: Cross-Process Barrier

RS's barrier is different from nnsight's — it crosses a process boundary:
1. Hook fires in Python (inside worker process)
2. Worker's Rust code reads tensor via `data_ptr()` (zero-copy, same address space)
3. Worker sends summary/data to daemon over IPC (Content-Length framed JSON-RPC)
4. Worker blocks waiting for daemon's "continue" command
5. Daemon inspects data, possibly queues intervention
6. Daemon sends "continue" (with optional intervention payload)
7. Worker applies intervention (if any), releases hook, forward pass resumes

The IPC round-trip IS the barrier. No threading.Event needed — the worker's stdin
read is the blocking point.

## 4. Module Tree & Adapter Design

### Architecture Detection

`config.model_type` from AutoConfig is the dispatch key:
- `"llama"` → Llama-2/3, Mistral, CodeLlama (identical structure)
- `"gpt_neox"` → GPT-NeoX, Pythia
- `"mixtral"` → Mixtral (MoE)
- `"gpt2"` → GPT-2 (CI model)

### Module Tree Structures

**Llama**: `model.layers[i].{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input_layernorm, post_attention_layernorm}`

**GPT-NeoX**: `gpt_neox.layers[i].{attention.{query_key_value, dense}, mlp.{dense_h_to_4h, dense_4h_to_h}, input_layernorm, post_attention_layernorm}`
- Fused QKV: single `query_key_value` Linear — cannot hook Q/K/V separately without post-hook splitting
- Parallel residual mode: attn and MLP compute in parallel from same input

**Mixtral (MoE)**: Same as Llama for attention. MLP replaced by:
`model.layers[i].mlp` is `MixtralSparseMoeBlock` with:
- `gate` (MixtralTopKRouter) — `nn.Parameter`, not nn.Linear
- `experts` (MixtralExperts) — 3D parameter tensors, NOT individual nn.Module submodules

**MoE gotcha**: Individual experts are NOT hookable via register_forward_hook because
they're batched 3D tensors, not separate modules. Must hook the MoE block and split
outputs in the hook callback. (Phase 6 concern, but adapter must not promise per-expert
hookability.)

### Adapter Mapping Pattern

Following pyvene's approach — static dict mapping canonical names to module paths:
```
canonical_name → (module_path_template, hook_type)
```

For Phase 1, the adapter must:
1. Walk `named_modules()` to build the full module inventory
2. Match against the per-family mapping dict
3. Unknown modules get `_raw.<original.path>` fallback name
4. Report hookable components to daemon
5. Discover execution order dynamically (first forward pass)

## 5. Shared Memory & Tensor Handoff

### Within Worker: Zero-Copy via PyO3

Worker embeds Python — same address space. Rust can read tensor data directly:
```rust
let data_ptr = tensor.getattr("data_ptr")?.call0()?.extract::<usize>()?;
let numel = tensor.getattr("numel")?.call0()?.extract::<usize>()?;
let slice = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, numel * dtype_size) };
```

Must hold `Py<PyAny>` reference to prevent GC. GPU tensors require `.cpu()` first.

### Worker → Daemon: Shared Memory Ring Buffer

- POSIX `shm_open()` + `mmap()` — Rust opens by name via `nix` crate
- macOS: POSIX shm works (no 4MB limit — that's SysV only). Name limit 31 chars.
- Python 3.13+: use `track=False` to prevent resource_tracker auto-unlink
- Notification: Unix domain socket (1-byte write per frame)
- SPSC ring buffer with atomic cursors (Release/Acquire ordering)

### Performance Budget (17MB Llama-3-8B residual tensor)

| Operation | Time |
|-----------|------|
| GPU→CPU DMA | ~2ms |
| BLAKE3 hash (single-thread) | ~5.7ms |
| memcpy to shm | ~2ms |
| **Total** | **~10ms** |

With optimizations (multi-thread BLAKE3, forward GPU stats via control channel): ~4ms.

### Key Optimization

Compute summary stats on GPU (PyTorch reduction ops), send stats via JSON-RPC control
channel alongside the data path. Daemon receives pre-computed stats, avoids CPU
recomputation (saves ~8ms per tensor).

## 6. Perfetto Trace Sink

- Vendor monolithic `perfetto_trace.proto` (19k lines), generate with `prost-build`
- Streaming append: protobuf repeated-field encoding = concatenation. Write packets
  to `BufWriter<File>` as generated — no in-memory Trace object needed.
- Mapping: session→Process, rank→Thread, layer→named sub-track, tick→duration event
  pair, probe firing→instant event, intervention→instant event
- String interning via `interned_data` saves ~80MB at 1000 forward passes
- File size: ~64-160KB per forward pass. 1000 passes ≈ 64-500MB. Within Perfetto UI's
  ~2GB browser limit.

## 7. Design-Impacting Findings Summary

| Finding | Impact on RS Design |
|---------|-------------------|
| Fast path optimization | Must install sentinel hooks on every module at attach time |
| named_modules() ≠ execution order | Must discover execution order dynamically |
| Mixtral experts not hookable individually | Adapter must not promise per-expert hooks (Phase 6 concern) |
| nnsight's ping-pong threading | RS uses IPC-based barrier instead — simpler, cross-process |
| Worker zero-copy via data_ptr() | No shared memory needed within worker process |
| Shared memory only for worker→daemon | Ring buffer with Unix socket notification |
| GPU stats via control channel | Avoids daemon CPU recomputation |
| Perfetto streaming append | No need for in-memory trace buffer |
| torch.compile breaks dynamic hooks | Phase 1 must reject compiled models (already done) |
| CUDA stream propagation | Worker must capture/restore stream for GPU interventions |
