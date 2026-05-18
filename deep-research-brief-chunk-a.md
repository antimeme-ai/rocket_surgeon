# Deep Research Brief: rocket_surgeon Phase 1 Chunk A

## What is rocket_surgeon?

A proper debugger and in-situ surgery tool for transformer neural networks. It operates natively on multi-GPU forward passes, letting you step through transformer internals (dense and MoE) one tick at a time, forward and backward, with full surgical intervention between steps.

Architecture: three-process (Rust daemon → Rust orchestrator → Rust+PyO3 worker). The worker embeds Python via PyO3 `auto-initialize` to interact with PyTorch models. Communication is Content-Length framed JSON-RPC over stdin/stdout at every IPC boundary.

The project is in Phase 1 — building the single-GPU eager-mode debugger. Phase 0 (protocol spec, TCK, probe grammar) is complete. We just finished WU 1.5 (model host skeleton: three-process spawn, model load/unload, basic IPC).

## What we're building next (Chunk A)

**Model Adapter** (WU 1.6): Given a loaded HuggingFace model, map its module tree to canonical probe-point names. Core adapter framework handles 1:1 (direct) and 1:N (fused) module mappings. Per-family mapping declarations are static data. Execution order discovered dynamically via a tracing forward pass (because `named_modules()` returns constructor order, not execution order).

**Hook Manager** (WU 1.7): Register PyTorch forward hooks on every mapped component. Implement a barrier gate that pauses the forward pass at tick boundaries. Capture tensor data based on active probe policy (none / summary stats on GPU / full tensor transfer). Two threads in worker: Rust IPC thread + Python forward thread, coordinated via `threading.Event` + `queue.Queue`.

The full design spec is included below.

## What we want researched

Please investigate the following open questions and risk areas. For each, we want: what you found, what the implications are for our design, and whether our current approach handles it correctly or needs revision.

### 1. threading.Event + queue.Queue barrier safety in PyTorch hooks

Our design uses `threading.Event.wait()` inside a PyTorch forward hook to pause the forward pass, with a `queue.Queue` for data handoff. Questions:

- Are there known issues with `threading.Event` inside PyTorch forward hooks? Any CPython or PyTorch-specific gotchas?
- When `Event.wait()` releases the GIL, can another thread safely call PyO3/Python operations? We need the Rust IPC thread to call into Python (e.g., `compute_tensor_stats`) while the forward thread is blocked.
- Is `queue.Queue` the right choice for cross-thread tensor handoff? It holds references to Python objects (tensors). Are there GC or reference counting issues when tensors are put on a Queue from the forward thread and consumed from the Rust thread?
- What happens if `Event.set()` is called before `Event.wait()` (race on first tick)? Our design calls `set()` then `clear()` — is there a window where the forward thread misses the signal?
- nnsight uses `_thread.allocate_lock()` (C-level locks) instead of `threading.Event`. Is there a reason to prefer C-level locks? What's the performance difference?

### 2. PyTorch forward hook behavior with mixed precision / autocast

Our design doesn't explicitly address `torch.amp.autocast` or mixed precision training/inference. Questions:

- When a model is running under `autocast`, what dtype are the tensors seen by forward hooks? The original dtype, or the autocasted dtype?
- If we call `tensor.mean()` / `tensor.std()` inside a hook under autocast, do the reduction ops run in the autocasted precision or full precision?
- Does `tensor.detach().cpu()` preserve the autocasted dtype or cast back?
- Are there any issues with holding references to autocasted tensors while the forward pass is paused? (Autocast context manager scope vs tensor lifetime)

### 3. Execution order discovery — edge cases

We run a tracing forward pass with lightweight pre-hooks on every module to discover execution order. Questions:

- Are there HuggingFace models where execution order changes between forward passes? (e.g., dynamic routing, early exit, conditional computation)
- What about models that use `torch.utils.checkpoint` (gradient checkpointing)? Does that affect hook firing order?
- KV-cache: on the second forward pass (generation), some models skip recomputing certain modules. Does this change the hook firing order vs the first pass? Should we discover on the first pass specifically?
- Are there models where a module's `forward()` is called multiple times per forward pass? (e.g., weight sharing, parameter tying) How do hooks behave — fire once or once per call?

### 4. Fused module patterns across the HuggingFace ecosystem

We designed 1:N fused mappings for cases like GPT-NeoX's `query_key_value`. Questions:

- What other fusion patterns exist in popular HuggingFace models? (e.g., fused gate+up projections in Llama variants, fused attention in newer architectures)
- Are there models where the fusion isn't a simple tensor split (chunk along a dimension) but requires more complex decomposition?
- How does HuggingFace handle fused projections in the newer `transformers` releases (4.40+)? Is there a trend toward or away from fused implementations?
- Mixtral's `MixtralExperts` stores expert weights as 3D tensors, not individual modules. Are there other MoE implementations that ARE per-module hookable? How does the ecosystem trend here?

### 5. Sentinel hook overhead

We install no-op sentinel hooks (`lambda _, __, out: out`) on every module to defeat PyTorch's fast-path optimization. Questions:

- What is the measured overhead of sentinel hooks? If a model has 500 modules and we hook all of them, how much does this slow down the forward pass compared to no hooks?
- nnsight uses the same pattern. Has anyone benchmarked this? Are there published numbers?
- Is there a lighter-weight way to defeat the fast path? (e.g., a single global hook instead of per-module sentinels?)
- Does the sentinel hook overhead change between PyTorch versions? (The fast path code has been refactored several times)

### 6. Memory implications of pausing mid-forward-pass

When we block in a hook via `Event.wait()`, all intermediate tensors from prior layers are alive (Python stack frames hold references). Questions:

- For a 7B parameter model paused at layer 16 of 32, approximately how much GPU memory is consumed by intermediate activations? (Rough estimate for fp32, fp16, bf16)
- Does `torch.no_grad()` (inference mode) change the memory footprint? (No autograd graph to hold)
- Are there any Python stack frame issues with long pauses? (e.g., does the Python GC do anything problematic when a thread is blocked for extended periods?)
- If the user pauses for a long time (minutes, while inspecting data), are there any CUDA driver timeouts or resource issues?

### 7. Module tree patterns in the latest HuggingFace transformers

We need accurate adapter mapping tables. For each of these model families, please provide the actual `named_modules()` output structure (from the latest `transformers` release):

- **LlamaForCausalLM** (Llama-3 / Llama-3.1 / Llama-3.2 — are there structural differences between these?)
- **GPT2LMHeadModel** (our CI model)
- **MistralForCausalLM** (is this identical to Llama or are there differences?)
- **Phi3ForCausalLM** (popular small model, may be a good CI candidate)
- **Qwen2ForCausalLM** (growing in popularity)

For each: module tree structure, any fused modules, any structural surprises vs Llama.

### 8. GIL behavior with PyO3 `auto-initialize` + threading.Event

Our worker binary initializes Python via `pyo3::prepare_freethreaded_python()` and then has two threads: the Rust main thread (IPC) and a Python forward thread. Questions:

- When the Python forward thread is blocked on `Event.wait()` (GIL released), can the Rust main thread acquire the GIL via `Python::with_gil()` to call Python functions?
- Are there any known issues with `pyo3::prepare_freethreaded_python()` + Python threads created from the Rust side vs Python side?
- If we spawn the forward thread from Python (`threading.Thread`), does the Rust thread need to do anything special to interact with it?
- What are the GIL acquisition patterns for PyO3 when another thread holds the GIL? (Blocking vs try-acquire?)

---

## Design Spec (for reference)

[The full spec from docs/specs/2026-05-18-adapter-hook-manager-design.md is included in the project context. Key sections: Core adapter framework with Direct/Fused/Container module mappings, per-family static declarations, execution order discovery + caching, sentinel + capture hook two-layer system, barrier via threading.Event + queue.Queue, tick cycle sequence diagram, capture policy (none/summary/full), bridge.py thin functions, worker crate Rust modules (adapter.rs, tick.rs, capture.rs), data flow diagrams, error handling, multi-GPU design-for.]

---

## What we DON'T need

- We don't need implementation advice or code. Just research findings.
- We don't need alternatives to our overall architecture (three-process, PyO3 embedding, JSON-RPC IPC). Those decisions are made.
- We don't need coverage of torch.compile / compiled models — we reject those at attach time.
- We don't need multi-GPU specifics — that's Phase 5.

## What would be most valuable

Anything that would cause us to revise the design BEFORE we write 10+ implementation tasks and discover the problem mid-build. Edge cases, failure modes, version-specific behavior changes, things that "work in the simple case but break at scale."
