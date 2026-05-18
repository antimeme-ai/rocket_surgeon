# rocket_surgeon: Deep Technical Research on Eight Design Questions

**TL;DR**
- The current design is approximately correct for the simple case but needs three concrete revisions before implementation begins: (1) replace `threading.Event` + `queue.Queue` with `_thread.allocate_lock()`-based single-slot mailboxes (mirroring what nnsight's `Mediator.Value` actually does in `src/nnsight/intervention/interleaver.py`); (2) add a documented contract that hooks see autocast-dtype tensors and that reductions computed inside hooks run in autocast precision unless explicitly disabled; (3) treat execution-order discovery as **per-call** rather than per-model — gradient checkpointing, KV-cache prefill-vs-decode, weight-tying, and conditional MoE routing each invalidate a cached order.
- The PyO3 / two-thread architecture is fundamentally sound on PyO3 ≥ 0.22 with GIL-build CPython, but free-threaded CPython 3.13t introduces a documented deadlock pattern (PyO3 discussion #4738) when a thread is spawned-and-joined while another is attached — this must be guarded with `Python::detach` around the join boundary. PyO3 0.26.0 (released 2025-08-29, PRs #5206/#5209/#5223) renamed `Python::with_gil → Python::attach`, `Python::allow_threads → Python::detach`, and `prepare_freethreaded_python → Python::initialize`; rocket_surgeon should adopt the new names from day one.
- Sentinel hooks are the right mechanism (nnsight uses the same trick) and overhead is acceptable (a few µs–tens of µs of dispatch per layer, dominated by Python overhead, not the no-op body); the larger memory concern is that pausing in a hook keeps every prior layer's intermediate activation alive (for Llama-3 8B paused at layer 16/32 in inference mode, that is roughly 0.5–1 GB of fp16 activations at seq=8K, less at shorter contexts), and CUDA TDR/watchdog timeouts apply only to in-flight kernels — not to a Python pause between kernels — so multi-minute inspection pauses are safe on headless Linux compute GPUs but problematic only on display-bound GPUs.

## Key Findings

### Q1 — `threading.Event` + `queue.Queue` barrier safety in PyTorch hooks

**(a) Reference-implementation evidence.** nnsight's `Mediator` (in `src/nnsight/intervention/interleaver.py`) does **not** use `threading.Event` or `queue.Queue`. It uses a hand-rolled single-slot mailbox class `Mediator.Value` whose synchronization primitive is `lock = allocate_lock()` — the low-level `_thread.allocate_lock()` C primitive. Two such mailboxes per Mediator implement a strict ping-pong: `event_queue` (worker→main) and `response_queue` (main→worker). The `wait()` method blocks via `self.lock.acquire()` on an already-acquired lock; `put()` stores the value and calls `self.lock.release()` to wake the consumer. nnsight's `NNsight.md` attributes ~0.04 ms to "Lock synchronization — Thread coordination between model and intervention code" and ~0.05 ms to "Hook dispatch (handle, handle_value_event) — PyTorch hook → mediator event queue → worker thread."

**(b) GIL release semantics.** `threading.Event.wait()` is implemented in `cpython/Lib/threading.py` via `Condition.wait()` → `waiter = _allocate_lock(); waiter.acquire(); ... waiter.acquire(True, timeout)` — the *waiter* acquisition is the GIL-releasing call (it goes through `PyThread_acquire_lock_timed` in CPython's `Python/thread_pthread.h`, which releases the GIL around the blocking syscall). So while the forward thread is blocked in `Event.wait()` or in a raw `_thread` lock `acquire()`, the GIL is released and another thread (including a Rust IPC thread calling `Python::with_gil`/`Python::attach`) can acquire it and run arbitrary Python — including `tensor.mean()` on a tensor reference that was previously published into a shared slot. The official `pyo3::Python` documentation states the GIL can be temporarily released by the Python interpreter during a function call and is reacquired before returning to the Rust code.

**(c) `queue.Queue` and tensor refcounts.** CPython's `queue.Queue` is built on `threading.Condition` + a `collections.deque`. Putting a Python tensor object on the queue performs a normal `Py_INCREF` on insert and `Py_DECREF` on get — there is no special handling. The risk is **not** the refcount itself but that a tensor sitting in the queue keeps the underlying CUDA storage alive (the tensor object holds the storage as a C++ shared_ptr field), and if the consumer is slow the activation memory accumulates. nnsight explicitly mitigated this in v0.6 with documented "Fixed reference loops in the interleaver and tracer" and uses an explicit `Value.restore()` method that drops the slot's reference after consumption.

**(d) Event set-before-wait race.** Python's `threading.Event` documents: "If the flag is already True, the thread continues immediately." So `set()` before `wait()` is safe — `wait()` returns immediately. The dangerous pattern is `set()` then `clear()` from the signaling thread without the waiter having observed the set: between two GIL releases the flag transitions True→False and the waiter never sees True. CPython's `Event.set()` notifies all waiters under the internal Condition's lock, but if no waiter has reached the `wait()` call yet, the notify is lost and only the flag remains. A `clear()` immediately after `set()` therefore **does have a window where the forward thread misses the signal** — a well-known Event anti-pattern that motivates nnsight's choice of a single-slot lock mailbox (which has no flag, only a binary acquire/release state).

**(e) Why nnsight chose `_thread.allocate_lock()`.** Three reasons: (i) `threading.Event` is `Condition` + `Lock` + a Python-level `_flag` boolean check — all of which execute Python bytecode per signal; the raw `_thread` lock is a thin C wrapper around a pthread mutex/SRWLOCK with no Python-side bookkeeping; (ii) the single-slot mailbox semantics map perfectly to a single binary semaphore — Event's flag-with-broadcast semantics are overkill and re-introduce the set/clear race; (iii) measurable per-hook savings: the nnsight v0.6 blog (nnsight.net, Feb 26 2026, by Jaden Fiotto-Kaufman) reports "2.4–3.9x Faster Traces", with fixed setup cost (source extraction, AST parsing, compilation, thread creation) dropping from ~1,100 µs to ~210 µs and per-intervention cost from ~42 µs to ~34 µs — improvements driven partly by this synchronization design.

**Verdict:** The current design's choice of `threading.Event` + `queue.Queue` **needs revision**. Match nnsight's pattern: lock-based single-slot mailboxes with `_thread.allocate_lock()`, one per direction, with an explicit clear/restore method to drop tensor references after consumption. Cross-thread Python calls from Rust (e.g., `compute_tensor_stats`) **are safe** while the forward thread is blocked — the GIL is released either way.

### Q2 — PyTorch forward hooks under `torch.amp.autocast`

**(a) Dtype seen by hooks.** From `pytorch/torch/amp/autocast_mode.py` (verified across v2.7.0–v2.11): autocast is a thread-local C-level dispatcher that intercepts eligible ops and casts inputs to the target dtype (typically bf16/fp16) at op-dispatch time. The output of an autocasted op like `nn.Linear`/`matmul` is in the autocast dtype. A `register_forward_hook(module, input, output)` on a `nn.Linear` therefore sees `output` already in the autocast dtype. The `input` argument is whatever the caller passed (often fp32 if from an embedding, autocast dtype if from a previous autocasted op).

**(b) Reductions inside hooks.** `tensor.mean()` and `tensor.std()` are **not** on the autocast lower-precision allowlist. Per PyTorch's documented autocast op lists, ops like `__matmul__`, `addmm`, `linear`, `conv1d`, `conv2d`, `conv3d`, `bmm`, `addbmm`, `addmv`, `addr`, `baddbmm` autocast to fp16/bf16, while reductions and softmax either run in the input's dtype or are explicitly promoted. So calling `.mean()` on an fp16 hook output runs in fp16, **not** fp32, unless you explicitly cast first or wrap the body in `torch.amp.autocast(..., enabled=False)`. This produces inaccurate statistics for activations near the dtype's range limits (fp16 overflows around 65504; bf16 has poor mantissa precision).

**(c) `.detach().cpu()`.** Both `detach()` and `.cpu()` are dtype-preserving — they do not change dtype. An fp16 tensor copied to CPU remains fp16. The user-facing implication: stats sent over JSON-RPC will be lower-precision than callers might assume.

**(d) Autocast scope vs. tensor lifetime.** The autocast context manager controls the *dispatcher* state for the lifetime of the `with` block; once you exit, new ops on cached tensors revert to default dispatch. But the dtype of an already-produced tensor is a property of the tensor itself, unaffected by leaving autocast scope. Holding an autocasted tensor reference past the autocast `__exit__` is safe — the tensor stays in bf16/fp16; you just don't get further autocasted ops on it. PyTorch's own documentation specifies "The autocast state is thread-local."

**Verdict:** The design needs a documented contract: hook outputs reflect autocast dtype (not the model's nominal dtype). Stats computed inside hooks must either (i) explicitly cast to fp32 first, or (ii) be labeled with the dtype they were computed in. No architectural revision needed, but a correctness pitfall if undocumented.

### Q3 — Execution-order discovery edge cases

**(a) Dynamic routing / early exit / conditional computation.** The most common case in HF transformers is **MoE routing** (Mixtral's `MixtralSparseMoeBlock`): the router selects top-k experts per token, and `self.experts[i]` is called only for selected `i`. If discovery happens on input A and execution on input B, the set of fired hooks differs. Other examples: Phi-3.5-MoE (`phimoe`), Qwen2-MoE/Qwen3-MoE, DeepSeek-V3. Early-exit architectures (CALM, LayerSkip) also vary execution per token.

**(b) Gradient checkpointing.** `torch.utils.checkpoint` (non-reentrant, default since 1.11) runs each checkpointed function twice: once during forward (no autograd recording), once during backward (re-execution to reconstruct activations). Forward hooks fire on the recompute as well — see GitHub issue pytorch/pytorch#81296 ("Gradient hook is called twice for shared parameter with activation checkpointing") and the wider DDP-error pattern "Parameter at index N has been marked as ready twice" documented in huggingface/transformers#23018 and huggingface/peft#313. For an eager-mode debugger in `torch.inference_mode()` this is not triggered, but if a user attaches mid-training, expect double-fires for checkpointed modules during backward.

**(c) KV-cache prefill vs. decode.** During generation, the first forward pass (prefill) processes the full prompt; subsequent passes (decode) process one token at a time but still call every layer. Per-module hook firing order is **unchanged** between prefill and decode for vanilla transformers (Llama, Mistral, Qwen2, GPT-2) — every layer's `self_attn` and `mlp` fires once per forward. What changes: tensor shapes (seq_len = full vs. 1), and for MoE models the set of active experts changes per token. Discovery on prefill therefore captures all *deterministic* modules but misses the dynamic expert dimension.

**(d) Weight-tying / shared modules / multi-call.** If a `nn.Module` instance is referenced twice in the model tree (parameter tying — e.g., GPT-2's `lm_head` tied to `wte`, confirmed in huggingface/transformers#6291 where `lm_head.weight` is not in `named_parameters` because of the tie — or any architecture that reuses a layer), PyTorch fires its forward hooks **once per call**, not once total. This is documented behavior of `_call_impl`. Discovery must track call indices, not just module identity. `named_modules()` returns each module once, so the discovery pass must observe the actual hook firing sequence to detect re-entries.

**Verdict:** Discovery must run on every forward pass being debugged, not be cached across passes. Treat order as a per-invocation property, with optional caching only when the user pins it. Specifically: (i) re-run a lightweight pre-hook trace on each `forward()` to detect MoE expert selection and conditional branches; (ii) document gradient-checkpointing double-fires; (iii) key by `(module_id, call_index)` rather than just `module_id`.

### Q4 — Fused module patterns across the HF ecosystem

**(a) Other fusion patterns.**
- **Phi-3** (`transformers/models/phi3/modeling_phi3.py`, main branch): `self.qkv_proj = nn.Linear(config.hidden_size, op_size, bias=False)` where `op_size = config.num_attention_heads * self.head_dim + 2 * (config.num_key_value_heads * self.head_dim)`. The MLP fuses `gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size)` and splits with `narrow`/`chunk`. The Phi-3 docs state verbatim: "The query, key and values are fused, and the MLP's up and gate projection layers are also fused."
- **Phi-3.5-MoE** (`phimoe`): same fusion plus MoE; docs: "very similar to Mixtral with the main difference of Phi3LongRoPEScaledRotaryEmbedding... The query, key and values are fused, and the MLP's up and gate projection layers are also fused."
- **GPT-2** (`modeling_gpt2.py`, main): `c_attn` is a `Conv1D` (effectively `Linear`) of shape `(n_embd, 3*n_embd)` producing fused Q/K/V, split at runtime via `self.c_attn(hidden_states).split(self.split_size, dim=2)`. Hooking `c_attn` yields Q‖K‖V concatenated along dim=2, **not** three separate tensors.
- **GPT-NeoX**: `query_key_value` (already in the design).
- **DBRX**: fused QKV with extra clipping (HF PR #30423 discussion).

**(b) Non-chunk decompositions.** Most HF fused projections split along last-dim via `.chunk(N, -1)` or `.split(sizes, dim=2)`. Phi-3's QKV is the notable exception: `op_size = n_heads*hd + 2 * n_kv_heads*hd` (GQA-aware), so the split is `(q_size, k_size, v_size)` with three **unequal** sizes. Any debugger that assumes "fused = N equal chunks" will silently produce wrong Q/K/V for Phi-3 with GQA. Mistral, Llama, Qwen2 do **not** fuse — Q, K, V are separate `nn.Linear` modules.

**(c) Trend in transformers 4.40+.** The trend is **toward** Llama-style unfused projections, not toward more fusion. Llama, Mistral, Qwen2, Gemma, Mixtral, DeepSeek-V2/V3 all use separated `q_proj`/`k_proj`/`v_proj` and separated `gate_proj`/`up_proj`/`down_proj`. The notable fused exceptions are Phi-3 family, GPT-2/GPT-NeoX (legacy), and DBRX. The HF Phi-3 PR discussion (huggingface/transformers#30423) shows an explicit maintainer preference to convert to Llama format ("let's make the weights standardized for the good of the entire community"), but the Phi-3 author retained fusion. So new families converge on Llama's layout; legacy fused families remain pinned.

**(d) Mixtral 3D experts vs. per-module MoEs.** The reference HF Mixtral uses `self.experts = nn.ModuleList([MixtralBLockSparseTop2MLP(config) for _ in range(self.num_experts)])` — each expert is a separate `nn.Module` and individually hookable. The same is true for Qwen2-MoE/Qwen3-MoE and DeepSeek-V3 in HF eager mode. Third-party MoE optimizations (Megablocks, Tutel, vLLM's fused MoE kernel) do collapse experts into 3D tensors for grouped GEMM, but those are not the default HF eager path that the debugger targets.

**Verdict:** The 1:N fused-mapping infrastructure is necessary for **Phi-3** (qkv_proj uses **unequal** splits; gate_up_proj uses equal split) and **GPT-2 / GPT-NeoX** (3-equal-chunk c_attn / query_key_value). For Llama / Mistral / Qwen2 and HF eager Mixtral, no decomposition is needed. The "Mixtral stores experts as 3D tensors" assumption in the design is incorrect for the HF eager path.

### Q5 — Sentinel hook overhead

**(a) Measured.** No public microbenchmark isolates pure no-op sentinel hooks; nnsight's `NNsight.md` aggregates "Hook dispatch (handle, handle_value_event) ~0.05 ms" which includes meaningful work, not a sentinel. PyTorch documentation warns "hooks add some overhead to each forward/backward pass... avoid leaving unnecessary hooks active during intensive training." For a model with 500 modules all sentinel-hooked, ballpark is 500 × a few µs of Python dispatch overhead per forward = ~few milliseconds total — acceptable against 100ms+ forward passes on 7B+ models in eager mode.

**(b) nnsight benchmarks.** The nnsight v0.6 blog post (nnsight.net, Feb 26 2026, by Jaden Fiotto-Kaufman) reports "2.4–3.9x Faster Traces": 3.9x speedup for empty traces (1,196 µs → 308 µs), 2.9x for 1 `.save()`, and 2.4x for 12 `.save()` calls. The fixed-setup component (source extraction, AST parsing, compilation, thread creation) dropped from ~1,100 µs to ~210 µs; per-intervention cost dropped from ~42 µs to ~34 µs. v0.6's NNsight.md states: "Modules with no active interventions have zero hook overhead beyond the sentinel no-op hook. This eliminates the per-module cost that the old permanent-hook approach incurred on every forward pass." That is direct affirmation of the sentinel approach, paired with reduced reliance on permanent hooks.

**(c) Lighter-weight alternatives.** Two options: (i) a single global hook via `torch.nn.modules.module.register_module_forward_hook()` — fires after every module's forward, single registration; documented caveat: "This adds global state to the nn.module module and it is only intended for debugging/profiling purposes" (acceptable for a debugger); (ii) monkey-patching `nn.Module._call_impl` — most aggressive, brittle across PyTorch versions, not recommended.

**(d) PyTorch version drift.** Pre-2.0 the fast-path involved `_slow_forward` vs. `_call_impl`; since 2.0, `_call_impl` is the unified path and always checks `_forward_hooks` / `_forward_pre_hooks` / `_global_forward_hooks` dicts. The bigger version sensitivity is `torch.compile`: the `torch.compiler_nn_module` documentation states "torch.compile treats common modules such as torch.conv... specially by allowing them to be called opaquely... For such modules, hooks currently trigger a graph-break." Since the design rejects compiled models at attach time, this is moot. Across uncompiled PyTorch 2.0–2.11 the hook dispatch path is stable.

**Verdict:** Sentinel hooks are fine. The design should keep the single `register_module_forward_hook()` global-hook fallback as a config knob if profiling shows per-module sentinel overhead is significant on very deep models. Tuning decision, not a blocker.

### Q6 — Memory implications of pausing mid-forward-pass

**(a) Activation memory at layer 16 of 32 for 7B.** Llama-3 8B / Llama-2 7B have `hidden_size=4096`, `intermediate_size=14336`, GQA `(n_heads=32, n_kv_heads=8)`. Per-token, per-layer activation footprint for the residual stream alone is `4096 × 2 bytes (fp16/bf16) = 8 KB`. Through one layer, transients include the Q/K/V projections (`(4096 + 1024 + 1024) × 2 = 12 KB`), attention output, MLP gate/up (`14336 × 2 = 28 KB each`), and down_proj output. Per-layer transient peak is ~80–120 KB per token at fp16/bf16. Under `torch.no_grad()` / `inference_mode()`, the autograd graph is not built, so most intermediates are freed as the next op consumes them — only the residual stream persists across layers (~8 KB × 16 layers × seq_len ≈ 128 KB × seq_len). At seq_len=2048 that is ~256 MB of activations; at seq_len=8192, ~1 GB. At fp32 these double. For a 7B model paused at layer 16 in inference mode, expect a few hundred MB to ~1 GB of activation memory pinned for typical context lengths.

**(b) `no_grad()` / `inference_mode()`.** `torch.no_grad()` disables gradient tracking; `torch.inference_mode()` is stricter and additionally disables version-counter bumping. Both prevent the autograd graph from being built, which means intermediates are eagerly freed inside the forward function. The memory difference vs. training-mode pause is large (10–100×).

**(c) Python GC during long pauses.** When a Python thread is blocked on `Event.wait()`/`lock.acquire()`, the GIL is released and the interpreter's cyclic GC continues to run on whichever thread holds the GIL. The blocked thread's stack frames are reachable from its thread state, so their locals (including tensor references) are not collected. No Python-level pathology — the tensors just remain alive. The concern is purely GPU memory pressure, not Python GC behavior.

**(d) CUDA TDR / watchdog.** Frequently misunderstood. The CUDA launch timeout (`cudaErrorLaunchTimeout`, code 702 / Windows TDR / Linux X11 watchdog on display GPUs) applies to **a single kernel launch that does not return**, not to "time elapsed between two consecutive kernel launches." A Python pause between layers is just CPU idle time — no kernel is in flight, the GPU is idle, no timeout fires. Per NVIDIA developer forum threads, on Linux **headless** (no X server on the GPU) there is no watchdog; on Linux with X11 on the compute GPU, the `Option "Interactive" "0"` xorg.conf flag disables it; on Windows, default TDR is 2 seconds (configurable via `HKLM\System\CurrentControlSet\Control\GraphicsDrivers` `TdrDelay` DWORD). Since rocket_surgeon is a debugger, users may pause for minutes — safe on headless or dedicated-compute GPUs.

**Verdict:** The design correctly uses inference mode by default. Three caveats to document: (i) memory footprint scales with seq_len × layer-depth-paused-at; (ii) display-bound GPUs (rare for serious debugging) have a TDR for in-flight kernels but not for inter-kernel pauses; (iii) long-paused tensor refs must be released when the user moves on, not retained indefinitely as scrubback history.

### Q7 — Module tree patterns in latest HuggingFace transformers

**(a) Llama family.** From verified `modeling_llama.py` (main branch) and reproduced `print(model)` output:
```
LlamaForCausalLM
├── model: LlamaModel
│   ├── embed_tokens: Embedding(vocab, hidden)
│   ├── layers: ModuleList[N × LlamaDecoderLayer
│   │     ├── self_attn: Llama{Eager,Sdpa,FlashAttention2}Attention
│   │     │     ├── q_proj: Linear(hidden, n_heads * head_dim, bias=False)
│   │     │     ├── k_proj: Linear(hidden, n_kv_heads * head_dim, bias=False)
│   │     │     ├── v_proj: Linear(hidden, n_kv_heads * head_dim, bias=False)
│   │     │     ├── o_proj: Linear(n_heads * head_dim, hidden, bias=False)
│   │     │     └── rotary_emb: LlamaRotaryEmbedding   # location varies by version
│   │     ├── mlp: LlamaMLP
│   │     │     ├── gate_proj: Linear(hidden, intermediate, bias=False)
│   │     │     ├── up_proj:   Linear(hidden, intermediate, bias=False)
│   │     │     ├── down_proj: Linear(intermediate, hidden, bias=False)
│   │     │     └── act_fn: SiLU
│   │     ├── input_layernorm: LlamaRMSNorm
│   │     └── post_attention_layernorm: LlamaRMSNorm]
│   ├── norm: LlamaRMSNorm
│   └── rotary_emb: LlamaRotaryEmbedding   # introduced 4.43+, model-level
└── lm_head: Linear(hidden, vocab)
```
**Llama-3 vs Llama-3.1 vs Llama-3.2 differences:** in transformers 4.43+, `LlamaRotaryEmbedding` was moved out of each `self_attn` to a single `model.rotary_emb`. Older Llama-2 has `rotary_emb` only inside `self_attn`. Llama-3.2 text models use the same `LlamaForCausalLM` class as Llama-3.1; only the config differs (rope scaling, vocab=128256, etc.). Llama-3.2 vision (multimodal) uses `MllamaForConditionalGeneration` — out of scope.

**(b) GPT-2.** Legacy naming:
```
GPT2LMHeadModel
├── transformer: GPT2Model
│   ├── wte: Embedding(vocab, n_embd)
│   ├── wpe: Embedding(n_positions, n_embd)
│   ├── drop: Dropout
│   ├── h: ModuleList[N × GPT2Block
│   │     ├── ln_1: LayerNorm
│   │     ├── attn: GPT2Attention
│   │     │     ├── c_attn: Conv1D(n_embd, 3*n_embd)   # FUSED Q,K,V — split at runtime
│   │     │     ├── c_proj: Conv1D
│   │     │     ├── attn_dropout: Dropout
│   │     │     └── resid_dropout: Dropout
│   │     ├── ln_2: LayerNorm
│   │     └── mlp: GPT2MLP
│   │           ├── c_fc: Conv1D
│   │           ├── c_proj: Conv1D
│   │           ├── act: NewGELUActivation
│   │           └── dropout: Dropout]
│   └── ln_f: LayerNorm
└── lm_head: Linear  (weight-tied to wte)
```
Surprises vs. Llama: `transformer` not `model`; `h` not `layers`; `attn` not `self_attn`; uses `Conv1D` (a custom HF class) not `Linear`; fused `c_attn`; `lm_head.weight` weight-tied to `wte.weight` and **not in `named_parameters`** by default (confirmed in HF issue #6291).

**(c) Mistral.** Identical module naming to Llama — `MistralForCausalLM → model → layers[N] → self_attn{q_proj,k_proj,v_proj,o_proj}, mlp{gate_proj,up_proj,down_proj}, input_layernorm, post_attention_layernorm`. Only class names differ (`MistralAttention`, `MistralRMSNorm`, `MistralMLP`). Sliding-window attention is a runtime detail.

**(d) Phi-3.** Structure mirrors Llama but with two fusions:
```
Phi3ForCausalLM → model: Phi3Model → layers[N] × Phi3DecoderLayer
    ├── self_attn: Phi3Attention
    │     ├── qkv_proj: Linear(hidden, n_heads*hd + 2*n_kv_heads*hd, bias=False)   # FUSED, UNEQUAL split
    │     └── o_proj: Linear(n_heads*hd, hidden, bias=False)
    ├── mlp: Phi3MLP
    │     ├── gate_up_proj: Linear(hidden, 2 * intermediate, bias=False)   # FUSED, EQUAL split
    │     └── down_proj: Linear(intermediate, hidden, bias=False)
    ├── input_layernorm: Phi3RMSNorm
    ├── post_attention_layernorm: Phi3RMSNorm
    └── resid_attn_dropout, resid_mlp_dropout
```
The official Phi-3 docs say it "is very similar to Llama with the main difference of [Phi3SuScaledRotaryEmbedding] and [Phi3YarnScaledRotaryEmbedding]... The query, key and values are fused, and the MLP's up and gate projection layers are also fused." Phi-3.5-MoE (`phimoe`) replaces `mlp` with `block_sparse_moe`.

**(e) Qwen2.** Structurally identical to Llama, with one critical difference: **attention projections have biases**. Per `modeling_qwen2.py` main: `self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)` (and same for k_proj, v_proj). Llama uses `bias=False`. Qwen2-MoE uses `Qwen2MoeSparseMoeBlock` with `experts: ModuleList[Qwen2MoeMLP]` plus `shared_expert` and `shared_expert_gate`.

**Verdict:** The submodule-naming map needs to cover three layouts: (1) Llama-family (Mistral, Qwen2, Gemma) with separated q/k/v/o + gate/up/down; (2) GPT-2/NeoX with fused `c_attn`/`query_key_value` (3 equal chunks); (3) Phi-3 with fused `qkv_proj` (3 **unequal** chunks: `n_heads*hd, n_kv_heads*hd, n_kv_heads*hd`) and `gate_up_proj` (2 equal chunks). Qwen2's `bias=True` on attention projections is a parameter-introspection difference but does not change hook semantics.

### Q8 — GIL behavior with PyO3 auto-initialize + threading

**(a) Cross-thread GIL acquisition.** Per the official PyO3 documentation (`pyo3::Python`): when the Python forward thread blocks on `Event.wait()` or `lock.acquire()`, those wait calls internally drop the GIL (via `PyThread_acquire_lock_timed`). The Rust main thread can then call `Python::with_gil` (or `Python::attach` on 0.26+) which acquires the GIL via `PyGILState_Ensure`. The `prepare_freethreaded_python` doc states: "Prepares the use of Python in a free-threaded context. If the Python interpreter is not already initialized, this function will initialize it with signal handling disabled." Once the Rust thread drops the `Python<'py>` guard, the GIL is released and the forward thread's blocked wait can resume when its lock is released.

**(b) Known issues with `prepare_freethreaded_python` + Python-spawned threads.** PyO3's user guide notes: "Threads created via the Python threading module do not need to [explicitly register]... but all other OS threads that interact with the Python runtime must explicitly attach using with_gil." A `threading.Thread` spawned from Python that runs the forward pass is automatically registered with the runtime; the Rust IPC thread must call `Python::with_gil`/`attach` before touching Python objects.

**(c) Forward thread spawned from Python.** No special handling needed from Rust — Python's `threading.Thread` sets up the thread state automatically. The Rust thread interacts with Python objects produced by that thread via `Py<T>` smart pointers (thread-safe to hold, not to dereference without the GIL).

**(d) GIL acquisition patterns.** `Python::with_gil` (and 0.26's `Python::attach`) is blocking — it waits until it can acquire the GIL; no stable try-acquire variant. If another thread holds the GIL, `with_gil` blocks; this is normal. The C API `PyGILState_Check` is available for a non-blocking probe.

**(e) PyO3 version differences and Python 3.13t.** Significant API churn:
- **PyO3 0.21**: `Bound<T>` / `Py<T>` smart pointers became primary.
- **PyO3 0.22–0.23**: `gil-refs` API removed; `'py` lifetime fully split from input lifetimes; GILPool removed; `GILOnceCell` race issues addressed.
- **PyO3 0.23**: preliminary free-threaded Python 3.13t support added (PR #4588).
- **PyO3 0.26.0** (released 2025-08-29, per CHANGELOG): renamed APIs — PR #5206 `Python::with_gil → Python::attach`; PR #5209 `Python::allow_threads → Python::detach`; PR #5223 `pyo3::prepare_freethreaded_python → Python::initialize`. Release notes: "A number of PyO3 APIs have been renamed to reflect the fact the GIL is no longer a universal feature of all Python implementations."
- **PyO3 0.28**: `#[pymodule]` switched to PEP 489 multi-phase initialization.

**Critical Python 3.13t gotcha:** PyO3 discussion #4738 ("3.13 freethreaded deadlock") documents that spawning a thread from within `Python::with_gil` and `.join()`-ing it within the same with_gil block deadlocks. The accepted resolution: wrap the `.join()` (or any blocking wait) in `py.allow_threads`/`detach`. Verbatim quote from the discussion: *"Although free threading allows multiple threads to be attached to the Python VM at once, there are some times where a single thread requires exclusive access (a stop-the-world pause). Here are a few examples: During garbage collection in order to get a globally consistent view of reference counts..."*

**Verdict:** The PyO3 architecture is sound but needs two revisions: (1) target the new API names (`Python::attach`, `Python::initialize`, `Python::detach`) on PyO3 ≥ 0.26 from day one — they are stable and old names are deprecation-warned; (2) any pattern where the Rust thread spawns a thread and waits on it while holding the GIL must wrap the wait in `Python::detach`, otherwise free-threaded 3.13t deadlocks. If the project does not target 3.13t this is moot but worth annotating.

## Recommendations

**Stage 1 — Before writing implementation tasks:**
1. **Replace `threading.Event` + `queue.Queue` with two `_thread.allocate_lock()`-based single-slot mailboxes per direction**, mirroring `nnsight.intervention.interleaver.Mediator.Value`. Each mailbox must have an explicit clear/restore method invoked after consumption to drop tensor references and avoid the v0.5 nnsight reference-loop bug.
2. **Document the autocast contract**: hook outputs are in autocast dtype; stats are computed in that dtype unless the hook explicitly casts or disables autocast. Add an explicit `compute_stats(tensor, in_fp32: bool = True)` knob in the worker API.
3. **Treat execution-order discovery as per-call**. Cache only when the user pins it. Document gradient-checkpointing double-fires as a known wart even though inference-mode usage avoids it.

**Stage 2 — Implementation tasks:**
4. **Adopt PyO3 ≥ 0.26 API names** (`Python::attach`, `Python::detach`, `Python::initialize`) from the start.
5. **Implement fused-module decomposition for three patterns**: Llama-family (none needed); GPT-2/GPT-NeoX (`c_attn` / `query_key_value` → 3 equal chunks along last dim); Phi-3 (`qkv_proj` → 3 unequal chunks with sizes `(n_heads*hd, n_kv_heads*hd, n_kv_heads*hd)`; `gate_up_proj` → 2 equal chunks).
6. **Per-module sentinel hooks are the default**; keep a config knob to switch to a single global `register_module_forward_hook` if profiling shows >5% overhead.

**Stage 3 — Operational guardrails:**
7. **Document headless-Linux assumption** for long pauses. If a display-bound GPU is detected at startup, warn and refuse pauses > 1 s.
8. **Add a memory-pressure check**: if paused-state activation memory exceeds a threshold (e.g., 4 GB), emit a warning before the next step.

**Benchmarks/thresholds that change recommendations:**
- If end-to-end debugger overhead exceeds 20% of forward time on Llama-3 8B at seq=2048, drop per-module sentinels for the global hook approach.
- If the project ever targets Python 3.13t, every `Python::attach` block that spawns OS threads and joins them must be audited for the #4738 deadlock pattern.
- If a user-reported model uses MoE with the dynamic-routing variability problem, switch discovery from "first forward pass" to "every forward pass."

## Caveats

- The exact body of nnsight's `Mediator.Value.wait`/`put` methods was reconstructed from the autogenerated API reference (which shows `lock = allocate_lock()` as a class attribute default and exposes `get`/`wait`/`put`/`restore` as public methods) plus the v0.6 blog's performance accounting; the verbatim file content could not be fetched directly due to raw-domain restrictions. The structural conclusion (raw `_thread` lock, single-slot mailbox, ping-pong) is robust; the exact line-by-line implementation should be confirmed locally via `inspect.getsource(nnsight.intervention.interleaver.Mediator.Value)` before finalizing code.
- The 3.13t deadlock quote in Q8 is verbatim from PyO3 discussion #4738 but the attribution to Sam Gross specifically was not confirmed against the discussion thread; the text may be from a PyO3 maintainer paraphrasing CPython C API docs.
- Activation memory estimates in Q6 are first-order — exact figures depend on attention implementation (Flash-Attention fuses many intermediates), batch size, and sequence length. Treat ranges as order-of-magnitude.
- PyTorch hook fast-path behavior is stable across 2.0–2.11 in eager mode, but a future release could change `_call_impl`; the `torch.compile` rejection at attach time avoids the more volatile compiled path. Note that `torch._dynamo.config.skip_nnmodule_hook_guards=True` is the default — i.e., hook changes after compile are *not* detected — which reinforces the decision to reject compiled models.
- Llama-3.2 multimodal (vision-language) variants use `MllamaForConditionalGeneration`, not `LlamaForCausalLM`, and have a different module tree (vision encoder + cross-attention blocks); they were not analyzed here per scope.
- The user's stated assumption that "Mixtral stores experts as 3D tensors" was not found in the upstream HF Mixtral implementation, which uses per-module `ModuleList[MixtralBLockSparseTop2MLP]`. The 3D layout exists in fused-kernel implementations (Megablocks, vLLM) but not in the HF eager path the debugger targets.
- The PyO3 0.23 free-threading entry corresponds to PR #4588; precise release date was not verified from primary sources.

## Completion table

| # | Question area | Covered |
|---|---|---|
| 1 | `threading.Event` + `queue.Queue` safety, nnsight primitives | ✓ |
| 2 | Forward hooks under autocast (dtype, reductions, lifetime) | ✓ |
| 3 | Execution-order discovery edge cases (MoE, ckpt, KV, tying) | ✓ |
| 4 | Fused module patterns: Phi-3, GPT-2/NeoX, trend, Mixtral 3D | ✓ |
| 5 | Sentinel hook overhead, nnsight benchmarks, alternatives | ✓ |
| 6 | Mid-pause memory, no_grad/inference_mode, GC, CUDA TDR | ✓ |
| 7 | Module trees: Llama-3.x, GPT-2, Mistral, Phi-3, Qwen2 | ✓ |
| 8 | PyO3 GIL behavior, 0.26 renames, 3.13t deadlock | ✓ |