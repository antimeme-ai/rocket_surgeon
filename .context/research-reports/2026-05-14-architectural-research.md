## rocket_surgeon: Architectural Research Report

**Audience:** rocket_surgeon core team, transition from lit review → written plan.
**Purpose:** Surface missing prior art, resolve the 11 architecture questions, name the biggest risks, critique the proposed three-layer design and tick model, and pin down a defensible MVP.
**Stance:** Opinionated. The goal is to make decisions, not catalog options.

---

## 1. Missed Prior Art (HIGHEST PRIORITY)

Your bibliography is strong on circuits, SAEs, GPU scheduling, and rr. The systematic gaps are: (a) the *inference-engine-native* interpretability stack that exploded in 2024–2026, (b) tracer/graph-rewrite approaches as an alternative to runtime hooks, (c) cloud/SaaS omniscient debuggers, (d) the JAX-side state of the art, and (e) several "adjacent-field" patterns that map directly onto your design problems. The following items are not in the 101-paper / 47-repo list you provided, and each is load-bearing for rocket_surgeon.

### 1.1 Inference-engine-native interpretability (you cannot avoid this anymore)

- **vLLM Hook (IBM, arXiv 2603.06588, "vLLM Hook v0: A Plug-in for Programming Model Internals on vLLM", Ko & Chen)** — a plug-in architecture that exposes attentions, attention heads, and activations of any vLLM-served model for both "passive programming" (capture) and "active programming" (steering). It uses a *config-file-driven* declarative hook spec rather than imperative `register_forward_hook`. This is direct prior art for your probe DSL.
- **vLLM-Lens (UK AISI / `UKGovernmentBEIS/vllm-lens`)** — fast interpretability tooling that scales to tensor-parallel + pipeline-parallel vLLM deployments. Critically, it solves problems you will hit: per-request hooks in a *continuously batched* environment, capture only on TP rank 0 (since residual streams are identical after all-reduce), steering applied on all TP ranks, lossless CPU-side compression of captured activations, and integration with the `Inspect` evals framework. It is the closest existing system to rocket_surgeon's "inspect/steer on a real production-grade multi-GPU model." Study its `WorkerExtension` hook injection pattern.
- **EasySteer (arXiv 2509.25175)** — a vLLM-integrated steering framework that extends vLLM's `forward_context` with *inference-stage markers* and token-level metadata, supports *positional/conditional* steering, concurrent application of multiple steering vectors with user-specified conflict resolution (additive superposition, priority), and reports 5.5–11.4× speedups over hook-based baselines. The forward-context-extension pattern is exactly what you want for "tick metadata" propagation through TP/PP shards.

**Why this matters:** vLLM, not raw HuggingFace `model.generate`, is how most teams will exercise a 70B+ model on a multi-GPU node. If rocket_surgeon does not have a vLLM-compatible execution path, it will not be usable on the very models it most needs to dissect. Your design should at least *accommodate* vLLM as a backend, not just `transformers` eager.

### 1.2 Penzai + Treescope (Google DeepMind, ICML 2024 MI Workshop, Johnson)

You list NNsight, TransformerLens, pyvene, NNterp. You do **not** list **Penzai/Treescope** (arXiv 2408.00211, `google-deepmind/penzai`, `google-deepmind/treescope`). This is a major omission for three reasons:

1. **Models-as-data.** Penzai exposes the forward pass *in the structure of the model object itself*: it is a pytree of declarative combinators, and the pretty-printed model is roundtrippable — pretty-print → edit → rebuild. Treescope visualizes arbitrary-dimensional NDArrays inline, with built-in support for *sharded* arrays across multiple accelerators (it color-codes which device each slice lives on). This is the visualization grammar rocket_surgeon's TUI should crib from. Treescope is also a stand-alone package that already renders PyTorch modules and tensors.
2. **Selector system.** `penzai.core.selectors` generalizes JAX's `.at[…].set(…)` to type-driven pytree traversal, enabling "find every attention layer in the model and replace it with X" in a single, declarative expression. This is a clean inspiration for your probe-point wildcard model `model:layer:component:event`.
3. **Sharding-aware visualization.** Treescope renders sharded `jax.Array` correctly out of the box, which is the exact problem your TUI must solve for DTensor and FSDP-sharded parameters.

Treat Penzai/Treescope as the *visualization and selector layer* reference, even if you stay PyTorch-native.

### 1.3 Goodfire Ember & Neuronpedia — the "what shape should the API take" priors

- **Goodfire Ember API** (goodfire.ai/blog/announcing-goodfire-ember, Dec 2024). The first *hosted* mechanistic interpretability API. SDK semantics — `features`, `steering`, `Auto Steer`, *conditional* feature steering for jailbreak prevention — are a useful target for what an LLM-driven client expects. Llama-3.3-70B and Llama-3.1-8B with open-source SAEs on HuggingFace under `Goodfire/`.
- **Neuronpedia** (neuronpedia.org, Lin, 2024–2026). Open-source platform: probe/latent/feature browsing, circuit tracer, custom-vector steering, OpenAPI-spec endpoints, Python + TypeScript SDKs. It hosts Gemma Scope, OpenAI SAEs, and others. The Neuronpedia OpenAPI spec is a free design reference for what a *machine-consumable* interpretability protocol looks like at the resource-and-method level.

Read both of these specifically as protocol-design priors. They have already paid for the lesson of "what verbs does an LLM need to drive interpretability."

### 1.4 The tracer/graph-rewrite alternative to runtime hooks

You list TransformerLens, NNsight, baukit (runtime hook approaches) and `torch.fx`. You do **not** list these, which are critical for the torch.compile / CUDA-graph problem:

- **NNsight's "intervention graph"** is itself the right abstraction even if you do not use NNsight directly. NNsight (`ndif-team/nnsight`, NNsight 0.6 announcement, Feb 2026) records a deferred computation graph during a `trace` context, then *interleaves* it with the model's normal forward graph at execution time. This sidesteps the "where do I register the hook" question because the intervention is described *symbolically* and applied when the graph is run — whether that run is eager, compiled, or remote. The Envoy system and intervention-graph serialization format are worth studying.
- **`torch._dynamo` custom backends + `torch.fx.Interpreter`** — Dynamo captures Python frames into FX graphs and hands them to a user backend. You can write a backend that wraps every `call_module` / `call_function` node with an instrumentation pre/post hook *inside the compiled graph*, which is the only way to keep hooks working when the user has called `torch.compile(model)`. The `pytorch/pytorch` issue #117758 confirms what you suspected: "hooks registered after compilation is triggered will not run." The fix is graph rewriting, not runtime registration. (See ezyang's "Ways to use torch.compile", blog.ezyang.com/2024/11.)
- **SimpleFSDP (arXiv 2411.00284, Meta)** — re-implements FSDP semantics using `DTensor` + parametrization + selective activation checkpointing so that the whole forward (including comms) is `torch.compile`-traceable as a single graph. This is the right model for how rocket_surgeon should view FSDP: do not fight FSDP's hook machinery, instead lower to DTensor + a graph-rewrite pass.
- **`make_graphed_callables` constraints (NVIDIA cuda-graph docs)** — explicit, authoritative confirmation: "Registering hooks before calling `make_graphed_callables()` is not allowed. Hooks must be registered after graphing." And: "During replay, submodule forward methods are never called — the graph executes as a monolithic sequence of pre-recorded CUDA kernels. Only the top-level module's hooks are invoked; all submodule hooks are skipped." This is a hard constraint, not a workaround target.
- **PyGraph (arXiv 2503.19779, ASPLOS 2025)** — extends torch.compile to make CUDA Graphs robust; specifically discusses static-shape and scalar-parameter handling. Useful to understand what *cannot* be intercepted post-compile.
- **`torch.distributed.fsdp.fully_shard` (FSDP2)** docs explicitly describe FSDP's *internal* use of pre-forward/post-forward and pre-backward/post-backward hooks to do all-gather/reshard. This is why your bibliography note "FSDP uses hooks internally" is right, but the implication is stronger than you have: registering *your own* forward hooks on a `fully_shard`-wrapped module *will* run, but the parameters they see depend on whether you are in the unsharded window. The right point of interception is between FSDP's pre-forward (which all-gathers) and its post-forward (which reshards).

### 1.5 Pernosco — the omniscient-debugger UX reference

rr is in your bibliography; **Pernosco (pernos.co)** is not. Pernosco takes an rr recording, processes it in the cloud, and exposes "instant access to the full details of any program state at any point in time" via dataflow analysis ("click a value, jump to where it was set"). The UX lessons rocket_surgeon should steal:

- **Notebook as primary artifact.** Pernosco's "notebook" is a shareable annotated trace, not a transient debugger session. For rocket_surgeon, this maps directly to "a debugging session = a reproducible JSON-RPC trace + annotations" — itself a perfect MCP resource.
- **Dataflow over stepping.** The killer feature is *not* step-back; it is "for this value, where did it come from?" For transformers, the equivalent is the *causal-graph* view: "this logit was 0.7 because of these residual-stream contributions in these layers." Patchscope and Tuned Lens point at this, but Pernosco's UX is the right organizing metaphor — make dataflow a first-class primitive in your protocol, not just stepping.
- **WinDbg Time Travel Debugging (TTD)** and **PANDA** (panda.re, QEMU-based whole-system record/replay) are additional industrial-strength record/replay systems worth a half-day of study. PANDA's hardware-simulator framing is closer to rocket_surgeon than rr is: rr replays a process, PANDA replays an entire machine — your tick model is closer to PANDA's instruction-level granularity.

### 1.6 Anthropic Garçon (transformer-circuits.pub/2021/garcon)

You don't list Garçon explicitly. Garçon is the *direct ancestor* of TransformerLens (Nanda credits it explicitly) and the reference design for "interpretability infrastructure on production-scale models." Key Garçon design choices to internalize:

- Server-per-model, launched with `python -m garcon.launch --snapshot PATH --name my-garcon`. Multiple notebooks attach to the same server. Loading is amortized across sessions.
- *Time-to-first-experiment* is the metric Garçon optimized for. You should make it yours.
- Architecture-agnostic via autodetection of model config and GPU count.

The "Garçon as a service that researchers attach to" pattern is what rocket_surgeon's daemon model should look like. Combine this with the Pernosco "notebook = session" pattern and you have an opinionated UX.

### 1.7 MAIA — automated interpretability *agent* (arXiv 2404.14394, ICML 2024)

MIT CSAIL's Multimodal Automated Interpretability Agent generates hypotheses, runs experiments, observes results, and refines until it can answer a query. It uses a Python API surface of tools — `synthesize_input`, `compute_max_activating`, `describe_results`. This is the *exact* persona of "LLM consuming rocket_surgeon's machine interface." Your protocol's tool surface should be designed against MAIA-style usage as the primary user, not against a human in a TUI. If MAIA would have a hard time using a verb, redesign the verb.

### 1.8 GPU observability you missed

- **`eunomia.dev`/`GPUprobe` line of work**, in particular *GPU Profiling Under the Hood* (eunomia.dev, Apr 2025), surveys how Nsight, HPCToolkit, TAU, and GPUprobe instrument CUDA. The eBPF angle is covered for AMD via roctracer and NVIDIA via CUPTI; this is more useful than NCCL Inspector alone.
- **NVIDIA Nsight Systems trace format / Perfetto compatibility.** Kineto (PyTorch profiler) emits Chrome trace JSON that Perfetto consumes. You already list Perfetto. The non-obvious win: **use Perfetto's protobuf trace format as your timeline format**. It is open, has a mature schema, has SDKs in Rust and Python, and lets you co-visualize with `nvtx` ranges, CUPTI kernel events, and your own probe events in a tool researchers already know. Do not invent a trace format.
- **NVIDIA Nsight Compute Section files** are a precedent for declarative, replayable kernel-level inspection.

### 1.9 Adjacent-field patterns worth stealing

- **JTAG / in-circuit emulator architecture.** Real ICEs distinguish *halt the device* from *attach inspector*. Halting is a hardware primitive; inspection is software. For rocket_surgeon, the analog is: *halt the forward pass* (insert a barrier kernel + spin) is the primitive; *inspection* is everything you do while halted. This separation gives you two-phase pause for free (see Q3).
- **Database query debuggers — specifically SQL `EXPLAIN ANALYZE` and CockroachDB's "statement bundle".** A statement bundle is a single file containing the SQL, the plan, the statistics, the execution trace, and the env state. A rocket_surgeon "tick bundle" should be similar: a self-contained zipfile of `{model_id, tokens, layer, captured tensors, intervention manifest, replay seed}`. This makes bug reports reproducible by *file*.
- **Distributed-tracing semantic conventions — OpenTelemetry.** Your `model:layer:component:event` schema should be an OTel-compatible *span* schema. Then you get Jaeger, Tempo, and Grafana for free as your timeline visualizer. Probe points become spans, interventions become span events, replay traces become a trace tree.
- **Game-engine hot-reload (Unreal Live Coding, Unity's Hot Reload).** The lesson is that the unit of hot-swap is the *module*, not the *function*: modules are versioned, references are reparented atomically. For interventions like "swap expert 3 for expert 7," design the swap at the module/component granularity with reference-counted lifetimes; do not patch raw tensors in-place.
- **eBPF verifier / static analysis (Gershuni 2019 you have).** What you're missing is the *userspace eBPF VM* — **uBPF** and **rbpf** (Solana's Rust eBPF VM). If you want your probe DSL to be safe and composable with "zero cost when off," compile probes to a tiny bytecode and JIT them with rbpf. This is the dtrace-meets-pyo3 path.
- **`tokio-console`** — a remote-attached observability TUI over a wire protocol (gRPC) for the Tokio runtime. This is exactly your TUI-as-client-of-protocol pattern, already battle-tested in Rust. Architecture and code are worth reading end-to-end.
- **DAP "Reverse Debugging" sub-spec (`stepBack`, `reverseContinue`).** DAP already has these requests; very few adapters implement them. rr's gdbserver implements them. You can extend DAP's reverse-stepping verbs to your tick model rather than inventing new ones.
- **LSP semantic tokens / inlay hints.** If you want the TUI to show "this attention head fires on punctuation," that is the inlay-hints pattern. Worth borrowing the contract.

### 1.10 MoE-specific work you don't have

- **MixtureKit (arXiv 2512.12121, MBZUAI)** — a library specifically for composing, training, *and visualizing* MoE models. Includes a token-routing visualization that distinguishes "high-level" (token colored by dominant expert) and "low-level" (per-expert percentages). This is the reference for your MoE TUI panel.
- **Latent Prototype Routing (arXiv 2506.21328)** — reframes expert routing as clustering. Reports Gini-coefficient improvements on DeepSeek-V3, Qwen3-MoE, Mixtral. The clustering frame is the right way to *summarize* router state for a TUI: don't show 256 expert probabilities, show K cluster centroids and assignment.
- **`moe-viz.martinalderson.com` / Alderson's MoE visualizer** — a working open implementation on `llama.cpp` showing per-token per-layer expert firing. Useful as a UI prior even though llama.cpp is C, not PyTorch.
- For DeepSeek-V3 and Mixtral specifically, the HuggingFace `output_router_logits=True` flag on `MixtralForCausalLM` returns *raw* router pre-softmax logits — your probe system should expose this as a built-in probe rather than rebuilding routing instrumentation.

### 1.11 Determinism: the 2025 state of the art

The "GPU Determinism Survey 2024" you have is fine, but the practical landscape has moved. The single most useful 2025 piece is **Thinking Machines / Horace He's "Defeating Nondeterminism in LLM Inference"** (thinkingmachines.ai blog, Sep 2025) which identifies that the dominant source of LLM-inference non-determinism is **non-batch-invariant reductions** in matmul/attention kernels (the output of an operation depends on the batch size because of varying split-K choices, not because of atomic reductions as commonly assumed). They demonstrate batch-invariant kernels for matmul, RMSNorm, and attention. For rocket_surgeon this is decisive: bit-exact replay is achievable for a fixed micro-batch shape if you use batch-invariant kernels and pin per-rank reduction order. Add this to your bibliography.

### 1.12 PyTorch-internals primary sources you do not cite

- **`torch._dynamo` design doc** in the pytorch repo and Ansel et al.'s "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation" (ASPLOS 2024, expanded internal docs). You list "Ansel 2024 PyTorch 2" — make sure you have the *guards*, *graph break*, and *frame-evaluation* internals, not just the paper.
- **`torch._functorch.aot_autograd`** README — the forward/backward graph split is essential for understanding what hooks survive `torch.compile`.
- **`torch._inductor`** — relevant for "can I inject between fused kernels."
- **DTensor design doc (`torch/distributed/_tensor/README.md`)** — DTensor is the abstraction that makes TP+PP+FSDP composable. rocket_surgeon's tensor inspector should be DTensor-aware: a DTensor argument has a `placements` attribute that tells you the sharding spec; gathering is a one-liner `tensor.full_tensor()`.
- **`torch.distributed.pipelining`** (the new pipeline API, replacing PiPPy) — the correct abstraction for pipeline-parallel tick boundaries.

### 1.13 Checkpointing-specific work you missed

- **DeepSpeed ZeRO-Inference / ZeRO-Offload** and **FlexGen (Sheng et al., ICML 2023)** — CPU/NVMe-offload tiered storage of model state during inference. The tiering pattern (GPU → pinned CPU → mmap'd disk) is exactly what you need for "hierarchical checkpointing" in Q4.
- **NVIDIA `cuda-checkpoint`** (you have it in repos) is good for whole-process. The complement is **CRIU `criu-ns`** for namespaced checkpoint and **GPUSnap / Singularity-CRIU** for container-level. You probably won't need them, but for full multi-tenant scenarios they exist.
- **TorchSnapshot** — async, sharded checkpointing API for PyTorch. Worth cribbing the chunked-tensor serialization scheme.
- **Safetensors mmap** — you have it as a repo. The non-obvious use is for *write*: you can mmap a sparse file, write activation snapshots as new safetensors blobs, and stream them to disk without ever touching Python heap.

### 1.14 Reverse-debugger architecture references

- **GDB Reverse Execution** (gnu.org gdb manual, "process record and replay") — the original instruction-record approach. Heavier than rr, but the API patterns (`record`, `reverse-step`, `reverse-continue`, `bookmark`) are the right verbs to copy into your DAP extension namespace.
- **`undo.io` LiveRecorder** — commercial Linux reversible debugger, integrates into CI. Their public docs on "snapshot every N instructions, replay deterministically forward" describe the same square-root checkpointing tradeoff you're facing, with real numbers.
- **OCaml's `time-rex`** and **Java's Chronon** — JVM time-travel debuggers showing that managed-runtime instrumentation can be cheap if done at the JIT level. Encourages you toward Dynamo-backend instrumentation over Python-level hooks.

### 1.15 The "missing repos" short list (cite verbatim)

Add to your repos-analyzed list, in priority order:
1. `google-deepmind/penzai` and `google-deepmind/treescope`
2. `UKGovernmentBEIS/vllm-lens`
3. `IBM/vLLM-Hook`
4. `ndif-team/nnsight` (you have it, but specifically re-read the intervention-graph and Envoy modules and NNsight.md)
5. `pernos.co` docs + the rr Pernosco-fork
6. `tokio-rs/console` (TUI-as-protocol-client reference impl in Rust)
7. `goodfire-ai/*` (`param-decomp`, `causalab`)
8. `multimodal-interpretability/maia` (MAIA agent code)
9. `martinalderson/moe-viz` (frontend MoE viz reference)
10. `MBZUAI-Paris/MixtureKit` (MoE composition + viz)
11. `EleutherAI/gpt-neox` and `NVIDIA/Megatron-LM` — for *how* their TP/PP boundaries are drawn (so you know where to put your tick points)
12. `huggingface/text-generation-inference` — server-side hook patterns at production scale
13. `linkedin/Liger-Kernel` — fused Triton kernels for HF models; useful for understanding what's been fused (so you know what's been hidden)
14. `state-spaces/mamba` (out of scope but worth knowing the SSM hook story)
15. `qubvel-org/segmentation_models.pytorch` — not relevant; ignore. (Listed only to acknowledge non-relevant adjacents.)

## 2. The 11 Hard Questions: Resolutions

Each subsection states the problem in one paragraph, gives the recommended decision in bold, then justifies it with concrete reference to systems, papers, and tradeoffs.

### Q1. The Hook Registration Problem

**Recommendation: Adopt a *layered* interception stack with three tiers, selected per-model at attach time. Do not rely on a single mechanism.**

- **Tier A — eager mode (default for MVP): `nn.Module.register_forward_pre_hook` / `register_forward_hook` with `prepend=True`, registered AFTER distributed wrapping (`DistributedDataParallel`, `fully_shard`) but BEFORE any `torch.compile` call.** This is the configuration that "just works": DDP's known issue is that hooks registered *before* `DDP(model)` are not seen for the inner module on rank > 0 because DDP replicates references; registering on `model.module` (the unwrapped inner) after DDP-wrapping resolves it. For FSDP2 (`fully_shard`), hooks on submodules fire inside the unsharded window (between pre-forward all-gather and post-forward reshard), which is precisely the inspection window you want.
- **Tier B — compiled mode: do *not* try to register hooks at all. Inject instrumentation as a `torch._dynamo` *custom backend* that wraps every node in the captured FX graph.** The pytorch/pytorch issue #117758 confirms hooks registered after compilation are silently dropped, and even pre-registered hooks are inlined into the compiled graph and never re-fire on the unchanged module. The correct approach is FX-graph rewriting: register a backend with `torch._dynamo.optimize(my_backend)` (or via `torch.compile(model, backend="rocket_surgeon")`), and in your backend transform the `fx.GraphModule` by inserting `call_module` nodes that invoke your probe dispatcher between every original node you want to observe. See `torch.fx.Interpreter` (pytorch/torch/fx/README.md) and ezyang's "Ways to use torch.compile" for the pattern. For users who do not call `torch.compile`, you never enter Tier B.
- **Tier C — CUDA-graphed / `make_graphed_callables` mode: do not attempt to intercept inside the graph; intercept *between graphs*.** Per NVIDIA docs: "Submodule forward methods are never called — only the top-level module's hooks are invoked." This is a hard CUDA constraint. The right design is to refuse single-tick granularity inside a captured graph and instead expose graph-level tick boundaries — i.e., the user gets layer-of-graph granularity, not component granularity. If they want finer granularity, they recapture with smaller graph regions. Document this explicitly; do not pretend.

Concrete advice:

- **Force `fullgraph=False` (the default).** This causes Dynamo to insert graph breaks at unsupported operations; your hook-based instrumentation works in the eager regions between breaks. This is how NNsight, vllm-lens, and TransformerLens survive `torch.compile`-using models in practice.
- **Detect the regime at attach time.** Walk the module tree. If you find `OptimizedModule` (the `torch.compile` wrapper class), `FullyShardedDataParallel`, `DistributedDataParallel`, or `make_graphed_callables` artifacts, switch tiers. Surface the chosen tier in the protocol's `capabilities` payload from `initialize`. Tell the LLM client explicitly that "this model uses CUDA graphs; minimum tick granularity is `layer`, not `component`."
- **Use `prepend=True` everywhere** so your hooks run before any user hooks (analyses, gradient compression, etc.).
- **For PyTorch's "fast path" when no hooks are registered** (e.g. `nn.MultiheadAttention`'s fused kernel) — register a sentinel hook (a no-op) on every module of interest at attach time, *before* the first forward. The fast path is keyed on `self._forward_hooks` being empty; one registration disables it. Document the perf cost (one Python dispatch per module per token); for inference of typical sizes the overhead is ~1–3%.
- **NNsight's intervention-graph model is the right escape hatch for "register hooks at all, ever" being unviable**: build a deferred symbolic graph, then *interleave* with the live forward graph at execution time. This is the same as Tier B in spirit. Implementing intervention-graph semantics on top of your tick model gives you a uniform programming model that maps to all three tiers underneath. **Strong recommendation: adopt the intervention-graph abstraction in your Rust state machine, regardless of which Tier executes it.**

### Q2. FlashAttention and Fused Kernels

**Recommendation: Adopt a "shadow execution" mode. By default run with the user's optimized kernels and only what FlashAttention naturally exposes (LSE, output). When the user requests attention-matrix inspection on a specific layer/head, *selectively unfuse only that layer* by swapping its attention implementation for the reference SDPA path for that single forward pass, replayed from the nearest checkpoint.**

Rationale: FlashAttention-2/3 fundamentally does not materialize the N×N attention matrix in HBM — that is the entire point of the algorithm. Dao 2022/2023/2024 are clear that the matrix exists only in SRAM tiles and is recomputed in the backward pass. Two viable strategies:

1. **LSE-only fast path.** FlashAttention stores log-sum-exp values; with LSE plus stored Q, K, V you can reconstruct any row of the softmax matrix on demand. For "show me the attention pattern for token 17," this is enough and cheap. Expose this as the default `attention` probe.
2. **Reference-path unfusing for a single layer.** PyTorch's `torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)` context manager forces the math (reference) backend, which materializes the full matrix. Use it scoped narrowly. For HF Transformers models, set `attn_implementation="eager"` on the specific layer (HF supports per-layer attention implementations as of `transformers` 4.40+). For Llama/Mistral/Mixtral the relevant module is `LlamaAttention.forward`; you can monkey-patch only that layer's `_attn_implementation` attribute.

Concretely, the user-facing protocol primitive is:

```
inspect attention layer=12 head=*  # uses LSE reconstruction, full row only on request
inspect attention layer=12 head=3 full_matrix=True  # triggers shadow replay
```

The `full_matrix=True` path: from the nearest checkpoint, replay forward to layer 12 with `attn_implementation="eager"` for layer 12 only (other layers stay FlashAttention). Cost: one extra layer's worth of recomputation. The same pattern applies to Liger fused RMSNorm+linear, fused SwiGLU, etc.

Do not run a parallel unfused forward by default — that is 30–50% throughput overhead always-on. The selective-shadow-replay pattern means you pay the unfusing cost only when an inspection requests it, and the cost is bounded.

For **FlexAttention** (which you cite, 2024) the story is better: FlexAttention has an `eager` mode that runs everything unfused, designed exactly for inspection. If the user has a FlexAttention model, exploit that.

### Q3. Multi-GPU Tick Synchronization

**Recommendation: A two-phase pause is mandatory. Implement it as a *barrier-gated forward* with explicit pre-collective and post-collective inspection windows. The barrier is a host-side rendezvous; the collective is never interrupted.**

NCCL is opaque and cannot be paused mid-collective; a torn collective deadlocks all ranks. NVIDIA's own `make_graphed_callables` documentation flags this and recommends releasing CUDA graphs *before* destroying the process group to avoid the hang in `destroy_process_group`. Treat NCCL collectives as atomic.

Concrete protocol:

1. **Barrier objects, not pauses.** A "tick" is a *barrier* injected into the model. Every rank reaches the barrier independently; the barrier blocks on a host-side `threading.Event` shared via the Rust core's IPC. The debugger holds the event; when the user steps, the event is set.
2. **Two windows per collective.** Each transformer block typically contains: (compute) → (all-reduce, all-gather, or reduce-scatter for TP/SP) → (compute). Insert barriers *before* the collective and *after* the collective. The pre-collective window lets you inspect *sharded* activations on each rank. The post-collective window lets you inspect the *gathered/reduced* tensor. Between the windows, the collective runs atomically.
3. **Collective inspection.** During the pre-collective window, expose each rank's local shard as a DTensor with its `placements` annotation; the protocol can offer a "gather to rank 0 for inspection" command that issues a *separate* all-gather (since the user's collective hasn't run yet, this does not interfere with computation ordering — but you must serialize: only one collective in flight per process group at a time).
4. **Watchdog.** NCCL has a watchdog (`TORCH_NCCL_BLOCKING_WAIT`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`). When the user pauses indefinitely between barriers, you will trip the watchdog. Set `TORCH_NCCL_BLOCKING_WAIT=0` and a *very* high `NCCL_TIMEOUT` (hours) for debug sessions. Document this. Detect it automatically and refuse to attach if the env is misconfigured.
5. **Pipeline parallelism is different.** With pipeline parallel, each rank has different layers, not different shards. Use `torch.distributed.pipelining`'s schedule machinery; a "tick" advances one micro-batch through one pipeline stage on the relevant rank. Barriers are per-stage, not per-rank.
6. **Heartbeat over JSON-RPC.** While paused, the daemon sends a `tick.heartbeat` notification every 1s with each rank's status, so the client (and the LLM driving it) does not believe the system is dead.

The right mental model is **JTAG halt-and-attach for a multi-GPU node**. Rust core is the JTAG controller; barriers are halt instructions; inspection happens between halts. The model code itself is unmodified.

A useful adjacent reference is `Demystifying NCCL` (2025, in your bibliography). The 2021 `Monitoring Collective Communication Among GPUs` paper you have describes an external observer; that is the right shape for *passive* observation but not for *active* pause-and-inspect. NCCL Inspector can run alongside and feed the timeline; do not put it on the critical path.

### Q4. Checkpoint Size vs Replay Speed Tradeoff

**Recommendation: A three-tier hierarchical scheme. Tier 1 = *event log* of probe captures (always-on, ~MBs). Tier 2 = sqrt(n) *activation checkpoints* of residual streams at √L layer boundaries (~hundreds of MB for 7B, GBs for 70B+). Tier 3 = *full GPU memory snapshot* on demand only, async, via `cuda-checkpoint` or `PhoenixOS`-style COW.**

This matches what `Chen 2016` (sublinear memory) does for backward training, applied to debugging. The decisive insight: for interactive debugging you do not need to support arbitrary reverse-step. You need to support **(a) re-run forward from a recent point with a new intervention** and **(b) jump back to any captured probe point** for re-inspection. (a) and (b) have very different cost profiles.

Concrete numbers (for sizing):
- 7B Llama-2, fp16, single GPU: 14 GB weights, 32 layers, residual stream `[batch=1, seqlen=2048, 4096]` ≈ 17 MB per layer in fp16. Checkpointing every layer = 32 × 17 MB = ~540 MB activations. Sqrt-n (every √32 ≈ 6 layers, 6 total checkpoints) = ~100 MB.
- 70B Llama-3.3 across 4 H100s: TP=4, per-rank residual ≈ 4 MB at the same shape, 80 layers, ~10 sqrt-n checkpoints, ~40 MB per rank, ~160 MB cluster-wide.
- 405B Llama-3.1 across 8 GPUs with TP=8: per-rank residual ≈ 8 MB at the same shape, 126 layers, ~12 sqrt-n checkpoints. Easy.

The full GPU snapshot path (which the question flags as "77s for 54GB, too slow") is *not* what you want for interactive use. Use it only for "save the entire debug session for later." PhoenixOS-style COW (185 ms overhead, per the paper you list) is excellent **if and only if** you can guarantee no in-place writes happen to the snapshotted regions during the COW window — which inference generally satisfies, but training does not. Adopt PhoenixOS for tier 3 but with a documented caveat.

For the protocol:

- Auto-checkpoint at every transformer block boundary by default (your existing instinct).
- Snapshot retention policy: keep last `K=8` full sqrt-n checkpoints (rolling), plus user-named bookmarks (Pernosco notebook pattern), pinned forever until explicit free.
- All checkpoints are CPU-pinned-memory by default with NVMe spillover (FlexGen-style tiering) when CPU is exhausted.
- Use **safetensors mmap** as your on-disk format. It is exactly designed for this: `safe_open(file, framework="pt", device="cuda")` is a zero-copy view.

The forward-replay-on-demand path is what handles "step back one component." From the nearest sqrt-n checkpoint, replay forward up to the requested point with all probes that were enabled at original-record-time re-firing. Replay cost is bounded by sqrt(L) layers. On A100 a 7B forward layer is ~1 ms; sqrt(32)=6 layers is ~6 ms per reverse step. Acceptable.

Critically: replay non-determinism is the failure mode (see Q6). Mitigate by recording inputs and RNG state at each sqrt-n checkpoint, replaying with `torch.use_deterministic_algorithms(True, warn_only=True)`, and accepting "ULP-close but not bit-identical" as the success criterion (cosine similarity > 0.9999 to original, gate replay verification on a configurable epsilon).

### Q5. MoE-Specific Design

**Recommendation: The MoE "tick" granularity has *four* levels, not three. Build the schema to expose all four from day one, even if MoE shipping is later in your phase plan.**

The four granularities:
1. **Router tick.** Pause *after the router emits logits, before top-k selection.* Inspect routing logits and route-attention statistics; the user can override before top-k selection runs. This is where the 2026 "Geometric Routing Causal Expert Control" intervention lives.
2. **Routing-decision tick.** Pause *after top-k selection, before dispatch.* Now the per-token expert assignment is concrete (a sparse `[batch*seq, top_k]` tensor of expert IDs). The user can force-reassign tokens to specific experts. This is the natural step granularity for most users.
3. **Per-expert tick.** Pause *inside a specific expert, post-dispatch.* Each expert runs on its tokens-bucket; the user can inspect that expert's input/output as a normal FFN.
4. **Layer tick.** Pause *after combine (post-expert all-reduce/all-to-all)*. The MoE layer is now a normal block boundary.

This maps cleanly to the 2026 polysemantic-experts-monosemantic-paths finding: probe expansion lives at granularities 1 and 2 (paths), and the polysemanticity-aware view of expert internals lives at granularity 3.

Concrete protocol primitives:

```
probe router model=mixtral layer=* event=pre_topk
probe routing_decision model=mixtral layer=12 event=post_topk
intervene routing_decision layer=12 token=4 force_expert=[3,7]
inspect expert_input layer=12 expert=3
```

Key MoE-specific protocol additions:
- **Expert capacity overflow indication.** When tokens exceed expert capacity factor, some are dropped; this is the silent failure mode of MoE. Always surface dropped-token count as a first-class field in the routing-decision tick's response.
- **Routing entropy summary** per token, per layer. Surface as a built-in interpretability view.
- **Cluster-projection view** (Latent Prototype Routing pattern). Don't show 256 expert probabilities; show K cluster centroids. Make K a parameter; default 8.
- **HF's `output_router_logits=True`** flag for Mixtral and DeepSeek must be set in the model config at attach time. Otherwise routing logits are not propagated.
- For DeepSeek-V3 specifically, the routing is *shared+routed* (one shared expert always fires, top-k routed). Protocol must distinguish "shared expert contribution" from "routed expert contribution" in inspection responses.

MixtureKit's two-level visualization (per-token dominant expert; per-layer per-expert percentage) is the right TUI default panel for MoE.

**Default component-tick for MoE = routing-decision tick (level 2).** It is the granularity at which the most interesting interventions happen and the least frequent decision per token.

### Q6. Determinism Requirements

**Recommendation: Pursue "close-enough determinism" via three layers, not bit-exact. Accept ULP-close replay as success.**

The bit-exact target is a trap: `torch.use_deterministic_algorithms(True)` disables ops you need (some `scatter_add`, certain `index_*`), FlashAttention's deterministic mode adds ~25% overhead, MoE `scatter_reduce` is inherently non-deterministic on GPU under permutation, and NCCL all-reduce reduction order can vary. Multi-GPU timing perturbation makes this worse.

The three layers:

1. **Seed and env capture at session start.** Record `torch.initial_seed()`, `torch.cuda.initial_seed()`, NumPy seed, `os.environ` slice (CUDA, NCCL, PYTHONHASHSEED, etc.), driver version, NCCL version, compute capability, and the *exact* set of model files (safetensors content hash). Refuse to replay across mismatches without an explicit `--unsafe-replay` flag.
2. **Op-level pinning where it matters.** Set `torch.use_deterministic_algorithms(True, warn_only=True)`. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Force FlashAttention `deterministic=True` during replay only. For MoE, pin the routing decision (re-feed the recorded top-k indices via your tick-2 interception rather than re-deriving them).
3. **Batch-invariant kernels (Thinking Machines, 2025) for the matmul/RMSNorm/attention path** when running in debug mode. This is the single largest practical win for "same input, same output across runs at different batch sizes." Adopt their batch-invariant matmul/RMSNorm/SDPA kernels (or vendor equivalents as they appear) in your shadow-replay path.

Tolerance: cosine similarity > 0.99995 between original tensor and replayed tensor at the same probe point counts as "matches." Maximum elementwise relative error of 1e-3 in fp16, 1e-5 in fp32. Log every mismatch above tolerance; do not fail. This is what `rr`'s "best-effort divergence detection" does and what every real-world TTD has settled on.

For multi-GPU, fix the **per-rank reduction order** in NCCL by setting `NCCL_ALGO=Ring` and `NCCL_PROTO=Simple` (slower, but eliminates one source of non-determinism). Use `NCCL_DEBUG_SUBSYS=INIT,COLL` to verify the chosen algorithm is stable across runs.

**Bottom line:** stop trying to make replay bit-exact. Make replay "indistinguishable for interpretability purposes" and verify it with a checksum at each tick.

### Q7. The Rust/Python Boundary

**Recommendation: Three-process architecture, *not* two. Rust core daemon (process A) ↔ Python "model host" (process B, one per model) ↔ Rust TUI (process C). Tensors live in Python until explicitly snapshotted to a Rust-owned shared-memory ring buffer. Rust never touches `Tensor` C++ objects directly.**

Why three processes:
- **Isolation.** A model crash should not kill the daemon. Python process is the riskiest component (GPU OOM, NCCL hang, segfault from cuDNN).
- **Multiple models per daemon.** The Garçon pattern: one daemon, many model hosts. The Pernosco pattern: one daemon, many sessions. You want this.
- **TUI client lifecycle independent of model state.** TUIs come and go; models stay loaded.

Boundary semantics:
- **Control plane: JSON-RPC over Unix domain sockets between Rust↔Python.** Latency ~50 μs; fine for interactive use. Use `tokio` on Rust side and `asyncio` on Python side. Schema is the same JSON-RPC schema the external clients see; the internal Rust↔Python channel is just one transport.
- **Data plane: shared-memory ring buffer (mmap'd file under `/dev/shm`) for tensor handoff.** Python writes tensor bytes directly into the shared region via `torch.Tensor.numpy().view(np.uint8)` or via safetensors serialization; Rust reads the region. **PyO3 should not own the tensor object lifetime — Python keeps the `torch.Tensor` alive; Rust gets a typed `&[u8]` view + metadata.** The shared-mem layout is a fixed-size record header (rank, layer, component, dtype, shape, offset) plus the bytes.
- **GIL management.** Long-running Python operations (model loading, forward passes) must release the GIL so Rust can keep the protocol loop running. Use `Python::allow_threads` for any tensor-copy operation that does not touch Python objects. Per PyO3 docs (`pyo3::marker::Python::detach` / `allow_threads`), this is well-understood. Most of your Python code's tensor work goes through PyTorch which itself releases the GIL during CUDA kernel launches, so the practical hot path is fine; the issue is your *own* probe-callback Python code, which must explicitly release the GIL during any synchronous wait.
- **Why not "Rust as PyO3 extension module in the Python process"?** You proposed this. It is wrong for production because: (a) Python GIL becomes a global serializer for the protocol loop and TUI traffic; (b) a Python crash takes Rust with it; (c) iteration-on-Rust requires rebuilding the wheel and restarting Python. PyO3-extension is fine for an MVP-only convenience build, but the *architecture* should be a separate Rust daemon.

So: **Rust is a daemon process. Python model host is a child process started by the daemon. PyO3 lives in the Python process as a thin embedded core that runs the probe-callback hot path with the GIL released.** TUI is a third process.

Concrete tensor-sharing recipe:
- Define a `ProbeFrame` shared-memory record: `{u32 rank, u32 layer_idx, u8 component_id, u8 dtype, u16 ndim, u32 shape[8], u64 offset, u64 size}` + the raw bytes.
- Python probe callback: capture the tensor → `tensor.detach().contiguous().to('cpu', non_blocking=True)` → memcpy into ring slot → publish slot index over a 64-bit `eventfd` to Rust.
- Rust reads the slot, builds a `TensorRef` (just metadata + slice), feeds the TUI / protocol clients zero-copy.

For the TUI specifically: send only *summaries* by default (shape, dtype, min/max/mean/std/histogram, top-k absolute values with indices). Send full bytes only when the user explicitly inspects a slice — and even then, send only the requested slice. This is the right answer for the 405B-model scaling concern (Q10).

### Q8. Protocol Design — DAP Extension vs Custom

**Recommendation: A hybrid, with three layers in the same JSON-RPC 2.0 wire format. (1) DAP-shaped *lifecycle and stepping* verbs (`initialize`, `launch`, `attach`, `disconnect`, `continue`, `next`, `stepIn`, `stepOut`, `pause`, `stepBack`, `reverseContinue`, `setBreakpoints`, `threads`, `stackTrace`, `scopes`, `variables`, `evaluate`). (2) A custom `rocket/*` namespace for *domain* verbs (`rocket/probe.define`, `rocket/probe.list`, `rocket/intervention.set`, `rocket/checkpoint.create`, `rocket/checkpoint.restore`, `rocket/replay`, `rocket/tensor.slice`, `rocket/router.override`, `rocket/sae.activate`). (3) An MCP wrapper that exposes the same verbs as MCP `tools` for LLM clients, with MCP `resources` exposing tensors, probe definitions, and checkpoints.**

Why this is the right call:
- **DAP for lifecycle** because the IDE ecosystem (VS Code, JetBrains, Eclipse via LSP4IJ, Nova) gets a working "attach to a running session" for free. DAP's `stepBack` and `reverseContinue` are real spec, supported by rr and Undo. Implementing them gives you free reverse-stepping UI in VS Code with zero per-IDE work. This is a >10x leverage feature.
- **Custom `rocket/*` namespace for domain ops** because DAP's `Variable`/`Scope`/`Source` types are wrong shape for tensors-layers-heads-experts-features. Trying to flatten everything into DAP's variable tree would lose the semantic structure. The `evaluate` DAP request can be the escape hatch for arbitrary expressions, but normal operation goes through structured RPCs.
- **MCP exposure for LLM-native use** because that is your declared primary user. But — and this is critical — MCP's streaming model is genuinely awkward (the "Streamable HTTP" replacement of HTTP+SSE requires HTTP/2 for true bidirectional streaming, which is not always documented; see modelcontextprotocol#598). So **do not make MCP the only transport.** Implement the protocol as JSON-RPC 2.0 over stdio (DAP-compatible), TCP, and WebSocket, and ship an *MCP server adapter* that wraps the JSON-RPC service. The adapter exposes:
  - `rocket_surgeon.step`, `rocket_surgeon.inspect`, `rocket_surgeon.intervene`, etc. as MCP tools (single-turn).
  - Pending-state and tensor data as MCP resources (resource URIs like `rocket://session/{id}/checkpoint/{cid}/tensor/{path}`).
  - A long-lived MCP "subscription" for tick events via MCP's notification mechanism, fallback to polling if the transport does not support streaming.
- **Capability negotiation at `initialize`** — keep this; DAP already specifies it.
- **State in every response** — this is correct and is consistent with DAP's pattern of returning `stopped` state in event payloads. Make the per-response state envelope `{status: stopped|running|terminated, position: {model_id, rank?, layer, component, event, tick_id}, last_event: ...}` and use the same envelope shape across all `rocket/*` responses.

This avoids both extremes: you get IDE compatibility without contorting your domain model into DAP variables, and you get LLM compatibility without depending on MCP's evolving streaming story.

Reference design touchstone: **`tokio-rs/console`'s wire protocol** is a gRPC service that exposes runtime state to a TUI client. The pattern of (a) protobuf-defined types + (b) bidirectional streaming via a single long-lived RPC + (c) a CLI client that's *just* a renderer of subscription events, is exactly what you want. Consider whether gRPC over HTTP/2 is more honest than "JSON-RPC over WebSocket" for your bidirectional case — gRPC gives you bidirectional streaming natively, has Rust and Python SDKs, and `tonic` (Rust gRPC) is mature. The MCP layer wraps gRPC to JSON.

**Strong opinion: JSON-RPC 2.0 is fine for the spec but gRPC is the better implementation.** Spec the protocol in a protobuf `.proto` *and* a JSON-Schema; transports include stdio framed JSON-RPC, TCP gRPC, WebSocket JSON-RPC, and an MCP adapter.

### Q9. Observability Stack — What Goes Where

**Recommendation: Two-tier observability. Tier 1 = your probe events, your JSON-RPC notifications, your trace. Tier 2 = "deep diagnostics" mode that turns on CUPTI, eBPF uprobes, NVML, NCCL Inspector. Always emit a Perfetto-compatible trace.**

Pinning down the layout:
- **PyTorch hooks / FX-graph instrumentation:** produces `probe.fired` events. Always on (zero-cost when no probes are defined).
- **CUPTI Activity API:** produces kernel-launch and -end events. Off by default. Toggle with `rocket/diag.cupti.enable`. When on, every CUDA kernel between two probe events appears as a span in the timeline.
- **eBPF uprobes on `libcuda.so` / `libnccl.so`:** off by default. Useful for "why did this collective stall." Toggle with `rocket/diag.ebpf.enable`. Use `bcc` or `bpftrace` for the prototype; consider compiling a CO-RE BPF program with `libbpf-bootstrap` for the shipped version. **Run only when explicitly enabled** — eBPF on libcuda symbols at full event rate is expensive and can perturb timing.
- **NVML:** poll once per second always-on, into a separate ring buffer. Provides per-GPU memory, power, utilization, ECC errors. Cost is negligible.
- **NCCL Inspector / `NCCL_DEBUG=INFO`** parsed via a side process: on by default but only at INFO; opt-in for TRACE. Feeds into the same trace timeline.
- **Unified timeline:** **emit Perfetto protobuf traces** as the canonical format. Perfetto SDK has C++/Rust bindings (`perfetto-sdk-rs`). All events — probe fires, CUPTI spans, NCCL ranges, NVML samples — go into one trace file. You get the Perfetto UI as your timeline viewer for free.
- The TUI shows a *summarized* view of the timeline; the full trace is exportable.

Do **not** invent a trace format. The eunomia.dev survey shows that PyTorch Kineto already converts to Chrome trace JSON; align with that. Perfetto consumes both Chrome JSON and protobuf, but protobuf is much smaller for long sessions.

### Q10. Scaling Concerns (1B → 405B+)

**Recommendation: Make *every* inspection lazy and bounded by default. Tensors are not values; they are *handles* with metadata. Materialization is an explicit verb.**

Design rules:
1. **Tensors flow as handles, not bytes.** Every `inspect` response includes `tensor_id`, shape, dtype, sharding (DTensor placements), summary statistics, but **not** the raw bytes. Bytes are fetched via a separate `rocket/tensor.slice` call with explicit slice indices. Default response size cap: 64 KB. LLM clients are quickly capped from accidentally consuming a 10GB tensor.
2. **Built-in hierarchical summaries.** For any tensor: mean/std/min/max/abs-max/histogram(32 bins)/top-k(by-abs)/sparsity. These are always cheap to compute (single reduction). The TUI and the LLM client mostly want these, not raw values.
3. **Adaptive checkpoint granularity.** sqrt(L) is right for 32-layer models; for 405B (126 layers) the natural cadence is every 12 layers, ~10 checkpoints. For 8B Llama (32 layers), every 6 layers. Make it a function of `L` and available memory, not a fixed knob.
4. **Streaming tensor inspection.** For the TUI rendering "the residual stream at layer 17," stream the requested slice (e.g. token 0..32, dim 0..64) chunk-by-chunk; never deliver the full `[1, 2048, 16384]`.
5. **Pipeline-parallel scale-out.** For pipeline parallel with stages on different ranks, the daemon must be aware of which rank owns which layer. Issue inspection RPCs to the *owning rank only*; do not all-gather.
6. **Attention pattern recomputation for 128K context.** Do not materialize the full `[H, 128K, 128K]` matrix; never. Always operate on sparse slices: "row T for tokens t1..t2." Materialization should refuse with an error message for matrices > 1 GB and require an explicit `--allow-large` flag.
7. **Multi-tensor probe responses use a "summary, then materialize" two-phase protocol.** The first response gives all probe metadata + summaries in one JSON message; the client follows up with byte-range fetches.

For 405B specifically: with TP=8, each rank has 1/8 of weights and 1/8 of attention heads. The "global" residual stream after the row-parallel + reduce-scatter pair is *replicated* across TP ranks (this is one of the key facts about Megatron-style TP), so the daemon can fetch from rank 0 only. The per-rank partial computations are visible only via the pre-collective inspection window (Q3) — and that is the right place to expose them via DTensor handles.

### Q11. What to Build First

**Recommendation: Reorder the phases. The protocol and the probe model are Phase 1; everything else slots in.**

Revised phase order:

- **Phase 0 (week 1–2): Protocol spec + golden TCK.** Write the JSON-RPC schema and the probe-point grammar. Write Gherkin scenarios for `initialize`, `attach`, `pause`, `step`, `inspect`, `intervene`, `checkpoint`, `replay`. Don't write any model code yet. Red tests fail because the daemon doesn't exist. This is the highest-leverage week.
- **Phase 1 (week 3–6): Single-GPU eager-mode daemon serving the protocol against GPT-2-small.** Rust daemon (process A), Python model host (process B). Tier A hooks (Q1). Component-level tick. Component = {attn_q_proj, attn_k_proj, attn_v_proj, attn_out_proj, attn_softmax, mlp_in, mlp_act, mlp_out, residual_pre, residual_post}. Inspect = summary + on-demand slice. No interventions yet. No checkpointing yet. No reverse-step. Validation: an external script can drive a 12-layer model and surface activation summaries via the protocol. **MVP candidate (see §5).**
- **Phase 2 (week 7–8): Interventions on a single GPU.** `set_activation`, `ablate`, `scale`, `add_steering_vector`. The intervention is a *recipe* the probe layer applies on the next tick. Validation: reproduce ITI (Inference-Time Intervention) end-to-end against the protocol.
- **Phase 3 (week 9–11): Checkpointing + reverse-step (sqrt-n).** Tier-2 from Q4. Reverse-step replays from nearest checkpoint with original probes re-firing. Validation: ROME-style locate-then-edit reproduction via reverse-step + intervention.
- **Phase 4 (week 12–14): TUI dogfood release.** Ratatui client of the protocol. Activation summary view, intervention panel, timeline. By this point the protocol is mature enough that the TUI is a *user* of it, not co-evolving with it.
- **Phase 5 (week 15–18): Multi-GPU (DDP first, then TP via DTensor).** Adds two-phase pause (Q3). Validation: same ROME experiment on Llama-2-70B with TP=4.
- **Phase 6 (week 19–21): MoE.** All four tick granularities (Q5). Validation: routing override on Mixtral 8x7B reproducing the 2026 Geometric Routing paper's interventions.
- **Phase 7 (week 22+): torch.compile / CUDA-graph (Tier B/C from Q1) + FlashAttention shadow replay (Q2) + FSDP.** These are the hard infrastructure problems; do not solve them in MVP.

Crucial corrections to your current ordering:
- **Protocol is Phase 0, not embedded inside Phase 1.** TCK-first methodology requires the protocol be specified before any code.
- **TUI is Phase 4, not 6.** Earlier is wrong because the TUI's needs co-evolving with the protocol pollutes both; later is wrong because the team needs dogfooding feedback before tackling the multi-GPU phase.
- **MoE is Phase 6, not 7.** But: the *protocol* must accommodate MoE tick granularity from Phase 0. Designing tick semantics that have to retrofit MoE is much worse than designing them with MoE in mind from day one (your instinct here is right; the change is "design for MoE early, ship for MoE late").
- **Reverse-stepping is Phase 3, not 2.** Forward-only debugging is genuinely useful (Garçon shipped without time travel for years). Defer time travel until forward stepping is rock solid.

## 3. Risk Analysis & Mitigations

Ranked by expected pain × probability.

### R1 (highest): PyTorch internals churn

**Likelihood: certain. Impact: severe.** Dynamo, AOTAutograd, Inductor, FSDP2, `torch.distributed.pipelining` are *all* on a 6-month evolution cadence. Issues like #117758 (hooks-after-compile) were filed in early 2024 and remain unresolved in the form that affects you. SimpleFSDP (Nov 2024) suggests FSDP itself is being redesigned. The PyTorch surface you build against today will move under you within 12 months.

**Mitigation:**
- **Pin PyTorch versions** in your CI matrix: cover the last 3 stable releases + nightly. Test on each.
- **Vendor critical interception points.** Your Tier B FX-rewrite pass and Tier A hook adapter should live behind a versioned `BackendAdapter` interface. When PyTorch changes, you replace the adapter, not the daemon.
- **Wire-protocol independence.** The protocol exists in protobuf/JSON-Schema and does not change when PyTorch changes. This is the single most important architectural decision protecting you from PyTorch churn.
- **Hire / contract a PyTorch internals expert** at 0.5 FTE if you don't have one. The cost of a wrong abstraction here is a quarter; the cost of an expert is a week.
- Lean on **vLLM-Lens, NNsight, Penzai** as canaries: if these libraries break under a new PyTorch, you have time to react.

### R2: Hook fragility across HuggingFace model diversity

**Likelihood: certain. Impact: high.** "HuggingFace models" is not a thing; it's 4000 partial implementations with idiosyncratic forward-pass structures. LlamaAttention is *not* MistralAttention is *not* Qwen2Attention is *not* MixtralAttention even though they look similar. NNterp exists precisely because TransformerLens's per-architecture rewrites were unmaintainable.

**Mitigation:**
- **Use NNsight as the model-introspection layer.** NNsight wraps arbitrary PyTorch models without rewriting them; let it do the heavy lifting for you and consume its `Envoy` tree.
- **Define a small canonical component vocabulary** (`attn_q`, `attn_k`, `attn_v`, `attn_o`, `attn_softmax`, `mlp_up`, `mlp_gate`, `mlp_down`, `ln1`, `ln2`, `residual_pre`, `residual_mid`, `residual_post`, `router`, `expert.{i}.up`, `expert.{i}.down`, `lm_head`). Build *per-model-family* adapters that map module paths in the HF model to canonical-vocabulary names. Start with Llama/Mistral/Qwen2/Mixtral/Gemma2/GPT-NeoX; everything else is "supported in best-effort module-path mode."
- **Build a model-conformance test suite.** For each supported family, run a fixed prompt and assert that probes fire at the canonical points in the expected order. Run nightly. When HF releases `transformers` 5, run the suite; failures are an early-warning system.

### R3: torch.compile / CUDA-graph incompatibility

**Likelihood: certain on production-grade serving setups. Impact: severe if not handled.** Per Q1: post-compile hook registration is silently ignored; CUDA Graphs skip submodule hooks entirely.

**Mitigation:** Tier B and Tier C from Q1. Specifically:
- Detect compiled/graphed state at attach time; downgrade granularity gracefully.
- Document the limitation in the protocol response (capability negotiation surfaces it).
- For Phase 1–4, simply **require eager mode** (`model.eval()`, no `torch.compile`). This is the right MVP boundary.
- Treat Phase 7 (compile/graph support) as a hard research problem on its own; budget 1 person-quarter.

### R4: Multi-GPU deadlock from interception

**Likelihood: high on first attempt. Impact: catastrophic if it happens in production.** Holding a barrier on one rank while NCCL is mid-collective hangs the whole node.

**Mitigation:**
- **Strict pre-collective / post-collective barriers (Q3).** Never insert a barrier *inside* a collective.
- **Watchdog with self-recovery.** A barrier held longer than `T_max` (default 5 min, configurable) auto-releases with an error event. Better to lose a debug session than wedge a node.
- **Fault injection in CI.** Use `pytest`-driven scenarios that deliberately hold barriers on rank 0 while ranks 1–7 advance; assert that the watchdog fires before NCCL times out.
- **Document the env requirements**: `TORCH_NCCL_BLOCKING_WAIT=0`, `NCCL_TIMEOUT=14400`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` during debug sessions.
- **Refuse to attach** if the env is wrong; surface the exact required env to the user.

### R5: Determinism failures cause "phantom bug" reports

**Likelihood: high. Impact: medium (erodes trust).** Users will replay, see different numbers, file bugs.

**Mitigation:**
- Spec the determinism guarantee *explicitly* in the protocol's `capabilities`: "Replay is ULP-close (rel error < 1e-3 fp16). Bit-exact replay is not guaranteed."
- Auto-verify on every replay; surface divergence in the response.
- Q6's three-layer recipe.
- Adopt Thinking Machines' batch-invariant kernels.

### R6: Performance overhead unacceptable in production

**Likelihood: medium. Impact: medium.** If always-on probes cost 30%, no one will use it for anything but isolated debug runs. Some users want always-on observability.

**Mitigation:**
- **Zero-cost when off.** Probe definitions are compiled to a no-op when no client is attached. PyTorch's fast-path-bypass cost (forced no-op hook) should be benchmarked and disclosed.
- **Tier the overhead** in the protocol: "passive" mode (NVML + selected NCCL counters, ~0% overhead), "interactive" mode (probes active, ~3–5%), "deep" mode (CUPTI + eBPF, ~20%).
- Publish overhead numbers per tier per model size in the docs.

### R7: Scope creep from "real models" requirements

**Likelihood: certain with a 2–4 person team. Impact: project failure.** Supporting dense+MoE × DDP+FSDP+TP+PP × compile+CUDA-graph × Llama+Mistral+Mixtral+Qwen+DeepSeek+Gemma = 6 × 5 × 3 × 6 = 540 cells in the support matrix. Even at 1 hour per cell, that's 13 person-weeks of pure validation.

**Mitigation:**
- **Explicit support matrix.** Public table of (architecture × parallelism × execution mode): Green, Yellow, Red. MVP supports a 3×1×1 slice. Phase 5 expands to 3×2×1. Phase 7 expands to 3×4×3.
- **Refuse to attach to unsupported configurations** with a clear error message pointing to the matrix.
- **One model family at a time.** Llama-2/3 first. Validate everything against Llama before adding Mistral. Then Qwen. Then MoE (Mixtral).

### R8: Team size (2–4 people) for an OS-debugger-grade product

**Likelihood: certain. Impact: high.** rr took several engineer-years. Pernosco took multi-engineer-years. NNsight has had ~5+ contributors continuously.

**Mitigation:**
- **Aggressive scope reduction (see MVP, §5).** Ship one valuable thing, not seven half things.
- **Build atop NNsight or Penzai, not from scratch.** Your bibliography lists NNsight; the team should make a serious decision: is rocket_surgeon a wrapper around NNsight that adds the GDB-like protocol + Rust core + TUI, or is it a from-scratch replacement? **Strong recommendation: rocket_surgeon's MVP is "the interactive debugger UX layer on top of NNsight."** This buys you the model adapter and intervention-graph machinery for free; you focus on the protocol, the daemon, the TUI, the checkpoint/replay engine, and MoE. That is a tractable 2–4-person 6-month scope.
- **Convert the bibliography review into a contribution funnel.** The original-authors of vLLM-Lens, Penzai, NNsight, Neuronpedia, EasySteer would likely advise you for free; reach out early.

### R9: Protocol design lock-in

**Likelihood: medium. Impact: medium.** The protocol you ship in MVP will become your *de facto* spec; clients will pin to it. A poor early decision (e.g. coupling tick semantics to Python list-index of layers, or making intervention recipes positional rather than named) is painful to undo.

**Mitigation:**
- **Version every message.** `jsonrpc: "2.0"`, `protocol_version: "0.1.0"`. Capability negotiation surfaces version.
- **Define a written deprecation policy** before v0.2.
- Borrow LSP/DAP versioning conventions (they are the reference for "long-lived debugger protocols").

### R10: Probe DSL safety

**Likelihood: medium. Impact: medium.** A "composable probe with arbitrary user code" can OOM the GPU, hang a barrier, or corrupt the model. If you go DTrace-style, you need DTrace-style safety: bounded execution, no allocation, no unsafe ops.

**Mitigation:**
- **Two-tier probe DSL.** Tier 1: declarative — `match`, `select`, `summary(stats=[mean,std])`, `slice`, `histogram` — compiled to safe code. Tier 2: arbitrary Python callbacks marked `unsafe=true`, run with a watchdog timer and an OOM guard. Default: Tier 1.
- **rbpf / uBPF as an option for Tier 1.** Compile declarative probes to BPF bytecode and JIT for ~zero-cost execution. Long-term option, not MVP.

### R11: Reproducibility of debug sessions

**Likelihood: medium. Impact: medium.** "It worked on my machine" for transformer debugging is brutal.

**Mitigation:**
- **Session bundle**: a tar.gz with model hash, prompts, probe definitions, intervention recipes, env capture, protocol-RPC trace, output. Like CockroachDB's statement bundle. Make this the primary bug-report artifact.

## 4. Architecture Critique

### 4.1 The three-layer architecture: what's right and what's wrong

**What's right:**
- Separating a Rust core from the Python PyTorch host.
- Putting the TUI in a separate process as a protocol client.
- Capability negotiation at `initialize`.
- DAP-inspired stopped-state model.
- Wildcard-queryable named probe points (`model:layer:component:event`).
- "Zero-cost when off."
- 7 composable primitives — though the *list* needs work (see below).

**What's wrong or under-specified:**

1. **"Rust as PyO3 extension module" is the wrong default architecture.** It is correct for a *prototype build*. It is wrong for the shipped product because: (a) GIL serializes the protocol loop; (b) Python crash kills Rust; (c) iteration speed on Rust requires Python rebuild; (d) you cannot run multiple models in one daemon. Refactor to *three* processes (Q7): Rust daemon ↔ Python model host(s) ↔ Rust TUI. PyO3 is then a thin embedding inside each model host, not the architectural backbone.
2. **"Python process as host owning PyTorch runtime" is correct but does not scale to multi-GPU multi-node.** For multi-rank, you need one Python *worker* per rank (because `torch.distributed` requires it), and one Rust *daemon* that fans out to all workers. The daemon owns the protocol; the workers do the work. Add this to the architecture diagram.
3. **The seven primitives — `step, inspect, intervene, probe, checkpoint, evaluate, status` — are close but missing two and one is wrong:**
   - **Missing: `attach` and `detach`.** Sessions must be attachable/detachable independently of step. Otherwise you cannot have multiple concurrent clients.
   - **Missing: `subscribe` / `unsubscribe`** for event streams (router decisions, OOM, NCCL events). Without these, the LLM client polls.
   - **`evaluate` is overloaded.** DAP `evaluate` is "run this expression in the debuggee's context." That's useful, but you also need `recipe` (declare a reusable intervention or probe template) and `replay` (re-run from checkpoint with possibly-different recipes). Split them.
   - Recommended primitives: `attach`, `detach`, `step`, `inspect`, `intervene`, `probe`, `checkpoint`, `replay`, `subscribe`, `status`. Drop `evaluate` (subsume into `inspect` with an `expr` field).
4. **"State in every response" is right but under-specified.** State must include `(session_id, model_id, rank?, layer, component, event, tick_id, mode: running|stopped|replaying|error)`. The `tick_id` is monotonic and is the primary key for everything (checkpoints reference it, probe firings reference it, interventions are attached at a tick).
5. **"DAP-inspired stopped-state inspection with scoped object references"** — good. But: DAP's `variablesReference` is a 32-bit handle. For tensors, you want richer handles: `tensor_id = sha256(content)` so the same tensor seen at two probe points has the same id (enables dedup in TUI). Use *content-addressable* tensor IDs everywhere; this is also how you make session bundles (R11) work.
6. **"MCP server exposure" should be an adapter, not a transport.** Per Q8: ship JSON-RPC + gRPC as native transports; MCP is an adapter that wraps them.
7. **"DTrace-inspired probe system" is correct in spirit; the bibliography is missing the userspace-eBPF/rbpf option** for actual zero-cost-when-off probe compilation. Otherwise "zero-cost when off" is aspirational. For MVP, "zero-cost when off" = "no Python callback registered." Be honest in docs.
8. **Probe-point namespace `model:layer:component:event` lacks the rank dimension and the token dimension.** A probe fires once per (rank × layer × component × event × token). The namespace should be `model:rank:layer:component:event` and the *firing* includes the token range. Otherwise on TP=8 you cannot distinguish "the value on rank 3" from "the gathered value."
9. **"Composable hooks" is great in principle but you have not specified the composition semantics.** Are probes a set, a list, or a tree? Order of execution? What happens when two probes match the same point with conflicting interventions? Specify: probes are an ordered list, executed in registration order, interventions compose via the EasySteer pattern (additive superposition by default, with optional `replace` and `priority`). Document.
10. **"Built-in interpretability views" is right, but the list isn't specified.** Concretely, ship these as built-in: `residual_stream_norm`, `attention_pattern` (with LSE-recon for FlashAttention), `head_output`, `logit_lens` (Tuned Lens + projection to vocab), `routing_decision`, `routing_entropy`, `sae_activation` (when SAEs are registered), `feature_attribution`. Make these *names* in the protocol, not user-defined.

### 4.2 The tick model — assessment

**"One tick = one atomic unit of forward pass execution. Granularity levels layer/component/head/expert. Default component. Each tick boundary a `cudaDeviceSynchronize` point."**

What's right: the layered granularity is the right idea. Component as default is the right default. Sync at boundaries is necessary.

What's wrong or missing:

1. **`cudaDeviceSynchronize` on every tick is too expensive.** It blocks all streams. For pipeline parallelism it is wrong — you want per-stage sync, not global. Use `torch.cuda.current_stream().synchronize()` scoped to the rank/stream, or better, use CUDA events: record an event at the tick boundary, host blocks on the event. Same correctness, less stream-stall.
2. **The granularity list is incomplete for MoE.** Per Q5, add `router_pretopk` and `router_posttopk` as distinct sub-granularities under "component" or as their own level. The protocol should also support **`tensor`** granularity — i.e., "step until this specific named tensor is written." This is the equivalent of a GDB watchpoint and is more useful in practice than blind stepping.
3. **"Head" and "expert" are not natural tick boundaries in vectorized code.** All heads run as a single batched-matmul; you cannot pause "between head 3 and head 4" without rewriting the attention call to split heads. Same for experts in fused-grouped-GEMM MoE implementations like Megablocks. **Be honest: head and expert granularity require unfused execution** (Q2 shadow-replay). Document this; do not pretend it's free.
4. **The tick model needs a "tick scope" concept.** A user may want to step at component-granularity inside one specific layer but layer-granularity elsewhere. Provide `set_step_granularity(scope=match("layer.12"), granularity="component")`.
5. **Backward-tick granularity is not specified.** Eventually users want backward-pass interventions (gradient ablation, Pearce-style backward patching). The tick model should be symmetric forward/backward from day one in the *schema*, even if backward implementation is deferred to Phase 8+.
6. **Tick ID semantics.** State that tick_id is monotonic, that replayed ticks get fresh tick_ids with a `replay_of: parent_tick_id` reference, and that bookmarks are first-class names for ticks.

### 4.3 The reverse-stepping design — assessment

**"Checkpoint + forward replay, rr/TTD model, auto-checkpoint every sqrt(n) layers."**

This is correct in shape. Refinements:

1. **Adopt Pernosco's *dataflow* view as the primary reverse-stepping UX, not just step-back.** Per Q1.5: the killer feature in omniscient debuggers isn't pressing "step back N times" — it's "click a value, jump to where it was produced." For transformers this maps to "click a logit dimension, jump to the layer/head/expert that contributed most." Spec this as a `rocket/dataflow.trace` verb from day one. The reverse-step verbs (DAP `stepBack`, `reverseContinue`) become *implementations* of dataflow traversal: step back = go to the immediately-prior probe fire.
2. **Auto-checkpoint cadence should be input-dependent.** For long-context inference (128K), per-token activations dwarf the residual stream and you cannot afford sqrt-L checkpoints; you need sqrt-N over *output tokens*. Make the cadence adaptive on `seq_len * L`.
3. **Replay determinism must be checked, not assumed.** Per Q6: every replayed tensor at a probe point is compared to the recorded summary; divergence above tolerance fires a `replay.divergence` event. The user sees "this replay diverges at layer 17; cosine 0.997, max-rel 4e-3" and can choose to proceed or escalate. This is the right honesty stance.
4. **Bookmarks** (named ticks) are required: this is Pernosco's notebook model.
5. **Session bundle export** at any tick: produces the reproducer artifact (R11).

### 4.4 Architecture decisions to make explicit in the written plan

- Transport: ship gRPC (`tonic`) + stdio JSON-RPC + MCP adapter. Drop "WebSocket" unless there's an explicit browser-client need.
- Tensor IDs: content-addressable (`sha256(bytes)`), 32-byte hex.
- Trace format: Perfetto protobuf.
- Probe DSL: declarative for MVP (no rbpf); add bytecode tier in Phase 7+.
- Backend adapter interface: versioned, separate from daemon, replaceable per PyTorch version.
- Build NNsight in as the model-introspection layer (R8). Cite it; co-design where possible.
- Determinism guarantee: documented as "ULP-close, not bit-exact." Verified per-tick.
- Multi-GPU: pre/post-collective barriers; never inside.
- MoE: four tick granularities defined in schema from day one, even if Phase 6 ships the implementation.

## 5. Minimum Viable Version

### 5.1 The MVP, in one sentence

**A daemon that, given a path to a single-GPU eager-mode HuggingFace Llama-3-8B and a prompt, lets an LLM client step through the forward pass at component granularity over JSON-RPC, inspect residual streams and attention patterns with summary-then-slice semantics, and apply ablation/scaling interventions — with a session bundle exportable on demand.**

That is the entire MVP. Everything else is later.

### 5.2 What is included

- Rust daemon process, JSON-RPC 2.0 over stdio + Unix socket.
- Schema in protobuf and JSON-Schema. Versioned. `initialize` capability negotiation.
- Python model host (one process) loaded by daemon. Loads HF Llama-3-8B in eager mode, `bf16`, single GPU.
- Tier A hooks only (Q1): registered after attach, before any user `torch.compile`. The MVP **refuses to attach** to a compiled model.
- Component-granularity tick (the 12-name component vocabulary from R2 mitigation, for Llama family only).
- `attach`, `detach`, `step`, `inspect`, `intervene`, `probe`, `status`, `subscribe` verbs. No `checkpoint`, no `replay`, no reverse-step.
- Tensor handles are content-addressable. Default response carries summaries only (mean/std/min/max/abs-max/sparsity/histogram-32). `rocket/tensor.slice` retrieves up to 64 KB at a time.
- Two built-in interpretability views: `residual_stream_norm` and `attention_pattern` (using `attn_implementation="eager"` for the MVP — we don't deal with FlashAttention until Phase 7's shadow-replay).
- Three interventions: `ablate` (zero out a component output), `scale` (multiply by a scalar), `add` (add a fixed vector). No SAE, no expert override, no patchscope yet.
- Session bundle export: tar.gz with model hash, prompt, protocol-RPC trace, all captured tensors as safetensors. This is the unit of reproducibility.
- MCP adapter: wraps the JSON-RPC service and exposes `step`, `inspect`, `intervene` as MCP tools. Resources are checkpoints (none yet) and tensors. Validated by having Claude or another LLM successfully drive the MVP from schema alone, end-to-end, to reproduce one IOI-paper-style probe.

### 5.3 What is explicitly out of MVP

- Multi-GPU. Single GPU only.
- TUI. The MVP ships with a CLI test harness in Rust, not Ratatui. The TUI ships in Phase 4.
- Checkpointing & reverse-step. Forward-only.
- MoE. Llama (dense) only.
- FlashAttention / fused-kernel inspection. `attn_implementation="eager"` is forced.
- `torch.compile`, FSDP, CUDA Graphs. Eager only.
- SAE manipulation. Defer.
- Expert routing override. Defer.
- Probe DSL composition. Probes are register-and-fire; no chained probes yet.
- gRPC. JSON-RPC stdio + Unix socket only.

### 5.4 Why this is the right MVP

- **It validates the protocol-first, schema-driven thesis.** If an LLM can drive this MVP from schema alone to do interpretability, the entire concept is proven. If not, no amount of multi-GPU support saves it.
- **It is achievable by 2–4 people in ~10 weeks** (Phase 0 + Phase 1 + Phase 2 of the revised plan in Q11), which is the right time-to-feedback for a JSMNTL flow.
- **It exercises every architectural seam** — Rust daemon, Python host, PyO3 boundary, protocol schema, capability negotiation, content-addressable tensors, MCP adapter, session bundles — but only one model, one parallelism mode, one execution mode.
- **It reproduces a known result** as the acceptance test. Recommended: reproduce **Indirect Object Identification (IOI, Wang 2023)** on Llama-3-8B via the protocol, driven by an LLM client. If the LLM can use rocket_surgeon to ablate the name-mover heads and verify the score drop, MVP is done.

### 5.5 Definition of done for MVP

1. JSON-RPC schema is frozen at v0.1.0 and published.
2. Daemon starts, attaches to Llama-3-8B in eager mode in <30s.
3. An automated test, driven by `claude-cli` or equivalent, runs the IOI ablation experiment using only the protocol and produces the expected score deltas (±5%).
4. Session bundle export reproduces the same result on a different machine of the same GPU class.
5. Overhead with zero probes registered is <5% on prefill, <2% on decode.
6. Documentation: protocol spec, attach guide, one tutorial (IOI reproduction).

### 5.6 Recommended team allocation

- 1 person on Rust daemon + protocol + transports + session bundles.
- 1 person on Python model host + Tier A hook layer + intervention engine.
- 0.5 person on MCP adapter + schema design + LLM-driven test harness.
- 0.5 person on docs + Gherkin TCK + model-conformance test suite.

If the team is 2 people, drop the MCP adapter from MVP (ship it Phase 2). If 4 people, keep MCP and add an early-spike on Tier B (FX-graph) for Phase 7 readiness.

# Closing Summary

The five strongest opinions in this report, restated for the plan:

1. **Build atop NNsight's intervention-graph abstraction, do not re-implement it.** The team is too small to redo NNsight. rocket_surgeon is the *protocol, daemon, checkpoint engine, time-travel UX, and TUI* on top of NNsight's model adapter and Envoy tree. This single decision changes the project from "infeasible in 6 months with 2 people" to "tight but feasible." Cite NNsight 0.6, Penzai/Treescope, vLLM-Lens, and the Garçon design as the four most influential references.

2. **Three-process architecture: Rust daemon ↔ Python model host(s) ↔ Rust TUI**, with JSON-RPC + gRPC + MCP adapter as transports. PyO3 is embedded inside the Python host as a thin hot-path accelerator, not the architectural backbone. Tensors flow as content-addressable handles; bytes are fetched lazily in summary-then-slice form. Perfetto protobuf is the trace format.

3. **Protocol-first, schema-frozen at v0.1.0 before Phase 1 code.** Ten verbs: `attach, detach, step, inspect, intervene, probe, checkpoint, replay, subscribe, status`. DAP-compatible lifecycle + reverse-stepping verbs; `rocket/*` namespace for domain operations. MCP exposure as an adapter, not a transport. The protocol must accommodate MoE's four tick granularities (router-pre-topk, router-post-topk, per-expert, layer) from day one even though MoE ships in Phase 6.

4. **Three-tier hook strategy (eager / Dynamo-FX-rewrite / CUDA-graph), with the MVP supporting only eager.** Pre/post-collective barriers for multi-GPU; never barrier inside a collective. Sqrt-N checkpointing with content-addressable tensor IDs, ULP-close replay tolerance (cosine > 0.99995), batch-invariant kernels (Thinking Machines, 2025) on the shadow-replay path. FlashAttention is handled by selective shadow-unfusing of inspected layers, not always-on reference execution.

5. **MVP is single-GPU eager Llama-3-8B + JSON-RPC + three interventions + MCP adapter, validated by an LLM client driving an IOI-paper reproduction from schema alone.** Everything else is a phase. Reorder phases so protocol is Phase 0, TUI is Phase 4 (mid-project dogfood, not endgame), reverse-step is Phase 3 not Phase 2, and `torch.compile`/CUDA-graph/FSDP are Phase 7 — explicitly *after* the MoE phase, because compile-mode interception is the hardest infrastructure problem in the project and should not block real research use.

The single most valuable hour the team can spend before writing the plan is reading vLLM-Lens, Penzai, and the Garçon write-up back-to-back; the single most valuable artifact to produce in week 1 is the JSON-RPC schema v0.1.0 with capability negotiation.