# ROCKET SURGEON: Deep Research Brief

## What We're Building

**rocket_surgeon** is a proper debugger and in-situ surgery tool operating natively on multi-GPU forward passes through large language models. Think GDB meets mechanistic interpretability: step one tick at a time through a transformer forward pass (forward and backward through time, not through autograd), pause at any layer/component/head/expert boundary, inspect every tensor with full fidelity, and perform arbitrary surgical interventions (ablation, scaling, steering, activation patching, SAE feature manipulation, expert routing overrides) before resuming.

This must work on:
- Dense transformers (LLaMA, Mistral, GPT-NeoX, Gemma, Qwen)
- MoE architectures (Mixtral, DeepSeek)
- Multi-GPU setups (DDP, FSDP, tensor parallelism, pipeline parallelism)
- Real HuggingFace models without re-implementation

The tool has two first-class interfaces:
1. **Machine interface** (JSON-RPC 2.0 protocol) — the PRIMARY interface, designed so an LLM can pick it up and use it from schema alone with zero system prompts, skill files, or wrappers
2. **TUI** (Ratatui terminal UI) — a client of the protocol, not the system itself

We are at the end of a deep research phase. We have read 101 papers, cloned and analyzed 47 reference implementations, and written 13 literature reviews and 7 deep-dive source code analyses. We are asking for architectural advice before committing to a plan.

---

## The Architecture We're Converging On

Three layers:

### Layer 1: Core Engine (Rust + Python via PyO3)
- Rust state machine managing stepping lifecycle, checkpoint storage, probe registry, protocol server
- Python hook layer for PyTorch integration (hooks require Python, no way around this)
- Python process is host (owns PyTorch runtime), Rust compiled as extension module via PyO3
- TUI runs as separate Rust process communicating over the protocol

### Layer 2: Machine Interface (JSON-RPC 2.0, DAP-inspired)
- 7 composable primitives: step, inspect, intervene, probe, checkpoint, evaluate, status
- State in every response (position, active probes, available actions)
- Capability negotiation at initialize (LSP pattern)
- DAP-inspired stopped-state inspection with scoped object references
- MCP server exposure for native LLM integration
- Transports: stdio (default), TCP, WebSocket

### Layer 3: Probe System (DTrace-inspired)
- Named probe points: `model:layer:component:event` (wildcard-queryable)
- Composable hooks: inspect, checkpoint, trace, aggregate, assert, intervene, sae_decompose
- Zero-cost when off (hooks registered only when probes armed)
- Built-in interpretability views: logit lens, attention patterns, SAE feature decomposition, routing tables

### Tick Model
One "tick" = one atomic unit of forward pass execution. Granularity levels: layer, component, head, expert. Default: component. Each tick boundary is a `cudaDeviceSynchronize` point.

### Reverse Stepping
Checkpoint + forward replay (rr/TTD model). Auto-checkpoint every sqrt(n) layers. "Step backward" = restore nearest checkpoint, replay forward to target. Requires deterministic execution (single CUDA stream, cuBLAS deterministic mode, fixed seeds, fixed batch composition, same GPU arch).

---

## What We've Learned from Reading Everything

### From the Reference Implementations (47 repos analyzed)

**nnsight** (ndif-team): Wraps existing HuggingFace models via Envoy proxy pattern. Lazy hook registration (zero overhead for uninspected modules). Thread-per-invoke with blocking queues. CUDA stream propagation to worker threads is essential. The execution-order constraint (interventions must be written in forward-pass order) is actually a feature for a step-through debugger. BUT: uses `sys.settrace()` for code capture (fragile), mediator threading model is extremely complex (6 event types, 2 queues per mediator), sentinel hooks needed as workaround for PyTorch's fast-path when no hooks registered.

**TransformerLens**: HookPoint identity module insertion — clean but requires model re-implementation. Context-level scoping for nested hook lifetimes. Activation caching via hooks is simple and effective. BUT: hooks are global state (biggest footgun), every HookPoint runs on every forward even with zero hooks (overhead), must maintain own model implementations (TransformerLens trap).

**pyvene**: IntervenableConfig serialization — interventions as JSON data, not code. Composable intervention types. BUT: 14-field config is over-designed, tightly coupled to its own model representation.

**baukit** (David Bau): `invoke_with_optional_args()` for flexible callback signatures. `StopForward` exception for early termination. `TraceDict` for multi-point activation capture. Elegantly simple.

**transformer-debugger** (OpenAI): Closest prior art. `DerivedScalarType` taxonomy (~100+ members) is the most thorough enumeration of inspectable transformer internals. Three-phase hook model (fwd/bwd/fwd2). GroupId batching for related requests. BUT: REST-only (no streaming, no bidirectional), SIGKILL on OOM, tight coupling to specific transformer implementation, NO stepping/tick model (entire forward pass is atomic), no multi-GPU, no MoE, no LLM interface.

**SAELens**: Error term computation is ESSENTIAL — without it, removing a feature also removes correlated reconstruction error, corrupting the model. Multiple SAE architectures supported (Standard, TopK, JumpReLU, Gated). Loader ecosystem for different SAE providers. SAE HookPoints give internal observability.

**ACDC** (Automated Circuit Discovery): Three edge types (ADDITION, DIRECT_COMPUTATION, PLACEHOLDER) — good vocabulary but needs ROUTING for MoE. Graph built from HookedTransformer hook names. No MoE support.

**tuned-lens**: Architecturally trivial (one Linear per layer, residual). `invert()` capability (L-BFGS to find hidden state producing target logits) enables goal-directed surgery.

**rr** (record & replay debugger): Checkpoint at boundary between user code and kernel (our boundary is between host code and GPU). Three-tier checkpoint strategy. Diversion concept (fork-and-modify for what-if analysis). TraceFrame format inspiration.

**DAP spec**: Capability negotiation at initialize. Stopped-state inspection waterfall (threads → stackTrace → scopes → variables maps to devices → layers → components → activations). Object reference lifetime scoped to current stopped state. StepBack as optional capability.

**MCP spec**: Tool schemas in JSON Schema (same format LLMs understand). Resource subscriptions for state updates. `outputSchema` + `structuredContent` for typed tool results. `annotations` (readOnlyHint, destructiveHint) for LLM safety reasoning.

**tinygrad**: Universal UOp IR — one data structure from tensor graph to compiled binary. Global deduplication (structural identity = object identity). PatternMatcher system for all transformations: `(UPat, callback)` pairs composed with `+`. 110 ops in 12 categories cover all of modern ML. TLSF-based memory planning. Scheduling via Kahn's toposort. The clearest demonstration that "capture → optimize → execute" is universal.

**Triton**: Python → Triton-IR → Triton-GPU IR → LLVM IR → PTX → cubin. CUDABackend runs 80+ MLIR passes. Instrumentation hooks exist in the compilation pipeline.

**vLLM**: GroupCoordinator for TP/PP/EP groups. Custom collective ops. GraphCaptureContext for CUDA graph management. PagedAttention manages KV cache as virtual memory pages.

**cuda-checkpoint**: Lock phase (drain pending work, block new APIs) is fast and usable for tick-pause without full checkpoint. Full checkpoint is O(GPU memory) = 77s for 54GB = too slow for interactive use. No NCCL support, no UVM support. R580 adds GPU migration (remap UUIDs on restore).

**flash-attention**: The N×N attention matrix NEVER EXISTS IN HBM. Tiled computation keeps everything in SRAM. The debugger cannot display attention patterns by reading memory — must recompute or infer from stored LSE (log-sum-exp) values. LSE must be a first-class observable.

**nccl**: Ring/tree algorithms, dynamic protocol selection (Simple/LL/LL128). Communicator state is opaque — no checkpoint support, no external inspection. Multi-GPU tick synchronization must happen at framework level, not transport level.

**Ratatui**: Immediate-mode API with retained-mode diffing. Cassowary constraint-based layout. trippy (network diagnostic TUI) is the best architectural model: clean workspace separation, snapshot-based data flow, freeze/unfreeze for pausing updates.

**safetensors**: Zero-copy tensor access via mmap. `View` trait (dtype, shape, data as `Cow<[u8]>`). Simple format (8-byte header length + JSON metadata + raw bytes). Right serialization for tensor data exchange between Python engine and Rust TUI.

**PyO3**: `py.detach(|| ...)` for GIL release during long Rust operations. Zero-copy tensor sharing via raw pointers. `#[pyclass(frozen)]` for thread-safe immutable objects. Free-threaded Python 3.14+ for true parallelism.

**Perfetto**: Strictly superior to Chrome Trace Format. Protobuf format, streaming-friendly, GPU-specific protos already exist. SQL-based analysis via trace_processor. TRACE_EVENT macros map to our tick model.

**ProfInfer** (eBPF LLM profiler): Multi-granularity probing (token/graph/operator/scheduler). Adaptive overhead control. MoE expert tracing. 1.7% overhead with libbpf. Most directly applicable reference for our observability stack.

**NeutriNo**: Instruction-level GPU kernel profiling via assembly probing. LD_PRELOAD hook driver pattern for libcuda.so interception. 1.04x overhead. DMAT visualization of memory access patterns.

**Key gap across ALL existing tools**: No existing library handles tensor parallelism or MoE interception correctly. Both must be designed from scratch.

### From the Papers (101 papers read)

**GPU determinism** is achievable but costly (10-40% slower): single CUDA stream + `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:16:8` + fixed seeds + fixed batch composition + same GPU arch. MoE routing via `scatter_reduce` is a particular trouble spot for non-determinism.

**Gradient/activation checkpointing** (Chen 2016): Store activations at sqrt(n)-th nodes, recompute during replay. 25-50% more compute for 40-60% less memory. Our checkpoint strategy should align with this existing pattern.

**PhoenixOS** (2024): Concurrent GPU checkpoint/restore using copy-on-write. Only 185ms overhead for Llama2-13B. Checkpointing at iteration start is cheapest (only activation buffers dirty). Context pool bypasses expensive creation.

**Circuits framework** (Elhage 2021, Olah 2020): Transformers as computational graphs of identifiable circuits. Mathematical framework for understanding attention heads as moving information and MLPs as storing/retrieving facts.

**Superposition** (Elhage 2022): Models represent more features than dimensions by using superposition. SAEs decompose this. Fundamental for understanding what we're looking at when we inspect activations.

**SAEs** (Cunningham 2023, Bricken 2023, Templeton 2024): Sparse autoencoders find interpretable features. Scaling to Claude 3 Sonnet shows millions of meaningful features. Error term tracking essential for intervention.

**ROME/MEMIT** (Meng 2022-2023): Causal tracing localizes factual associations to specific MLP layers. Rank-one edits modify facts. MEMIT scales to thousands of edits simultaneously.

**Steering vectors** (Turner 2023, Zou 2023, Li 2023): Contrastive activation differences define behavioral directions. Adding/subtracting vectors steers model behavior. Representation engineering treats representations as the fundamental unit.

**IOI circuit** (Wang 2023): Full reverse-engineering of the indirect object identification circuit. 26 attention heads across 7 types. The gold standard for what "understanding a circuit" means.

**ACDC** (Conmy 2023): Automated circuit discovery via iterative edge pruning with activation patching. Makes circuit analysis tractable at scale.

**Patchscope** (Ghandeharioun 2024): Decode any hidden representation through any model's generation. Generalizes logit lens.

**MoE interpretability** (2026 papers): "Polysemantic Experts, Monosemantic Paths" — experts are polysemantic but token-routing paths are monosemantic. "The Expert Strikes Back" — experts develop semantic specialization. "Geometric Routing" — causal expert control via geometric methods. This is bleeding-edge and directly relevant to our MoE stepping/routing features.

---

## Hard Questions We Need Help With

### 1. The Hook Registration Problem

PyTorch's hook behavior is a minefield:
- DDP silently ignores hooks registered before `DistributedDataParallel()` wrapping
- `torch.compile()` silently ignores hooks registered after compilation
- FSDP uses hooks internally and custom hooks can interfere
- CUDA Graphs bypass hook entry points entirely
- PyTorch fast-paths when no hooks registered (nnsight's sentinel hook workaround)

We need a hook registration strategy that works reliably across all these configurations. Register before compile but after distributed wrapping? Force `fullgraph=False`? Use torch.compile's FX graph interception instead of runtime hooks for compiled models? What's the right layered approach?

### 2. FlashAttention and Fused Kernels

The attention matrix never exists in HBM with FlashAttention. We can store the LSE values, but reconstructing full attention patterns requires recomputation. More broadly, kernel fusion (TorchInductor, Triton) merges operations that we want to inspect individually.

How should we handle the tension between optimized execution (fused) and debuggable execution (unfused)? Run an unfused "debug mode" forward pass alongside the fused one? Selectively unfuse only the layers being inspected? Accept that some intermediate state is only available via recomputation?

### 3. Multi-GPU Tick Synchronization

NCCL is opaque — no checkpoint support, no external inspection of communicator state. For a multi-GPU model, collective operations (all-reduce, all-gather, reduce-scatter) must complete atomically or the system deadlocks.

How do we synchronize a "pause" across multiple GPUs? The collective must finish on all devices before any device can pause for inspection. But what if we want to inspect activations BEFORE the collective? Do we need a two-phase pause (pre-collective inspection window, then collective completion, then post-collective inspection window)?

### 4. The Checkpoint Size vs Replay Speed Tradeoff

Options:
- **Full GPU checkpoint** (cuda-checkpoint): 77s for 54GB. Captures everything. Way too slow.
- **Activation-only checkpoints** at every layer: ~1GB for 7B model. Fast save/restore. But replay from distant checkpoint = recompute many layers.
- **sqrt(n) checkpoints**: ~180MB for 7B model. Optimal recomputation. But max recompute is sqrt(32) ≈ 6 layers.
- **Hierarchical**: frequent lightweight checkpoints + periodic heavy ones.

What's the right strategy for interactive debugging where users step forward and backward frequently? The PhoenixOS copy-on-write approach (185ms overhead) is intriguing — could we adapt it?

### 5. MoE-Specific Design

No existing tool handles MoE well. The challenges:
- Router decisions are per-token — different tokens go to different experts
- Expert execution is parallel (tokens dispatched to different experts simultaneously)
- We need to visualize routing decisions AND allow forcing specific expert assignments
- Expert capacity limits and load balancing add complexity
- The "tick" model for MoE: is one tick the router decision? One expert's computation? All experts for one layer?

The 2026 MoE interpretability papers suggest experts are polysemantic but paths are monosemantic. How does this inform our probe/visualization design?

### 6. Determinism Requirements

We need deterministic replay for backward stepping. But:
- `torch.use_deterministic_algorithms(True)` disables some ops entirely (no fallback)
- FlashAttention's `deterministic=True` flag exists but adds overhead
- MoE routing via `scatter_reduce` is non-deterministic
- NCCL collectives may have non-deterministic reduction order
- Multi-GPU timing introduces non-determinism

Can we achieve "deterministic enough" for debugging purposes without requiring bit-exact replay? What if we checkpoint more aggressively and accept that replayed tensors are "close but not identical"?

### 7. The Rust/Python Boundary

The Rust core handles state machine, protocol, TUI, checkpoint storage. Python handles PyTorch hooks, model loading, SAE integration. The boundary is critical:
- Tensor data must cross it efficiently (inspection, checkpoint save/restore)
- Commands flow Rust → Python (step, intervene), results flow Python → Rust
- GIL management during long operations
- The TUI needs tensor data for visualization without copying through Python

Our current thinking: mmap'd safetensors files for zero-copy tensor sharing between processes, plus JSON-RPC for control commands. Is this the right split? Should the Rust side ever touch tensors directly, or always go through Python?

### 8. Protocol Design: DAP Extension vs Custom

DAP gives us IDE ecosystem compatibility but doesn't fit our domain well (call stacks, source lines, variable scoping). Our domain has layers, components, tensors, probe points.

Options:
- Extend DAP with custom request/response types (keep compatibility, awkward mapping)
- Custom JSON-RPC protocol inspired by DAP (clean domain fit, no ecosystem)
- DAP for lifecycle + custom namespace for domain operations (hybrid)
- Full MCP server (native LLM integration, but MCP lacks streaming/bidirectional)

What's the right call? The LLM-native interface is the priority — we'd rather have perfect LLM ergonomics than VS Code integration.

### 9. Observability Stack: What Goes Where

We have multiple instrumentation layers:
- PyTorch hooks (tensor operations, module boundaries)
- CUPTI (kernel launches, memory ops, synchronization)
- eBPF uprobes on libcuda.so (driver-level syscalls)
- NVML (hardware state: utilization, temperature, memory)
- NCCL Inspector (collective operations)

How should these compose? Should the user see a unified timeline or separate views? Does the eBPF layer run always-on for diagnostics, or only when explicitly enabled? Perfetto vs custom trace format?

### 10. Scaling Concerns

This needs to work on models from 1B to 405B+ parameters:
- At 405B across 8 GPUs, checkpoint size at every layer is ~64GB
- Attention pattern recomputation for 128K context models is expensive
- The protocol must handle large tensor payloads without choking the LLM client
- TUI must render useful information about tensors with millions of elements

What are the right abstractions for scale? Hierarchical summarization? Adaptive checkpoint granularity? Streaming tensor inspection?

### 11. What Should We Build First?

Given JSMNTL methodology (specs → red tests → implementation), what's the right build order? Our current thinking:

Phase 1: Single-GPU, single-model, basic stepping + inspection
Phase 2: Checkpoint/replay for backward stepping
Phase 3: Probe system
Phase 4: Interventions
Phase 5: Multi-GPU
Phase 6: TUI
Phase 7: MoE

Is this right? Should the protocol be Phase 1 (since everything depends on it)? Should the TUI be earlier (for dogfooding)? Should MoE be integrated from the start rather than bolted on?

---

## Bibliography

101 papers and web documents informing this project, organized by domain.

### Interpretability & Circuits

- Olah et al. 2020. "Zoom In: An Introduction to Circuits." Distill. [HTML]
- Elhage et al. 2021. "A Mathematical Framework for Transformer Circuits." Anthropic. [PDF]
- Elhage et al. 2022. "Toy Models of Superposition." arXiv 2209.10652. [PDF]
- Bricken et al. 2023. "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning." Anthropic. [PDF]
- Templeton et al. 2024. "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." Anthropic. [HTML]
- Bills et al. 2023. "Language Models Can Explain Neurons in Language Models." OpenAI. [HTML]
- Anthropic. 2025. "Circuit Tracing: Revealing Computational Graphs in Language Models." [HTML]
- Cunningham et al. 2023. "Sparse Autoencoders Find Highly Interpretable Features in Language Models." arXiv 2309.08600. [PDF]
- Conmy et al. 2023. "Towards Automated Circuit Discovery for Mechanistic Interpretability." arXiv 2304.14997. [PDF]
- Wang et al. 2023. "Interpretability in the Wild: A Circuit for Indirect Object Identification." arXiv 2211.00593. [PDF]
- Belinkov. 2022. "Probing Classifiers: Promises, Shortcomings, and Advances." arXiv 2102.12452. [PDF]
- Hewitt & Manning. 2019. "A Structural Probe for Finding Syntax in Word Representations." Stanford NLP. [PDF]
- Elazar et al. 2021. "Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals." arXiv 2006.00995. [PDF]
- Belrose et al. 2023. "Eliciting Latent Predictions from Transformers with the Tuned Lens." arXiv 2303.08112. [PDF]
- Ghandeharioun et al. 2024. "Patchscope: A Unifying Framework for Inspecting Hidden Representations." arXiv 2401.06102. [PDF]
- Geva et al. 2021. "Transformer Feed-Forward Layers Are Key-Value Memories." arXiv 2012.14913. [PDF]
- Geva et al. 2022. "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts." arXiv 2203.14680. [PDF]
- Geva et al. 2023. "Dissecting Recall of Factual Associations in Auto-Regressive Language Models." arXiv 2304.14767. [PDF]
- Dai et al. 2022. "Knowledge Neurons in Pretrained Transformers." arXiv 2104.08696. [PDF]
- VISIT. 2023. "Semantic Information Flow." arXiv 2305.13417. [PDF]
- "Attention Head Intervention." 2026. arXiv 2601.04398. [PDF]
- Chen. 2019. "Gradient Checkpointing." arXiv 1904.10631. [PDF]

### MoE Interpretability

- "Polysemantic Experts, Monosemantic Paths." 2026. arXiv 2604.17837. [PDF]
- "The Expert Strikes Back." 2026. arXiv 2604.02178. [PDF]
- "Geometric Routing: Causal Expert Control." 2026. arXiv 2604.14434. [PDF]

### Surgery & Steering

- Meng et al. 2022. "Locating and Editing Factual Associations in GPT (ROME)." arXiv 2202.05262. [PDF]
- Meng et al. 2023. "Mass-Editing Memory in a Transformer (MEMIT)." arXiv 2210.07229. [PDF]
- Turner et al. 2023. "Activation Addition: Steering Language Models Without Optimization." arXiv 2308.10248. [PDF]
- Zou et al. 2023. "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv 2310.01405. [PDF]
- Li et al. 2023. "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." arXiv 2306.03341. [PDF]
- Heimersheim. 2024. "How to Use and Understand Activation Patching." arXiv 2404.15255. [PDF]
- Vig & Gehrmann. 2020. "Causal Mediation Analysis for Interpreting Neural NLP." arXiv 2004.12265. [PDF]
- "Model Surgery: Parameter Editing." 2024. arXiv 2407.08770. [PDF]
- "Contrastive Weight Steering." 2025. arXiv 2511.05408. [PDF]
- "SAE Steering: Refusal." 2024. arXiv 2411.11296. [PDF]
- "SAE Steering: Knowledge Selection." 2024. arXiv 2410.15999. [PDF]
- "SAE-SSV: Supervised Steering." 2025. arXiv 2505.16188. [PDF]
- "Feature Guided Activation Additions." 2025. arXiv 2501.09929. [PDF]

### Tools & Frameworks

- Fiotto-Kaufman et al. 2024. "NNsight and NDIF: Democratizing Access to Foundation Model Internals." arXiv 2407.14561. [PDF]
- Wu et al. 2024. "pyvene: A Library for Understanding and Improving PyTorch Models via Interventions." arXiv 2403.07809. [PDF]
- Ferrando et al. 2024. "Inseq: An Interpretability Toolkit for Sequence Generation Models." arXiv 2302.13942. [PDF]
- "NNterp." 2025. arXiv 2511.14465. [PDF]
- Paszke et al. 2019. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." arXiv 1912.01703. [PDF]
- Abadi et al. 2016. "TensorFlow: A System for Large-Scale Machine Learning." arXiv 1603.04467. [PDF]
- Frostig et al. 2018. "Compiling Machine Learning Programs via High-Level Tracing." SysML. [PDF]
- Tillet et al. 2019. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MAPL/PLDI. [PDF]
- Lattner et al. 2020. "MLIR: A Compiler Infrastructure for the End of Moore's Law." arXiv 2002.11054. [PDF]
- Lattner et al. 2021. "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation." CGO. [PDF]
- Ansel et al. 2024. "PyTorch 2: Faster ML Through Dynamic Python Bytecode Transformation and Graph Compilation." ASPLOS. [PDF]
- Reed et al. 2022. "torch.fx: Practical Program Capture and Transformation for Deep Learning." arXiv 2112.08429. [PDF]
- "GraphMend: Code Transformations for Fixing Graph Breaks in PyTorch 2." 2025. arXiv 2509.16248. [PDF]
- "Flex Attention: A Programming Model for Generating Optimized Attention Kernels." 2024. arXiv 2412.05496. [PDF]
- Kwon et al. 2023. "Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)." arXiv 2309.06180. [PDF]
- "PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch." 2025. arXiv 2503.19779. [PDF]
- "Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel." 2026. arXiv 2604.13327. [PDF]
- "Mirage: Mega-Kernelizing Tensor Programs with Persistent Kernels." 2025. arXiv 2512.22219. [PDF]
- "Evaluating Cross-Architecture Performance Modeling Using StableHLO." 2026. arXiv 2604.12090. [PDF]

### GPU Architecture & Scheduling

- "GPU Determinism Survey." 2024. arXiv 2408.05148. [PDF]
- "Hummingbird: SLO-Oriented GPU Scheduling." 2026. arXiv 2601.04071. [PDF]
- Dao et al. 2022. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." arXiv 2205.14135. [PDF]
- Dao. 2023. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv 2307.08691. [PDF]
- Dao et al. 2024. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv 2407.08608. [PDF]
- Jooybar et al. 2013. "GPUDet: A Deterministic GPU Architecture." ASPLOS. [PDF]
- Park et al. 2021. "GPUReplay: A 50-KB GPU Stack for Client ML." arXiv 2105.05085. [PDF]
- Tanasic et al. 2014. "Enabling Preemptive Multiprogramming on GPUs." ISCA. [PDF]
- "GPU Context-Aware Preemptive Scheduling." 2024. ECRTS. [PDF]
- Fan et al. 2025. "GPU Preemptive Scheduling Made General and Efficient." USENIX ATC. [PDF]
- Capodieci et al. 2018. "Deadline-Based Scheduling for GPU with Preemption Support." RTSS. [PDF]
- Li et al. 2019. "Evaluating Modern GPU Interconnect: PCIe, NVLink, NV-Switch and GPUDirect." IEEE TPDS. [PDF]
- Li et al. 2022. "MISO: Exploiting Multi-Instance GPU Capability on Multi-Tenant GPU Clusters." SoCC. [PDF]
- Li et al. 2024. "GPU-to-GPU Communication on Supercomputer Interconnects." arXiv 2408.14090. [PDF]
- Olmedo et al. 2020. "Dissecting the CUDA Scheduling Hierarchy." RTAS. [PDF]
- Zhao et al. 2023. "Effectively Scheduling Computational Graphs of Deep Neural Networks." OSDI. [PDF]
- "Demystifying NCCL: In-Depth Analysis." 2025. arXiv 2507.04786. [PDF]
- Cai et al. 2020. "Synthesizing Optimal Collective Algorithms." arXiv 2008.08708. [PDF]
- "Monitoring Collective Communication Among GPUs." 2021. arXiv 2110.10401. [PDF]

### Checkpointing

- "CRIUgpu: Transparent Checkpointing of GPU-Accelerated Workloads." 2025. arXiv 2502.16631. [PDF]
- Chen et al. 2016. "Training Deep Nets with Sublinear Memory Cost." arXiv 1604.06174. [PDF]
- "PhoenixOS: Concurrent OS-Level GPU Checkpoint and Restore." 2024. arXiv 2405.12079. [PDF]

### Debuggers

- O'Callahan et al. 2017. "Engineering Record And Replay For Deployability." USENIX ATC. [PDF]
- O'Callahan et al. 2017. "rr Extended Technical Report." arXiv 1705.05937. [PDF]
- "Architecture of Open Source Applications, Volume 2: GDB Chapter." [PDF]

### Systems Observability & eBPF

- Huang et al. 2025. "NeutriNo: Fine-grained GPU Kernel Profiling via Programmable Probing." USENIX OSDI. [PDF]
- McCanne & Jacobson. 1993. "The BSD Packet Filter: A New Architecture for User-level Packet Capture." USENIX. [PDF]
- Cassagnes et al. 2024. "The eBPF Runtime in the Linux Kernel." arXiv 2410.00026. [PDF]
- Gershuni et al. 2019. "Simple and Precise Static Analysis of Untrusted Linux Kernel Extensions." PLDI. [PDF]
- "DepSurf: Revealing the Unstable Foundations of eBPF-Based Kernel Extensions." 2025. EuroSys. [PDF]
- Zheng et al. 2025. "ProfInfer: An eBPF-based Fine-Grained LLM Inference Profiler." arXiv 2601.20755. [PDF]
- Malony et al. 2011. "Parallel Performance Measurement of Heterogeneous Parallel Systems with GPUs." ICPP. [PDF]
- Gregg. 2017. "Visualizing Performance with Flame Graphs." USENIX ATC. [PDF/Slides]

### RTOS & Real-Time Systems

- Klein et al. 2009. "seL4: Formal Verification of an OS Kernel." SOSP. [PDF]
- Reghenzani et al. 2019. "The Real-Time Linux Kernel: A Survey on PREEMPT_RT." ACM Computing Surveys. [PDF]
- "Performance Evaluation of Xenomai 3." [PDF]

### LLM Tool Use & UX

- Patil et al. 2023. "Gorilla: Large Language Model Connected with Massive APIs." arXiv 2305.15334. [PDF]
- Yao et al. 2023. "ReAct: Synergizing Reasoning and Acting in Language Models." arXiv 2210.03629. [PDF]
- Schick et al. 2023. "Toolformer: Language Models Can Teach Themselves to Use Tools." arXiv 2302.04761. [PDF]

### TUI

- Bui et al. 2026. "Building Effective AI Coding Agents for the Terminal." arXiv 2603.05344. [PDF]

---

## Reference Implementations Analyzed (47 repos)

### ML Interpretability & Surgery
nnsight, TransformerLens, pyvene, CircuitsVis, nnterp, transformer-debugger (OpenAI), inseq, SAELens, baukit, rome, repeng, sae (EleutherAI), Automatic-Circuit-Discovery, tuned-lens, transformer-utils, steering-vectors, LIT (Google PAIR), ecco

### Debugger & Protocol
rr, debug-adapter-protocol, mcp-spec, mcp-servers

### TUI
ratatui, textual, taskwarrior-tui, trippy, git-cliff

### GPU/CUDA Infrastructure
cuda-samples, cuda-checkpoint, nccl, open-gpu-kernel-modules, DCGM, flash-attention, triton

### ML Frameworks
pytorch, jax, tinygrad, vllm

### Systems Observability
bcc, bpftrace, libbpf-bootstrap, perfetto

### Profilers
py-spy, scalene

### Build Infrastructure
pyo3, safetensors, ROCm

---

## What We're Asking

Given everything above — the architecture we're converging on, what we've learned from reading source code and papers, the hard questions we've identified — **how should we build this?**

Specifically:
1. What's wrong with our proposed architecture? What are we missing?
2. How should we resolve each of the 11 hard questions?
3. What should the build order be?
4. What are the biggest risks and how do we mitigate them?
5. Are there patterns, papers, or systems we should know about that aren't in our bibliography?
6. What's the minimum viable version that would be useful enough to validate the concept?

We follow JSMNTL methodology: lit review → written plan → TCK specs (Gherkin) → red tests → implementation → review. We are at the transition from lit review to written plan. This response should help us write that plan.
