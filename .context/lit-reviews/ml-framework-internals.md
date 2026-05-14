---
topic: ML framework internals — PyTorch dispatcher/autograd/compile, JAX/XLA, TensorFlow, tinygrad, MLIR, Triton
status: draft
created: 2026-05-14
sources: PyTorch docs/blogs, JAX docs, TensorFlow docs, tinygrad docs, MLIR docs, Triton
---

# ML Framework Internals: Lit Review

How major ML frameworks execute computation, their compilation pipelines, and extension/interception points.

## PyTorch Internals

### ATen Dispatcher
- Central routing mechanism: DispatchKey (CPU, CUDA, Autograd, Sparse, etc.) + DispatchKeySet (bit vector per tensor)
- Call chain: torch.add(a,b) -> pybind11 -> ATen at::add() -> Dispatcher lookup (DispatchKeySet) -> Backend kernel -> C10 storage -> GPU execution
- Boxing/Unboxing for generic dispatch through IValue containers
- **Interception point**: hook dispatch keys to intercept all tensor operations before GPU execution

### Autograd Engine
- Dynamic DAG of Node/Edge objects built during forward pass
- Node = grad_fn (differentiable op), Edge = connection with function pointer + input number
- Backward: topological sort via topological_nr_, execute in reverse order, gradients accumulate at leaves via +=
- sequence_nr_ for ordering, gradient buffers as unordered_map from nodes to variable lists
- **Interception**: register_full_backward_hook (module-level), tensor.register_hook (tensor-level)

### torch.compile Pipeline
1. **TorchDynamo** (frontend): custom Python bytecode interpreter, traces arbitrary Python -> torch.fx graphs. Captures ~99% of graphs correctly (vs TorchScript's 50%).
2. **torch.fx**: functional graph IR of ops and data flow. Inspectable via GraphModule.code. Debuggable with print_readable() and node inspection.
3. **AOTAutograd**: separates forward/backward graphs, performs functionalization (mutations -> functional updates)
4. **TorchInductor** (backend): lowers FX graph to Triton kernels (CUDA) or C++. Operator fusion, memory optimization.
5. **Graph breaks**: when Dynamo encounters unsupported ops, ends current graph, executes eagerly, starts new graph. Major performance culprit.

### c10 Library: TensorImpl & DispatchKeySet
- TensorImpl: holds storage_, metadata (sizes/strides), dtype_, device_, DispatchKeySet
- DispatchKeySet: bit vector of applicable implementations. key_set() method returns the set.
- Sparsity, quantization detection via non-virtual methods querying DispatchKeySet

### torch.distributed
- c10d: ProcessGroup (abstract interface for NCCL/Gloo/UMAP backends), collective (all_reduce, all_gather, reduce_scatter) and P2P (send, recv) APIs
- DDP: Reducer C++ component managing gradient bucketing into GradBucket objects, overlaps communication with backward
- FSDP: shards parameters and gradients across processes via ProcessGroup
- SymmetricMemory: direct GPU-to-GPU access (NVLink) via NCCLSymmetricMemory, bypassing standard collective buffers

### Custom Operators
- TORCH_LIBRARY / STABLE_TORCH_LIBRARY macro for registration
- C++ functions forward to CUDA kernels (.cu files)
- Build ahead-of-time (setuptools) or JIT (torch.utils.cpp_extension.load())

### GPU Memory: CUDACachingAllocator
- Allocates blocks larger than requested (2MB for <1MB, 20MB for <10MB) for reuse via splitting
- Causes fragmentation if varying sizes allocated/freed. Cannot move fragmented blocks.
- expandable_segments=True: uses CUDA virtual memory APIs, reserves large VA space, maps/unmaps physical memory on demand
- GPU OOM can occur at <70% utilization due to fragmentation

### TorchScript -> torch.compile History
- TorchScript: tracing + scripting modes, struggled to acquire graphs (50%), significant overhead, required code changes
- torch.compile: Dynamo acquires 99% correctly, no code changes, handles arbitrary Python via graph breaks
- Reflects move from static graph-centric to dynamic eager + optional compilation

## JAX Internals

### XLA Compilation Pipeline
- Tracing: Python function -> jaxpr (JAX primitive language, functional IR)
- Lowering: jaxpr -> StableHLO -> HLO (XLA's format)
- HLO: hardware-independent + hardware-dependent optimizations
- Final compilation to device-specific machine code
- StableHLO: portability layer between frameworks (JAX, TF, PyTorch) and compilers (XLA, IREE)

### Function Transformations
- **vmap**: vectorizes single-example functions across batch dimensions. Fused operations.
- **pmap**: SPMD programs. Applies jit(), compiles with XLA, executes in parallel across GPUs/TPU cores. Now implemented via jit() + shard_map().
- **shard_map**: modern replacement for pmap. More flexible, composable, operates on device mesh shards. Eager for debugging.
- **Composability**: transformations layer naturally: vmap(jit(grad(f))). Key differentiator.

### Pallas (Custom GPU/TPU Kernels)
- Kernels defined in Python using JAX primitives
- Lowered to Mosaic (TPU), Triton (GPU), or Mosaic GPU
- Grid-based parallelism, explicit memory hierarchy management (tiling, HBM<->VMEM/SMEM transfers)

### JAX Debugging
- jax.experimental.io_callback() for impure functions
- jax.debug.callback(), jax.debug.print(), jax.debug.breakpoint()
- **Caveat**: adding debug statements changes computation sent to XLA (different fusions -> numeric discrepancies)
- Control flow: lax.fori_loop, lax.while_loop, lax.cond — replace Python if/for/while for JIT

### JAX vs PyTorch
- JAX: functional programming, trace-time compilation, composable transformations
- PyTorch: imperative/OOP, runtime graph construction, hooks for interception
- JAX harder to debug inside jit (must use callbacks). PyTorch eager execution enables step-through.
- **For us**: PyTorch's eager execution + hooks are far more natural for a step-through debugger

## TensorFlow

### tf.function & AutoGraph
- Traces Python functions into TF graphs. ConcreteFunction for each unique input shape/dtype.
- AutoGraph: Python if/else -> tf.cond, while -> tf.while_loop
- Graph optimization: constant folding, operation independence analysis, common subexpression elimination

### XLA Integration
- TF graph -> canonicalization -> cluster identification -> TF-to-HLO -> XLA HLO IR -> optimizations -> device code
- TF 2.x defaults to eager. @tf.function + jit_compile=True triggers XLA.

### GradientTape
- Records operations in a "tape" during eager execution
- persistent=True keeps tape for multiple calls (increases memory)
- Debug in eager mode, then decorate with @tf.function for optimization

## tinygrad

### Architecture
- **Lazy evaluation**: every op appends node to in-memory DAG. Computation only on Tensor.realize() or numpy conversion.
- **UOp (Unified Operation)**: single IR node throughout entire stack — high-level tensor graph -> scheduled kernel graph -> linearized instructions
- **Structural interning**: identical UOps are same Python object (O(1) equality, global dedup)
- **Scheduler**: converts DAG of UOps into list of ExecItem. One ExecItem = one GPU kernel.
- **RISC-like**: only 12-13 primitive ops needed for modern networks

### Backend System
- Pluggable: CUDA, Metal, OpenCL, WebGPU, Qualcomm, AMD, CPU
- Auto-detects best device
- Mesa NIR backend (v0.12, Jan 2026): full free software stack

### Why tinygrad matters
- Exposes ML compilation from first principles in ~10K LOC
- Shows that the "capture -> schedule -> codegen" pipeline is universal across all frameworks
- UOp model is the clearest minimal example of how tensor IR works

## Cross-Cutting Infrastructure

### MLIR (Multi-Level Intermediate Representation)
- LLVM project, modular/extensible IR for domain-specific compilers
- Models computations at various abstraction levels, progressively lowering toward machine code
- **StableHLO**: portability layer between frameworks and compilers
- **TOSA**: portable, quantization-friendly operators targeting CPU/GPU/NPU

### Triton (OpenAI)
- GPU programming language: Python -> Triton-IR -> Triton-GPU IR -> LLVM-IR -> PTX -> cubin
- Vertical integration with torch.compile: Dynamo captures Python, Inductor lowers to Triton kernels
- FlexAttention (PyTorch 2.5): generates fused FlashAttention kernels from pure Python via Triton

### CUDA Graphs + Persistent Kernels + Event Tensors
- CUDA Graphs: capture kernel sequence, replay with minimal overhead (5x speedup)
- Persistent kernels (2025): fuse operators into single kernel, eliminate launch overhead
- Event Tensors (2025): semaphore-based synchronization as first-class tensors in compiler IR

## Key Interception Points for rocket_surgeon

1. **PyTorch dispatcher**: hook dispatch keys before GPU execution
2. **Autograd nodes**: full_backward_hooks for gradient flow/accumulation order
3. **torch.compile FX graphs**: transform GraphModule before Inductor codegen
4. **c10d ProcessGroup**: instrument collective communication
5. **CUDA kernels**: CUDA Graphs + Event Tensors for persistent kernel capture
6. **NCCL operations**: trace ring/tree topology execution

## Execution Models to Debug
- **PyTorch**: dynamic eager + optional JIT. Tape-based autograd with topological sort.
- **JAX**: trace-time compilation. Functional primitives composable. Control flow constraints.
- **TensorFlow**: AutoGraph + XLA. GradientTape eager execution.
- **tinygrad**: lazy evaluation DAG. Scheduler -> kernel batches. UOp IR.

**Universal pattern across all frameworks**: capture -> optimize -> execute. Debugger's power comes from instrumenting at capture and optimization stages, before computation scatters across GPUs.

## Sources

- PyTorch dispatcher walkthrough (github wiki)
- Red Hat: PyTorch call stack deep dive, Understanding ATen, Autograd engine
- torch.compile: official tutorial, ezyang blog (Aug 2025), GraphMend paper, vLLM blog
- c10: TensorImpl.h, DispatchKeySet.h (pytorch github)
- JAX: aot.html, StableHLO tutorial, Pallas docs, callbacks docs, control flow docs
- TensorFlow: tf.function guide, XLA tutorial, GradientTape guide
- tinygrad: docs.tinygrad.org, tinygrad-notes (mesozoic-egg.github.io)
- MLIR: overview articles, StableHLO (openxla.org), TOSA dialect (mlir.llvm.org)
- Triton: PyTorch blog (compilation stages), github triton-lang/triton
- CUDA Graphs: NVIDIA blog, PyTorch blog, PyGraph paper
