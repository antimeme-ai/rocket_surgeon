# Framework Compilation Pipeline Analysis

Analysis of ML framework internals for rocket_surgeon instrumentation design.
Sources: tinygrad (source), Triton (source + Tillet2019), vLLM (source + Kwon2023),
torch.compile (Ansel2024 + Reed2022), CUDA Graphs (Ghosh2025), FlexAttention (2024 paper).

---

## 1. tinygrad Deep Dive

### 1.1 The UOp Universal IR

Everything in tinygrad is a UOp. One data structure for tensor graphs, scheduled kernels,
linearized programs, compiled binaries, and execution. The definition
(`tinygrad/uop/ops.py`):

```python
class UOp:
  op: Ops        # enum from FastEnum (IntEnum subclass)
  dtype: DType
  src: tuple[UOp, ...]
  arg: Any
  tag: Any       # used for beam search marking
```

Key property: **global deduplication**. `UOpMetaClass.ucache` is a `WeakKeyDictionary` that
deduplicates all UOps by `(op, dtype, src, arg)`. Two structurally identical subgraphs
collapse to the same Python object. This means graph identity IS structural equality.

The Ops enum (`tinygrad/uop/__init__.py`, 110 ops) is ordered to control toposort:

| Category | Examples | Purpose |
|----------|----------|---------|
| Defines | DEFINE_VAR, SPECIAL, DEFINE_LOCAL | GPU thread/block dims, vars |
| Non-op | PARAM, FUNCTION, CALL, PROGRAM, LINEAR, SOURCE, BINARY | Compilation artifacts |
| Structure | SINK, AFTER, GROUP | Dependency/grouping |
| Load/Store | INDEX, LOAD, STORE | Memory ops |
| Math | ADD, MUL, EXP2, WHERE, WMMA... (ALU) | Compute |
| Control | BARRIER, RANGE, IF, END, ENDIF | Control flow |
| Tensor graph | RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP | Movement ops |
| Reduce | REDUCE, ALLREDUCE | Reductions |

`GroupOp` provides semantic categories: `ALU`, `Elementwise`, `Movement`, `Commutative`,
`Associative`, `Idempotent`, `Comparison`. These are used by pattern matchers.

### 1.2 Pattern Matcher System

All transformations in tinygrad are `PatternMatcher` + `UPat` rewrites applied via
`graph_rewrite()`. This is the single mechanism for: optimization, scheduling,
codegen, lowering, compilation, and execution dispatch.

`UPat` is a structural pattern over UOps:
- Matches on `op`, `dtype`, `src` patterns, `arg`, named captures
- Supports `allow_any_len`, `.or_casted()`, `.f(op)` chaining
- `PatternMatcher` is a list of `(UPat, callback)` pairs
- Matchers compose with `+` operator (concatenation)

`graph_rewrite(uop, pm, ctx=None, name=None, bottom_up=False)` walks the graph
and applies the pattern matcher until fixpoint. Optional `ctx` is threaded through
rewrites. Optional `bottom_up` for bottom-up traversal. Optional `enter_calls` to
recurse into CALL nodes (used for scheduling).

### 1.3 Tensor Layer

`Tensor` (`tinygrad/tensor.py`) is a thin wrapper:

```python
class Tensor:
  __slots__ = "uop", "requires_grad", "grad"
```

All tensors tracked in `all_tensors` (WeakValueDictionary). `_apply_map_to_tensors`
does bulk UOp graph substitution across every live tensor -- this is how schedule
results get wired back. `_apply_uop` creates new Tensors from UOp operations.

Lazy evaluation: tensor ops build UOp graph nodes but no computation happens.
`Tensor.realize()` triggers the full pipeline.

### 1.4 Scheduling Pipeline

The schedule converts a lazy UOp tensor graph into a LINEAR (sequence of kernel calls).

**Entry point**: `create_linear_with_vars(big_sink)` in `tinygrad/schedule/__init__.py`.

Pipeline:

1. **`pm_schedule`**: Pattern matcher wrapping `lower_sink_to_linear()`.
   Calls `get_kernel_graph(function)` (from `schedule/rangeify.py`) to convert the
   tensor-level function UOp into a kernel dependency graph. Then
   `create_schedule(sched_sink)` topologically sorts kernel dependencies via in-degree
   counting (Kahn's algorithm) to produce a flat `LINEAR` of CALL nodes.

2. **`pm_resolve_linear_call`**: Resolves CALL(LINEAR) by substituting PARAMs with
   actual BUFFERs (`pm_post_sched_cache`) and flattening nested LINEARs.

3. **`memory_plan_rewrite(linear, held_bufs)`**: TLSF allocator-based memory planning.
   Computes buffer lifetimes, groups by device+lane (copy vs compute), sub-allocates
   from shared arenas. Replaces individual BUFFERs with BUFFER_VIEW into arenas.

4. **JIT capture**: If `capturing` is active (2nd call in `TinyJit`), the schedule is
   captured instead of executed.

Schedule caching: `schedule_cache` maps `function.key` (bytes) to the resulting LINEAR,
so repeated identical subgraphs skip re-scheduling.

### 1.5 Rangeify and Kernel Formation

`schedule/rangeify.py` contains the pattern matchers that transform tensor-graph UOps
into kernel-level UOps:

- **`pm_mops`**: Movement ops on INDEX -- folds RESHAPE/PERMUTE/EXPAND/PAD/SHRINK/FLIP
  into index arithmetic. Moves movement ops through AFTER nodes.
- **`pm_syntactic_sugar`**: Concatenates nested INDEX ops, early-rangeifies elementwise ops.
- **`pm_store_ranges`**: Adds RANGE loops around STORE ops based on tensor shapes.
- **`get_kernel_graph(function)`**: The top-level function that converts a tensor
  FUNCTION UOp into a scheduled kernel graph with AFTER dependency edges.

### 1.6 Codegen Pipeline

`tinygrad/codegen/__init__.py` defines `full_rewrite_to_sink()` -- the monolithic
codegen pipeline as a chain of ~20 `graph_rewrite` passes:

1. **Preprocessing**: `pm_mops + pm_syntactic_sugar + pm_store_ranges` (bottom-up)
2. **Load collapse**: `pm_load_collapse`
3. **Range splitting**: `pm_split_ranges + pm_flatten_range`
4. **Symbolic simplification**: `sym + pm_flatten_range`
5. **Range simplification**: `pm_flatten_range + pm_simplify_ranges`
6. **Optimization**: `apply_opts(sink, ren, beam=...)` -- BEAM search or hand-coded opts
7. **Expander**: `pm_pre_expander + pm_group_for_reduce + expander`
8. **Local buffers**: `pm_add_buffers_local + rangeify_codegen`
9. **Reduce removal**: `pm_reduce + gep_pushing` (with `ReduceContext`)
10. **GPU dims**: `pm_add_gpudims`
11. **Add loads**: `pm_add_loads`
12. **Image buffers**: `pm_make_images` (for QCOM/CL targets)
13. **Devectorize**: `devectorize + load_store_folding + correct_load_store + load_store_indexing`
14. **Lower index dtype**: `pm_lower_index_dtype`
15. **Renderer pre-matcher**: `ren.pre_matcher`
16. **Decompositions**: Late rewrite patterns, dtype decomps, transcendentals
17. **Gate movement**: `pm_move_gates_from_index`
18. **Final rewrite**: `pm_decomp + pm_render + extra_matcher + pm_split_ends`
19. **Linearize**: `pm_add_control_flow` (IF/ENDIF insertion)

Then `pm_to_program` converts the codegen result through stages:
```
PROGRAM(SINK, DEVICE) -> do_linearize -> LINEAR added
PROGRAM(SINK, DEVICE, LINEAR) -> do_estimates -> estimates computed
PROGRAM(..., LINEAR(INS)) -> do_assemble -> BINARY (for asm renderers)
PROGRAM(..., LINEAR) -> do_render -> SOURCE (text)
PROGRAM(..., SOURCE) -> do_compile -> BINARY (compiled)
```

`to_program()` caches by `(ast.key, renderer_type, target, config_tuple)`.

### 1.7 Compilation and Execution

`tinygrad/engine/realize.py`:

- **`compile_linear(linear)`**: Chain of pattern matcher rewrites:
  `pm_validate -> pm_beam_search -> pm_compile -> pm_optimize_local_size`

- **`run_linear(linear, var_vals)`**: Iterates LINEAR.src CALL nodes, dispatches via
  `pm_exec` which maps to: `exec_view`, `exec_copy`, `exec_kernel`, `exec_encdec`,
  `exec_graph`, `exec_validate`.

- **`get_runtime(key, device)`**: Caches compiled programs. Loads device runtime from
  `runtime/ops_*.py` lazily via `Device` singleton.

### 1.8 JIT System

`tinygrad/engine/jit.py`:

- `TinyJit`: Captures on 2nd call (1st call runs eagerly to collect shape info).
  `jit_lower()`: parametrize -> compile -> `memory_plan_rewrite` -> `graph_split_rewrite`.

- `GraphRunner`: Manages CUDA-graph-like kernel batching. `graph_split_rewrite`
  partitions kernels into graph-compatible groups based on dependency analysis.
  Each `GraphRunner` captures a batch of kernels for replay.

- `CapturedJit`: Holds compiled linear for replay. Tracks which UOps were written
  (outputs).

### 1.9 Device Abstraction

`tinygrad/device.py`:

- **`Compiled`**: device name, allocator, renderer, runtime, graph (optional).
- **`Buffer`**: size, dtype, device, base/offset for views, allocate/deallocate/copyin/copyout.
- **`LRUAllocator`**: Caches freed buffers by `(size, options)`. `_offset` method for
  sub-allocation support.
- **`Device`**: Singleton, lazy device loading from `runtime/ops_*.py` files.

### 1.10 Instrumentation-Relevant Properties

**UOp deduplication**: Graph identity = structural equality. Any instrumented wrapper
must preserve or explicitly break dedup semantics.

**Pattern matcher extensibility**: All transforms are composable PatternMatcher chains.
Instrumentation can be injected as additional patterns prepended/appended to existing
matchers, or as entirely new graph_rewrite passes.

**Lazy evaluation boundary**: The realize() call is the single materialization point.
Everything before is graph construction; everything after is execution. Instrumentation
must decide which side of this boundary to operate on.

**Explicit graph structure**: Unlike PyTorch's implicit autograd graph, tinygrad's UOp
graph is fully explicit and inspectable at every stage. The `toposort()` method and
`tag` field (currently used for beam search) provide natural extension points.

**JIT capture**: TinyJit's capture mechanism provides a natural checkpoint/replay
model. The `capturing` list and `CapturedJit.add_linear()` show how to intercept
the schedule-to-execution boundary.

---

## 2. Triton Compilation Pipeline

### 2.1 Architecture Overview

Triton compiles Python kernel functions to GPU binary through a staged MLIR pipeline.
The backend is pluggable (NVIDIA CUDA, AMD HIP) with a common `BaseBackend` interface.

Source: `triton/compiler/compiler.py` `compile()` function + `nvidia/backend/compiler.py`.

### 2.2 Frontend: Python AST to Triton-IR (TTIR)

`ASTSource.make_ir()` calls `ast_to_ttir()` from `code_generator.py`:

- Python AST is walked by a custom `CodeGenerator` visitor
- Triton language constructs (`tl.load`, `tl.store`, `tl.dot`, `tl.program_id`) become
  MLIR operations in the Triton dialect
- Type inference happens during AST walk (every `base_value` carries a `base_type`)
- `constexpr` values are evaluated at compile time, enabling specialization
- `JITFunction` decorators handle signature hashing, caching, autotuning

### 2.3 NVIDIA Backend Stages

`CUDABackend.add_stages()` defines the pipeline:

```
ttir -> ttgir -> llir -> ptx -> cubin
```

**Stage 1: make_ttir** (Triton-IR optimization):
- Inlining, canonicalization, combine, reorder_broadcast, CSE, DCE, loop unroll.
- On pre-Hopper: rewrites tensor descriptors to pointers.

**Stage 2: make_ttgir** (Triton-GPU-IR -- the core optimization stage):
- `convert_to_ttgpuir`: Assigns data layouts (blocked, shared, distributed)
- Coalescing: Ensures memory access patterns align with warp structure
- `f32_dot_tc`: Emulates TF32 tensor core operations
- `plan_cta`: CTA (thread block) planning
- Layout conversion removal (multiple passes)
- `accelerate_matmul`: Maps dots to tensor core operations
- `optimize_dot_operands`: Optimizes operand layouts for tensor cores
- Architecture-specific:
  - **Hopper (sm_9x)**: Warp specialization (`hopper_warpspec`), TMA lowering,
    loop fusion, scheduling, pipelining
  - **Blackwell (sm_10x)**: TMEM (tensor memory) allocation/optimization,
    partition warp optimization, 2-CTA matmul support
  - **Ampere (sm_8x)**: Prefetching, async copy coalescing
- Fence insertion, MMA lowering, SCCP, CSE

**Stage 3: make_llir** (TTGIR to LLVM-IR):
- Shared memory allocation
- Tensor memory allocation (Blackwell)
- `to_llvmir`: Converts MLIR to LLVM-IR MLIR dialect
- Warp specialization lowering to LLVM
- NVGPU ops to LLVM intrinsics
- NVVM to LLVM conversion
- LLVM optimization (O3)

**Stage 4: make_ptx** (LLVM-IR to PTX):
- `llvm.translate_to_asm()` with target triple, features, fusion flags
- PTX version patching, debug flag management

**Stage 5: make_cubin** (PTX to cubin):
- External `ptxas` invocation with arch-specific flags
- Register allocation, line info, FMA control

### 2.4 Caching Architecture

Multi-level caching:
- **Compilation cache**: SHA256 of `(source, backend, options, env_vars)` -> file cache
- **Override manager**: User can override any intermediate IR by hash
- **Dump manager**: IR at each stage can be dumped for inspection

`knobs.runtime.add_stages_inspection_hook` allows injecting custom stages -- this is a
natural instrumentation point.

### 2.5 Instrumentation Points

Triton has built-in instrumentation support:
- `CUDABackend.instrumentation` class variable -- can `patch()` at specific pipeline points
- `instrumentation_mode` option supports: `fpsan` (FP sanitizer), `consan` (constraint sanitizer),
  `gsan` (global sanitizer), `iisan` (instruction-level sanitizer)
- `knobs.compilation.listener` callback receives `(src, metadata, metadata_group, times, cache_hit)`

### 2.6 Tile-Based Programming Model (from Tillet2019 paper)

Triton's programming model is SPMD at the tile level, not the thread level:
- Programs operate on tiles (multi-dimensional blocks of data)
- `tl.program_id(axis)` provides block index
- `tl.arange(0, BLOCK_SIZE)` creates tile indices
- `tl.load(ptr + offsets, mask)` and `tl.store(ptr + offsets, val, mask)` for masked tile I/O
- `tl.dot(a, b)` maps to tensor cores

The compiler manages thread-level decomposition, shared memory, synchronization.
Machine-independent IR (Triton-IR) handles tile-level semantics; machine-dependent
passes (in TTGIR) handle the actual hardware mapping.

---

## 3. vLLM Multi-GPU Architecture

### 3.1 Distributed State Management

`vllm/distributed/parallel_state.py`:

Central abstraction is `GroupCoordinator`:
- Wraps PyTorch `ProcessGroup` for a group of processes
- Manages both CPU group (`gloo` backend) and device group (`nccl` backend)
- Tracks: `rank`, `local_rank`, `rank_in_group`, `world_size`
- Creates `DeviceCommunicatorBase` for optimized device communication
- Optional `MessageQueue` broadcaster for shared memory IPC

Initialization flow:
1. `init_distributed_environment()` -- sets up torch.distributed
2. `initialize_model_parallel()` -- creates TP/PP groups
3. Code uses GroupCoordinator for all collectives
4. `destroy_model_parallel()` / `destroy_distributed_environment()` -- cleanup

### 3.2 Communication Operations

Custom ops registered with `direct_register_custom_op()`:
- `all_reduce(tensor, group_name)`: Out-of-place all-reduce
- `reduce_scatter(tensor, dim, world_size, group_name)`
- `all_gather(tensor, dim, world_size, group_name)`
- `fused_scaled_matmul_reduce_scatter`: FP8 matmul + reduce_scatter fusion

Each op has a `fake_impl` for torch.compile tracing (returns `torch.empty_like`).

`GroupCoordinator` methods:
- `_all_reduce_out_place`, `_reduce_scatter_out_place`, `_all_gather_out_place`
- P2P send/recv via `Handle` protocol (async with `is_completed()`/`wait()`)

Device communicators hierarchy:
- `DeviceCommunicatorBase` (abstract)
- `pynccl.py`: Python NCCL bindings
- `custom_all_reduce.py`: Custom all-reduce for small messages
- `shm_broadcast.py`: Shared memory for metadata broadcasting
- `cuda_communicator.py`: CUDA-specific optimizations

### 3.3 PagedAttention and KV Cache (from Kwon2023 paper)

Core innovation: virtual memory paging for KV cache.
- KV cache divided into fixed-size blocks (e.g., 16 tokens per block)
- Block table maps logical blocks to physical blocks (like page tables)
- Enables: non-contiguous storage, dynamic allocation, copy-on-write sharing
- Eliminates pre-allocation waste (up to 3.5x throughput over baselines)

Scheduling: centralized scheduler on CPU decides which sequences to run.
Workers are SPMD: all execute the same model partition, synchronized via NCCL.

### 3.4 Expert Load Balancing

`distributed/eplb/` -- Expert Parallel Load Balancer:
- For MoE models with expert parallelism
- Dynamic redistribution of experts across devices based on load statistics

### 3.5 Elastic Expert Parallelism

`distributed/elastic_ep/` -- Elastic expert parallelism:
- Dynamic scaling of expert parallel degree at runtime

### 3.6 Multi-GPU Execution Model

Architecture:
- Single centralized scheduler process (CPU-based)
- Multiple worker processes (one per GPU), SPMD execution
- Workers execute identical model code; collectives handle synchronization
- Tensor parallelism: shards weights across GPUs within a layer
- Pipeline parallelism: shards layers across GPUs
- Expert parallelism: shards MoE experts across GPUs

Communication patterns:
- TP: all-reduce after each layer's linear ops
- PP: point-to-point send/recv at pipeline stage boundaries
- EP: all-to-all for expert routing

---

## 4. torch.compile Pipeline

### 4.1 TorchDynamo (from Ansel2024)

Frame evaluation hook using PEP 523 (`_PyInterpreterState_SetEvalFrameFunc`):

1. **Guard generation**: For each input, generates guards (type, shape, value checks).
   30+ guard types. Guards determine when cached compiled code is valid.

2. **Symbolic evaluation**: Python bytecodes executed symbolically.
   `TensorVariable` wraps tensors, `SymNodeVariable` wraps symbolic shapes.
   Side effects tracked (global mutations, list mutations).

3. **Graph capture**: Builds `torch.fx.Graph` from symbolic execution.
   Pure tensor operations become graph nodes. Python control flow, data-dependent
   branches, and unsupported ops cause "graph breaks."

4. **Graph breaks**: When Dynamo encounters code it can't trace:
   - Splits into subgraphs
   - Creates "continuation functions" for the un-traceable code
   - Resumes tracing after the break
   - Each subgraph compiled independently

### 4.2 torch.fx IR (from Reed2022)

Six-instruction IR:
- `placeholder`: function input
- `get_attr`: access module attribute
- `call_function`: call a Python callable
- `call_method`: call a method on a value
- `call_module`: call an `nn.Module`
- `output`: return value

`Graph` is a doubly-linked list of `Node` objects. `Node.args` and `Node.kwargs`
contain references to other Nodes (def-use chains). `GraphModule` wraps a Module +
Graph for standard PyTorch interop.

Key capabilities:
- `graph.eliminate_dead_code()`, `graph.lint()`
- Shape propagation via `ShapeProp`
- Subgraph matching via `SubgraphMatcher`
- Passes compose as Graph -> Graph functions

### 4.3 AOTAutograd

Traces forward AND backward on FakeTensors (no data, only metadata):
- Joint graph captures both forward and backward
- Min-cut partitioning decides what to save vs. recompute
- Produces separate forward and backward FX graphs
- Both submitted to TorchInductor independently

### 4.4 TorchInductor (from Ansel2024)

FX Graph -> Loop-level IR -> Generated code:

1. **Lowering**: FX graph nodes lowered to 54 loop-level IR primitives.
   "Define-by-run": IR is built by executing Python, not by pattern matching.

2. **Scheduling**: Fuses operations into kernels.
   - `can_fuse(node1, node2)`: Legal check
   - `score_fusion(node1, node2)`: Heuristic score
   - Iterative: fuse highest-scoring pairs until no more profitable fusions

3. **Code generation**:
   - GPU: Triton kernels (Python source generated, then compiled by Triton)
   - CPU: C++ with OpenMP
   - Templates for matmul, conv (e.g., `aten.mm -> triton_mm_template`)

4. **Autotuning**: For templated kernels, multiple configurations tried.
   Results cached by `(graph_hash, input_shapes)`.

### 4.5 FlexAttention (from 2024 paper)

Compiler-driven composable attention:
- User defines `score_mod(score, b, h, q_idx, kv_idx)` and `mask_mod(b, h, q_idx, kv_idx)`
- These are Python callables, traced by TorchDynamo into FX graphs
- `flex_attention()` lowers to handwritten Triton attention templates
- `BlockMask`: Block-level sparsity mask computed from `mask_mod`
  - Evaluates mask for each block, skips fully-masked blocks
  - 2x-8x speedup for sparse patterns (sliding window, causal, etc.)
- Compositions: `and_masks`, `or_masks` for combining mask_mods

### 4.6 Pipeline Integration

Full torch.compile flow:
```
Python code
  -> TorchDynamo (PEP 523 hook)
    -> Symbolic eval + Guard gen
    -> FX Graph (+ graph breaks -> subgraphs)
  -> AOTAutograd
    -> Joint forward+backward trace on FakeTensors
    -> Min-cut partition
    -> Separate forward/backward FX graphs
  -> TorchInductor
    -> Lower to loop-level IR
    -> Fuse into kernels
    -> Generate Triton/C++ code
  -> Triton compiler (for GPU kernels)
    -> TTIR -> TTGIR -> LLVM -> PTX -> cubin
  -> Runtime
    -> Guards check -> dispatch to compiled code or recompile
```

---

## 5. CUDA Graphs and PyGraph

### 5.1 CUDA Graphs Background

CUDA Graphs capture a sequence of GPU operations (kernel launches, memory copies) as a
graph, then replay the entire graph with a single CPU-side launch. Eliminates per-kernel
launch overhead (can be ~10us per kernel).

Problem: Standard CUDA Graph usage requires:
- Static shapes (no dynamic allocation)
- Fixed kernel arguments (pointers can't change)
- No CPU-GPU synchronization within the graph

### 5.2 PyGraph Contributions (from Ghosh2025)

Three problems and solutions:

**Problem 1: CG-oblivious code**
Many PyTorch ops don't account for CUDA Graph semantics (e.g., allocating new tensors).
- **CGCT (CG Code Transformations)**: Automatic transformations that make existing code
  CG-compatible. Identifies problematic patterns and rewrites them.

**Problem 2: Parameter copy overhead**
CUDA Graphs fix pointer addresses at capture time. When model parameters change
(optimizer step), all pointers must be re-copied into the graph.
- **Parameter Indirection**: Uses pointer-to-pointer scheme. Graph contains indirection
  buffers that point to actual parameter storage. Updating parameters only requires
  updating the indirection buffer, not re-capturing the graph.

**Problem 3: CGs can hurt performance**
Not all kernel sequences benefit from graph capture. Small kernels benefit; large
compute-bound kernels don't (launch overhead is negligible relative to kernel time).
- **Selective CG**: Cost-benefit profiling determines which subsequences to capture.
  Profiles kernel execution time + launch overhead. Only captures sequences where
  launch overhead dominates.

Result: 29% geometric mean speedup over PyTorch 2 CUDA Graph support.

### 5.3 Implications for Instrumentation

CUDA Graphs create a fundamental tension with instrumentation:
- Captured graphs are opaque -- can't insert probes between kernels
- Parameter indirection pattern is relevant for rocket_surgeon's activation modification
- Selective capture means some kernels remain individually launchable (instrumentable)
- Graph replay prevents step-through debugging within a captured sequence

---

## 6. Implications for rocket_surgeon Instrumentation

### 6.1 Where to Intercept: Framework-Specific Points

**tinygrad**:
1. **UOp graph construction** (pre-realize): Insert/modify UOp nodes before scheduling.
   Natural because graphs are explicit and inspectable.
2. **Pattern matcher injection**: Add custom `PatternMatcher` rules to existing
   `graph_rewrite` passes. E.g., prepend instrumentation patterns to `pm_compile`.
3. **Schedule boundary** (`create_linear_with_vars`): Intercept the LINEAR before
   compilation. Can inspect, modify, or replace kernel sequences.
4. **`pm_exec` dispatch**: Add custom exec handler for instrumentation ops.
5. **JIT capture** (`CapturedJit.add_linear`): Intercept the JIT's capture mechanism
   to record/replay with modifications.
6. **Buffer operations**: `Buffer.copyin`/`copyout`/`allocate` -- natural points for
   activation capture without modifying the computation graph.

**torch.compile**:
1. **Guard system**: Custom guards could trigger re-compilation with instrumented graphs.
2. **FX graph transforms**: Insert instrumentation nodes as FX graph passes between
   AOTAutograd and Inductor. `torch.fx.passes` provides the infrastructure.
3. **Inductor scheduler**: Custom fusion rules could prevent fusing instrumented ops.
4. **Triton template hooks**: `knobs.runtime.add_stages_inspection_hook` for inspecting
   generated kernels.
5. **AOTAutograd joint graph**: Intercept before min-cut to preserve observability.

**Triton**:
1. **Stage hooks**: `knobs.compilation.listener` for observing compilation.
   `knobs.runtime.add_stages_inspection_hook` for injecting custom stages.
2. **IR override**: `fn_override_manager` allows replacing any intermediate IR.
3. **Instrumentation modes**: Built-in `fpsan`/`consan`/`gsan`/`iisan` patterns
   show how to inject instrumentation at MLIR level.
4. **Backend.instrumentation.patch()**: Explicit patch points in the MLIR pipeline.

**vLLM**:
1. **GroupCoordinator**: Wrap or subclass for communication interception.
2. **Custom ops**: `direct_register_custom_op` pattern for adding instrumented collectives.
3. **Scheduler**: Centralized CPU scheduler is the control point for multi-GPU
   sequencing -- natural place for step-through control.
4. **KV cache block table**: PagedAttention's block table provides a natural
   indirection layer for cache inspection/modification.

### 6.2 The "Tick" Abstraction

A "tick" in rocket_surgeon must map to different things depending on the framework:

| Framework | One Tick | Granularity Control |
|-----------|----------|-------------------|
| tinygrad | One UOp in LINEAR | Pattern matcher can split/merge |
| torch.compile | One FX Node | Graph pass can adjust |
| Triton | One TTIR/TTGIR op | MLIR pass level |
| vLLM | One scheduler step | Scheduler iteration |
| CUDA Graph | One captured graph | Selective CG boundary |

For transformer internals specifically:
- **Layer level**: One transformer layer = one tick
- **Sub-layer level**: Attention / FFN / LayerNorm = one tick
- **Operation level**: Each matmul / softmax / activation = one tick
- **Element level**: Individual tensor elements (only for targeted surgery)

### 6.3 Multi-GPU Considerations

**DDP/FSDP**: All-reduce between gradient computation and parameter update.
Instrumentation must handle: (a) partial gradients per rank, (b) synchronized
stepping across ranks, (c) communication timing.

**Tensor Parallelism**: Sharded tensors. Viewing activations requires gathering
from all ranks. Modification requires coordinated scatter.

**Pipeline Parallelism**: Micro-batches in flight. Tick-by-tick stepping must
account for pipeline bubbles and inter-stage communication.

**MoE Expert Parallelism**: All-to-all routing. Instrumentation must handle:
(a) routing decisions (which tokens go where), (b) expert activations per rank,
(c) load balancing state.

### 6.4 Critical Design Decisions for rocket_surgeon

**1. IR Level**: tinygrad's UOp is the most natural target for a debugger because
it is a single universal IR. PyTorch has multiple IRs (FX, Inductor loop IR, Triton)
with information loss at each boundary. Recommendation: primary instrumentation at
UOp level for tinygrad-based models; FX graph level for PyTorch models.

**2. Lazy vs Eager boundary**: Instrumentation before realize() can modify the
computation graph (surgical intervention). Instrumentation after realize() can only
observe/modify data (activation viewing). Both are needed.

**3. Pattern Matcher as extension mechanism**: tinygrad's pattern matcher system
is the ideal extension point. Instrumentation rules expressed as `(UPat, callback)`
pairs compose naturally with existing optimization passes. The question is ordering:
should instrumentation run before or after optimization?

**4. CUDA Graph compatibility**: Must support both:
(a) Breaking CUDA Graphs for fine-grained stepping (disable graph capture)
(b) Operating within CUDA Graphs for low-overhead observation (parameter indirection)

**5. Communication interception**: For multi-GPU, must intercept NCCL collectives.
vLLM's `GroupCoordinator` pattern (wrapper with custom ops) is the right model.
Register instrumented versions of all-reduce/all-gather that can pause, inspect,
and modify tensors at the communication boundary.

**6. Memory planning awareness**: tinygrad's TLSF-based memory planner (BUFFER_VIEW
into shared arenas) means buffer addresses are not stable across runs. Instrumentation
must track buffer identity through the BUFFER_VIEW indirection, not raw pointers.

### 6.5 Recommended Instrumentation Architecture

```
                    User / LLM Client
                           |
                    Protocol Layer (structured + TUI)
                           |
                    Tick Controller
                    /      |       \
           UOp Hooks   FX Hooks   Comm Hooks
              |           |           |
         tinygrad     PyTorch      NCCL/Gloo
         realize()   torch.compile  GroupCoordinator
```

**UOp Hooks** (tinygrad): Custom PatternMatcher injected into `pm_compile` and
`pm_exec`. Can intercept any UOp before/after compilation/execution.

**FX Hooks** (PyTorch): FX graph passes that insert instrumentation nodes.
Works with both eager and torch.compile modes.

**Comm Hooks** (multi-GPU): Wrapped GroupCoordinator that intercepts all collective
operations. Provides synchronized stepping across ranks.

**Tick Controller**: Central state machine that coordinates stepping across all
three hook types. Manages breakpoints, watchpoints, and surgical modifications.

### 6.6 Key Technical Risks

1. **JIT invalidation**: Modifying activations between ticks may invalidate JIT-captured
   graphs or compiled kernels (shape changes, dtype changes). Must handle gracefully.

2. **NCCL deadlocks**: Pausing one rank while others continue will deadlock NCCL
   collectives. Must pause ALL ranks simultaneously at collective boundaries.

3. **Memory overhead**: Capturing activations at every tick can exhaust GPU memory.
   Need selective capture with spill-to-CPU strategy (analogous to vLLM's KV cache
   swapping).

4. **Compilation overhead**: Re-compiling kernels after surgical modification is
   expensive. Should cache pre/post-modification compiled variants.

5. **Backward pass entanglement**: Modifying forward activations changes backward
   gradients. AOTAutograd's min-cut decisions may become invalid. Need to either
   (a) re-trace after modification or (b) operate below AOTAutograd.

6. **Graph deduplication**: tinygrad's UOp deduplication means modifying one tensor's
   UOp may affect others sharing the same subgraph. Must clone before modification.
