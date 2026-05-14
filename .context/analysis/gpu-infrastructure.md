# GPU Infrastructure Analysis for rocket_surgeon

Date: 2026-05-14
Status: Active reference document

## Sources

### Papers
- CRIUgpu (2502.16631) — Transparent GPU checkpoint/restore via CRIU plugin + cuda-checkpoint
- PhoenixOS (2024) — Concurrent OS-level GPU C/R with validated speculation
- Chen et al. (2016) — Sublinear memory cost via gradient checkpointing
- Dao (2022) — FlashAttention: IO-aware exact attention
- GPU Determinism Survey (2408.05148) — FPNA, deterministic parallel sums, PyTorch non-det ops
- GPUDet (Jooybar 2013) — Hardware deterministic GPU architecture
- GPUReplay (Park 2021) — Record-replay at CPU/GPU boundary

### Repos (quarantine/)
- cuda-checkpoint — NVIDIA's cuda-checkpoint utility and CRIU CUDA plugin
- cuda-samples — CUDA stream callbacks, NVTX profiling examples
- flash-attention — FlashAttention-3 (Hopper SM90) + FlashAttention-4 (CuTeDSL)
- nccl — NVIDIA Collective Communication Library
- open-gpu-kernel-modules — NVIDIA UVM kernel driver (page migration, fault handling)

---

## 1. Checkpoint/Restore Mechanisms

### 1.1 cuda-checkpoint Utility (r550-r580)

The `cuda-checkpoint` binary operates on a running process by PID. The protocol is a four-phase state machine:

```
RUNNING -> lock -> LOCKED -> checkpoint -> CHECKPOINTED -> restore -> LOCKED -> unlock -> RUNNING
```

**Lock phase**: All CUDA driver APIs that launch work, manage resources, or impact GPU state are blocked. CPU threads continue running; they can still access `cudaMallocHost` memory. Already-submitted work (including stream callbacks) drains to completion.

**Checkpoint phase**: Device memory is copied to host-side shadow allocations managed by the CUDA driver. All GPU resources (contexts, streams, allocations) are released. The process no longer references any GPU hardware at the OS level.

**Restore phase**: GPUs are re-acquired. Device memory is copied back. GPU memory mappings are restored at their original virtual addresses. CUDA objects (streams, contexts) are reconstructed.

**Unlock phase**: Blocked CUDA APIs are unblocked.

From source code (`r580-migration-api.c`), the CUDA 13.0+ driver API surface is:
```c
CUcheckpointLockArgs lock_args;
CUcheckpointCheckpointArgs checkpoint_args;
CUcheckpointRestoreArgs restore_args;
CUcheckpointUnlockArgs unlock_args;

cuCheckpointProcessLock(pid, &lock_args);
cuCheckpointProcessCheckpoint(pid, &checkpoint_args);
cuCheckpointProcessRestore(pid, &restore_args);
cuCheckpointProcessUnlock(pid, &unlock_args);
```

**R580 GPU Migration**: The restore phase accepts a `gpuPairs` array mapping old GPU UUIDs to new GPU UUIDs, enabling transparent migration of CUDA state between GPUs. Every GPU accessible to CUDA must be specified in the map, even unused ones. The CLI equivalent is `--device-map oldUuid1=newUuid1,oldUuid2=newUuid2`.

**R570 CRIU Integration**: A CRIU plugin (`cuda_plugin.so`) bridges cuda-checkpoint into CRIU's dump/restore workflow. The demo (`r570-features.c`) shows: parent process calls `cuCheckpointProcessGetState(child, &state)` to confirm CUDA is running, then uses `criu dump` with `--libdir` pointing to the plugin, and `criu restore` to bring it back.

**Current limitations (as of r570)**:
- x64 only
- No UVM or IPC memory support
- Waits for all submitted CUDA work to finish before checkpoint completes
- No NCCL support (communicators, active collectives)
- No graceful error recovery if unsupported features are encountered

### 1.2 CRIUgpu Paper Findings

Performance on H100 with LLaMA 3.1 8B (54GB GPU memory):
- Checkpoint: 77.4s total (GPU memory copy dominates at 80-97% of time)
- Restore: 38.83s
- Scales linearly with GPU count

GPU memory dominates checkpoint size. For a 405B model distributed across 8 GPUs: checkpoint ~10 minutes, restore ~5 minutes. The paper confirms the approach is transparent (no application modification needed) but heavy for interactive use.

### 1.3 PhoenixOS Findings

PhoenixOS takes a fundamentally different approach: concurrent checkpoint/restore at the OS level using validated speculation on kernel arguments to trace GPU memory read/write sets. Key differences from cuda-checkpoint:

- **Concurrent C/R**: Uses copy-on-write (CoW) and recopy protocols to checkpoint without fully stopping GPU work
- **Context pool**: Bypasses expensive GPU context creation on restore by pooling pre-initialized contexts
- **Optimal checkpoint timing**: Checkpointing at iteration start is cheapest because only activation buffers are dirty (weights are clean)
- **Performance**: Llama2-13B shows only 185ms overhead. 1-12% runtime overhead from the validator. 70-160% less stall than Singularity

### 1.4 Implications for rocket_surgeon

**The checkpoint/restore mechanism is directly usable for our "tick" model.** At each tick boundary, we can:
1. Lock CUDA (blocking new work)
2. Drain pending work (implicit in lock)
3. Read GPU state for display/inspection without doing a full checkpoint
4. Allow user modifications (activation surgery, weight patching)
5. Unlock to proceed to next tick

However, the current cuda-checkpoint approach has problems for us:
- **Full checkpoint is too slow for interactive debugging** (77s for 54GB is unacceptable between ticks)
- **No NCCL support** means multi-GPU collective communication state would be lost
- **No UVM support** is a gap if the model uses managed memory
- The **lock/unlock without checkpoint** pattern may be fast enough for just pausing execution between ticks, if we don't need to serialize full state to disk

**Design decision needed**: Do we use cuda-checkpoint's lock/unlock for tick-pause, or do we need our own interception layer? The lock phase alone (without checkpoint/restore) may be sufficient for step-through debugging if we can read GPU memory while locked.

---

## 2. CUPTI Instrumentation Patterns

### 2.1 Available in quarantine

The cuda-samples repo does not contain dedicated CUPTI examples (no CUPTI headers or API calls found). The relevant samples are:

- **simpleCallback**: Demonstrates `cudaStreamAddCallback` — CPU callbacks triggered after GPU stream operations complete. This is the CUDA Runtime callback mechanism, not CUPTI.
- **kernelNsysProfile**: A Python sample using `cuda.core` to compile and launch kernels with NVTX annotations for Nsight Systems profiling. Uses `nvtx.annotate()` context managers for structured profiling regions.

### 2.2 NVTX Integration in NCCL

NCCL provides extensive NVTX instrumentation natively. Every collective operation is wrapped:
```c
NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    NVTX3_PAYLOAD(comm->commHash, count * ncclTypeSize(datatype), op));
```

This means Nsight Systems can already capture NCCL collective boundaries, data sizes, and operation types. For rocket_surgeon, we can leverage NCCL's built-in NVTX markers to identify collective communication boundaries in the forward pass.

### 2.3 CUPTI Instrumentation Strategy (from literature)

CUPTI (CUDA Profiling Tools Interface) provides four API levels relevant to us:

1. **Callback API**: Register callbacks for CUDA Runtime/Driver API entry/exit. Can intercept every `cudaLaunchKernel`, `cudaMemcpy`, etc. Low overhead per callback (~1us) but cumulative cost matters.
2. **Activity API**: Asynchronous recording of GPU activities (kernel execution, memcpy, memset) with timestamps. Buffer-based, lower overhead than synchronous callbacks.
3. **Profiling API**: Hardware performance counter collection. High overhead, not suitable for interactive debugging.
4. **PC Sampling API**: Statistical sampling of instruction pointers. Useful for hotspot analysis but not for deterministic step-through.

**For rocket_surgeon**, the Callback API on `cudaLaunchKernel` is the primary hook point. We intercept kernel launches to:
- Identify which layer/operation is about to execute
- Pause at tick boundaries
- Record the launch parameters (grid, block, shared memory, stream) for the TUI display

The Activity API provides post-execution timing data for the LLM-facing protocol (structured query about what happened in the last tick).

### 2.4 Implications for rocket_surgeon

**CUPTI Callback API is our primary instrumentation mechanism for kernel-level tick identification.** However, we need to be aware that:
- CUPTI callbacks are per-context, and multi-GPU means multiple contexts
- The callback overhead must be minimized for interactive debugging (avoid profiling-level instrumentation)
- We should use NCCL's existing NVTX markers rather than trying to intercept NCCL internals

---

## 3. FlashAttention Architecture and Implications

### 3.1 Kernel Structure (from source)

FlashAttention (Hopper/SM90) is a single fused CUDA kernel. The key data structures from `flash.h`:

**Flash_fwd_params** contains:
- Q, K, V pointers with batch/row/head strides
- Output O pointer
- `softmax_lse_ptr` — log-sum-exp of softmax (O(N) per head, stores `log(sum(exp(scores)))`)
- `softmax_lseaccum_ptr` — for split-KV accumulation
- FP8 descale pointers (q_descale, k_descale, v_descale)
- Paged KV cache support (page_table, page_size)
- Causal/local masking flags
- Split-KV parameters (num_splits)
- Dropout probability and RNG state
- Rotary embedding cos/sin pointers

**Flash_bwd_params** extends fwd with:
- dO, dQ, dK, dV pointers
- `dq_accum_ptr`, `dk_accum_ptr`, `dv_accum_ptr` for accumulation
- `dsoftmax_sum` — derivative of softmax normalization
- `softmax_lse_log2_ptr` — log2 of LSE for efficient backward
- `deterministic` flag — controls whether backward uses deterministic accumulation
- Semaphores for dQ, dK, dV (for multi-block synchronization)

### 3.2 Online Softmax Implementation (from softmax.h)

The `Softmax` struct maintains per-row state:
- `row_max` — running maximum across all K blocks (TensorT, one per M-tile row)
- `row_sum` — running sum of exp(scores - max) across all K blocks
- `softmax_scale_log2` — precomputed `softmax_scale * log2(e)` for using exp2f instead of expf

The online algorithm processes K blocks incrementally:
1. `max_get_scale`: Compute new row_max, derive rescaling factor `exp2f((old_max - new_max) * scale)`, rescale row_sum
2. `online_softmax`: Apply `exp2f(score * scale - max_scaled)` to current scores tile, accumulate into row_sum
3. `rescale_o`: Rescale accumulated output O by the scale factor from max update
4. `finalize`: Final normalization — compute `1/row_sum`, produce LSE as `row_max * scale + log(row_sum)`

The use of `exp2f` instead of `expf` is deliberate: it maps directly to a single PTX instruction and enables `ffma` fusion with the scale multiplication.

### 3.3 Tiling and Memory Architecture

From the SM90 forward kernel:
- Uses CUTLASS pipeline abstraction with separate producer (load) and consumer (MMA) warp groups
- `NumLoadWarpGroups = 1`, `NumMmaWarpGroups = 1 or 2 or 3` (template parameter)
- Shared memory is overlapped between mainloop (Q, K, V tiles) and epilogue (O tiles) — `smem_v` and `smem_o` share the same memory
- TMA (Tensor Memory Accelerator) descriptors are prefetched
- K and V are loaded through pipelined stages (`MainloopPipelineK`, `MainloopPipelineV`)
- Cluster-level barriers (`ClusterTransactionBarrier`) coordinate TMA loads

The backward kernel (`FlashAttnBwdSm90`) has:
- Two MMA operations: `TiledMmaSdP` (for S = Q @ K^T, dP computation) and `TiledMmadKV` (for dK, dV accumulation)
- Separate pipelines for Q loading and dO loading
- Option for `dKV_swapAB` (transposing the GEMM operands for dK/dV computation)

### 3.4 Critical Implications for rocket_surgeon

**The N x N attention matrix never exists in HBM.** FlashAttention computes S = Q_tile @ K_tile^T in SRAM (shared memory), applies softmax in-place, then immediately multiplies by V_tile to accumulate into O. The intermediate attention scores matrix (S) and post-softmax probabilities (P) are ephemeral — they exist only in shared memory/registers during the tiled computation.

This means:
1. **Our debugger cannot display the full attention matrix** by reading GPU memory at a tick boundary. It literally does not exist anywhere readable.
2. **To inspect attention patterns, we need one of**:
   - A special "debug mode" FlashAttention kernel that materializes S/P to HBM (at O(N^2) memory cost, defeating FlashAttention's purpose)
   - Recomputation: run the forward pass in standard (non-flash) mode for a specific head/layer on demand
   - Approximate visualization: use the stored LSE values and O to infer aggregate attention statistics (row entropy, top-k attended positions)
3. **The LSE (log-sum-exp) is our primary observable.** It is stored per (batch, head, seq_pos) and tells us the effective "temperature" of attention at each position. Sharp attention = low LSE, diffuse = high.
4. **The `deterministic` flag in backward params** is directly relevant to our determinism story — FlashAttention backward can be run deterministically at a small performance cost (uses semaphore-coordinated accumulation instead of atomics).
5. **Dropout state**: FlashAttention uses a per-kernel RNG state (`rng_state` pointer). For reproducible debugging, we need to capture and replay this state.

**Design decision**: We should expose LSE as a first-class observable in the debugger, with an option to "expand" a specific (layer, head, position range) into full attention scores via recomputation when the user needs it.

---

## 4. NCCL Internals and Multi-GPU Communication

### 4.1 Algorithms and Protocols

From `collectives.cc` and `device.h`, NCCL supports 7 algorithms:
- **RING**: Classic ring-based collective. Data split into chunks, pipelined around a ring of GPUs.
- **TREE**: Binary tree reduction/broadcast. Root has up to 2 children, interior nodes up to 3 (NCCL_MAX_TREE_ARITY).
- **COLLNET_DIRECT**: CollNet direct offload to network hardware.
- **COLLNET_CHAIN**: CollNet with chained topology.
- **NVLS**: NVLink SHARP — hardware multicast via `cuMulticastCreate` for NVLink-connected GPUs.
- **NVLS_TREE**: NVLS combined with tree for cross-node.
- **PAT**: Pattern-based algorithm (newer).

And 3 protocols:
- **LL (Low Latency)**: 8-byte atomicity with inline flags. Each 8-byte line carries 4 bytes data + 4 bytes flag. Good for small messages.
- **LL128**: 128-byte lines with 112 bytes data + 16 bytes flags. Better bandwidth than LL.
- **SIMPLE**: Direct buffer transfers with head/tail pointer signaling. Best bandwidth for large messages.

### 4.2 Channel and Ring Structure

From `device.h` and `comm.h`:

```c
struct ncclRing {
    int prev;           // Previous rank in ring
    int next;           // Next rank in ring
    int* userRanks;     // Maps internal index -> user rank
    int* rankToIndex;   // Maps user rank -> internal index
    int index;          // This rank's position in ring
};

struct ncclTree {
    int depth;
    int up;                          // Parent
    int down[NCCL_MAX_TREE_ARITY];   // Children (max 3)
};
```

Each communicator has up to `MAXCHANNELS` (64) channels, each with its own ring and tree topology. Multiple channels enable pipelining and bandwidth aggregation.

### 4.3 Transport Layer

NCCL has 4 transport backends, selected based on topology:

- **P2P** (`transport/p2p.cc`): Direct GPU-to-GPU via PCIe P2P or NVLink. Four subtypes:
  - `P2P_DIRECT`: Direct memory access between GPUs
  - `P2P_INTERMEDIATE`: Via intermediate GPU
  - `P2P_IPC`: Via CUDA IPC memory handles
  - `P2P_CUMEM`: Via `cuMem` API (newer unified memory API)
  
  Connection check (`p2pCanConnect`) queries topology graph, falls back to `cudaDeviceCanAccessPeer`, checks legacy IPC support.

- **SHM** (`transport/shm.cc`): Shared memory between processes on same node.

- **NET** (`transport/net.cc`): Network transport (InfiniBand, RoCE, TCP).

- **NVLS** (`transport/nvls.cc`): NVLink SHARP multicast. Creates multicast groups via `cuMulticastCreate`, maps memory with `cuMulticastAddDevice`. Handles both POSIX FD and fabric handle types for cross-process sharing.

### 4.4 Work Scheduling and Enqueue

From `enqueue.cc`, NCCL batches operations into `ncclDevWorkBatch` structures:
- Each batch has a `workType` (collective, P2P, broadcast)
- P2P operations are grouped into epochs with max `NCCL_MAX_DEV_WORK_P2P_PER_BATCH` ops per batch
- The scheduler selects algorithm and protocol based on message size, topology, and tuning
- Kernel launch uses persistent kernels that consume work from a FIFO

### 4.5 Implications for rocket_surgeon

**Multi-GPU tick synchronization is the hardest problem.** At a tick boundary in a distributed forward pass:

1. **AllReduce in tensor parallelism**: After each attention/MLP block, ranks do AllReduce on partial results. A tick boundary between layers must wait for the AllReduce to complete on ALL ranks before any rank can proceed. This is already handled by NCCL's synchronous semantics if we pause after the collective returns.

2. **Pipeline parallelism**: Different ranks execute different layers. Tick synchronization means pausing all pipeline stages simultaneously, which requires a barrier protocol on top of NCCL.

3. **NCCL state is opaque**: We cannot checkpoint NCCL communicators (cuda-checkpoint explicitly doesn't support NCCL). For tick-step debugging, we don't need to serialize NCCL state — we just need all ranks to reach the same tick boundary.

4. **Interception strategy**: We should intercept at the PyTorch/framework level (before NCCL calls) rather than inside NCCL. Each collective call is a natural tick boundary. The NVTX markers NCCL already emits can help identify which collective is executing.

**Key data structures to expose in the debugger**:
- Per-rank communicator info (rank, nRanks, algorithm, protocol)
- Channel topology (ring prev/next, tree structure)
- Transport type per connection (P2P direct, IPC, NVLS, NET)
- Current operation and data sizes

---

## 5. GPU Determinism

### 5.1 Sources of Non-Determinism

From the GPU Determinism survey (2408.05148):

**Floating-Point Non-Associativity (FPNA)** is the primary source. Parallel reductions (sum, max) across threads/warps/blocks produce different results depending on execution order because `(a + b) + c != a + (b + c)` in IEEE 754.

**Specific non-deterministic PyTorch operations**:
- `scatter_reduce` (used in sparse operations, MoE expert routing)
- `index_add` / `index_put` / `index_copy` (used in embedding lookups, sparse updates)
- `ConvTranspose` (transposed convolution backward)
- `cumsum` on CUDA (prefix sum)
- Atomic operations in general (used throughout CUDA kernels for reduction)

**Hardware-level sources**:
- Thread scheduling order within a warp is deterministic (SIMT), but across warps/blocks is not
- Memory access patterns from different warps can lead to different reduction orders
- L2 cache behavior can affect which data is resident, changing access patterns

### 5.2 Deterministic Alternatives

**SPTR (Sorting with Parallel Tree Reduction)**: Deterministic parallel sum using tree-structured reduction with fixed ordering. Overhead: <1% for large arrays.

**SPRG (Sorting with Parallel Reduction and Grouping)**: Groups partial sums by exponent range before accumulating. Better numerical accuracy. Overhead: 3.5-7.8%.

**FlashAttention deterministic backward**: The `deterministic` flag in `Flash_bwd_params` forces semaphore-coordinated accumulation for dQ instead of atomicAdd. This ensures the same reduction order on every run at a cost of additional synchronization.

**PyTorch deterministic mode**: `torch.use_deterministic_algorithms(True)` forces deterministic implementations where available and raises errors where not. This is a good starting point for rocket_surgeon users.

### 5.3 GPUDet Architecture

GPUDet proposes hardware changes for deterministic GPU execution:
- Three-phase quantum: parallel execution -> commit phase -> serial phase
- Wavefront-level determinism exploits SIMT (all threads in a wavefront execute in lockstep)
- Z-Buffer hardware provides deterministic ordering for commit phase
- ~2x average slowdown — too expensive for production, but the principles inform software approaches

### 5.4 GPUReplay: Record-Replay

GPUReplay provides record-replay at the CPU/GPU boundary:
- Records all register accesses and memory dumps at kernel launch boundaries
- 50KB replayer replaces the entire GPU software stack for replay
- Handles non-determinism via three strategies: prevention (force deterministic execution), tolerance (accept small differences), elimination (record and replay exact execution order)

### 5.5 Implications for rocket_surgeon

**Determinism is required for meaningful step-through debugging.** If stepping forward from the same state produces different results each time, the debugger is useless for investigating specific behaviors.

Our strategy should be layered:
1. **Mandatory**: `torch.use_deterministic_algorithms(True)` as a prerequisite for debugging sessions
2. **FlashAttention**: Use the `deterministic=True` flag for backward passes
3. **NCCL**: Enforce deterministic reduction order (NCCL's ring algorithm is inherently deterministic in reduction order; tree may vary)
4. **Custom kernels**: Identify and flag non-deterministic operations in the forward pass; offer to substitute deterministic alternatives
5. **Seed control**: Capture and replay RNG state at tick boundaries (dropout, sampling)
6. **Record-replay fallback**: For operations that cannot be made deterministic, record the exact output and replay it (GPUReplay-inspired approach)

**The MoE router is a particular concern.** Expert routing uses `scatter_reduce` which is non-deterministic. We need either a deterministic scatter implementation or a record-replay approach for the routing decisions.

---

## 6. Lightweight Checkpointing Strategy

### 6.1 What We Actually Need vs. What Exists

Full process checkpoint (cuda-checkpoint + CRIU) is designed for fault tolerance and migration. It serializes everything to disk. We need something much lighter: the ability to capture and restore the minimum state needed to re-execute a forward pass segment.

### 6.2 Chen et al. Gradient Checkpointing Insights

The key insight from Chen (2016) that applies to us:

**O(sqrt(n)) memory for n layers**: Divide the forward pass into sqrt(n) segments. Store activations only at segment boundaries. To recompute a specific layer's activations, replay only within its segment. Cost: 1 extra forward pass.

**Recursive version**: O(log n) memory with O(n log n) computation. Not practical for interactive debugging (too many recomputations).

**Budget-based planning (Algorithm 3)**: Given a memory budget B, search for the optimal segmentation that minimizes recomputation. This is directly applicable: we can let the user choose how much GPU memory to dedicate to activation checkpoints, and compute the optimal placement.

### 6.3 Proposed Lightweight Checkpoint Strategy

For rocket_surgeon, "checkpointing" serves two purposes:
1. **Tick state capture**: Snapshot activations at tick boundaries for display/inspection
2. **Rewind**: Return to a previous tick to re-execute with modified state

**Tier 1 — Hot state (always captured)**:
- Layer input activations at tick boundaries (the tensor being passed between layers)
- Attention LSE values (small: batch * heads * seqlen * sizeof(float))
- MoE routing decisions (which experts were selected, what gates were assigned)
- RNG states for dropout/sampling

Memory cost: O(batch * seqlen * hidden_dim) per tick boundary = one activation tensor. For a 7B model with seqlen=2048, batch=1: ~32MB per checkpoint. For 32 layers: ~1GB total.

**Tier 2 — Warm state (captured on demand)**:
- Full KV cache state
- Optimizer states (only relevant for backward debugging)
- Weight deltas if surgery was performed

**Tier 3 — Cold state (recomputed)**:
- Intermediate activations within attention (S, P matrices — see FlashAttention section)
- Intermediate MLP activations
- Recomputed using Chen-style segment replay when inspected

### 6.4 Multi-GPU Checkpoint Coordination

For multi-GPU forward passes:
1. All ranks reach the tick boundary (barrier)
2. Each rank independently captures its local Tier 1 state
3. The debugger controller knows which rank holds which data (from the parallelism strategy)
4. For tensor-parallel layers, the debugger reconstructs the full tensor by gathering shards on demand (not eagerly)

**Critical optimization from PhoenixOS**: Checkpoint at iteration boundaries (between layers), not mid-kernel. This is when the least state is dirty. For rocket_surgeon, this aligns naturally with our tick-at-layer-boundary model.

### 6.5 Rewind Protocol

To rewind to tick T from current tick T+N:
1. Restore Tier 1 state from tick T
2. Restore RNG state from tick T
3. If weight surgery was performed between T and T+N, those modifications are already in-place (weights are persistent) — user must explicitly revert if desired
4. Re-execute forward from T

For multi-GPU rewind, all ranks must rewind to the same tick. The controller broadcasts the rewind command and waits for all ranks to acknowledge.

### 6.6 UVM Considerations

From the open-gpu-kernel-modules UVM source:

UVM manages page migration between CPU and GPU memory transparently via page faults. The kernel module (`uvm_migrate.c`) handles:
- Two-pass migration (first pass migrates, second pass handles stragglers)
- `block_migrate_add_mappings`: After migration, updates page table mappings for all relevant processors
- Fault handling (`uvm_gpu_replayable_faults.c`): Batched fault processing (default batch size 256), replay policies, prefetch fault support
- Access counters: Track page access patterns to drive proactive migration

**Relevance to rocket_surgeon**: If a model uses CUDA Unified Memory (managed memory), page faults and migrations will occur transparently during the forward pass. This complicates our tick model because:
- A tick boundary might trigger page faults as the debugger reads data that was migrated away
- UVM page migration is non-deterministic (depends on access patterns and fault timing)
- cuda-checkpoint does not support UVM (confirmed in README limitations)

**Recommendation**: For initial implementation, require that debugged models use explicit device allocations (not managed memory). Document this as a known limitation. UVM support can be added later by either:
- Using the UVM tools API to pin pages before tick boundaries
- Implementing our own page tracking layer

---

## Cross-Cutting Concerns

### Tick Boundary Definition

A "tick" in rocket_surgeon should be defined as:
1. **Primary tick**: Between transformer layers (post-LayerNorm output of one layer = input to next)
2. **Sub-tick (optional)**: Between attention and MLP within a layer
3. **Micro-tick (optional)**: Between Q/K/V projection, attention computation, output projection
4. **Collective tick**: After each NCCL collective operation completes

The granularity is user-selectable. Finer granularity means more interception points and more overhead.

### The Lock-Without-Checkpoint Pattern

The most promising approach for interactive debugging:
1. Use cuda-checkpoint's **lock** operation (or equivalent) to pause CUDA
2. While locked, read GPU memory directly for display (no serialization to disk)
3. Allow modifications to GPU memory (activation surgery)
4. **Unlock** to proceed

This avoids the 77-second full checkpoint cost entirely. The lock operation itself should be fast (just draining pending work and blocking new submissions). The key question is whether `cuCheckpointProcessLock` allows reading device memory while locked — the README says "device memory is copied to host" only during the checkpoint phase, suggesting memory is still on-device during lock. We need to verify this empirically.

### Forward vs. Backward Step-Through

Forward step-through: Straightforward — intercept kernel launches, pause between ticks, display state.

Backward step-through: More complex because:
- Backward operates in reverse layer order
- Gradient accumulation uses atomics (non-deterministic without FlashAttention's deterministic flag)
- Gradient checkpointing (Chen-style) means some forward recomputation happens during backward
- The user needs to see both gradients AND forward activations simultaneously

### MoE-Specific Considerations

Mixture-of-Experts layers add complexity:
- **Routing decisions**: The gate network assigns tokens to experts. These decisions must be captured at each MoE tick.
- **Expert parallelism**: Different experts may be on different GPUs. Inspecting a specific expert's computation requires knowing which rank it's on.
- **Load imbalance**: Some experts may process many tokens, others few. The tick timing varies per expert.
- **All-to-All communication**: MoE uses All-to-All (ncclAlltoAll) to redistribute tokens to expert-owning ranks. This is a natural tick boundary.
