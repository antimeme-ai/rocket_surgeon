---
topic: RTOS fundamentals, real-time GPU control, determinism, checkpointing, GPU scheduling
status: draft
created: 2026-05-14
sources: FreeRTOS/Zephyr/PREEMPT_RT docs, NVIDIA docs, seL4 papers, various research
---

# RTOS + Real-Time GPU: Lit Review

What real-time systems thinking teaches us about controlling GPU execution deterministically.

## RTOS Fundamentals

### Hard vs Soft Real-Time
- **Hard**: failure to meet deadline = system failure (safety-critical)
- **Soft**: deadline misses degrade but don't break
- RTOSes: event-driven, preemptive, bound WCET (worst-case execution time), minimize jitter

### FreeRTOS
- Fixed-priority preemptive scheduling + round-robin time-slicing for equal priority
- Task Control Block (TCB) stores all metadata. Scheduler always runs highest-priority-ready.
- Open-source, 10KB kernel, supports ARM Cortex-M, RISC-V, x86
- Dominant in embedded. Simple, deterministic.

### Zephyr
- Monolithic kernel (since v1.6). Supports MMU-rich and constrained devices.
- Two thread types: cooperative (negative priority, yield voluntarily) and preemptible (non-negative priority)
- **Tickless scheduler**: event-driven, reschedules only on Ready state changes. Reduces jitter.

### PREEMPT_RT
- **Merged into mainline Linux v6.12 (Sept 2024)** for x86, ARM64, RISC-V
- Converts Linux into fully preemptible kernel
- Spinlocks reimplemented as preemptible rt_mutex ("sleeping spinlocks")
- Interrupt handlers run as kernel threads
- **Priority inheritance** via rt_mutex prevents priority inversion
- ~100µs latencies typical. Soft real-time, not hard.
- **For us**: enables host-side deterministic scheduling for GPU control operations

### Xenomai (Dual-Kernel)
- Real-time microkernel runs beneath Linux. RT tasks bypass Linux scheduler entirely.
- Near-zero latency with µs-scale jitter
- Higher complexity, steeper learning curve
- Better on ≤4 cores; PREEMPT_RT better on >4 cores (SMP)
- Likely overkill unless we need <10µs determinism

### seL4
- First OS kernel with machine-checked formal proof of functional correctness
- WCET analysis integrated with correctness proof
- Scheduling Contexts: time as first-class resource, mixed-criticality systems
- Incremental operations: long-running ops broken into short sub-ops, abortable
- Theoretical foundation for proving debugger safety properties, but overkill for prototype

## GPU Determinism

### Sources of Non-Determinism
1. **Floating-point non-associativity**: (a+b)+c ≠ a+(b+c) due to finite precision. Different operation orderings -> different results.
2. **Atomic operations**: accumulation order depends on thread timing
3. **Thread block scheduling**: affects operation order
4. **Multiple CUDA streams**: cuBLAS may choose different internal implementations
5. **Kernel selection**: different GPUs/SMs may select different algorithms
6. **Memory access patterns**: cache behavior affects timing -> affects atomic order
7. **Batch composition**: cloud inference batching changes FP intermediate results

### Solutions
- cuBLAS: all routines bit-wise reproducible on same GPU arch within single stream
- `torch.use_deterministic_algorithms()` + single stream: 10-40% slower, some ops unsupported
- Set CUDA, cuDNN, Python random seeds (doesn't fix FP non-associativity)
- `CUBLAS_WORKSPACE_CONFIG=:16:8` + `CUDA_LAUNCH_BLOCKING=1`

### For rocket_surgeon
Forward passes must be deterministic for replay/stepping:
1. Force single CUDA stream per GPU
2. cuBLAS deterministic mode
3. Fixed batch composition (same-size, deterministic order)
4. Match GPU architectures across debugging sessions

## GPU Preemption

### Preemption Types
- **Instruction-level (CILP)**: finest granularity, huge context switch cost (hundreds of registers)
- **Thread-block-level**: medium granularity, current NVIDIA streams don't support this
- **Wait-for-idle (WFI)**: coarsest, no mid-kernel preemption

### Practical Reality
- NVIDIA priority streams do NOT preempt executing thread blocks
- Can only prevent new blocks from scheduling
- Can't pause mid-kernel. Must wait for completion.
- **"Step backward" requires checkpointing, not instruction rollback.**

### CUDA Stream Priorities
- Three priority levels; scheduler prioritizes highest-priority stream's new blocks
- Priorities don't preempt running work, only affect scheduling of new blocks
- Only hints, not guaranteed for memory transfers
- Same stream: strictly sequential. Different streams: can interleave.
- **For us**: single stream per GPU context guarantees deterministic execution

## Multi-Process GPU Control

### CUDA Context Management
- Each process manages a CUDA context (driver state, memory space)
- Context creation: ~100-200ms. Switching: ~10-50ms. Expensive.
- IPC: share GPU memory pointers between processes
- VMM: unified VA space across GPU+CPU, allows debugger to inspect inference GPU memory

### GPU Virtualization
- **MIG**: hardware partitioning (A100 -> up to 7 instances). Dedicated compute + memory + L2 per instance. Guaranteed isolation.
- **MPS**: software sharing, single CUDA context, concurrent thread blocks. No isolation.
- **vGPU**: time-sliced (VM gets time slice) or MIG-backed (VM gets MIG instance). SR-IOV + IOMMU for secure isolation.
- **For us**: MIG recommended for isolating debugger from inference workload

### Priority Inversion
- High-priority H blocks on resource held by low-priority L. Medium-priority M runs, preventing L from releasing. H misses deadline.
- Classic: Mars Pathfinder (1997)
- **Solutions**: priority inheritance (boost L to H's priority), priority ceiling protocol, lock-free data structures
- **GPU context**: occurs if high-priority inference waits for low-priority debug holding GPU context
- **Mitigation**: MIG (no contention), rt_mutex (PREEMPT_RT), minimize critical sections

### Watchdog / TDR (Timeout Detection and Recovery)
- GPU task initiated -> wait 3s -> attempt preemption -> wait 2s -> reset GPU + report error
- TDR reset clears GPU context, all pending work lost
- Long-running debugger stepping might trigger TDR
- **Mitigation**: launch short sub-kernels, set TdrDelay high, monitor watchdog in debugger loop

## Deterministic Replay for Backward Stepping

### Requirements
1. Fixed seed management (all RNG sources)
2. Deterministic algorithm selection (cuBLAS/cuDNN modes)
3. Fixed batch composition
4. Same GPU architecture
5. Single stream (avoid multi-stream optimization paths)

### GPU-Specific Challenges
- Reduction kernels: atomic summation order non-deterministic
- Attention softmax: intermediate sums affect output
- Thread scheduling varies by SM configuration
- Memory allocation order affects layout and cache behavior

### GPUReplay Approach
- Record GPU stack interactions at development time
- Encode "how GPU stack interacts with GPU"
- At deployment: invoke recorded executions with new inputs
- Enables deterministic inference across heterogeneous hardware

### Practical Strategy for rocket_surgeon
1. After each transformer block, save GPU memory state (input activations, attn_weights, intermediate states)
2. On "step backward", restore checkpoint and re-execute block forward
3. Forward-only replay (don't reverse autograd)
4. Aligns with rr/TTD model from debugger lit review

## Process Checkpointing

### CRIU (Checkpoint/Restore In Userspace)
- Standard Linux: checkpoints entire process tree (memory, FDs, thread state, signal masks)
- **Doesn't know about GPU state**

### NVIDIA cuda-checkpoint
- Extends CRIU for CUDA state
- Checkpoint: lock CUDA driver APIs -> wait for submitted work -> copy device memory to host -> release GPU resources
- Restore: re-acquire GPUs -> restore device memory -> reinstate CUDA objects -> unlock APIs
- Checkpoint time: O(GPU memory size) -> 10-100s for 40GB+ GPUs
- CRIUgpu (2025): transparent container checkpointing

### For rocket_surgeon
- Full checkpointing too slow for interactive stepping
- **Lightweight alternative**: save intermediate activations (not full GPU memory), recompute from saved state. Much faster.

## Real-Time Scheduling for Debugger Control

### Nsight GPU Debugging
- "Freeze Mode" for stepping through GPU execution
- Scheduler Locking Resume All: all blocks progress together
- Block-level freeze: step individual warps
- Designed for debugging, not production

### Practical Approach for rocket_surgeon
Application-level scheduling (not kernel-level):
1. Debugger CPU thread calls next transformer kernel
2. Kernel runs to completion
3. cudaDeviceSynchronize() ensures GPU work done
4. Inspect GPU state via cudaMemcpy (device -> host)
5. User commands: step, inspect, resume
- CPU-side isolation via PREEMPT_RT: ensure debugger thread can preempt inference thread

## Inference Optimization (TensorRT, ONNX Runtime)

### TensorRT
- Kernel fusion (LayerNorm + MatMul + Bias + Activation -> single kernel)
- Quantization (FP32 -> FP16 -> INT8 -> FP8/INT4). Per-channel, requires calibration.
- Auto-tuning: profiles implementations, selects fastest per hardware + batch size
- **Challenge for debugging**: fused ops hide granular bottlenecks. Need unfuse mode.

### ONNX Runtime
- Cross-platform: same model format, multiple execution providers (CUDA, CPU, TensorRT, CoreML, QNN)
- Useful for testing same model across backends

## Key Insight

True hard real-time guarantees are difficult on GPUs (no mid-kernel preemption). The practical approach is **checkpoint-based replay with deterministic forward-only execution**, complemented by **PREEMPT_RT scheduling on the CPU side** for reliable control flow. This aligns perfectly with the rr/TTD architecture from the debugger lit review.

## Sources

- FreeRTOS Kernel Book (github)
- Zephyr Project docs
- PREEMPT_RT wiki (Linux Foundation)
- seL4 whitepaper, SOSP '09 paper
- NVIDIA CUDA Programming Guide (streams, events, async execution)
- NVIDIA: Compute Sanitizer, CUDA-GDB, CUPTI docs
- GPU determinism: arxiv 2408.05148, thinkingmachines.ai
- GPU preemption: NVIDIA DRIVE OS docs, Deadline-based GPU scheduling (RTSS '18)
- MIG User Guide, vGPU User Guide
- Priority inversion: Wikipedia, embedded.com
- TDR: NVIDIA GameWorks docs, Microsoft WDDM docs
- CRIU: criu.org, cuda-checkpoint (NVIDIA github), CRIUgpu (arxiv 2502.16631)
- TensorRT: NVIDIA docs (transformers, quantization)
- ONNX Runtime: onnxruntime.ai
