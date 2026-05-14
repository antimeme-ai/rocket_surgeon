---
topic: GPU compute platforms — CUDA runtime/driver, streams, NCCL, ROCm, Metal, Vulkan, multi-GPU management
status: draft
created: 2026-05-14
sources: NVIDIA docs, AMD ROCm docs, Apple Metal docs, various research papers
---

# GPU Compute Platforms: Lit Review

The runtime/driver layer between ML frameworks and hardware. What rocket_surgeon must understand to control multi-GPU execution.

## CUDA Architecture

### Runtime vs Driver API
- **Runtime**: high-level, implicit context management, most apps never need driver API
- **Driver**: low-level, explicit context, earlier access to new features (Virtual Memory Management)
- No performance difference — choice is about control vs convenience
- **For us**: Driver API context manipulation needed for pause/checkpoint/manage across devices

### Streams and Asynchronous Execution
- Stream = sequence of operations executing in-order, async dispatch returns immediately to host
- Different streams can overlap; same stream is strictly sequential
- Default stream synchronizes w.r.t. all other streams
- **Debugging challenge**: can't "pause" a GPU mid-kernel. Must work at stream sync boundaries.

### CUDA Graphs
- Capture kernel launch sequence as reusable template. Instantiation pays overhead once, replays are cheap (~5x speedup).
- **Hook implications**: graphs bypass normal API entry points. Need graph-aware instrumentation (capture/instantiation events).
- Can't selectively pause individual ops within a captured graph.

### Events and Synchronization
- cudaEventBlockingSync: host blocks until event completes
- cudaEventDisableTiming: lightweight, no timing overhead
- Async barriers (CUDA 12.2+): fine-grained non-blocking coordination
- **Events are natural breakpoint locations** for multi-GPU debugging

### Memory Model
- **Unified Memory**: single VA space across CPU+GPUs via cudaMallocManaged(). Page Migration Engine (Pascal+) handles faulting + migration.
- **Coherence**: explicitly exposed scopes, enforced at sync boundaries. __syncthreads() required.
- **Peer-to-Peer**: GPUDirect P2P over NVLink/PCIe. cudaCanDeviceAccessPeer() checks, cudaEnablePeerAccess() enables.
- **For us**: memory coherence violations hard to catch. Need memory access tracking at kernel granularity.

### Compilation Pipeline
- PTX: platform-independent virtual GPU assembly (forward compatible)
- cubin: machine code for specific architecture (sm_90 etc.)
- Driver JIT-compiles PTX for target GPU at runtime. Milliseconds per kernel, paid once.
- **Hook point**: intercept PTX before driver compiles

### CUDA Debugging Tools
- **Compute Sanitizer** (replaced cuda-memcheck): memcheck, racecheck, initcheck, synccheck
- **CUDA-GDB**: GPU kernel debugging, coredump inspection
- **CUPTI**: primary instrumentation hook. Callback API (inject at entry/exit of CUDA calls), Activity API (trace executed work).
- CUPTI is the foundation for Nsight Systems, Nsight Compute, and third-party profilers.

### CUDA MPS (Multi-Process Service)
- Alternative CUDA impl: shared copy of GPU resources, multiple processes use same scheduler
- No memory/fault isolation (unlike MIG)
- 1-2ms round-robin context switching
- CUDA-GDB can generate coredumps from Volta MPS

## NCCL & GPU Interconnects

### NCCL Collective Operations
- all-gather, all-reduce, broadcast, reduce, reduce-scatter, P2P send/recv
- **Ring algorithm**: best for large messages. Reduce-scatter then all-gather. 2(k-1) steps for k GPUs.
- **Tree algorithm**: best for small messages. Double binary trees.
- **LL128 protocol**: optimized for NVLink, fully leverages high bandwidth
- Dynamic protocol selection (Simple/LL/LL128) based on topology, arch, message size
- NCCL Inspector: real-time visibility, per-communicator per-collective logging (JSON)

### NVLink & NVSwitch
- H100 (NVLink 4.0): 900 GB/s. Blackwell (Gen 5): 1,800 GB/s. Rubin (Gen 6): 3.6 TB/s.
- NVSwitch: non-blocking packet switch, any GPU to any GPU at full bandwidth
- 256 H100s via NVSwitch = 57.6 TB/s bisection bandwidth
- Same NVSwitch = no latency penalty. Different switches = up to 2x latency.

## AMD ROCm

### HIP & HIPIFY
- HIP: C++ enabling code on both NVIDIA and AMD GPUs
- hipify-clang (Clang-based, AST) and hipify-perl (pattern matching) for CUDA translation
- HIP_PLATFORM=amd -> clang + ROCclr. HIP_PLATFORM=nvidia -> NVCC + CUDA runtime.

### RCCL (AMD's NCCL)
- Same API: all-reduce, all-gather, etc. Optimized for AMD Instinct GPUs.
- MSCCL++ integration for efficient GPU-GPU communication primitives
- Intra-node: PCIe + xGMI (AMD's interconnect). Inter-node: InfiniBand, RoCE, TCP/IP.

### ROCm Profiling
- rocprof: CLI counter collection (raw CSV)
- Omniperf: system performance profiler, graphical bottleneck analysis
- Omnitrace: comprehensive C/C++/HIP/Python profiling, binary instrumentation, web-based visualization
- All open-source (vs NVIDIA's proprietary CUPTI)

## Apple Silicon

### Metal
- Low-level, low-overhead hardware-accelerated compute/graphics API
- Threads -> threadgroups -> simdgroups (size 32, like CUDA warps)
- Metal Performance Shaders (MPS): highly optimized kernel library
- Tight integration with Apple silicon, lower power, weaker portability vs CUDA

### PyTorch Metal Backend Limitations
- Buffer size limits: attention with seq_length >12,000 often exceeds Metal max
- NDArray limit: 2^32 elements
- Some PyTorch ops not implemented on MPS (CPU fallback)
- Dispatch overhead for small tensors (1-element multiply 3-5x slower than exp/tanh)
- Unified memory pool shared with OS, sensitive to pressure

### Apple AMX
- 32x32 grid of compute units, 16-bit multiply-accumulate
- Undocumented ARM64 ISA extension, only accessible via Accelerate library
- Reverse-engineered but no official developer API
- 2 AMX cores on M1/M2, 4 on M4

### Apple Neural Engine (ANE)
- Dedicated NPU since A11 Bionic. M4: up to 38 TOPS (INT8), 16 cores.
- CoreML auto-selects CPU/GPU/ANE per operation
- Bandwidth-bound for short sequences (parameter fetch dominates)
- Orion: first system enabling direct ANE programming beyond CoreML (reverse-engineered)

## Other Platforms

### Vulkan Compute
- Cross-platform: runs on Android, Linux, BSD, QNX, Nintendo Switch, etc.
- ARM Mali, Qualcomm Adreno, Imagination PowerVR support
- Portability Initiative: layered over Metal/DX12 for macOS/iOS
- Less mature than CUDA for large-scale training

### SYCL / Intel oneAPI
- C++17/20 with GPU offload semantics
- Compiles for CPUs, GPUs, FPGAs via DPC++ compiler
- Growing in HPC/cloud but less widespread than CUDA

### WebGPU / WGSL
- W3C standard, shipped in Chrome/Firefox/Safari/Edge (2025)
- First-class GPGPU support (compute shaders)
- Useful for visualization/monitoring dashboards, not for training

## Multi-GPU Management

### GPU Scheduling
- **Time-slicing**: software-based round-robin, 1-2ms per context. No memory/fault isolation.
- **MIG (Ampere+)**: hardware partitioning, up to 7 instances. Dedicated compute, memory, L2 cache per instance. Guaranteed performance isolation.
- **Hummingbird (research)**: SLO-oriented scheduling, µs-scale preemption

### Topology & Peer Access
- Frameworks query GPU topology at startup, build optimal communication graphs
- cudaCanDeviceAccessPeer() checks, cudaEnablePeerAccess() enables direct GPU-GPU transfer
- Same-package (NVLink) vs different packages (PCIe) = massive performance difference

### Transformer-Specific Considerations
- Attention is memory-bandwidth-bound. Flash Attention: avoid writing intermediates to HBM, keep in SRAM.
- Packed sequences: concatenate variable-length with position IDs, ~2x throughput vs padding
- Gradient checkpointing: save activations at sqrt(n)-th nodes, recompute during backward. 25-50% more compute for 40-60% less memory.
- Operator fusion: merge sequential ops into single kernel. TorchInductor + Triton do this automatically.

## Critical Runtime Layers to Instrument

1. **CUPTI hooks**: kernel launches, memory ops, synchronization
2. **CUDA Driver context management**: low-level pause/resume
3. **Stream & Event APIs**: natural breakpoint locations for checkpointing
4. **CUDA Graphs**: need graph-aware instrumentation (capture/instantiation events bypass normal API)
5. **NCCL topology & algorithm selection**: which ring/tree traversal is active
6. **NCCL Inspector**: real-time collective operation visibility

## Debugging Challenges

- **No true pause**: GPU has no native pause instruction. Checkpoint-restart at sync boundaries.
- **Async execution**: host returns immediately from kernel launch; GPU executes later.
- **Memory coherence**: only enforced at sync points; transient inconsistencies invisible.
- **Graphs & fusion**: optimizations bypass normal tracing; need alternative instrumentation.
- **MPS/MIG**: process-level and hardware-level virtualization add indirection.

## Platform Support Strategy

- **NVIDIA CUDA**: primary target. CUPTI, NCCL, NVLink — well-documented, mature.
- **AMD ROCm**: secondary. Functionally similar, RCCL analog, open-source profiling.
- **Apple Silicon**: limited. Metal backend immature for training. ANE/AMX undocumented.
- **Cross-platform (Vulkan, SYCL)**: emerging. Useful for portability, not mainstream for training.

## Sources

- NVIDIA CUDA Programming Guide, Runtime/Driver API docs
- CUPTI docs, Compute Sanitizer docs, CUDA-GDB docs
- NCCL docs, NVLink/NVSwitch docs
- AMD ROCm docs, HIP porting guide, RCCL docs
- Apple Metal docs, CoreML, ml-ane-transformers, AMX research
- Vulkan.org, Intel oneAPI, WebGPU MDN
- Various research papers: Hummingbird (arxiv 2601.04071), Flash Attention, PyGraph (arxiv 2503.19779)
