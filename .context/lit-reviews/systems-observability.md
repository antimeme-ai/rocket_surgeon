---
topic: Low-level userland, kernel, and systems observability for GPU compute
status: draft
created: 2026-05-14
sources: eBPF docs, Brendan Gregg, NVIDIA docs, kernel docs, various research papers
---

# Systems Observability: Lit Review

Low-level observability tools for understanding what happens at the OS/hardware level during GPU compute.

## eBPF Ecosystem

### eBPF Core
- Sandboxed VM in Linux kernel (fully available since 4.4)
- Executes bytecode attached to hook points in response to system events
- **Verifier**: 10,000+ LOC, ensures programs run to completion, validates all execution paths before JIT compilation
- **BPF Maps**: persistent kernel data structures (arrays, hashmaps) for data sharing between eBPF programs and userspace
- **Hook types**: kprobes (kernel functions), uprobes (userland functions), tracepoints (predefined kernel points), USDT (application-level markers), perf_events (hardware counters)
- **For GPU debugging**: trace GPU driver syscalls (/dev/nvidia ioctl patterns), kernel function calls, userspace PyTorch tensor ops with µs precision and zero crash risk

### bpftrace
- awk/C-like high-level tracing language compiling to eBPF bytecode
- One-liners for kernel/userland tracing
- Aggregation via BPF maps (histograms, percentiles)
- Can trace cudaMalloc calls, ioctl patterns, memory operations in running PyTorch process

### BCC (BPF Compiler Collection)
- Python bindings + C for eBPF
- Pre-built tools: funccount (count calls), argdist (histogram arguments), trace (capture args/returns), profile (CPU profiling)
- Custom Python programs that compile eBPF on the fly

### libbpf + CO-RE (Compile Once, Run Everywhere)
- C library for portable eBPF programs
- Uses BTF (BPF Type Format) from running kernel (/sys/kernel/btf/vmlinux)
- Automatically adjusts structure field accesses at load time
- Write one eBPF program that traces GPU driver internals across different NVIDIA driver versions and kernel versions

## Classic Tracing

### DTrace (macOS/Solaris)
- Probe model: provider:module:function:name
- D language for scripting
- Available on macOS but SIP breaks many features
- eBPF has largely superseded it on Linux (10x broader scope, JIT, better kernel integration)
- **For us**: still viable if targeting macOS + Apple Silicon, otherwise eBPF

### strace / ltrace
- **strace**: traces all syscalls. For GPU workloads reveals:
  - ioctl patterns to /dev/nvidia* and /dev/nvidia-uvm
  - Memory mapping calls (mmap to GPU BAR, UVM regions)
  - Synchronization syscalls (futex, epoll)
  - ~50% slowdown but raw call visibility
- **GPU-specific ioctl patterns**: NV_ESC_ATTACH_GPUS_TO_FD (ioctl 212), NV_ESC_WAIT_OPEN_COMPLETE, custom driver codes
- **ltrace**: user-level library calls — traces cudaMalloc, cudaKernel, cudaLaunch with parameters and returns
- **gpuTrace**: research tool, strace/ltrace specifically for GPU interaction

### SystemTap vs eBPF
- SystemTap: compiles script -> C -> kernel module (slow, limited safety)
- eBPF: compiles script -> LLVM -> BPF bytecode (fast, verifier-enforced safety)
- eBPF has largely replaced SystemTap for new projects

## Kernel Observability

### ftrace
- Built-in kernel function tracer using -pg gcc instrumentation
- function_tracer: records all kernel function calls with timestamps
- function_graph_tracer: call hierarchy with return times (latency analysis)
- trace_pipe: live streaming
- Interface: pure debugfs (echo commands to files)
- **For GPU**: trace nvidia_drm_*, amdgpu_* driver functions, call graphs, latency profiling

### /proc and /sysfs
- /proc/driver/nvidia/: GPU model, IRQ, BIOS version, runtime driver state, per-process GPU memory
- /sys/devices/pci*/...: GPU device identification, vGPU config, power management
- How the OS sees the GPU: resource allocation, power states, process accountability

### perf_event_open
- The syscall underneath `perf`
- Hardware PMU access: CPU cycles, instructions, cache misses, branch mispredicts
- Sampling mode (interrupt after N events, capture stack) and counting mode
- **For GPU**: monitor CPU-side driver overhead, memory bus utilization, CPU idle time waiting for GPU

## Brendan Gregg Methodology

### USE Method (Utilization, Saturation, Errors)
Applied to GPU:
- **Utilization**: SM% occupied, memory controller% busy
- **Saturation**: kernel queue depth, memory access latency
- **Errors**: page faults, thermal throttling, permission errors

### Flame Graphs
- Stack trace visualization: each box = frame, width = time spent
- On-CPU, off-CPU, differential variants

### Latency Heat Maps
- 2D: time vs latency percentile
- Shows distribution modes, outliers, temporal patterns
- **For forward pass debugging**: where do stalls happen? which layers see outlier latencies?

## GPU-Specific Observability

### NVIDIA NVML
- C API underlying nvidia-smi
- GPU utilization (% SMs with active warps), memory (per-process allocated bytes), temperature, power, thermal throttling
- Programmatic access: query GPU state continuously during training

### GPU Driver Internals
- **cudaMalloc flow**: app -> CUDA runtime ioctl to /dev/nvidia-uvm -> driver allocates VA range + physical VRAM -> returns pointer
- **Unified Virtual Memory (UVM)**: 103K LOC open-source (nvidia-uvm.ko), automatic page migration CPU<->GPU, thrashing detection, multi-GPU coherence, hardware-assisted access tracking
- **ioctl interface**: VA space management, memory management/migration, 2MB granularity blocks
- **Observability hook**: trace ioctl calls to monitor allocation patterns, migration traffic, page fault frequency

### CUDA Memory Allocator
- **CudaMallocAsync / Memory Pools** (CUDA 11.2+): stream-ordered allocation (async, no OS calls), large pool (512MB+) and small pool (<512MB), defragmentation via VA remapping, configurable release threshold
- **torch.cuda.memory.memory_stats()**: allocated/freed per pool, active/inactive blocks, fragmentation ratio
- Instrument per-tensor to trace memory lifecycle

### DCGM (Data Center GPU Manager)
- Production monitoring: background health checks, prologue/epilogue diagnostics, telemetry
- Prometheus exporter for Kubernetes/datacenter monitoring
- Fleet-level GPU health and historical telemetry

### GPU Memory Hierarchy
- Registers (per-thread, 100s bytes) -> L1 cache (per-SM, hardware-managed) -> Shared memory (per-SM, 48-96KB, manual) -> L2 cache (global, small) -> HBM (~1TB/s bandwidth)
- **Bank conflicts**: shared memory mapped to 32 banks, multiple warps accessing same bank serializes (up to 32x slowdown)
- **Warp scheduling**: GTO (Greedy Then Oldest) prioritizes oldest warp, others hide latency while one stalls

### GPU Context Switching
- Opaque scheduling (no userspace interface)
- Context switches ~1-3µs
- No user control over preemption
- Nsight Systems shows context switch events

## Resource Isolation

### cgroups + namespaces for GPU
- cgroup v2 controls access to top-level GPU devices
- Mount namespaces overlay per-container access to /proc/driver/nvidia/capabilities

### Multi-Instance GPU (MIG)
- Partition GPUs into isolated instances (7-instance split on H100)
- Each instance: dedicated compute (SMs), dedicated memory, guaranteed performance isolation
- Which MIG instance affects memory bandwidth and compute throughput

### NUMA Topology
- 2-3x slowdown with poor NUMA placement
- GPUs connect to ONE socket (NUMA node)
- Higher bandwidth/lower latency to local socket's CPU cores
- numactl, lstopo show topology

## Advanced Research

### NeutriNo
- Instruction-level GPU kernel profiling via assembly-layer probing
- LD_PRELOAD injection to hook GPU workload capture
- Microsecond-granularity tracing

### eInfer (Distributed LLM Tracing)
- Fine-grained tracing for distributed transformer inference
- Hierarchical data reduction: eBPF filters locally, critical events always captured
- Kernel launch events with metadata (function name, grid dimensions, stream ID)
- Memory operation tracking, synchronization point detection
- **This is a roadmap for us**: kernel-side eBPF filtering + userspace PyTorch operation correlation

## Observability Architecture for rocket_surgeon

1. **eBPF foundation**: libbpf+CO-RE for portable programs, bpftrace for ad-hoc, BCC for production. Trace GPU driver ioctl, memory ops, synchronization.
2. **Userspace instrumentation**: PyTorch hooks for tensor alloc/dealloc, NVML for GPU state, CUDA events for kernel timing, NVTX markers for logical phases.
3. **Timeline correlation**: Nsight Systems for baseline, custom tool correlates eBPF events with PyTorch ops via timestamps.
4. **Memory debugging**: torch.cuda.memory.memory_stats() for pool fragmentation, NVML for process memory, eBPF for cudaMalloc/cudaFree patterns.
5. **Multi-GPU communication**: NCCL Inspector for collective observability, strace/eBPF for inter-GPU syscalls.
6. **Performance analysis**: USE method applied to GPU, latency heat maps of forward pass stages, flame graphs of CPU-side overhead.

## Sources

- ebpf.io, bpftrace.org, iovisor.github.io/bcc
- brendangregg.com (perf, flamegraphs, USE method, latency heat maps)
- kernel.org/doc/html (ftrace)
- developer.nvidia.com (NVML, CUPTI, DCGM, Nsight, NCCL Inspector)
- NVIDIA open-gpu-kernel-modules (github)
- NeutriNo: Fine-grained GPU Kernel Profiling (USENIX OSDI '25)
- eInfer: Fine-Grained Tracing for Distributed LLM (ACM)
- Various GPU architecture and scheduling research
