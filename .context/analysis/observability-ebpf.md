# Observability and eBPF for GPU Tracing: Analysis for rocket_surgeon

**Date:** 2026-05-14
**Sources:** BCC, bpftrace, libbpf-bootstrap, Perfetto repos; Huang2025 (NeutriNo), McCanne1993 (BPF), Cassagnes2024 (eBPF Runtime), Zheng2025 (ProfInfer), Malony2011 (Parallel Perf Measurement GPUs)

---

## 1. eBPF for GPU Tracing: Concrete Patterns

### The fundamental constraint

GPU kernels are opaque to the host OS. The GPU executes code atomically from the host's perspective -- eBPF cannot attach probes inside GPU code, cannot read GPU registers, and cannot intercept GPU-side execution. This is the central limitation stated explicitly in both Huang2025 (NeutriNo, p.333: "GPU kernel is considered atomic to the host OS") and Cassagnes2024.

What eBPF *can* observe is the **host-side boundary**: every GPU operation passes through a user-space driver library (libcuda.so, libcudart.so) and ultimately reaches the kernel via ioctl syscalls to the nvidia device nodes.

### What we can trace via uprobes on libcuda.so

The CUDA driver API is a shared library with exported ELF symbols. Key functions we can attach uprobes/uretprobes to:

**Kernel launch path:**
- `cuLaunchKernel` / `cuLaunchCooperativeKernel` -- the actual kernel dispatch
- `cuModuleLoad` / `cuModuleLoadData` -- loading GPU code modules
- `cuModuleGetFunction` -- resolving kernel handles from modules
- `__cudaRegisterFunction` -- registration at CUDA runtime init (TAU/VampirTrace intercept this per Malony2011)

**Memory operations:**
- `cuMemAlloc` / `cuMemAllocManaged` / `cuMemFree` -- device memory lifecycle
- `cuMemcpyHtoD` / `cuMemcpyDtoH` / `cuMemcpyDtoDAsync` -- host-device transfers
- `cuMemsetD8` / `cuMemsetD16` / `cuMemsetD32` -- device memory initialization

**Synchronization:**
- `cuCtxSynchronize` / `cuStreamSynchronize` / `cuEventSynchronize` -- sync points
- `cuEventRecord` / `cuEventElapsedTime` -- GPU-side timing

**Stream management:**
- `cuStreamCreate` / `cuStreamDestroy` -- execution stream lifecycle

**Context:**
- `cuCtxCreate` / `cuCtxSetCurrent` / `cuDevicePrimaryCtxRetain` -- GPU context management

NeutriNo's hook driver (Section 4.1) takes a different approach: it builds a complete shim library matching libcuda.so's symbol table, using `dlsym`/`dlopen` to forward calls to the real driver. This is more comprehensive than selective uprobes but has higher integration cost. For rocket_surgeon, the eBPF uprobe approach gives us the observation points we need without requiring a full shim.

### ioctl tracing pattern

All CUDA driver operations ultimately become ioctl calls to `/dev/nvidia*` device nodes. The ioctl command numbers encode the operation type. This is the lowest-level host-observable boundary before the GPU takes over.

Key tracepoints:
- `tracepoint:syscalls:sys_enter_ioctl` -- captures fd, cmd, arg for every ioctl
- Filter by file descriptor pointing to `/dev/nvidia*` or `/dev/nvidiactl`
- The `cmd` argument encodes the NVIDIA-specific operation (undocumented but reverse-engineerable from open-gpu-kernel-modules, which is in quarantine)

### What ProfInfer demonstrates concretely

ProfInfer (Zheng2025) implements exactly this pattern for llama.cpp inference profiling, providing a direct reference implementation. Their BPF API usage (Table 1):

**Kernel-space helpers used:**
- `bpf_ktime_get_ns` -- timestamps
- `bpf_get_current_pid_tgid` -- process/thread identification
- `bpf_get_num_processor_id` -- CPU core identification
- `bpf_probe_read_user` -- reading user/kernel address space
- `map.lookup`, `map.update` -- BPF map operations
- `map.perf_read` -- reading hardware performance counters
- `ringbuf(perf).submit` -- submitting data to user-space

**User-space attachment:**
- `attach_u(ret)probe` -- attaching to function entry/return
- `attach_tracepoint` -- kernel tracepoints (sched_switch, sched_wakeup)
- `open_perf_event` -- hardware performance counter monitoring
- `perf(ring)_buffer_poll` -- polling perf/ring buffers

**Their multi-granularity probe architecture (Table 2):**

| Function | Type | Level | Traced |
|---|---|---|---|
| `llama_decode` | u(ret)probe | Token | batch size |
| `ggml_backend_graph_compute_async` | u(ret)probe | Graph | backend type |
| `ggml_compute_forward` | u(ret)probe | OP(CPU) | tensor info, PMC |
| `ggml_cl_compute_forward` | u(ret)probe | OP(GPU) | tensor info |
| `sched_switch` | tracepoint | kernel | thread IDs |
| `sched_wakeup` | tracepoint | kernel | thread IDs |

This architecture maps directly to rocket_surgeon's needs: we need token-level, layer-level, and operator-level granularity.

---

## 2. bpftrace One-Liners: GPU Workload Tracing Recipes

bpftrace compiles a high-level tracing language to eBPF bytecode via LLVM (confirmed in bpftrace CLAUDE.md). The language supports uprobes on shared libraries directly -- the `sslsnoop.bt` tool in the repo demonstrates the exact pattern we need (attaching to `libssl` functions), which translates directly to `libcuda.so`.

### Kernel launch tracing

```
# Count CUDA kernel launches per second
bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuLaunchKernel { @launches = count(); }
             interval:s:1 { print(@launches); clear(@launches); }'

# Measure kernel launch latency (host-side dispatch time)
bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuLaunchKernel { @start[tid] = nsecs; }
             uretprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuLaunchKernel /@start[tid]/ {
               @launch_latency_us = hist((nsecs - @start[tid]) / 1000);
               delete(@start[tid]);
             }'

# Trace kernel launches with PID/comm context
bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuLaunchKernel {
               printf("%-10u %-8d %-16s cuLaunchKernel\n", elapsed/1000, pid, comm);
             }'
```

### Memory operation tracing

```
# Track CUDA memory allocation sizes
bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuMemAlloc_v2 {
               @alloc_bytes = hist(arg1);
             }'

# Trace host-to-device memory copies with sizes
bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuMemcpyHtoD_v2 {
               @htod_bytes = hist(arg2);
               printf("%-8d H->D %lu bytes\n", pid, arg2);
             }'

# Track device memory lifecycle (alloc/free balance)
bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuMemAlloc_v2 { @allocs = count(); }
             uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuMemFree_v2 { @frees = count(); }'
```

### Synchronization and ioctl patterns

```
# Measure synchronization latency (blocking waits for GPU)
bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuCtxSynchronize { @start[tid] = nsecs; }
             uretprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuCtxSynchronize /@start[tid]/ {
               @sync_latency_ms = hist((nsecs - @start[tid]) / 1000000);
               delete(@start[tid]);
             }'

# Count ioctl calls to NVIDIA devices (requires filtering by fd)
bpftrace -e 'tracepoint:syscalls:sys_enter_ioctl /comm == "python3"/ {
               @ioctl_cmds[args.cmd] = count();
             }'

# Trace stream synchronization patterns
bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuStreamSynchronize { @start[tid] = nsecs; }
             uretprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuStreamSynchronize /@start[tid]/ {
               @stream_wait_us = hist((nsecs - @start[tid]) / 1000);
               delete(@start[tid]);
             }'
```

### Multi-GPU awareness

```
# Track cuCtxSetCurrent to identify which GPU context is active
bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuCtxSetCurrent {
               printf("%-8d %-16s ctx_switch ctx=%lx\n", pid, comm, arg0);
             }'

# Track device selection
bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuDevicePrimaryCtxRetain {
               printf("%-8d device_retain dev=%d\n", pid, arg1);
             }'
```

### NCCL collective operation tracing (for multi-GPU)

```
# Trace NCCL all-reduce calls (nccl is in quarantine)
bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libnccl.so:ncclAllReduce {
               @start[tid] = nsecs;
             }
             uretprobe:/usr/lib/x86_64-linux-gnu/libnccl.so:ncclAllReduce /@start[tid]/ {
               @allreduce_us = hist((nsecs - @start[tid]) / 1000);
               delete(@start[tid]);
             }'
```

### Scheduler tracing (from ProfInfer pattern)

```
# Track thread scheduling for CUDA-using processes
bpftrace -e 'tracepoint:sched:sched_switch /comm == "python3"/ {
               printf("switch: %s -> %s on cpu %d\n", args.prev_comm, args.next_comm, cpu);
             }'
```

---

## 3. libbpf CO-RE: Portable eBPF Programs Across Kernel/Driver Versions

### Architecture from libbpf-bootstrap

The libbpf-bootstrap repo provides the canonical scaffolding for writing portable CO-RE eBPF programs. The pattern has three files per program:

1. **`program.bpf.c`** -- the eBPF kernel-side program, compiled with `clang -target bpf`
2. **`program.c`** -- the user-space loader and event handler
3. **`program.h`** -- shared data structures between kernel and user space

The build produces a **skeleton header** (`program.skel.h`) via `bpftool gen skeleton`, which provides type-safe C APIs for loading, attaching, and reading maps.

### CO-RE mechanism

CO-RE (Compile Once Run Everywhere) solves the portability problem via BTF (BPF Type Format):

- Programs are compiled against `vmlinux.h` (generated from the current kernel's BTF)
- `BPF_CORE_READ()` macro generates relocations that libbpf resolves at load time
- The kernel's BTF information tells libbpf where struct fields actually live in the running kernel
- This means a program compiled on kernel 5.15 can run on kernel 6.8 without recompilation, as long as the field semantics are preserved

From Cassagnes2024 (Section 3.4): BTF provides a compact debug format (order of magnitude smaller than DWARF) that ships with the kernel. The verifier uses BTF for type safety, and CO-RE uses it for field offset resolution.

### Key patterns for rocket_surgeon

**uprobe.bpf.c example from the repo:**
```c
SEC("uprobe")
int BPF_KPROBE(uprobe_add, int a, int b) {
    bpf_printk("uprobed_add ENTRY: a = %d, b = %d", a, b);
    return 0;
}

SEC("uretprobe")
int BPF_KRETPROBE(uretprobe_add, int ret) {
    bpf_printk("uprobed_add EXIT: return = %d", ret);
    return 0;
}
```

**bootstrap.bpf.c shows the production pattern:**
- Ring buffer for high-throughput event delivery (`BPF_MAP_TYPE_RINGBUF`)
- Hash maps for correlating entry/exit events (`BPF_MAP_TYPE_HASH`)
- `bpf_ktime_get_ns()` for nanosecond timestamps
- `BPF_CORE_READ()` for portable kernel struct access
- `bpf_ringbuf_reserve()` / `bpf_ringbuf_submit()` for zero-copy event delivery

**User-space loader pattern (uprobe.c):**
```c
skel = uprobe_bpf__open_and_load();
uprobe_opts.func_name = "target_function";
skel->links.my_probe = bpf_program__attach_uprobe_opts(
    skel->progs.my_probe,
    -1 /* all pids */,
    "/path/to/libcuda.so",
    0 /* offset */,
    &uprobe_opts);
```

### Portability considerations for GPU tracing

CO-RE handles kernel struct changes but **not** user-space library changes. The CUDA driver library (`libcuda.so`) has versioned symbols (e.g., `cuMemAlloc_v2`), and function signatures can change between CUDA versions. Our approach:

1. Use symbol versioning (`_v2`, `_v3` suffixes) to target specific API versions
2. Fall back to ioctl tracing for forward compatibility (ioctl command numbers are more stable)
3. Use the open-gpu-kernel-modules (in quarantine) to decode ioctl commands

### Build system

libbpf-bootstrap uses a Makefile that:
1. Generates `vmlinux.h` from the running kernel's BTF via `bpftool btf dump`
2. Compiles `.bpf.c` files with `clang -target bpf -g -O2`
3. Generates skeleton headers via `bpftool gen skeleton`
4. Compiles user-space code linking against libbpf and the skeleton

For rocket_surgeon, we would integrate this into our build system, generating skeletons at build time and shipping pre-compiled BPF object files for common kernels.

---

## 4. Perfetto Integration: Trace Format and SQL Analysis

### Why Perfetto over Chrome Trace Format

Chrome Trace Event Format (JSON) is what ProfInfer currently uses for timeline visualization. Perfetto is strictly superior for our use case:

**Protobuf-based format:**
- Binary format, dramatically smaller than JSON (important for multi-GPU traces that can be GBs)
- Streaming-friendly -- traces can be written incrementally, no need to buffer the entire trace in memory
- Schema-defined -- `trace.proto` is the root, containing `repeated TracePacket packet`
- The format already has GPU-specific protos: `GpuRenderStageEvent`, `GpuCounterEvent`, `GpuMemTotalEvent`

**SQL-based trace analysis:**
- Perfetto's trace_processor loads traces into an SQLite-based engine
- Pre-built SQL tables: `slice`, `counter`, `thread`, `process`, `track`, `gpu_counter`, `gpu_slice`
- Custom SQL queries can extract exactly the metrics we need
- Example: `SELECT ts, dur, name FROM slice WHERE track_id IN (SELECT id FROM track WHERE name LIKE '%cuda%')`

**Existing GPU support in Perfetto protos:**

`gpu_render_stage_event.proto`:
- `event_id`, `duration` (ns), `hw_queue_iid`, `stage_iid`, `gpu_id` (multi-GPU), `context`, `submission_id`
- `ExtraData` for arbitrary key-value annotations
- This maps cleanly to CUDA kernel launches (event = kernel, hw_queue = stream, stage = kernel function)

`gpu_counter_event.proto`:
- Per-GPU counter values (int or double) with counter descriptors
- Maps to CUPTI/DCGM hardware counters

`gpu_mem_event.proto`:
- `gpu_id`, `pid`, `size` -- per-process GPU memory tracking

**TrackEvent for custom instrumentation:**

The SDK provides `TRACE_EVENT` macros for custom instrumentation:
```cpp
TRACE_EVENT("rendering", "DrawPlayer", "player_number", player_number);
TRACE_EVENT_BEGIN("category", "name");
TRACE_EVENT_END("category");
TRACE_COUNTER("category", "counter_name", value);
```

This is exactly the pattern rocket_surgeon needs for emitting tick-by-tick events as the debugger steps through the forward pass.

### Concrete integration plan

1. **eBPF probes emit events to ring buffers** (libbpf CO-RE programs)
2. **User-space daemon reads ring buffers and writes Perfetto TracePackets** (using the C SDK or raw protobuf)
3. **Custom TrackDescriptors** for:
   - Per-GPU tracks (one track per device)
   - Per-stream tracks (one track per CUDA stream per device)
   - Per-layer tracks (one track per transformer layer)
   - MoE expert tracks (which experts fire per layer)
4. **TrackEvents** for:
   - Kernel launches (slices with begin/end)
   - Memory transfers (slices with size annotations)
   - Synchronization points (instant events)
   - Activation snapshots (counter events with tensor norms/stats)
5. **Trace processor SQL** for post-hoc analysis:
   - Layer-by-layer latency breakdown
   - Memory transfer overlap with computation
   - Expert activation patterns over time (MoE)
   - GPU utilization per device in multi-GPU setups

---

## 5. NeutriNo Approach: Instruction-Level GPU Profiling via Assembly Probing

### Core insight

NeutriNo (Huang2025, OSDI'25) is the first platform-independent, programmable GPU kernel profiler that works at the assembly level. Its key insight: **GPU parallel assemblies (PTX for NVIDIA, GCNAsm for AMD) are the highest common layer** between ahead-of-time compilation (CUDA C++ via nvcc) and just-in-time compilation (Triton via LLVM). Probing at this layer achieves both fine granularity and broad compatibility.

### Three-component probe design

1. **Snippet:** The probe's actual code, written as assembly instructions. Uses helpers like `SAVE` for storing values to maps, `OUT`/`IN1`/`IN2` for reading registers. Uses `S_MEMTIME` for time profiling via special GPU clock registers (`%clock`, `%globaltimer`).

2. **Tracepoint:** Where the probe is injected, at the finest instruction level. Examples: `thread:start`, `thread:end`, specific `ld/st.global` instructions. Probes are inserted *before* or *after* matched instructions.

3. **Structured Map:** eBPF-inspired persistence format. Two levels:
   - Thread-level: `[#Grid, #Block, cap]` for value profiling (every thread saves)
   - Warp-level: `[#Grid, #Warp, cap]` for time profiling (only warp leader saves, reducing overhead)

### Virtualized execution model

NeutriNo probes are *virtual* -- they don't interfere with the original program:
- **Time separation:** Due to SIMT model, instructions within a thread execute sequentially. Probes inserted between instructions are guaranteed temporal separation.
- **Resource separation:** Probe registers are independent register groups. Logical registers are declared, and the assembler allocates physical registers without disturbing the original program.
- **Verification:** Probes that modify original registers, change control flow, or use shared memory are rejected.

### Hook driver architecture (directly relevant to rocket_surgeon)

NeutriNo's hook driver uses **LD_PRELOAD** to intercept the CUDA driver library. It:
1. Catches `cuModuleLoad` to capture loaded GPU binaries
2. Catches `cuLaunchKernel` to intercept kernel launches
3. For each uncached kernel: objdumps the binary, invokes the probe engine, reassembles, launches the probed kernel
4. Allocates probe buffers on GPU, launches probed kernel, copies results back to CPU

This is a production-ready pattern for intercepting GPU execution that rocket_surgeon should adopt, but we would use it for *intervention* (modifying activations between ticks) rather than just profiling.

### Performance characteristics

- **Lightweight probes** (block_sched, gmem_bytes, tensorop_count): 1.04x average slowdown, +4 registers
- **Heavy probes** (dmat -- densified memory access timeline): 7.12x slowdown, but memory requirements scale sublinearly with model size
- Profiled Llama-3-8B with batch size 256 using only 64MB additional GMEM for lightweight probes
- NeutriNo latency is decomposed into <1% prologue, kernel overhead, and epilogue (vs. Nsight Compute which has much higher exposed latency)

### Relevance to rocket_surgeon

NeutriNo operates at a *complementary* layer to eBPF:
- eBPF sees the host-side boundary (kernel launches, memory ops, synchronization)
- NeutriNo sees *inside* the GPU kernel (memory access patterns, instruction timing, warp scheduling)
- Together they provide full-stack observability

The DMAT visualization is particularly relevant: it shows **densified memory access timelines** with page-level granularity and parallelism density. This would let rocket_surgeon users visualize how attention patterns, weight accesses, and activation memory behave at the hardware level -- information that no Python-level tool can provide.

---

## 6. ProfInfer: eBPF-Based LLM Inference Profiling -- Directly Relevant Patterns

### Architecture

ProfInfer (Zheng2025) is the most directly applicable reference for rocket_surgeon's observability layer. It targets llama.cpp but the patterns generalize to any inference engine.

**Three-layer trace architecture:**
1. **ProfDAG** -- computational graph structure (which operators, how they connect)
2. **ProfTime** -- timeline visualization (when operators execute, on which threads/backends)
3. **ProfStat** -- statistical analysis (per-operator performance, cross-token trends, MoE expert patterns)

### Multi-granularity probing (the key innovation)

ProfInfer attaches probes at four levels simultaneously:

1. **Token level:** uprobe on `llama_decode` captures batch size and timestamps for TTFT and TPOT metrics
2. **Graph level:** uprobe on `ggml_backend_graph_compute_async` captures backend type and graph-level timing
3. **Operator level:** uprobe on `ggml_compute_forward` (CPU), `ggml_cl_compute_forward` (GPU) captures tensor info and PMC data
4. **Scheduler level:** kernel tracepoints `sched_switch` and `sched_wakeup` capture thread scheduling

### Adaptive overhead control

ProfInfer dynamically adjusts tracing overhead based on inference speed:
- Monitors decoding speed (tokens/sec) against QoS threshold
- When performance degrades, disables operator-level probes (most expensive)
- Token and graph-level probes (lowest overhead) remain active
- This is implemented by writing to BPF maps that control probe behavior

Overhead measurements (Table 5):
- Token + graph tracing only: 0.1% speed decrease
- Full tracing with BCC: 2.8-4.0% speed decrease
- Full tracing with libbpf: 1.7% speed decrease (C-based, lower overhead)

### MoE expert tracing (Section 3.5)

ProfInfer demonstrates MoE-specific profiling by attaching a uprobe to `ggml_compute_forward_mul_mat_id` and reading expert IDs through two-level pointer dereferencing of `ggml_tensor`. This reveals:
- Which experts activate per iteration
- Expert activation frequency distributions
- Correlation between expert distance (reuse) and execution time (memory eviction)
- The bottleneck of MoE inference is **disk I/O** (expert weight paging), not memory bandwidth

This is directly relevant to rocket_surgeon's MoE support requirement.

### PMC integration (Section 3.4)

ProfInfer reads hardware performance counters per operator:
- `l3d_cache_refill` -- L3 cache misses (per-core)
- `mem_access_wr` -- memory writes (per-core)
- `major-faults` -- page faults (software)
- `cycles` / `idle-backend-cycles` -- CPU utilization

Uses `open_perf_event` to configure PMC file descriptors and `perf_read` in probe handlers to read counters at operator entry/exit.

---

## 7. Practical Observability Stack for rocket_surgeon

### Layer model

```
+------------------------------------------------------------------+
| Layer 5: rocket_surgeon TUI / LLM Protocol                      |
|   Tick-by-tick display, intervention UI, structured protocol     |
+------------------------------------------------------------------+
| Layer 4: Perfetto Trace Format + SQL Analysis                    |
|   Protobuf TracePackets, trace_processor SQL, ui.perfetto.dev    |
+------------------------------------------------------------------+
| Layer 3: User-Space Trace Daemon                                 |
|   Reads BPF ring buffers, correlates events, emits TracePackets  |
|   Manages probe lifecycle, adaptive overhead control             |
+------------------------------------------------------------------+
| Layer 2: eBPF Probes (libbpf CO-RE)                              |
|   uprobes on libcuda.so / libnccl.so / libcudnn.so              |
|   Kernel tracepoints: sched_switch, sched_wakeup                |
|   ioctl tracing on /dev/nvidia* for low-level driver ops         |
+------------------------------------------------------------------+
| Layer 1: GPU-side Probing (NeutriNo-style, future)               |
|   Assembly-level probes for intra-kernel profiling               |
|   DMAT for memory access pattern visualization                   |
|   Requires hook driver (LD_PRELOAD) for kernel interception      |
+------------------------------------------------------------------+
```

### Phase 1: Host-side eBPF observability (build first)

**Tool:** Custom libbpf CO-RE programs (not bpftrace -- we need programmatic control and lower overhead)

**What to build:**
1. `cuda_tracer.bpf.c` -- uprobes on libcuda.so functions:
   - `cuLaunchKernel` entry/exit (kernel name via cuModuleGetFunction correlation)
   - `cuMemAlloc_v2` / `cuMemFree_v2` (device memory lifecycle)
   - `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2` (transfer tracking with sizes)
   - `cuCtxSynchronize` / `cuStreamSynchronize` (sync point latency)

2. `nccl_tracer.bpf.c` -- uprobes on libnccl.so (multi-GPU collective ops):
   - `ncclAllReduce` / `ncclAllGather` / `ncclReduceScatter`
   - Captures timing, data sizes, communicator info

3. `sched_tracer.bpf.c` -- kernel tracepoints:
   - `sched_switch` / `sched_wakeup` for thread scheduling visibility
   - Filtered to CUDA-using PIDs only

4. `rs_trace_daemon` -- user-space daemon:
   - Reads ring buffers from all probes
   - Correlates kernel launches with module/function names
   - Emits Perfetto TracePackets to a file or streaming endpoint
   - Implements adaptive overhead control (ProfInfer pattern)

**Data flow:**
```
libcuda.so call -> uprobe fires -> BPF program -> ring buffer -> trace daemon -> Perfetto trace
```

### Phase 2: Perfetto integration (trace format and analysis)

**Custom Perfetto tracks:**
- `gpu.{device_id}.stream.{stream_id}` -- per-stream kernel execution
- `gpu.{device_id}.memory` -- memory allocation/deallocation counter track
- `gpu.{device_id}.transfer` -- H2D/D2H transfer slices
- `nccl.{communicator_id}` -- collective operation slices
- `model.layer.{N}` -- per-layer timing (correlated from kernel names)
- `model.layer.{N}.expert.{E}` -- MoE expert activation (from ProfInfer pattern)

**SQL analysis queries for the debugger:**

```sql
-- Layer-by-layer forward pass breakdown
SELECT layer_id, SUM(dur) as total_ns, COUNT(*) as kernel_count
FROM gpu_slice
WHERE name LIKE 'layer_%'
GROUP BY layer_id ORDER BY layer_id;

-- Memory transfer overlap with computation
SELECT s1.name as kernel, s2.name as transfer,
       MAX(s1.ts, s2.ts) as overlap_start,
       MIN(s1.ts + s1.dur, s2.ts + s2.dur) as overlap_end
FROM gpu_slice s1, gpu_slice s2
WHERE s1.track_id != s2.track_id
  AND s1.ts < s2.ts + s2.dur AND s2.ts < s1.ts + s1.dur;

-- MoE expert activation frequency
SELECT expert_id, COUNT(*) as activations, AVG(dur) as avg_ns
FROM slice WHERE category = 'moe_expert'
GROUP BY expert_id ORDER BY activations DESC;
```

### Phase 3: GPU-side probing (future, NeutriNo-style)

This is the highest-value, highest-effort layer. Requires:
1. Hook driver (LD_PRELOAD shim for libcuda.so)
2. Probe engine (PTX/SASS disassembly, probe injection, reassembly)
3. DSL compiler (Python tracing DSL to platform-specific assembly probes)

**What it would give us that eBPF cannot:**
- Per-instruction timing within GPU kernels
- Memory access patterns (which pages, when, by which threads)
- Warp scheduling behavior (tailing effects, synchronization costs)
- DMAT visualization of attention kernel memory behavior
- Register pressure and spilling analysis

### Tool selection rationale

| Need | Tool | Why |
|---|---|---|
| Rapid prototyping, one-off investigations | bpftrace | One-liner syntax, zero build step, immediate results |
| Production tracing in rocket_surgeon | libbpf CO-RE | Low overhead (1.7% per ProfInfer), portable, programmatic |
| Trace storage and analysis | Perfetto | Protobuf format, SQL queries, GPU-aware schema, web UI |
| Intra-kernel profiling | NeutriNo (future) | Only tool that sees inside GPU kernels, assembly-level |
| Hardware counters (CUPTI) | Direct CUPTI + Perfetto | CUPTI provides GPU-side PMC data, emit as Perfetto counters |
| Multi-GPU collective ops | NCCL uprobes | Observe AllReduce/AllGather timing and data flow |

### What NOT to use

- **BCC Python API for production:** Higher overhead than libbpf (ProfInfer measured 2.8-4.0% vs 1.7%). Fine for prototyping but not for a debugger that needs to minimize perturbation.
- **Chrome Trace Format:** JSON-based, enormous file sizes, no SQL analysis. Use Perfetto instead (ProfInfer already converts to Chrome Trace for Perfetto visualization).
- **CUPTI alone:** Provides GPU-side counters but requires synchronization after every kernel for counter reads (per Malony2011), dramatically altering execution behavior. Use selectively, not as primary tracing mechanism.
- **nsys/Nsight Systems:** Proprietary, high overhead, cannot be embedded in our tool. Use for validation/comparison only.

### Critical limitation to document

eBPF observes the **host-side control plane** of GPU execution. It cannot observe:
- What happens inside a GPU kernel (that is NeutriNo's domain)
- GPU-side timing with sub-microsecond accuracy (only host-side timestamps)
- GPU memory access patterns
- Warp-level scheduling decisions
- Cache hit/miss rates on the GPU

For rocket_surgeon's "step through one tick at a time" paradigm, eBPF provides the **orchestration layer** (when kernels launch, what memory moves, how long sync takes) while NeutriNo-style probing would provide the **introspection layer** (what happened inside that kernel). The combination is greater than either alone.

---

## References

- Huang & Wu, "NeutriNo: Fine-grained GPU Kernel Profiling via Programmable Probing," OSDI'25
- McCanne & Jacobson, "The BSD Packet Filter: A New Architecture for User-level Packet Capture," USENIX Winter 1993
- Gbadamosi et al., "The eBPF Runtime in the Linux Kernel," arXiv:2410.00026v2, 2024
- Zou et al., "ProfInfer: An eBPF-based Fine-Grained LLM Inference Profiler," arXiv:2601.20755v2, 2025
- Malony et al., "Parallel Performance Measurement of Heterogeneous Parallel Systems with GPUs," 2011
