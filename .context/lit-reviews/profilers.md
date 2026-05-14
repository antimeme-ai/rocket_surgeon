---
topic: CPU and GPU profilers — perf, flamegraphs, py-spy, Nsight, CUPTI, torch.profiler, ROCm
status: draft
created: 2026-05-14
sources: Brendan Gregg, NVIDIA docs, PyTorch docs, various profiler repos
---

# Profilers: Lit Review

CPU and GPU profiling tools relevant to debugging neural network execution at high fidelity.

## CPU Profilers

### perf (Linux perf_events)
- Hardware-based profiling via Performance Monitoring Units (PMUs)
- Counts/samples: instructions, cache misses, branch mispredictions
- Sampling mode: fixed-rate statistical sampling of call stacks (low overhead)
- `perf record` captures data, `perf report`/`perf stat` analyze
- Works with Python but better on C/C++ (symbol resolution)
- **For us**: CPU-side analysis of Python dispatch overhead, hardware counter data for microarchitecture issues

### Brendan Gregg's Flamegraph Tooling
- Hierarchical visualization of profiled stack traces
- X-axis = time spent (width), Y-axis = call depth, color = function
- Interactive SVG output
- **Variants**:
  - On-CPU: functions consuming CPU time
  - Off-CPU: time spent waiting (sync, I/O) — bidirectional layout showing waker/sleeping stacks
  - Differential: overlay/compare two profiles
- **For us**: off-CPU analysis reveals GPU synchronization bottlenecks, differential graphs compare before/after surgical interventions

### py-spy
- Out-of-process sampling profiler written in Rust
- Reads CPython interpreter state directly from target process memory (process_vm_readv)
- No instrumentation required, ~1% overhead
- `--native` flag captures C/C++/Cython frames alongside Python
- **For us**: profile Python-to-CUDA dispatch, capture PyTorch C++ kernel implementations in call stacks

### Austin
- Pure C frame stack sampler for CPython
- Higher sampling rates than py-spy, lower overhead
- Delegates aggregation to external tools
- Better for production/continuous profiling

### Scalene
- CPU+GPU+memory profiler at line level
- 10-20% overhead
- Separates Python time from native code time
- GPU time reporting on NVIDIA systems
- Per-line memory allocation tracking, leak detection
- AI-powered optimization suggestions
- **For us**: combined CPU+GPU+memory view identifies memory-bound vs compute-bound transformer bottlenecks

### Intel VTune
- Microarchitecture-level analysis using PMUs
- Top-down microarchitecture analysis (TMA): front-end, back-end, execution pipeline
- Cache analysis, vectorization analysis, roofline analysis
- Not GPU-focused but critical for CPU bottleneck diagnosis

## GPU Profilers

### NVIDIA Nsight Systems
- System-wide profiling via CUPTI
- Unified CPU-GPU timeline visualization
- **Key capabilities**:
  - CUDA kernel tracing: API time, queue time, kernel time
  - Memory operation tracing
  - Synchronization detection: GPU starvation, unnecessary waits
  - Timeline correlation: synchronized CPU-GPU timestamps
- Identifies where GPU sits idle due to CPU not launching kernels
- **For us**: critical for multi-GPU debugging, shows CPU-GPU sync issues, timeline reveals staggered compute

### NVIDIA Nsight Compute
- Kernel-level profiling for individual CUDA kernels
- Expert system for automated bottleneck identification
- **Analysis areas**:
  - Occupancy: CTA occupancy, physical resource constraints (registers, shared memory)
  - Memory: throughput as % of peak, cache behavior (L1/L2/L3)
  - Warp-level: scheduling overhead, load imbalance, instruction-level detail
  - SM-level: efficiency, Tensor Core activity, instruction throughput
  - Speed of Light (SOL): bottleneck classification
- **For us**: diagnoses why individual transformer kernels are slow, reveals if attention kernels underutilize hardware

### NVIDIA CUPTI (CUDA Profiling Tools Interface)
- Low-level C/Python API underneath all Nsight tools
- **Core APIs**:
  - Activity API: records executed work with minimal overhead
  - Callback API: notifications on CUDA events
  - Range Profiling API: selective profiling of code regions (Turing+)
  - PC Sampling API: program counter sampling for instruction-level analysis
  - SASS Metric API: low-level shader assembly metrics
- Worker thread design minimizes perturbation
- **For us**: foundation layer for custom profiling, can build custom instrumentation for surgical intervention

### torch.profiler
- Built-in PyTorch profiler, records CPU and CUDA operations with correlation
- Per-operation timing and resource usage
- Chrome trace format export (`prof.export_chrome_trace()`)
- Memory profiling with `record_memory=True`
- Stack trace capture for attribution
- Works with torch.compile, supports distributed training
- **For us**: native integration, direct correlation between torch operations and CUDA kernels

### AMD ROCm Profiling
- **rocprofv3**: command-line tracing of device activity, raw GPU counter collection
- **rocprof-sys**: unified trace of host, device, MPI communication; call-stack sampling; binary instrumentation
- **omnitrace/rocprof-compute**: comprehensive parallel profiling (C, C++, HIP, OpenCL, Python); multi-mode (binary instrumentation, sampling, user regions); interactive web-based visualization
- AMD tools are open-source vs NVIDIA's proprietary CUPTI
- Comparable maturity but smaller ecosystem

## Cross-Cutting

### Chrome Trace Format
- JSON array of event objects with fields: ph (phase), ts (timestamp µs), dur, pid, tid, name, cat, args
- Event types: B/E (begin/end slices), X (complete), I (instant), C (counter), M (metadata)
- De facto standard: PyTorch, TensorFlow, custom instrumentation all emit it
- Viewed by Chrome's chrome://tracing or Perfetto (modern, web-based, SQL queries)
- **For us**: natural output format, multi-GPU traces combine into single timeline, surgical interventions annotatable as custom events

### Perfetto
- Production-grade open-source tracing framework (Google)
- System-wide and app-level traces in unified timeline
- High-performance tracing daemons
- SQL-based analysis library for programmatic queries
- Scales to multi-GB traces
- **For us**: modern alternative to Chrome tracing, SQL analysis enables programmatic metric extraction

### OpenTelemetry for Distributed Tracing
- Standardized APIs for traces/metrics across services
- Trace = tree of spans, each recording one operation
- Context propagation carries trace IDs across process/network boundaries
- GPU metrics via DCGM exporter (clock speeds, ECC errors, NVLink throughput)
- **For us**: multi-GPU distributed debugging at scale, correlate request latency with GPU state

## Transformer-Specific Profiling

### Memory Profiling
- Model weights (static), gradients (2-4 bytes/param), optimizer states (up to 12 bytes/param Adam), activations (grow with batch*seq_len)
- `torch.cuda.memory._record_memory_history()` + `_dump_snapshot()`: fine-grained allocation timeline with stack traces
- Memory snapshot viewer: active memory timeline, allocation events, OOM failure analysis
- **For us**: activation memory bloat identification, opportunity detection for activation checkpointing

### NCCL Communication Profiling
- NCCL Inspector: plugin-based, per-communicator per-collective logging (JSON)
- Metrics: algorithmic bandwidth, bus bandwidth, execution time, message sizes
- Loaded via environment variable (no code changes)
- **For us**: critical for multi-GPU, reveals if communication steals compute cycles

### Kernel Fusion
- Merging sequential GPU operations into single fused kernel
- Eliminates intermediate memory reads/writes
- 5µs overhead per kernel launch; modern LLMs ~3000 launches/token with aggressive fusion
- **For us**: can insert unfusion for fine-grained profiling, selectively fuse for optimization

## Design Implications for rocket_surgeon

1. **Multi-layer profiling stack**: torch.profiler (Python ops) + CUPTI (kernel details) + Chrome trace format (output)
2. **Timeline correlation**: synchronize CPU Python execution with GPU kernel timeline
3. **Memory debugging**: PyTorch memory snapshots for activation lifetime, track per-tensor allocation through forward pass
4. **Multi-GPU communication**: NCCL Inspector for collective bottlenecks, per-GPU timeline aggregation
5. **Output format**: Chrome trace format primary (JSON, widely viewable), Perfetto for advanced analysis
6. **Flame graphs**: off-CPU analysis for synchronization bottlenecks, differential for before/after intervention comparison

## Sources

- brendangregg.com (perf, flamegraphs, off-CPU analysis, AI flame graphs)
- github.com/benfred/py-spy
- github.com/P403n1x87/austin
- github.com/plasma-umass/scalene
- developer.nvidia.com (Nsight Systems, Nsight Compute, CUPTI docs)
- docs.pytorch.org (profiler, memory)
- rocm.docs.amd.com (profiling tools)
- perfetto.dev
- opentelemetry.io
- NCCL Inspector blog (developer.nvidia.com)
