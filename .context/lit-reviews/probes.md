---
topic: Probes — DTrace, USDT, eBPF, tracepoints, kprobes/uprobes, OpenTelemetry, neural network probes, probe design patterns
status: draft
created: 2026-05-14
sources: DTrace docs, eBPF docs, Linux kernel docs, OpenTelemetry, Prometheus, ML interpretability research
---

# Probes: Lit Review

Probes as a first-class abstraction — from systems tracing to neural network interpretability — and what rocket_surgeon should steal.

## Systems Probes

### DTrace Probe Model
- Canonical probe description: `provider:module:function:name`
- **Provider**: source of probes (syscall, fbt, pid, etc.)
- **Module**: kernel module or library
- **Function**: specific function being instrumented
- **Name**: event within the function (entry, return, specific point)
- D language for predicates, aggregations, printf-like output
- Zero overhead when disabled — probes are NOPs until enabled
- **Key insight**: the 4-tuple naming convention makes probes discoverable and composable

### USDT (User Statically-Defined Tracing)
- Application-embedded probe points (like DTrace probes in userspace)
- Compile-time NOP instructions, dynamically patched to traps when enabled
- Python: `sys.monitoring` (3.12+) for low-overhead event callbacks
- Ruby, Node.js, JVM all support USDT
- **For us**: embed USDT probes in rocket_surgeon's core engine for zero-cost-when-off observability

### Linux Tracepoints
- Static probe points in kernel source (TRACE_EVENT macro)
- Predefined, stable ABI (unlike kprobes which break across versions)
- Categories: sched, irq, block, net, gpu (driver-specific)
- `/sys/kernel/debug/tracing/available_events` lists all
- **For GPU**: NVIDIA driver exposes some tracepoints for context switches, memory operations

### kprobes / uprobes
- **kprobes**: dynamic instrumentation of ANY kernel function entry/return
- **uprobes**: same for userspace functions — patches instructions at runtime
- kretprobes for return value capture
- Break across kernel/library versions (no stable ABI guarantee)
- eBPF programs attach to k/uprobes for safe, fast tracing
- **For us**: uprobes on libcuda.so functions for dynamic GPU API tracing

### eBPF as Probe Infrastructure
- Probes attach at: kprobes, uprobes, tracepoints, USDT, perf_events, cgroup hooks, XDP, TC
- BPF maps for stateful aggregation (histograms, counters, per-CPU arrays)
- Tail calls for probe chaining (up to 33 deep)
- Ring buffers for efficient kernel-to-userspace data transfer
- **CO-RE**: write once, run across kernel versions via BTF relocation
- **Probe composition**: multiple eBPF programs on same probe point, each with different logic

### Prometheus / Metrics
- Pull-based model: scrape `/metrics` endpoint at intervals
- Four metric types: counter (monotonic), gauge (up/down), histogram (bucketed distribution), summary (quantiles)
- Labels for dimensionality (gpu_id, layer_name, operation_type)
- PromQL for querying and alerting
- **For us**: expose rocket_surgeon metrics (checkpoint latency, step duration, memory usage) as Prometheus metrics for dashboard integration

### OpenTelemetry
- Unified framework: traces (distributed request flow), metrics (aggregated measurements), logs (discrete events)
- **Traces**: spans with parent-child relationships, context propagation across process boundaries
- **Semantic conventions**: standardized attribute names for common concepts
- Auto-instrumentation for Python, Java, Go, etc.
- **For us**: instrument multi-GPU stepping as distributed traces — each GPU's work is a span, collective operations connect them

### Structured Logging
- JSON-structured events with typed fields
- Correlation IDs for request-level tracing
- Log levels as probe granularity control (DEBUG = fine, INFO = coarse)
- **For us**: every stepping event emits structured JSON — same data feeds both human TUI and LLM client

## Neural Network Probes

### Linear Probes (Diagnostic Classifiers)
- Train simple classifier on frozen intermediate representations
- If linear probe achieves high accuracy → information is linearly accessible at that layer
- **Probing methodology**: freeze model, extract representations at layer L, train logistic regression on downstream task
- Standard probe tasks: POS tagging, dependency parsing, NER, semantic role labeling
- Probe complexity matters: overly expressive probes find information that isn't meaningfully "in" the representation

### Structural Probes
- Test whether syntax trees are embedded in representation geometry
- Distance probe: L2 distance between word representations ≈ tree distance
- Depth probe: norm of transformed representation ≈ tree depth
- Learned linear transformation B maps representations into structural subspace

### Causal vs Correlational Probes
- **Correlational**: linear probe shows information IS there (necessary but not sufficient)
- **Causal**: intervention shows information is USED for the task
- Amnesic probing: remove information via null-space projection, measure task degradation
- Causal probing is what rocket_surgeon enables: intervene on representations, observe downstream effects

### Logit Lens / Tuned Lens
- **Logit lens**: project intermediate layer representations through final unembedding matrix → see what token the model "would predict" at each layer
- **Tuned lens**: learned affine transformation per layer (better calibrated than raw logit lens)
- **Patchscope**: generalization — decode any hidden representation through any model's generation
- **For us**: logit lens is a natural "probe view" for the debugger — show predicted token distribution at each tick

### Activation Patching / Causal Tracing
- Replace activations at specific positions/layers with clean/corrupted versions
- Measure effect on output to localize which components matter
- **Path patching**: trace causal paths through the computational graph
- Foundation of mechanistic interpretability (Meng et al., Conmy et al.)
- **For us**: this IS the surgery — rocket_surgeon makes this interactive and iterative

### Sparse Autoencoders (SAEs) as Probes
- Decompose activation vectors into sparse, interpretable features
- Each feature = direction in activation space with semantic meaning
- Can probe individual features rather than raw dimensions
- **For us**: SAE features as named probe points — "probe the 'is_proper_noun' feature at layer 12"

## Probe Design Patterns

### Probe Points vs Probe Hooks
- **Probe point**: WHERE to observe (a location in the computational graph)
- **Probe hook**: WHAT to do when the point fires (log, aggregate, intervene, checkpoint)
- Separation enables: same point, different hooks; same hook, different points
- **Registration**: declarative (specify what you want) not imperative (specify how to get it)

### Static vs Dynamic Probes
- **Static**: compiled into the system (USDT, tracepoints). Zero overhead when off. Must be planned.
- **Dynamic**: attached at runtime (kprobes, uprobes, eBPF). Flexible. Slight overhead even when off.
- **For us**: static probes at layer boundaries (always available), dynamic probes for arbitrary tensor inspection

### Probe Multiplexing
- Multiple consumers on single probe point (fan-out)
- Each consumer sees same data, applies own logic
- Example: one hook logs, another checkpoints, third checks invariants
- Priority ordering for hook execution
- **For us**: TUI visualization hook + LLM analysis hook + checkpoint hook all on same layer boundary

### Probe Filtering and Sampling
- Predicates: only fire when condition met (e.g., "only when loss > threshold")
- Sampling: fire every Nth event, or probabilistic
- Aggregation: compute statistics in-probe, emit summaries not raw data
- **Critical for GPU**: can't inspect every tensor at every layer at every token — need smart filtering

### Probe Composition
- Chain probes: output of one feeds input of next
- Conditional probes: probe A enables/disables probe B
- Hierarchical: coarse probe triggers fine-grained sub-probes
- **For us**: "if attention entropy is high at layer 8, enable per-head inspection at layers 8-12"

### Probe Lifecycle
- **Registration**: declare probe point + hook
- **Arming**: enable the probe (transition from NOP to active)
- **Firing**: probe triggers, hook executes
- **Disarming**: disable without removing
- **Deregistration**: remove entirely
- DTrace and eBPF both follow this lifecycle. rocket_surgeon should too.

## Design Implications for rocket_surgeon

### Probe as First-Class Abstraction
```
probe = {
  point: "layer.12.attention.output",   // WHERE (hierarchical naming)
  hook: "checkpoint",                     // WHAT (from registry)
  filter: "entropy > 2.0",               // WHEN (predicate)
  enabled: true                           // lifecycle state
}
```

### Naming Convention (DTrace-inspired)
- `model:layer:component:event`
- Examples:
  - `llama:12:attention:output` — attention output at layer 12
  - `llama:*:mlp:input` — MLP input at ALL layers (wildcard)
  - `llama:12:attention.head.7:output` — specific head
  - `mixtral:8:router:decision` — MoE routing decision

### Built-in Hook Types
1. **inspect**: read tensor, return summary stats or full data
2. **checkpoint**: save state for backward stepping
3. **intervene**: modify tensor in-place (the surgery)
4. **assert**: check invariant, break if violated
5. **trace**: emit structured event to timeline
6. **aggregate**: accumulate statistics across ticks

### Zero-Cost When Off
- Probe points at layer boundaries are always present but NOPs when no hook registered
- Dynamic probes for arbitrary tensor locations attached at runtime
- GPU-side: probes trigger at cudaDeviceSynchronize boundaries, not mid-kernel

### Probe Discovery
- `list probes` returns all available probe points with types and descriptions
- Wildcard queries: `list probes llama:*:attention:*`
- Self-documenting: each probe point declares its tensor shape, dtype, semantic meaning
- **LLM ergonomic**: an LLM client can discover what's observable without documentation

## Sources

- DTrace: Dynamic Tracing in Oracle Solaris, Mac OS X, and FreeBSD (Gregg & Mauro)
- eBPF docs (ebpf.io), bpftrace reference
- Linux kernel tracing docs (ftrace, tracepoints, kprobes)
- OpenTelemetry specification (opentelemetry.io)
- Prometheus docs (prometheus.io)
- Belinkov (2022): Probing Classifiers — survey of neural network probing
- Hewitt & Manning (2019): Structural Probes for Finding Syntax in Word Representations
- Elazar et al. (2021): Amnesic Probing
- nostalgebraist: Logit Lens blog post
- Belrose et al. (2023): Tuned Lens
- Ghandeharioun et al. (2024): Patchscope
- Cunningham et al. (2023): Sparse Autoencoders for interpretable features
- Meng et al. (2022): ROME — causal tracing
- Conmy et al. (2023): Automated Circuit Discovery
