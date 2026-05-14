---
date: 2026-05-14
summary: Project kickoff — repo scaffolding, full literature sweep across 13 domains
---

# Session: 2026-05-14 — Project Kickoff

## What happened
- Initialized git repo (local only, no remotes)
- Set up JSMNTL directory structure mirroring ../evals/ patterns
- Launched 13 parallel research agents across 3 waves covering the full landscape
- Wrote 13 lit reviews from research results

### Wave 1 (6 domains)
traditional-debuggers, pytorch-hooks-internals, tui-frameworks, nnsight-transformerlens, pyvene-inseq-circuitsvis, transformer-surgery-papers

### Wave 2 (5 domains)
profilers, systems-observability, ml-framework-internals, gpu-compute-platforms, rtos-realtime-gpu

### Wave 3 (2 domains)
probes, llm-native-ux

## Literature Sweep: Cross-Cutting Synthesis

### The Architecture That Emerges
Every research thread converges on the same three-layer architecture:

1. **Core Engine** — Pure state machine for transformer forward pass inspection and intervention. No UI. Checkpoint + replay model (from rr/TTD) for reverse stepping.

2. **Machine Interface** — Structured protocol (JSON-RPC or DAP-like) as the PRIMARY interface. Everything goes through this. LLMs and humans both consume it. Proven by GDB-MI, DAP, Neovim RPC, tmux control mode.

3. **TUI** — Ratatui-based terminal UI that consumes the machine interface. A client, not the system. Other clients (Python scripts, notebooks, web, LLM orchestrators) are equally first-class.

### Key Technical Decisions Emerging

**Interception mechanism**: PyTorch forward hooks as primary. But with serious caveats:
- DDP silently ignores pre-registered hooks (must register inside forward())
- torch.compile silently ignores post-compilation hooks
- FSDP uses hooks internally and custom hooks can interfere
- MoE routing requires hooking gating network, per-expert forwards, auxiliary loss

**Don't re-implement models** (TransformerLens trap). Instead:
- Wrap existing HuggingFace models (nnsight approach)
- Standardize naming across architectures (nnterp approach)
- Use hook pairs (pre + post) for layer I/O capture

**Reverse stepping = checkpoint + forward replay** (rr insight). Don't reverse operations. Record forward pass state at strategic points, replay forward from nearest checkpoint. Activation checkpointing alignment: checkpoint every sqrt(n)-th layer.

**Interventions as data, not code** (pyvene insight). Serialize as JSON for versioning, sharing, LLM-driven synthesis. Decouple specification from execution.

**SAEs are essential** for interpretable interventions. Support loading pre-trained SAEs, querying features, steering via coefficient manipulation.

### Competitive Landscape
- **nnsight**: closest to our approach (wraps existing models, multi-GPU via vLLM) but not a debugger — research tool, deferred execution semantics, no step-through
- **TransformerLens**: best ecosystem (50+ papers) but re-implements models, no multi-GPU, no structured protocol
- **OpenAI Transformer Debugger**: closest to our UX concept but no multi-GPU, no LLM-native interface, no step-through
- **pyvene**: best intervention framework (serializable configs) but not a debugger

**Our unique angle**: the debugger metaphor (breakpoints, step, inspect, modify) applied to neural nets, with LLM-native interface as first-class, multi-GPU support, and MoE awareness.

### Waves 2+3 Synthesis: What the Additional Research Changes

**Probe model as unifying abstraction**: DTrace's `provider:module:function:name` naming maps directly to `model:layer:component:event`. Systems probes (eBPF, USDT, tracepoints) and neural network probes (linear probes, logit lens, SAE features) are the SAME pattern — observation points with composable hooks. Zero-cost-when-off via NOP placeholders.

**LLM-native protocol design**: 5-7 composable primitives (step, inspect, intervene, probe, checkpoint, evaluate, status). State in every response. DAP/LSP-inspired capability negotiation at init. Strict JSON schemas. No system prompt dependency. The tool's interface IS the documentation.

**GPU determinism is achievable**: single CUDA stream + cuBLAS deterministic mode + fixed batch composition + same GPU arch = bit-reproducible forward passes. Checkpoint + replay validated by both rr/TTD architecture and NVIDIA's cuda-checkpoint (though full GPU checkpoint too slow — lightweight activation-only checkpoints needed).

**PREEMPT_RT for host-side control**: merged into mainline Linux 6.12, gives ~100µs latency for CPU-side debugger scheduling. Not hard real-time but sufficient for interactive stepping where operations complete at cudaDeviceSynchronize boundaries.

**Instrumentation stack**: CUPTI (kernel launches, memory ops) → PyTorch hooks (tensor operations) → eBPF (driver-level syscalls) → Chrome Trace Format output. Multi-layer, each catches what the others miss.

**Framework internals**: PyTorch dispatcher (DispatchKey interception) is the primary hook point. torch.compile/Dynamo/Inductor pipeline has specific interception stages (FX graph transformation before codegen). CUDA Graphs bypass normal API entry points — need graph-aware instrumentation.

## Open Questions
- Language split: pure Rust? Rust core + Python orchestration? (leaning Rust core + Python bindings via PyO3)
- Hook registration strategy for compiled/distributed models (torch.compile silently drops hooks)
- Protocol: extend DAP vs custom JSON-RPC? (leaning DAP extension — proven pattern, IDE ecosystem)
- MoE routing visualization in TUI
- SAE integration: bundle pre-trained or load externally?
- eBPF vs CUPTI for primary GPU instrumentation (likely both, different layers)
- Apple Silicon support: Metal backend immature, ANE undocumented — worth it for dev ergonomics?

## Next Steps (per JSMNTL)
1. **BEAD-0001**: Architecture plan synthesizing all 13 lit reviews
2. ADR: language split decision
3. ADR: protocol design decision (DAP extension vs custom)
4. ADR: probe model design
5. TCK specs for core stepping semantics
6. BEAD-0002: Clone reference implementations to quarantine
