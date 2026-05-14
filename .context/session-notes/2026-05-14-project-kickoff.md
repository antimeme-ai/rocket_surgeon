---
date: 2026-05-14
summary: Project kickoff — repo scaffolding, full literature sweep across 6 domains
---

# Session: 2026-05-14 — Project Kickoff

## What happened
- Initialized git repo (local only, no remotes)
- Set up JSMNTL directory structure mirroring ../evals/ patterns
- Launched 6 parallel research agents covering the full landscape
- Wrote 5 lit reviews from research results

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

## Open Questions
- Language split: pure Rust? Rust core + Python orchestration? Need to decide.
- Hook registration strategy for compiled/distributed models
- Protocol design: extend DAP? Custom JSON-RPC? Something new?
- How to handle MoE routing visualization in a TUI
- SAE integration: bundle pre-trained SAEs or load externally?

## Next Steps (per JSMNTL)
1. Written sub-plan for architecture
2. ADR: language split decision
3. ADR: protocol design decision
4. TCK specs for core stepping semantics
5. Reference implementations to quarantine (nnsight, TransformerLens, pyvene, circuitsvis)
