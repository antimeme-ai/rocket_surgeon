# rocket_surgeon: Architecture Plan

Synthesized from 13 lit reviews across debuggers, ML frameworks, GPU platforms, profilers, systems observability, RTOS, probes, and LLM-native UX.

## System Overview

Three-layer architecture where the machine interface is the center of gravity, not the TUI.

```
┌─────────────────────────────────────────────────────────┐
│                      Clients                            │
│   ┌─────────┐   ┌──────────┐   ┌────────────────────┐  │
│   │  TUI    │   │  Python  │   │  LLM (via MCP or   │  │
│   │ (Ratatui│   │  Scripts │   │  JSON-RPC direct)  │  │
│   │  /Rust) │   │  /REPL   │   │                    │  │
│   └────┬────┘   └────┬─────┘   └────────┬───────────┘  │
│        │             │                   │              │
│        └─────────────┼───────────────────┘              │
│                      │                                  │
│              ┌───────▼────────┐                          │
│              │   Machine      │  JSON-RPC 2.0           │
│              │   Interface    │  DAP-inspired            │
│              │   (Protocol)   │  stdio / TCP / WebSocket │
│              └───────┬────────┘                          │
│                      │                                  │
│              ┌───────▼────────┐                          │
│              │   Core Engine  │  Rust (PyO3 bindings)   │
│              │   State machine│  + Python hook layer    │
│              │   + Checkpoint │                          │
│              └───────┬────────┘                          │
│                      │                                  │
│         ┌────────────┼────────────┐                     │
│         │            │            │                     │
│    ┌────▼───┐  ┌─────▼────┐ ┌────▼───┐                 │
│    │PyTorch │  │ CUDA/    │ │ Probe  │                 │
│    │Hooks   │  │ CUPTI    │ │ System │                 │
│    │Layer   │  │ Layer    │ │        │                 │
│    └────────┘  └──────────┘ └────────┘                 │
└─────────────────────────────────────────────────────────┘
```

## Layer 1: Core Engine

The stateful kernel. Pure state machine managing the forward pass stepping lifecycle.

### Language: Rust + Python

**Rust** for:
- State machine (stepping, checkpoints, probe registry)
- Protocol server (JSON-RPC parsing, connection management)
- Memory management (checkpoint storage, tensor buffer pooling)
- TUI rendering (Ratatui)

**Python** for:
- PyTorch hook registration and management (hooks must be Python — no way around this)
- Model loading and configuration (HuggingFace ecosystem is Python)
- SAE integration (existing SAE libraries are Python)
- User-defined intervention scripts

**Bridge**: PyO3 for Rust↔Python. Core state machine in Rust, called from Python process that owns the PyTorch runtime. The Rust side never touches PyTorch directly.

### State Machine

```
                    load_model
    UNLOADED ─────────────────► LOADED
                                  │
                          attach  │
                                  ▼
                               ATTACHED
                                  │
                        run/step  │
                                  ▼
                    ┌──────────PAUSED◄──────────┐
                    │             │              │
                step│      inspect│        step  │
                    │             ▼              │
                    │         INSPECTING         │
                    │             │              │
                    │    intervene│              │
                    │             ▼              │
                    │        INTERVENING─────────┘
                    │
                    ▼
                 STEPPING──────►PAUSED
```

States:
- **UNLOADED**: no model, only `load_model` valid
- **LOADED**: model in memory, can configure probes/checkpoints before attaching
- **ATTACHED**: hooks registered, ready to run
- **PAUSED**: stopped at a tick, can inspect/intervene/step/checkpoint
- **STEPPING**: executing forward pass between ticks (transient)
- **INSPECTING**: reading state (concurrent reads allowed, no mutation)
- **INTERVENING**: mutating state (exclusive access)

### Tick Model

A "tick" is one atomic unit of forward pass execution. Granularity options:

| Level | What fires | Density | Use case |
|-------|-----------|---------|----------|
| **layer** | Between transformer blocks | ~32 ticks for LLaMA-8B | Coarse navigation |
| **component** | Between attention/MLP/norm within a layer | ~128 ticks | Standard debugging |
| **head** | Between individual attention heads | ~1024 ticks | Fine-grained attention analysis |
| **expert** | Between MoE experts (per-token routing) | Variable | MoE-specific debugging |

Default: **component** level. User can change granularity dynamically.

Each tick boundary is a `cudaDeviceSynchronize` point — GPU work complete, tensors inspectable on host.

### Checkpoint + Replay

Reverse stepping via rr/TTD model:
1. Auto-checkpoint every `sqrt(n)` layers (activation checkpointing alignment)
2. Manual checkpoints at any tick via `checkpoint save`
3. "Step backward" = restore nearest checkpoint before target, replay forward
4. Forward replay is deterministic (single stream, cuBLAS deterministic mode, fixed seeds)

Checkpoint storage:
- **Lightweight**: activation tensors only (input to each checkpointed layer)
- **NOT full GPU memory dump** (cuda-checkpoint is O(GPU memory), too slow for interactive use)
- Store on host memory (CPU RAM), with option to spill to disk for large models
- Estimated: ~4MB per checkpoint for 8B model at FP16 (hidden_dim * batch * seq_len * 2 bytes)

### Multi-GPU Execution

For DDP/FSDP/tensor-parallel models:
- One probe coordinator (Rust) communicates with per-GPU Python workers
- Workers register hooks on their local model shard
- Coordinator synchronizes stepping: all GPUs pause at same tick boundary
- Collective operations (all-reduce, all-gather) execute as atomic sub-ticks
- NCCL Inspector integration for collective visibility

```
┌─────────────┐
│ Coordinator │  (Rust, single process)
│  (Protocol  │
│   Server)   │
└──┬──┬──┬────┘
   │  │  │   gRPC / shared memory
┌──▼┐┌▼──┐┌▼──┐
│GPU││GPU ││GPU│  (Python workers, one per device)
│ 0 ││ 1 ││ 2 │
└───┘└────┘└───┘
```

## Layer 2: Machine Interface (Protocol)

The primary interface. Everything goes through this — TUI, Python scripts, LLM clients.

### Protocol: JSON-RPC 2.0 with DAP-Inspired Semantics

Why not pure DAP:
- DAP assumes sequential execution with a call stack — neural nets don't have call stacks
- DAP's variable model doesn't map well to tensors (shape, dtype, statistics)
- DAP's breakpoint model doesn't map to probe points

Why DAP-inspired:
- Request/Response/Event message types (proven pattern)
- Capability negotiation at `initialize` (from LSP)
- Async event notifications for state changes
- Token-based request-response correlation

Transport: stdio (default, simplest), TCP (remote debugging), WebSocket (browser clients).

### Core Operations (7 primitives)

| Operation | Verb | Description | Idempotent |
|-----------|------|-------------|------------|
| **step** | mutating | Move forward/backward N ticks | No |
| **inspect** | read | Read tensor/state at current position | Yes |
| **intervene** | mutating | Modify tensor/state at current position | No |
| **probe** | config | Attach/detach/list observation hooks | Yes (list), No (attach/detach) |
| **checkpoint** | mixed | Save/restore/list named states | Yes (list/restore), No (save) |
| **evaluate** | read | Run expression against current state | Yes |
| **status** | read | Full state dump | Yes |

Plus lifecycle: `initialize`, `load_model`, `attach`, `detach`, `shutdown`.

### Response Contract

Every response includes:
```json
{
  "state": "PAUSED",
  "position": {
    "tick": 42,
    "layer": 3,
    "component": "attention",
    "phase": "output"
  },
  "result": { ... },
  "active_probes": [...],
  "available_actions": ["step", "inspect", "intervene", "probe", "checkpoint", "evaluate", "status"]
}
```

State in every response. No hidden state. An LLM can pick up any response cold and know exactly where it is and what it can do.

### Error Contract

```json
{
  "error": {
    "code": "CHECKPOINT_NOT_FOUND",
    "message": "No checkpoint named 'baseline' exists",
    "context": {"requested": "baseline", "available": ["init", "layer_8"]},
    "suggestions": ["checkpoint list", "checkpoint save baseline"]
  }
}
```

Errors are actionable. Code for programmatic handling, suggestions for recovery.

### MCP Server

rocket_surgeon exposes itself as an MCP server:
- **Resources**: model info, current state, checkpoint list, probe catalog
- **Tools**: the 7 core operations
- **Prompts**: none (no system prompt dependency)

Any MCP-capable LLM connects and operates the debugger natively. No wrappers.

## Layer 3: Probe System

Probes are the observation and intervention mechanism — the bridge between systems tracing and neural network interpretability.

### Naming Convention (DTrace-inspired)

`model:layer:component:event`

Examples:
- `llama:12:attention:output` — attention output at layer 12
- `llama:*:mlp:input` — MLP input at all layers (wildcard)
- `llama:12:attention.head.7:output` — specific attention head
- `mixtral:8:router:decision` — MoE routing decision
- `llama:*:*:output` — all outputs at all layers (very verbose)

### Probe Anatomy

```
probe = {
  point:   "llama:12:attention:output",  // WHERE
  hook:    "checkpoint",                  // WHAT (from hook registry)
  filter:  "norm > 50.0",                // WHEN (predicate, optional)
  enabled: true,                         // lifecycle state
  priority: 0                            // execution order (lower = first)
}
```

### Built-in Hook Types

| Hook | Mutates | Description |
|------|---------|-------------|
| **inspect** | No | Return tensor summary (shape, stats) or full data |
| **checkpoint** | No | Save state for backward stepping |
| **intervene** | Yes | Modify tensor (scale, zero, add, replace, clamp) |
| **assert** | No | Check invariant, pause if violated |
| **trace** | No | Emit structured event to timeline |
| **aggregate** | No | Accumulate statistics across ticks (running mean, histogram) |
| **sae_decompose** | No | Decompose through loaded SAE, return top features |

### Probe Lifecycle
1. **Register**: declare probe point + hook
2. **Arm**: enable (NOP → active)
3. **Fire**: hook executes at tick boundary
4. **Disarm**: disable without removing
5. **Deregister**: remove entirely

### Probe Discovery

`probe list` returns all available probe points with metadata:
```json
{
  "points": [
    {
      "name": "llama:0:attention:output",
      "tensor_shape": [1, 32, 128, 128],
      "dtype": "float16",
      "description": "Attention output after all heads, before projection"
    }
  ]
}
```

LLM clients discover what's observable without documentation.

### Neural Network Probes (Interpretability Layer)

Beyond raw tensors, structured interpretability views:

| View | Backing | Description |
|------|---------|-------------|
| **logit_lens** | Unembedding projection | Token distribution at each layer |
| **attention_pattern** | Raw attention weights | Per-head attention maps |
| **sae_features** | Sparse autoencoder | Top-k active features with labels |
| **activation_stats** | Tensor statistics | Norm, mean, std, sparsity per component |
| **routing_table** | MoE gate logits | Which experts selected for which tokens |

These are built on top of the raw probe system — sugar, not separate mechanism.

## Intervention System

Interventions modify model state between ticks. Serializable as JSON (pyvene insight).

### Intervention Types

| Type | Target | Example |
|------|--------|---------|
| **ablate** | Head, expert, neuron | Zero out attention head 7 at layer 12 |
| **scale** | Any tensor | Multiply MLP output by 0.5 |
| **steer** | Activation vector | Add steering vector to residual stream |
| **patch** | Activation | Replace activation from a different run |
| **clamp** | Any tensor | Clamp values to range |
| **sae_feature** | SAE coefficient | Set feature #1842 coefficient to 3.0 |
| **route** | MoE gate | Force token to specific expert |

### Intervention as Data

```json
{
  "interventions": [
    {
      "id": "ablate_head_7",
      "type": "ablate",
      "target": "llama:12:attention.head.7:output",
      "params": {}
    },
    {
      "id": "steer_honesty",
      "type": "steer",
      "target": "llama:12:residual:post_attention",
      "params": {"vector": "sae_feature:1842", "coefficient": 2.0}
    }
  ]
}
```

Serializable. Versionable. Shareable. LLM-synthesizable.

## PyTorch Integration

### Hook Strategy

Primary mechanism: `register_forward_hook` and `register_forward_pre_hook` pairs on `nn.Module` submodules.

**Critical gotchas** (from lit review):
1. **DDP**: hooks registered before `DistributedDataParallel()` wrapping are silently ignored. Register inside `forward()` or after wrapping.
2. **torch.compile**: hooks registered after `torch.compile()` are silently ignored. Register BEFORE compilation, or use `fullgraph=False`.
3. **FSDP**: internal hooks for parameter gathering can conflict. Must coordinate.
4. **CUDA Graphs**: captured graphs bypass hook entry points. Must instrument at capture time.

**Strategy**: register hooks on the UNWRAPPED model, then wrap. For torch.compile, register before compile. Provide clear errors when hook conflicts detected.

### Model Loading

Wrap existing HuggingFace models (nnsight approach, not TransformerLens re-implementation):
1. Load model via `transformers.AutoModelForCausalLM`
2. Walk module tree, identify layers/components by type
3. Register hook pairs at each probe-eligible boundary
4. Standardize naming across architectures (nnterp approach)

Supported architectures (initial):
- LLaMA family (LLaMA 2/3, Mistral, Mixtral)
- GPT-NeoX family (Pythia)
- Gemma
- Qwen

### Determinism Requirements

For replay correctness:
- `torch.use_deterministic_algorithms(True)`
- `CUBLAS_WORKSPACE_CONFIG=:16:8`
- Single CUDA stream per device
- Fixed random seeds (Python, NumPy, PyTorch, CUDA)
- Fixed batch composition (same input data, same order)
- Same GPU architecture across replay sessions

## Observability Stack

Multi-layer instrumentation, each catching what others miss:

| Layer | Tool | Catches |
|-------|------|---------|
| **Python** | PyTorch hooks, torch.profiler | Tensor ops, autograd, module boundaries |
| **Runtime** | CUPTI callbacks/activities | Kernel launches, memory ops, sync events |
| **Driver** | eBPF (uprobes on libcuda.so) | ioctl patterns, UVM migrations, page faults |
| **Hardware** | NVML, perf_events | GPU utilization, temperature, memory bus |
| **Communication** | NCCL Inspector | Collective operations, ring/tree topology |

Output format: Chrome Trace (JSON) for timeline visualization, structured JSON events for programmatic analysis.

## Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Core engine | Rust | Memory safety, performance, Ratatui ecosystem |
| Python bridge | PyO3 | Mature, zero-copy tensor sharing possible |
| Protocol | JSON-RPC 2.0 | Simple, LLM-native, DAP-compatible semantics |
| Transport | stdio (default) | Works everywhere, debugger convention |
| TUI | Ratatui | Immediate mode, sub-ms rendering, Rust-native |
| Hook layer | Python | PyTorch hooks require Python; no alternative |
| Tensor format | safetensors | Fast, memory-mapped, Rust + Python support |
| Config format | JSON | LLM-native, no YAML footguns |

## What This Is NOT

- **Not a training tool**: inference/forward-pass only. No gradient computation, no backward pass, no optimizer steps.
- **Not a model editor**: interventions are ephemeral (applied during stepping session). Persistent model editing (ROME-style) is out of scope for v1.
- **Not a profiler**: we use profilers (CUPTI, torch.profiler) but the goal is debugging and surgery, not performance optimization.
- **Not a framework**: doesn't replace PyTorch/JAX/TF. Wraps PyTorch models specifically.

## Build Order

Per JSMNTL: specs first, then red tests, then implementation.

### Phase 1: Foundation
1. ADR: language split (Rust + Python via PyO3)
2. ADR: protocol design (JSON-RPC 2.0 with DAP-inspired semantics)
3. ADR: probe model (DTrace-inspired naming, hook registry)
4. TCK: core stepping semantics (step forward, step backward, tick granularity)
5. TCK: probe registration and lifecycle
6. TCK: checkpoint save/restore
7. TCK: protocol request/response contract

### Phase 2: Core Engine (Rust)
1. State machine implementation
2. Checkpoint storage (host memory, activation-only)
3. Protocol server (JSON-RPC over stdio)
4. Probe registry

### Phase 3: PyTorch Integration (Python)
1. Model wrapper (HuggingFace model loading + hook registration)
2. Hook manager (forward/pre-forward pairs, DDP/compile-aware)
3. Determinism enforcement
4. PyO3 bridge to Rust state machine

### Phase 4: Probe System
1. Built-in hooks (inspect, checkpoint, trace, aggregate)
2. Intervention hooks (ablate, scale, steer, patch)
3. SAE integration (load pre-trained, decompose, steer by feature)
4. Logit lens / tuned lens views

### Phase 5: Multi-GPU
1. Per-GPU worker process
2. Coordinator synchronization
3. NCCL collective visibility
4. Cross-device checkpoint coordination

### Phase 6: TUI
1. Ratatui shell (status, command input)
2. Tensor visualization (heatmaps, sparklines)
3. Attention pattern rendering
4. MoE routing visualization
5. Timeline view

### Phase 7: MoE Support
1. Router hook (gating network output)
2. Per-expert stepping
3. Expert activation visualization
4. Selective expert forcing
