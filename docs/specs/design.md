# rocket_surgeon: Comprehensive Design Specification

Supersedes `architecture.md`. Synthesized from 13 lit reviews, 7 deep-dive analyses, sky claude
architectural consultation, and deep study of 60 reference implementations (NNsight, pyvene,
baukit, nnterp, vLLM-Lens, vLLM-Hook, tokio-console, trippy, ratatui, Penzai/Treescope,
transformer-debugger, rr, DAP, MCP, Megatron-LM, FlashAttention, NCCL, and others).

**Audience**: rocket_surgeon core team.
**Purpose**: The definitive design document from which TCK specs, ADRs, and implementation derive.
**Status**: Living document. Version 0.1.0.

---

## 1. Vision & Scope

### What rocket_surgeon is

A proper debugger and in-situ surgery tool operating natively on multi-GPU transformer forward
passes. Step through internals (dense and MoE) one tick at a time, forward and backward, with
full surgical intervention between ticks.

Dual-interface: TUI for humans, structured protocol for LLMs. LLM ergonomics are first-class —
LLMs as end users is inalienable.

### What it is not

- **Not a training tool.** Inference/forward-pass only. No backward pass, no optimizer. Gradient
  inspection is a future extension (Phase 8+), not a design constraint.
- **Not a model editor.** Interventions are session-scoped. Persistent edits (ROME-style) are out
  of scope.
- **Not a profiler.** We consume profiler data (CUPTI, Perfetto) but the goal is understanding and
  surgery, not performance.
- **Not a wrapper around NNsight or any other library.** We study prior art deeply but reimplement
  everything. The only runtime dependencies are PyTorch, HuggingFace transformers, and safetensors.

### Design axioms

1. **Protocol is the product.** The wire protocol is the center of gravity. TUI, Python scripts,
   and LLM clients are all equal consumers. If the protocol can't express it, it doesn't exist.
2. **Tensors are handles, not values.** Every tensor flows as metadata + summary until explicitly
   materialized. An LLM client should never accidentally consume 10 GB.
3. **State in every response.** No hidden state. Any client can pick up any response cold and know
   exactly where the debugger is and what it can do.
4. **Interventions are data.** Every intervention is a JSON-serializable recipe. Versionable,
   shareable, LLM-synthesizable, reproducible.
5. **Honest about limitations.** If a fused kernel hides the attention matrix, say so. If
   CUDA Graphs skip submodule hooks, say so. Capability negotiation surfaces what's possible.
6. **Zero cost when off.** With no probes active and no client attached, overhead is negligible.

---

## 2. Architecture: Three-Process Model

Three independent OS processes communicating over well-defined protocols. This is the decisive
architectural choice, informed by Garçon (server-per-model, multiple clients attach), Pernosco
(sessions outlive viewers), and tokio-console (instrumentation/api/tui as three crates).

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Clients                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ rs-tui   │  │ Python   │  │ LLM (MCP     │  │ IDE (DAP        │   │
│  │ (Rust)   │  │ scripts  │  │ adapter)     │  │ adapter)        │   │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └───────┬─────────┘   │
│       │              │               │                  │             │
│       └──────────────┴───────┬───────┴──────────────────┘             │
│                              │                                        │
│                    JSON-RPC 2.0 / Unix socket / TCP                   │
│                              │                                        │
│                 ┌────────────▼─────────────┐                          │
│                 │   Process A: rs-daemon   │                          │
│                 │   (Rust)                 │                          │
│                 │   • Protocol server      │                          │
│                 │   • State machine        │                          │
│                 │   • Checkpoint index     │                          │
│                 │   • Probe registry       │                          │
│                 │   • Session manager      │                          │
│                 │   • Tensor handle store  │                          │
│                 │   • Perfetto trace sink  │                          │
│                 └────────────┬─────────────┘                          │
│                              │                                        │
│                    JSON-RPC 2.0 / Unix socket                        │
│                    + shared-memory data plane                         │
│                              │                                        │
│      ┌───────────────────────┼───────────────────────┐               │
│      │                       │                       │               │
│  ┌───▼──────────────┐  ┌────▼─────────────┐  ┌─────▼────────────┐  │
│  │ Process B: host-0│  │ Process B: host-1│  │ Process B: host-N│  │
│  │ (Python)         │  │ (Python)         │  │ (Python)         │  │
│  │ • PyTorch runtime│  │ • PyTorch runtime│  │ • PyTorch runtime│  │
│  │ • Model shard    │  │ • Model shard    │  │ • Model shard    │  │
│  │ • Hook manager   │  │ • Hook manager   │  │ • Hook manager   │  │
│  │ • Intervention   │  │ • Intervention   │  │ • Intervention   │  │
│  │   engine         │  │   engine         │  │   engine         │  │
│  │ • Tensor capture │  │ • Tensor capture │  │ • Tensor capture │  │
│  │ • Barrier gate   │  │ • Barrier gate   │  │ • Barrier gate   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│         GPU 0                 GPU 1                 GPU N            │
└────────────────────────────────────────────────────────────────────────┘
```

### Why three processes

| Concern | Rationale |
|---------|-----------|
| **Isolation** | Python crash (GPU OOM, NCCL hang, segfault) does not kill the daemon. Debug session survives model restart. |
| **Multiple models** | One daemon, many model hosts. Garçon pattern: loading is amortized across sessions. |
| **TUI lifecycle** | TUIs come and go; models stay loaded. Multiple TUIs attach simultaneously. |
| **Iteration speed** | Rust daemon and Python host compile independently. No wheel rebuilds to iterate on protocol. |
| **Multi-GPU** | One Python worker per rank is mandatory for `torch.distributed`. Daemon fans out to all workers. |

### Inter-process communication

**Control plane**: JSON-RPC 2.0 over Unix domain sockets. Same schema externally and internally —
the daemon↔host channel uses the same verbs as daemon↔client. Latency ~50 μs per message.

**Data plane**: Shared-memory ring buffer (`/dev/shm/rs-<session>-<rank>`) for tensor handoff.
Python writes tensor bytes directly into the shared region; Rust reads them zero-copy. Layout:

```
┌──────────────────────────────────────────────────────┐
│  ProbeFrame record (fixed 128-byte header)           │
│  ┌─────────┬──────┬───────┬──────┬───────┬─────────┐ │
│  │ rank:u32│layer │comp_id│dtype │ndim:u8│shape    │ │
│  │         │:u32  │:u16   │:u8   │       │:[u32;8] │ │
│  ├─────────┼──────┴───────┴──────┴───────┴─────────┤ │
│  │ tick_id │ offset:u64  │ size:u64  │ flags:u32   │ │
│  │ :u64    │             │           │             │ │
│  └─────────┴─────────────┴───────────┴─────────────┘ │
│  [raw tensor bytes at offset...]                     │
└──────────────────────────────────────────────────────┘
```

Python probe callback flow:
1. `tensor.detach().contiguous().to('cpu', non_blocking=True)`
2. `torch.cuda.current_stream().synchronize()` (scoped, not global)
3. memcpy into ring slot
4. publish slot index via `eventfd` to Rust daemon

Rust reads the slot, builds a `TensorRef` (metadata + `&[u8]` view), serves clients zero-copy.

---

## 3. Process A — Rust Daemon (`rs-daemon`)

The stateful kernel. Owns the protocol, the state machine, the probe registry, and the session.

### State machine

```
                      ┌──────────────────────────────────────────┐
                      │              UNINITIALIZED               │
                      └────────────────┬─────────────────────────┘
                               initialize(capabilities)
                      ┌────────────────▼─────────────────────────┐
                      │              INITIALIZED                 │
                      │  (no model, protocol negotiated)         │
                      └────────────────┬─────────────────────────┘
                               attach(model_path, config)
                      ┌────────────────▼─────────────────────────┐
                      │              ATTACHING                   │
                      │  (model loading, hooks registering)      │
                      └────────────────┬─────────────────────────┘
                               attached(model_info, capabilities)
                      ┌────────────────▼─────────────────────────┐
              ┌──────►│              STOPPED                     │◄────────┐
              │       │  (at tick boundary, can inspect/intervene)│        │
              │       └───┬──────────┬───────────┬───────────────┘        │
              │     step  │   inspect│   intervene│   detach              │
              │           │          │            │       │               │
              │    ┌──────▼───┐  ┌───▼────┐  ┌───▼────┐  │               │
              │    │ STEPPING │  │INSPECT │  │MODIFY  │  │               │
              │    │(transient│  │(read,  │  │(excl.  │  │               │
              │    │ between  │  │concurrent│ │access) │  │               │
              │    │ ticks)   │  │allowed)│  │        │  │               │
              │    └──────┬───┘  └───┬────┘  └───┬────┘  │               │
              │           │          │           │       │               │
              └───────────┴──────────┴───────────┘       │               │
                                                         │               │
                      ┌──────────────────────────────────▼───────┐        │
                      │              DETACHING                   │        │
                      │  (hooks removing, cleanup)               │        │
                      └──────────────────────────────────┬───────┘        │
                                                         │               │
                      ┌──────────────────────────────────▼───────┐        │
                      │              INITIALIZED                 ├────────┘
                      │  (can re-attach to same or different model)       │
                      └──────────────────────────────────────────┘
```

States carry a session envelope:

```
SessionState {
    session_id:    Uuid,
    model_id:      String,        // content hash of model files
    status:        Status,        // stopped | stepping | inspecting | modifying | replaying
    position:      TickPosition,  // { tick_id, rank?, layer, component, event }
    tick_id:       u64,           // monotonic, never reused
    capabilities:  Capabilities,  // negotiated at initialize
    active_probes: Vec<ProbeId>,
    checkpoints:   Vec<CheckpointRef>,
}
```

`tick_id` is the primary key for everything. Checkpoints reference it. Probe firings reference it.
Interventions are attached at a tick. Replayed ticks get fresh tick_ids with a `replay_of` field
pointing to the original.

### Daemon responsibilities

1. **Protocol server**: Accept connections (Unix socket, TCP), parse JSON-RPC, dispatch to
   state machine, serialize responses. Every response includes the full `SessionState` envelope.
2. **State machine**: Enforce valid transitions. Reject out-of-state requests with actionable errors.
3. **Probe registry**: Store probe definitions, match firings, route events to subscribers.
4. **Checkpoint index**: Track checkpoint metadata (tick_id, layer, content hash, storage location).
   Actual tensor bytes live in shared memory or on disk; daemon holds the index.
5. **Tensor handle store**: Content-addressable. `tensor_id = blake3(bytes)`. Same tensor at two
   probe points has the same id. Dedup in TUI and session bundles.
6. **Session manager**: Multiple sessions (one per model), multiple clients per session.
7. **Trace sink**: Receive probe events, write Perfetto protobuf traces.
8. **Heartbeat**: While stopped, send `tick.heartbeat` notification every 1s with per-rank status.

### Capability negotiation

At `initialize`, client declares what it supports and daemon responds with its capabilities:

```json
{
  "protocol_version": "0.1.0",
  "supports_reverse_step": false,
  "supports_checkpointing": false,
  "supports_moe": false,
  "supports_backward": false,
  "supports_sae": false,
  "execution_mode": "eager",
  "parallelism": "single_gpu",
  "tick_granularities": ["layer", "component"],
  "intervention_types": ["ablate", "scale", "add", "patch", "clamp"],
  "built_in_views": ["residual_stream_norm", "attention_pattern"],
  "head_granularity": "unavailable",
  "transports": ["stdio", "unix_socket"],
  "wire_formats": ["json"],
  "max_response_bytes": 65536,
  "model_family": "llama",
  "model_id": "sha256:abc123...",
  "num_layers": 32,
  "num_heads": 32,
  "hidden_dim": 4096,
  "num_ranks": 1,
  "num_experts": null,
  "top_k_experts": null
}
```

Capabilities evolve across phases. Clients adapt to what's available. An LLM client reading this
knows exactly what verbs will work and at what granularity.

---

## 4. Process B — Python Model Host (`rs-host`)

One instance per GPU rank. Owns the PyTorch runtime and the model shard. The daemon starts these
as child processes.

### Responsibilities

1. **Model loading**: Load HF model via `AutoModelForCausalLM`, distribute across ranks.
2. **Model adapter**: Walk the module tree, identify components, build canonical name mapping.
3. **Hook manager**: Register/remove PyTorch hooks per the active tier.
4. **Barrier gate**: Block the forward pass at tick boundaries via `threading.Event`.
5. **Intervention engine**: Apply intervention recipes to tensors at barrier points.
6. **Tensor capture**: Snapshot tensors at probe points, write to shared-memory ring buffer.
7. **Summary computation**: Compute tensor statistics (mean, std, min, max, abs-max, histogram,
   sparsity, top-k) on GPU before transfer — cheap single-reduction ops.

### Model adapter

The adapter wraps an arbitrary HuggingFace model and maps its module tree to a canonical vocabulary.
This is our own implementation, informed by NNsight's Envoy and nnterp's `StandardizedTransformer`,
but designed for our tick-based stepping model.

**Canonical component vocabulary**:

| Canonical name | Description |
|---------------|-------------|
| `embed_tokens` | Token embedding layer |
| `layers[i].ln1` | Pre-attention layer norm |
| `layers[i].attn` | Attention module (composite) |
| `layers[i].attn.q_proj` | Query projection |
| `layers[i].attn.k_proj` | Key projection |
| `layers[i].attn.v_proj` | Value projection |
| `layers[i].attn.o_proj` | Output projection |
| `layers[i].attn.scores` | Attention scores (virtual — captured, not a module) |
| `layers[i].ln2` | Pre-MLP layer norm |
| `layers[i].mlp` | MLP module (composite) |
| `layers[i].mlp.gate_proj` | Gate projection (SwiGLU) |
| `layers[i].mlp.up_proj` | Up projection |
| `layers[i].mlp.down_proj` | Down projection |
| `layers[i].residual_pre` | Residual stream entering the block |
| `layers[i].residual_mid` | Residual stream between attention and MLP |
| `layers[i].residual_post` | Residual stream leaving the block |
| `ln_final` | Final layer norm |
| `lm_head` | Language model head |

For MoE models, additional components:

| Canonical name | Description |
|---------------|-------------|
| `layers[i].router` | MoE gating network |
| `layers[i].router.logits` | Raw router logits (pre-softmax, pre-top-k) |
| `layers[i].router.decision` | Top-k selection result |
| `layers[i].experts[j]` | Expert j (composite) |
| `layers[i].experts[j].gate_proj` | Expert gate projection |
| `layers[i].experts[j].up_proj` | Expert up projection |
| `layers[i].experts[j].down_proj` | Expert down projection |
| `layers[i].shared_expert` | Shared expert (DeepSeek-V3) |

**Per-architecture adapters** map HuggingFace module paths to canonical names:

```python
LLAMA_ADAPTER = {
    "model.embed_tokens":                "embed_tokens",
    "model.layers.{i}.input_layernorm":  "layers[{i}].ln1",
    "model.layers.{i}.self_attn":        "layers[{i}].attn",
    "model.layers.{i}.self_attn.q_proj": "layers[{i}].attn.q_proj",
    "model.layers.{i}.self_attn.k_proj": "layers[{i}].attn.k_proj",
    "model.layers.{i}.self_attn.v_proj": "layers[{i}].attn.v_proj",
    "model.layers.{i}.self_attn.o_proj": "layers[{i}].attn.o_proj",
    "model.layers.{i}.post_attention_layernorm": "layers[{i}].ln2",
    "model.layers.{i}.mlp":             "layers[{i}].mlp",
    "model.layers.{i}.mlp.gate_proj":   "layers[{i}].mlp.gate_proj",
    "model.layers.{i}.mlp.up_proj":     "layers[{i}].mlp.up_proj",
    "model.layers.{i}.mlp.down_proj":   "layers[{i}].mlp.down_proj",
    "model.norm":                        "ln_final",
    "lm_head":                           "lm_head",
}

MIXTRAL_ADAPTER = {
    # ...extends LLAMA_ADAPTER with:
    "model.layers.{i}.block_sparse_moe.gate":            "layers[{i}].router",
    "model.layers.{i}.block_sparse_moe.experts.{j}":     "layers[{i}].experts[{j}]",
    "model.layers.{i}.block_sparse_moe.experts.{j}.w1":  "layers[{i}].experts[{j}].gate_proj",
    "model.layers.{i}.block_sparse_moe.experts.{j}.w2":  "layers[{i}].experts[{j}].down_proj",
    "model.layers.{i}.block_sparse_moe.experts.{j}.w3":  "layers[{i}].experts[{j}].up_proj",
}
```

Initial architecture support: Llama/Llama-2/Llama-3, Mistral, Mixtral, Qwen2, Gemma2, GPT-NeoX.
Others run in "best-effort module-path mode" with auto-detected hook points.

**Model conformance test suite**: For each supported family, run a fixed prompt and assert probes
fire at canonical points in the expected order. Run nightly against latest `transformers` release.

### Hook manager

Three-tier interception stack, selected per-model at attach time based on automatic detection.

**Tier A — Eager mode (default, MVP)**

Standard `nn.Module.register_forward_hook` / `register_forward_pre_hook`:

- Register AFTER distributed wrapping (DDP, `fully_shard`) but BEFORE `torch.compile`.
- Register on the unwrapped inner module (`model.module` for DDP).
- Use `prepend=True` so our hooks run before any user hooks.
- Register a sentinel no-op hook on every module of interest at attach time to disable PyTorch's
  fast path (which skips hook dispatch when `_forward_hooks` is empty).
- For FSDP2: hooks fire inside the unsharded window (between pre-forward all-gather and
  post-forward reshard). This is the correct inspection window.

Hook lifecycle mirrors NNsight's lazy one-shot pattern, adapted for our tick model:
- Hooks registered once at attach time (not per-tick).
- Each hook checks the active probe set before doing work.
- If no probes match this component, the hook body is a no-op (fast exit, ~50 ns overhead).
- Hooks self-arm/disarm based on the probe registry, not re-registered each tick.

**Tier B — Compiled mode (Phase 7)**

For `torch.compile`-using models, hooks registered after compilation are silently dropped
(pytorch/pytorch #117758). The correct approach: register a `torch._dynamo` custom backend that
wraps `call_module` / `call_function` nodes in the captured FX graph with instrumentation.

Implementation sketch:
```python
def rocket_surgeon_backend(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.op == "call_module":
            # Insert pre/post probe dispatch around this node
            insert_probe_dispatch(gm.graph, node, probe_registry)
    gm.recompile()
    return gm
```

Alternative: force `fullgraph=False` (the default), which inserts graph breaks at unsupported ops.
Hook-based instrumentation works in the eager regions between breaks.

**Tier C — CUDA Graph mode (Phase 7)**

Per NVIDIA docs: "Submodule forward methods are never called — the graph executes as a monolithic
sequence of pre-recorded CUDA kernels. Only the top-level module's hooks are invoked."

This is a hard constraint. We intercept *between* captured graphs, not inside them. Minimum tick
granularity is graph-level (typically layer or multi-layer). If the user wants finer granularity,
they must recapture with smaller graph regions.

**Tier detection at attach time**: Walk the module tree. If `OptimizedModule` (torch.compile
wrapper) is found → Tier B. If `make_graphed_callables` artifacts → Tier C. If
`FullyShardedDataParallel` or `DistributedDataParallel` → Tier A with distributed awareness.
Surface the chosen tier in the `capabilities` response.

### Barrier gate

The mechanism that pauses the forward pass at tick boundaries:

```python
class BarrierGate:
    def __init__(self):
        self._event = threading.Event()
        self._event.set()  # initially open
        self._tick_id = 0

    def wait(self, component: str, layer: int) -> None:
        """Called from hook. Blocks until daemon signals continue."""
        if not self._should_pause(component, layer):
            return
        self._event.clear()
        self._tick_id += 1
        # notify daemon: we're stopped at this tick
        self._notify_stopped(self._tick_id, component, layer)
        self._event.wait()  # block until daemon calls step()

    def release(self) -> None:
        """Called by daemon when client steps."""
        self._event.set()
```

On multi-GPU: each rank has its own BarrierGate. The daemon coordinates: all ranks must reach
a barrier before any client sees `STOPPED`. When the client steps, all ranks release together.

### Intervention engine

Interventions are **recipes** — declarative data structures applied by the hook at barrier points.
Two tiers:

**Tier 1 — Declarative (safe, serializable, LLM-composable)**

```python
@dataclass
class InterventionRecipe:
    id: str
    type: Literal["ablate", "scale", "add", "patch", "clamp", "route_override"]
    target: str          # canonical probe point
    params: dict         # type-specific parameters
    condition: str|None  # optional predicate ("norm > 50.0")
    priority: int        # execution order (lower = first)
```

Types:

| Type | Params | Effect |
|------|--------|--------|
| `ablate` | `{}` | Zero out the target tensor |
| `scale` | `{"factor": 0.5}` | Multiply by scalar |
| `add` | `{"vector": [...]  or tensor_id}` | Add vector to tensor |
| `patch` | `{"source_tensor_id": "..."}` | Replace with tensor from another run |
| `clamp` | `{"min": -1.0, "max": 1.0}` | Clamp to range |
| `route_override` | `{"token": 4, "experts": [3, 7]}` | Force MoE routing |

Composition semantics: interventions at the same probe point execute in priority order. Multiple
interventions compose additively by default (EasySteer pattern). `priority=0` runs first.
A `replace` mode overwrites previous interventions' effects.

**Tier 2 — Python callbacks (unsafe, for power users)**

```python
def custom_intervention(tensor: torch.Tensor, metadata: ProbeMetadata) -> torch.Tensor:
    # arbitrary code
    return modified_tensor
```

Registered via the Python API, not the protocol. Marked `unsafe=true`. Run with a watchdog timer
(default 5s) and OOM guard. Cannot be serialized into session bundles.

---

## 5. Process C — Rust TUI Client (`rs-tui`)

A Ratatui-based terminal UI that connects to the daemon over JSON-RPC. Architecturally modeled
on tokio-console and trippy: a pure protocol client with no awareness of PyTorch or model internals.

### Architecture

```
┌─────────────────────────────────────────────────┐
│  rs-tui                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ Conn     │  │ State    │  │ View          │ │
│  │ (client) │─►│ (model)  │◄─│ (ratatui)     │ │
│  └──────────┘  └──────────┘  └───────────────┘ │
│       │              ▲              │           │
│       │              │              │           │
│       └──────────────┴──────────────┘           │
│              main event loop                    │
└─────────────────────────────────────────────────┘
```

Three layers (tokio-console pattern):

1. **Connection** (`conn`): Manages JSON-RPC connection with automatic reconnection. Holds
   subscription streams. Offers `step()`, `inspect()`, `intervene()` methods that issue RPCs.
2. **State** (`state`): Central in-memory model of the debug session. Ingests protocol responses,
   maintains probe registry mirror, tensor summaries, checkpoint list, tick history.
3. **View** (`view`): Renders state via ratatui. Handles keyboard input. Updates state and
   connection in response to user actions.

Main event loop (tokio select, biased):
```rust
loop {
    tokio::select! { biased;
        input = terminal_events.next() => {
            view.handle_input(input, &mut state, &mut conn);
        },
        message = conn.next_message() => {
            state.update(message);
        },
    }
    terminal.draw(|frame| view.render(frame, &state))?;
}
```

### TUI panels

| Panel | Content |
|-------|---------|
| **Status bar** | Session state, tick position, model info, attached probes count |
| **Activation summary** | Per-component: norm, mean, std, sparsity (sparkline history) |
| **Tensor inspector** | Selected tensor: heatmap, histogram, top-k values, slice viewer |
| **Attention pattern** | Per-head attention heatmap for selected layer (LSE-reconstructed) |
| **Intervention panel** | Active interventions, recipe editor |
| **Timeline** | Perfetto-style trace view of probe events, ticks, collectives |
| **MoE routing** | Per-token expert assignment, routing entropy, cluster projection |
| **Command bar** | Protocol command input with autocomplete |

Tensors flow as summaries by default. Full data fetched only when the user drills into a tensor
(spawn-on-detail pattern from tokio-console). Response size cap: 64 KB per message.

---

## 6. Wire Protocol

### Foundation: JSON-RPC 2.0

All communication uses JSON-RPC 2.0 messages. Three message types:

- **Request**: `{ "jsonrpc": "2.0", "id": 1, "method": "rocket/step", "params": {...} }`
- **Response**: `{ "jsonrpc": "2.0", "id": 1, "result": {...} }`
- **Notification**: `{ "jsonrpc": "2.0", "method": "rocket/tick.stopped", "params": {...} }`

### Transports

| Transport | Use case | MVP |
|-----------|----------|-----|
| **stdio** | CLI integration, DAP compatibility | Yes |
| **Unix socket** | Local multi-client, daemon↔host internal | Yes |
| **TCP** | Remote debugging | Phase 5 |

MCP is an **adapter**, not a transport. An MCP server wraps the JSON-RPC service and exposes:
- `rocket_surgeon.step`, `rocket_surgeon.inspect`, etc. as MCP **tools** (single-turn RPCs).
- Tensor data and checkpoints as MCP **resources** (`rocket://session/{id}/tensor/{tid}`).
- Tick events as MCP notifications (fallback to polling if transport doesn't support streaming).

### Protocol verbs: 10 primitives

Organized into three namespaces:

**Lifecycle (DAP-compatible)**

| Verb | Direction | Description |
|------|-----------|-------------|
| `initialize` | request/response | Capability negotiation. Client declares support; daemon responds with capabilities. |
| `attach` | request/response | Load model, start host process(es), register hooks. Returns model info. |
| `detach` | request/response | Remove hooks, unload model, terminate host(s). |

**Domain operations (`rocket/*` namespace)**

| Verb | Direction | Mutating | Description |
|------|-----------|----------|-------------|
| `rocket/step` | request/response | Yes | Advance N ticks forward (or backward if checkpoints exist). Params: `direction`, `count`, `granularity`. |
| `rocket/inspect` | request/response | No | Read tensor/state at current position. Returns summary by default. Params: `target` (probe point), `detail` (summary\|slice\|full). |
| `rocket/intervene` | request/response | Yes | Set/clear intervention recipe at a probe point. |
| `rocket/probe` | request/response | Mixed | Define/list/enable/disable probes. |
| `rocket/checkpoint` | request/response | Mixed | Create/restore/list/delete named checkpoints. |
| `rocket/replay` | request/response | Yes | Restore checkpoint and replay forward with (possibly different) interventions. |
| `rocket/status` | request/response | No | Full session state dump. |

**Events (daemon → client notifications)**

| Event | Description |
|-------|-------------|
| `rocket/tick.stopped` | Forward pass paused at a tick boundary. Includes full position. |
| `rocket/tick.heartbeat` | Sent every 1s while stopped. Per-rank status. Keeps clients alive. |
| `rocket/probe.fired` | A probe captured data. Includes tensor summary. |
| `rocket/replay.divergence` | Replayed tensor diverges from original beyond tolerance. |
| `rocket/error` | Unrecoverable error (OOM, NCCL hang, etc.). |

**Subscription**

Clients register for events via `rocket/subscribe`:
```json
{
  "method": "rocket/subscribe",
  "params": {
    "events": ["tick.stopped", "probe.fired"],
    "filter": { "layer": [10, 11, 12] }
  }
}
```

### Response envelope

Every response includes the session state. No exceptions.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "state": {
      "status": "stopped",
      "session_id": "abc-123",
      "model_id": "sha256:def456",
      "position": {
        "tick_id": 42,
        "rank": 0,
        "layer": 3,
        "component": "attn.o_proj",
        "event": "output"
      },
      "active_probes": ["p1", "p2"],
      "available_actions": ["step", "inspect", "intervene", "probe", "checkpoint"]
    },
    "data": { ... }
  }
}
```

### Error contract

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32001,
    "message": "Cannot step: model is in ATTACHING state",
    "data": {
      "error_code": "INVALID_STATE",
      "numeric_code": -32001,
      "severity": "recoverable",
      "current_state": "attaching",
      "valid_states": ["stopped"],
      "suggestion": "Wait for attach to complete, then retry"
    }
  }
}
```

Errors are actionable. Machine-readable code, human-readable message, structured context,
recovery suggestion. An LLM client can parse `error_code` and `valid_states` to self-correct.

### Protocol versioning

- `protocol_version: "0.1.0"` in every `initialize` response.
- Clients that send an unsupported version get a clear error with the server's supported range.
- Wire-protocol independence from PyTorch versions is the single most important architectural
  decision protecting against PyTorch internals churn.

---

## 7. Tick Model

A "tick" is one atomic unit of forward-pass execution. At each tick boundary, the forward pass is
paused at a barrier, tensors are inspectable, and interventions can be applied.

### Granularity levels

| Level | What fires | Approx. count (32-layer) | Use case |
|-------|-----------|--------------------------|----------|
| `layer` | Between transformer blocks | ~32 | Coarse navigation |
| `component` | Between attn/MLP/norm within a block | ~192 | Standard debugging (default) |
| `head` | Per attention head (requires unfused execution) | ~1024 | Fine attention analysis |

For MoE models, four additional sub-granularities within each MoE layer:

| Level | What fires | Use case |
|-------|-----------|----------|
| `router_pre_topk` | After router emits logits, before top-k | Routing inspection/override |
| `router_post_topk` | After top-k selection, before dispatch | Assignment inspection/override |
| `expert` | Inside a specific expert, post-dispatch | Per-expert tensor inspection |
| `moe_layer` | After combine (post-expert reduce) | Layer-level MoE inspection |

### Tick boundary implementation

**NOT `cudaDeviceSynchronize`** (too expensive — blocks all streams). Instead, use CUDA events
scoped to the relevant stream:

```python
event = torch.cuda.Event()
event.record(torch.cuda.current_stream())
event.synchronize()  # blocks host, not other streams
```

For pipeline parallelism, sync is per-stage, not global. Each rank synchronizes its own stream.

### Tick scoping

Users can set different granularities for different regions:

```json
{
  "method": "rocket/probe",
  "params": {
    "action": "set_granularity",
    "scopes": [
      { "match": "layers[12]", "granularity": "component" },
      { "match": "layers[*]", "granularity": "layer" }
    ]
  }
}
```

Specific scopes override general ones. This avoids stepping through 192 ticks when you only care
about layer 12.

### Tick identity

- `tick_id: u64` — monotonic, never reused, never reset within a session.
- Replayed ticks get fresh tick_ids with `replay_of: Option<u64>` referencing the original.
- Bookmarks: named references to ticks. `bookmark("before_ablation") -> tick_id 42`.
  Bookmarks are the Pernosco notebook pattern — persistent annotations on the tick timeline.

### Head and expert granularity honesty

Head-level and expert-level ticks require **unfused execution**. In vectorized code, all heads run
as a single batched matmul; all experts may run as a fused grouped GEMM (Megablocks). You cannot
pause "between head 3 and head 4" without splitting the computation.

When the user requests head or expert granularity:
1. If the model uses eager attention: heads can be inspected individually by slicing the output
   tensor (no unfusing needed for inspection, but intervention requires per-head execution).
2. If the model uses fused kernels: trigger shadow replay of that layer with unfused execution
   (same mechanism as FlashAttention unfusing — see §13).

The protocol's capabilities response states this honestly: `"head_granularity": "requires_unfused"`.

### Backward-tick schema (future, Phase 8+)

The tick model is symmetric forward/backward in the *schema*, even though backward implementation
is deferred:

```
TickPosition {
    tick_id:    u64,
    direction:  forward | backward,
    rank:       Option<u32>,
    layer:      u32,
    component:  String,
    event:      pre | post,
}
```

This ensures the protocol doesn't need breaking changes when backward-pass support ships.

---

## 8. Probe System

### Namespace: `model:rank:layer:component:event`

Five-level hierarchical naming, extended from the original DTrace-inspired four-level scheme with
a `rank` dimension (sky claude correction — without rank, you cannot distinguish "the value on
rank 3" from "the gathered value" on TP=8).

Examples:
```
llama:*:12:attn.o_proj:output        — attention output at layer 12, all ranks
llama:0:*:mlp:input                  — MLP input at all layers, rank 0
mixtral:*:8:router:pre_topk          — router logits at layer 8, all ranks
llama:*:*:residual_post:*            — all residual post outputs (very verbose)
llama:0:12:attn.scores:*             — attention scores at layer 12 (virtual probe)
```

Wildcards (`*`) match any value at that level.

### Probe definition

```json
{
  "id": "p1",
  "point": "llama:*:12:attn.o_proj:output",
  "action": "capture",
  "config": {
    "summary": true,
    "capture_tensor": false,
    "filter": "norm > 50.0"
  },
  "enabled": true,
  "priority": 0
}
```

**Actions**:

| Action | Mutates | Description |
|--------|---------|-------------|
| `capture` | No | Capture tensor summary and optionally full data |
| `checkpoint` | No | Create a checkpoint at this point |
| `trace` | No | Emit a structured event to the Perfetto timeline |
| `assert` | No | Check predicate, pause + alert if violated |
| `aggregate` | No | Accumulate statistics across ticks (running mean, histogram) |
| `intervene` | Yes | Apply an intervention recipe |

### Probe lifecycle

```
  define ──► armed ──► firing ──► armed (repeats)
               │                    │
           disarm ◄────────────── disarm
               │
           remove
```

Probes persist across ticks until removed. "Armed" means the hook checks this probe's predicate
on every tick at the matching point. "Firing" means the predicate passed and the action executes.

### Composition semantics

Probes at the same point execute in **priority order** (lower priority number = earlier execution).
Multiple probes can match the same point:
- Non-mutating actions (capture, trace, assert) compose freely — all execute.
- Mutating actions (intervene) compose additively by default. Multiple `add` interventions sum
  their vectors. An intervention with `mode: "replace"` overrides all prior interventions at that
  point. Explicit `priority` determines order.

### Built-in interpretability views

These are named, pre-configured probe + post-processing pipelines exposed as protocol primitives:

| View name | Backing | Output |
|-----------|---------|--------|
| `residual_stream_norm` | L2 norm of residual stream at each layer | `[f32; num_layers]` |
| `attention_pattern` | Attention weights (LSE-reconstructed for FlashAttention) | `[f16; H, T, T]` sparse |
| `head_output` | Per-head contribution to residual stream | `[f16; H, T, D_head]` |
| `logit_lens` | Unembed residual stream at each layer to vocab | `[f32; L, T, top_k]` |
| `routing_decision` | MoE router assignments per token | `[u16; T, top_k_experts]` + probs |
| `routing_entropy` | Shannon entropy of routing distribution | `[f32; T]` |
| `feature_attribution` | Gradient-based attribution (when backward available) | `[f32; T, D]` |
| `sae_activation` | SAE feature activations (when SAE loaded) | `[f32; T, top_k_features]` |

Built-in views are sugar over the raw probe system, not a separate mechanism.

---

## 9. Tensor Handling

### Content-addressable identity

`tensor_id = blake3(raw_bytes)` (32 bytes, hex-encoded). The same tensor observed at two different
probe points has the same tensor_id. Benefits:
- Dedup in TUI (highlight "same tensor" across views).
- Dedup in session bundles (store each unique tensor once).
- Cache-friendly (daemon can cache summaries by tensor_id).

### Summary-then-slice protocol

Every tensor inspection follows a two-phase pattern:

**Phase 1: Summary (always returned, ~200 bytes)**
```json
{
  "tensor_id": "a1b2c3...",
  "shape": [1, 32, 2048, 128],
  "dtype": "float16",
  "device": "cuda:0",
  "sharding": null,
  "stats": {
    "mean": 0.0012,
    "std": 0.342,
    "min": -4.21,
    "max": 3.87,
    "abs_max": 4.21,
    "sparsity": 0.023,
    "l2_norm": 14.7,
    "histogram": { "bins": 32, "edges": [...], "counts": [...] }
  },
  "top_k": [
    { "index": [0, 7, 1024, 42], "value": 4.21 },
    { "index": [0, 3, 512, 99], "value": -3.98 }
  ]
}
```

**Phase 2: Slice (on demand, bounded)**
```json
{
  "method": "rocket/tensor.slice",
  "params": {
    "tensor_id": "a1b2c3...",
    "slices": [[0, 0], [7, 7], [1020, 1028], [0, 128]],
    "format": "float16"
  }
}
```

Response size cap: 64 KB per slice request. For larger requests, paginate.

Attention matrices at 128K context: **never materialize the full `[H, 128K, 128K]`**. Always
operate on sparse slices ("row T for tokens t1..t2"). Refuse materialization for matrices > 1 GB
unless `--allow-large` flag is explicitly set.

### Summary computation

All summary stats are computed on-GPU before CPU transfer — single-reduction ops that are cheap:
```python
stats = {
    "mean": tensor.mean().item(),
    "std": tensor.std().item(),
    "min": tensor.min().item(),
    "max": tensor.max().item(),
    "abs_max": tensor.abs().max().item(),
    "l2_norm": tensor.norm(2).item(),
    "sparsity": (tensor.abs() < 1e-6).float().mean().item(),
}
histogram = torch.histc(tensor.float(), bins=32)
top_k_vals, top_k_flat_idx = tensor.abs().flatten().topk(8)
```

### DTensor-aware inspection

For distributed models, tensors may be DTensors with `placements` attributes (Shard, Replicate,
Partial). The inspect response includes sharding info:

```json
{
  "sharding": {
    "mesh": "tp",
    "placements": [{"type": "Shard", "dim": 0}],
    "local_shape": [1, 8, 2048, 128],
    "global_shape": [1, 32, 2048, 128]
  }
}
```

`rocket/tensor.gather` command issues a separate all-gather to materialize the full tensor on
rank 0 for inspection. Only executed on explicit request — never automatic.

---

## 10. Checkpoint & Replay Engine

### Three-tier hierarchy

| Tier | Content | Size (7B model) | Trigger | Retention |
|------|---------|-----------------|---------|-----------|
| **1: Probe log** | Tensor summaries + metadata at each probe firing | ~MBs | Always-on | Session lifetime |
| **2: Activation checkpoints** | Residual streams at √L layer boundaries | ~100 MB | Auto at tick boundaries | Rolling last K=8 |
| **3: Full snapshot** | Complete model + optimizer + RNG state | ~14 GB (7B fp16) | Manual only | Named bookmarks |

### Sqrt-N activation checkpoints

Auto-checkpoint at every √L transformer blocks:
- 32-layer model (Llama-8B): checkpoint every 6 layers → 6 checkpoints → ~100 MB
- 80-layer model (Llama-70B, TP=4): checkpoint every 9 layers → ~10 checkpoints → ~40 MB/rank
- 126-layer model (Llama-405B, TP=8): checkpoint every 12 layers → ~12 checkpoints → easy

Cadence adapts to model depth and available memory. Not a fixed knob.

Each checkpoint records:
```
ActivationCheckpoint {
    tick_id:          u64,
    layer_idx:        u32,
    residual_stream:  TensorRef,   // the activation tensor
    rng_state:        RngState,    // torch + cuda RNG snapshots
    input_ids:        Tensor,      // for replay verification
    env_hash:         String,      // driver version, NCCL version, etc.
}
```

### Storage hierarchy

- **GPU → CPU pinned memory** (default): `torch.Tensor.to('cpu', non_blocking=True)` with
  pinned memory allocation. Fast enough for interactive use (~2 ms per checkpoint).
- **CPU → NVMe spillover**: When CPU memory pressure exceeds threshold, spill oldest checkpoints
  to disk via safetensors mmap. FlexGen-style tiering.
- **On-disk format**: safetensors. Zero-copy mmap for reload: `safe_open(path, framework="pt",
  device="cuda")`.

### Reverse stepping

"Step backward" = restore nearest checkpoint before target + replay forward.

1. Find the √N checkpoint at or before the requested tick.
2. Restore residual stream and RNG state from checkpoint.
3. Replay forward with all probes that were enabled at original-record-time re-firing.
4. Cost: bounded by √L layers. On A100, a 7B forward layer is ~1 ms; √32 = 6 layers → ~6 ms.

### Replay determinism

Replay is **ULP-close, not bit-exact**. This is a deliberate, honest design decision.

Three-layer determinism strategy:

**Layer 1 — Seed and env capture at session start**

Record: `torch.initial_seed()`, `torch.cuda.initial_seed()`, NumPy seed, relevant env vars
(CUDA, NCCL, PYTHONHASHSEED), driver version, NCCL version, compute capability, model file
content hash. Refuse to replay across mismatches without `--unsafe-replay`.

**Layer 2 — Op-level pinning**

- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- FlashAttention `deterministic=True` during replay only
- For MoE: pin the routing decision (re-feed recorded top-k indices rather than re-deriving)
- Single CUDA stream per device during replay

**Layer 3 — Batch-invariant kernels (future)**

Adopt batch-invariant matmul/RMSNorm/SDPA kernels on the shadow-replay path. The dominant source
of inference non-determinism is non-batch-invariant reductions in matmul/attention kernels.

### Replay verification

Every replayed tensor at a probe point is compared to the recorded summary:
- Cosine similarity > 0.99995 → match
- Max elementwise relative error < 1e-3 (fp16), < 1e-5 (fp32) → match
- Divergence above tolerance fires a `rocket/replay.divergence` event:
  ```json
  {
    "method": "rocket/replay.divergence",
    "params": {
      "tick_id": 48,
      "original_tick_id": 42,
      "probe_point": "llama:0:17:residual_post:output",
      "cosine_similarity": 0.997,
      "max_relative_error": 4e-3,
      "message": "Replay diverges at layer 17"
    }
  }
  ```

The user sees the divergence and can choose to proceed or escalate. This is the right honesty
stance — log every mismatch, don't fail silently, don't pretend it's exact.

### Bookmarks

Named references to ticks. The Pernosco notebook pattern:
```json
{ "method": "rocket/checkpoint", "params": { "action": "bookmark", "name": "before_ablation", "tick_id": 42 } }
```

Bookmarks are first-class in the protocol and session bundles. A debugging session is a sequence
of bookmarked states with annotations — shareable as a reproducible narrative.

---

## 11. Multi-GPU Coordination

### Pre/post-collective barriers

NCCL collectives are **atomic**. A torn collective deadlocks all ranks. Never insert a barrier
inside a collective.

Each transformer block with tensor parallelism follows:
```
(compute) → [all-reduce / all-gather / reduce-scatter] → (compute)
```

Insert barriers **before** the collective and **after** the collective:

```
  ┌─ Rank 0 ──────────────────────────────────────────┐
  │ compute → [BARRIER_PRE] → all-reduce → [BARRIER_POST] → compute │
  └────────────────────────────────────────────────────┘
  ┌─ Rank 1 ──────────────────────────────────────────┐
  │ compute → [BARRIER_PRE] → all-reduce → [BARRIER_POST] → compute │
  └────────────────────────────────────────────────────┘
```

**Pre-collective window**: inspect *sharded* activations on each rank. Each rank's local shard
is exposed as a DTensor with its placements annotation. A "gather to rank 0 for inspection"
command issues a *separate* all-gather (serialized: only one collective in flight per process
group at a time).

**Post-collective window**: inspect the *gathered/reduced* tensor. On Megatron-style TP, the
global residual stream is replicated across TP ranks after row-parallel + reduce-scatter. Daemon
can fetch from rank 0 only.

### Rank coordination

The daemon is the JTAG controller; barriers are halt instructions; inspection happens between halts.

1. Daemon sends `step` to all host workers simultaneously.
2. Each worker's forward pass advances until the next barrier.
3. Each worker notifies daemon: "stopped at tick T, layer L, component C."
4. Daemon waits until ALL ranks report stopped (with timeout).
5. Daemon surfaces `STOPPED` state to clients.
6. Client issues inspect/intervene commands. Daemon routes to appropriate rank(s).
7. Client issues `step`. Daemon releases all barriers.

### Pipeline parallelism

Different from tensor parallelism: each rank has different layers, not different shards.

Use `torch.distributed.pipelining`'s schedule machinery. A "tick" advances one micro-batch through
one pipeline stage on the relevant rank. Barriers are per-stage, not per-rank.

### Watchdog

NCCL has a watchdog (`TORCH_NCCL_BLOCKING_WAIT`). When the user pauses indefinitely between
barriers, the watchdog will trip.

Required env at attach time (daemon sets these automatically):
```
TORCH_NCCL_BLOCKING_WAIT=0
NCCL_TIMEOUT=14400  # 4 hours
TORCH_NCCL_ASYNC_ERROR_HANDLING=0
```

If the env is misconfigured, refuse to attach with a clear error message listing what to set.

Self-recovery: a barrier held longer than `T_max` (default 5 minutes, configurable) auto-releases
with an error event. Better to lose a debug session than wedge a node.

### Heartbeat

While stopped, daemon sends `rocket/tick.heartbeat` every 1s:
```json
{
  "method": "rocket/tick.heartbeat",
  "params": {
    "tick_id": 42,
    "ranks": [
      { "rank": 0, "status": "stopped", "gpu_memory_used_gb": 24.3 },
      { "rank": 1, "status": "stopped", "gpu_memory_used_gb": 23.8 }
    ],
    "elapsed_stopped_sec": 15.2
  }
}
```

---

## 12. MoE Architecture

MoE support is designed into the protocol and tick model from day one, even though the implementation
ships in Phase 6. Retrofitting MoE tick granularities would be much worse than designing them upfront.

### Four tick granularities

```
  Input hidden states
        │
        ▼
  ┌─────────────┐
  │   Router    │──── [TICK: router_pre_topk] ── inspect/override router logits
  │  (gate net) │
  └──────┬──────┘
         │ softmax + top-k
         ▼
  ┌─────────────┐
  │  Routing    │──── [TICK: router_post_topk] ── inspect/override assignments
  │  Decision   │
  └──────┬──────┘
         │ dispatch (all-to-all for EP)
         ▼
  ┌─────────────────────────────────────┐
  │        Per-expert compute           │
  │  ┌─────┐ ┌─────┐ ┌─────┐          │
  │  │Exp 0│ │Exp 1│ │Exp N│ ─── [TICK: expert] ── per-expert inspection
  │  └─────┘ └─────┘ └─────┘          │
  └──────────────┬──────────────────────┘
                 │ combine (weighted sum + reduce)
                 ▼
  ┌─────────────┐
  │  MoE output │──── [TICK: moe_layer] ── post-combine inspection
  └─────────────┘
```

**Default granularity for MoE layers: `router_post_topk`** — the most interesting intervention
point (force-assign tokens to specific experts) at the lowest frequency per token.

### Protocol primitives for MoE

```json
// Inspect routing
{ "method": "rocket/inspect", "params": { "target": "mixtral:*:8:router:pre_topk", "detail": "summary" } }
// → returns: router logits shape [batch*seq, num_experts], entropy per token

// Override routing
{ "method": "rocket/intervene", "params": {
    "recipe": {
      "type": "route_override",
      "target": "mixtral:*:8:router:post_topk",
      "params": { "token": 4, "experts": [3, 7] }
    }
  }
}

// Inspect specific expert
{ "method": "rocket/inspect", "params": { "target": "mixtral:*:8:experts[3]:output", "detail": "summary" } }
```

### MoE-specific protocol fields

Every MoE tick response includes:
- **Dropped token count**: tokens exceeding expert capacity factor that were silently dropped.
  This is the silent failure mode of MoE — always surface it.
- **Routing entropy**: Shannon entropy of routing distribution, per token, per layer.
- **Expert load**: token count per expert (for load imbalance detection).
- **Shared expert contribution**: for DeepSeek-V3 style shared+routed architectures, distinguish
  shared expert output from routed expert output.

### HuggingFace integration

At attach time, set `output_router_logits=True` in the model config (required for Mixtral,
DeepSeek). This flag propagates routing logits through the model's output object, making them
available to our hooks without additional instrumentation.

---

## 13. FlashAttention & Fused Kernel Strategy

FlashAttention does not materialize the N×N attention matrix in HBM. This is fundamental — the
matrix exists only in SRAM tiles and is recomputed in the backward pass.

### Two-path inspection

**Fast path (default): LSE reconstruction**

FlashAttention returns log-sum-exp values (shape `[batch, heads, seq_q]`, fp32). With LSE plus
stored Q, K, V, reconstruct any row of the softmax matrix on demand:

```python
# For a specific query position t:
scores = (Q[t] @ K.T) / sqrt(d_k)
attention_weights = softmax(scores)  # or: exp(scores - lse[t])
```

For "show me the attention pattern for token 17," this is enough and cheap. Exposed as the
default `attention_pattern` built-in view.

**Shadow path (on demand): selective unfusing**

When full attention matrix inspection is requested for a specific layer/head:

1. From the nearest checkpoint, replay forward to the target layer.
2. For that layer only, force `attn_implementation="eager"` (HF supports per-layer attention
   implementations as of transformers 4.40+).
3. All other layers stay FlashAttention — no throughput penalty for uninspected layers.
4. Cost: one extra layer's worth of recomputation.

Protocol:
```json
{ "method": "rocket/inspect", "params": {
    "target": "llama:*:12:attn.scores:output",
    "detail": "full",
    "options": { "full_matrix": true }
  }
}
```

`full_matrix=true` triggers shadow replay. The response warns about the cost.

### Other fused kernels

The same shadow-replay pattern applies to any fused kernel (Liger fused RMSNorm+linear, fused
SwiGLU, etc.): replay the specific layer with the reference (unfused) implementation. The
intervention is scoped to one layer and one forward pass — not always-on.

### FlexAttention

FlexAttention (2024+) has a built-in `eager` mode designed for inspection. If the model uses
FlexAttention, exploit that — no shadow replay needed.

### MVP stance

For MVP: force `attn_implementation="eager"` globally. No FlashAttention in MVP. Shadow replay
ships in Phase 7.

---

## 14. Observability

### Two tiers

**Tier 1 — Probe events (always on, ~0% overhead when idle)**

Probe firings, tick events, intervention applications, checkpoint operations. Emitted as structured
events through the protocol and written to the Perfetto trace.

**Tier 2 — Deep diagnostics (opt-in, 5-20% overhead)**

| Source | What it captures | Toggle |
|--------|-----------------|--------|
| CUPTI Activity API | Kernel launches, memory ops, sync events | `rocket/diag.cupti.enable` |
| eBPF uprobes | `libcuda.so` / `libnccl.so` calls | `rocket/diag.ebpf.enable` |
| NVML | Per-GPU memory, power, utilization, ECC errors | Always on (1/s poll, negligible) |
| NCCL Inspector | Collective operations, topology | Default INFO; TRACE opt-in |

### Trace format: Perfetto protobuf

All events — probe fires, CUPTI spans, NCCL ranges, NVML samples, tick boundaries — go into one
Perfetto protobuf trace file. Benefits:
- Open format with mature schema.
- Rust and Python SDKs (`perfetto-sdk-rs`).
- Co-visualize with `nvtx` ranges, CUPTI kernel events, and our probe events.
- Researchers already know the Perfetto UI.
- PyTorch Kineto already converts to Chrome trace JSON; Perfetto consumes both.

Do **not** invent a trace format.

The TUI shows a summarized timeline view; the full trace is exportable as `.perfetto-trace`.

---

## 15. Session Bundles

A session bundle is a self-contained reproducibility artifact — like CockroachDB's "statement
bundle" but for transformer debugging.

### Contents

```
session-bundle-<id>.tar.gz
├── manifest.json           # session metadata, protocol version, timestamps
├── model-info.json         # model hash, architecture, config
├── env.json                # GPU, driver, NCCL, CUDA versions, env vars
├── protocol-trace.jsonl    # complete JSON-RPC request/response log
├── prompt.json             # input tokens
├── tensors/                # captured tensors as safetensors files
│   ├── <tensor_id>.safetensors
│   └── ...
├── interventions.json      # all intervention recipes applied
├── checkpoints/            # named checkpoint state
│   └── ...
├── trace.perfetto-trace    # Perfetto timeline
└── bookmarks.json          # named tick bookmarks with annotations
```

### Use cases

- **Bug reports**: "run this bundle on the same GPU class, see the same result"
- **Sharing**: send a colleague the exact debugging session
- **LLM context**: an LLM can read the bundle and understand what happened
- **Reproducibility**: replay from a bundle to verify findings

### Export

```json
{ "method": "rocket/session.export", "params": { "path": "/tmp/debug-session.tar.gz" } }
```

Exported at any tick. Includes all tensors captured up to that point.

---

## 16. Scaling (1B → 405B+)

### Design rules

1. **Every inspection is lazy and bounded by default.** Tensors are handles with metadata.
   Materialization is an explicit verb.
2. **Built-in hierarchical summaries.** Mean/std/min/max/abs-max/histogram/top-k/sparsity —
   always cheap, always returned. The TUI and LLM client mostly want these.
3. **Adaptive checkpoint granularity.** √L adapts to model depth and available memory.
4. **Streaming tensor inspection.** For TUI rendering, stream the requested slice chunk-by-chunk;
   never deliver the full `[1, 2048, 16384]`.
5. **Pipeline-parallel rank routing.** Issue inspection RPCs to the owning rank only; don't
   all-gather for inspection.
6. **TP-aware capture.** After all-reduce, residual streams are replicated. Capture from rank 0
   only (vLLM-Lens pattern). Pre-collective, capture per-rank shards.

### Concrete scaling numbers

| Model | GPUs | Residual per checkpoint | √N checkpoints | Total checkpoint memory |
|-------|------|------------------------|-----------------|------------------------|
| Llama-3-8B | 1 | 17 MB | 6 (√32) | ~100 MB |
| Llama-3-70B | 4 (TP=4) | 4 MB/rank | 10 (√80) | ~160 MB cluster |
| Llama-3-405B | 8 (TP=8) | 8 MB/rank | 12 (√126) | ~768 MB cluster |

### Response size budget

- Summary response: ~200 bytes (fits in one JSON message)
- Slice response: ≤64 KB (configurable cap)
- Full tensor download: streaming, chunked at 1 MB
- Attention matrix at 128K: refuse full materialization; row-only access

---

## 17. Support Matrix

Explicit, public, honest. Green = supported and tested. Yellow = best-effort. Red = not supported.

### MVP (Phase 0-2)

| | Eager | torch.compile | CUDA Graph |
|--|-------|---------------|------------|
| **Llama-3** | **Green** | Red | Red |
| **Mistral** | Yellow | Red | Red |
| **Other** | Red | Red | Red |

| | Single GPU | DDP | TP | PP | FSDP |
|--|-----------|-----|----|----|------|
| **MVP** | **Green** | Red | Red | Red | Red |

### Phase 5 (Multi-GPU)

| | Single GPU | DDP | TP | PP | FSDP |
|--|-----------|-----|----|----|------|
| **Llama-3** | Green | Green | Green | Yellow | Yellow |
| **Mistral** | Green | Green | Green | Yellow | Yellow |
| **Qwen2** | Yellow | Yellow | Yellow | Red | Red |

### Phase 6 (MoE)

| | Eager | torch.compile |
|--|-------|---------------|
| **Mixtral** | Green | Red |
| **DeepSeek-V3** | Yellow | Red |

### Phase 7

torch.compile and CUDA Graph support across the matrix.

**Refuse to attach to unsupported configurations** with a clear error message pointing to the
matrix and the roadmap.

---

## 18. Phase Plan

Protocol-first. TCK-first. Reordered from the original plan based on sky claude's analysis.

### Phase 0 — Protocol Spec + Golden TCK (weeks 1-2)

Write the JSON-RPC schema (JSON-Schema + protobuf definitions) and the probe-point grammar. Write
Gherkin scenarios for all 10 verbs: `initialize`, `attach`, `detach`, `step`, `inspect`,
`intervene`, `probe`, `checkpoint`, `status`, `subscribe`. Include MoE tick granularity in the
schema even though MoE ships in Phase 6.

Red tests fail because the daemon doesn't exist. This is the highest-leverage work.

**Deliverables**:
- `protocol/schema/v0.1.0.json` — JSON-Schema for all messages
- `protocol/rocket_surgeon.proto` — protobuf definitions (for future gRPC)
- `tck/*.feature` — Gherkin behavioral specs
- ADR-0004: Three-process architecture
- ADR-0005: Tick model and granularity design

### Phase 1 — Single-GPU Eager Daemon (weeks 3-6)

Rust daemon (Process A) + Python model host (Process B). Tier A hooks only. Component-level tick.
Loads HF Llama-3-8B in eager mode, bf16, single GPU.

**Deliverables**:
- `rs-daemon`: Protocol server over stdio + Unix socket. State machine. Probe registry.
- `rs-host`: Model loading, model adapter (Llama family), hook manager, barrier gate.
- `rocket_surgeon._rs`: PyO3 thin bridge for hot-path operations.
- Shared-memory tensor handoff.
- `rocket/step`, `rocket/inspect`, `rocket/status`, `rocket/probe` verbs working.
- Tensor handle store with content-addressable IDs.
- TCK green for stepping, inspection, probe lifecycle.

### Phase 2 — Interventions (weeks 7-8)

Five intervention types: `ablate`, `scale`, `add`, `patch`, `clamp`. Intervention recipes as data.

**Deliverables**:
- Intervention engine in rs-host.
- `rocket/intervene` verb working.
- Intervention composition (priority, additive, replace).
- Session bundle export (without checkpoints).
- TCK green for interventions.

**MVP gate**: At this point, an external script can drive Llama-3-8B through the protocol, ablate
name-mover heads, and verify score deltas. IOI reproduction as acceptance test.

### Phase 3 — Checkpoint + Reverse Step (weeks 9-11)

√N activation checkpoints. Forward replay from checkpoint. Replay verification.

**Deliverables**:
- Checkpoint engine: auto-checkpoint at √L boundaries, manual bookmarks.
- `rocket/checkpoint` and `rocket/replay` verbs.
- Reverse-step via checkpoint restore + forward replay.
- Replay divergence detection and reporting.
- Safetensors on-disk spill.
- TCK green for checkpointing and replay.

### Phase 4 — TUI Dogfood (weeks 12-14)

Ratatui client of the protocol. The TUI is a *user* of the protocol, not co-evolving with it.
By this point the protocol is mature enough.

**Deliverables**:
- `rs-tui`: Connection, state, view layers.
- Activation summary panel, tensor inspector, attention pattern viewer.
- Intervention panel, command bar.
- Timeline view (Perfetto trace rendering).
- Dogfood feedback → protocol refinements.

### Phase 5 — Multi-GPU (weeks 15-18)

DDP first, then TP via DTensor.

**Deliverables**:
- Per-rank Python workers. Daemon fans out.
- Pre/post-collective barriers.
- DTensor-aware tensor inspection.
- TP-aware capture (rank 0 for replicated tensors).
- NCCL watchdog and env validation.
- Heartbeat with per-rank status.
- TCK green for multi-GPU stepping and inspection.
- Validation: Llama-3-70B with TP=4.

### Phase 6 — MoE (weeks 19-21)

All four tick granularities. Routing inspection and override.

**Deliverables**:
- Router hook, routing decision capture.
- Per-expert stepping and inspection.
- Routing entropy, expert load, dropped token reporting.
- Route override intervention.
- Mixtral adapter.
- TCK green for MoE.
- Validation: Mixtral 8x7B routing override.

### Phase 7 — torch.compile + CUDA Graph + FlashAttention (weeks 22+)

The hard infrastructure problems. Explicitly *after* MoE because compile-mode interception is the
hardest problem and should not block research use.

**Deliverables**:
- Tier B: Dynamo custom backend FX-graph rewriting.
- Tier C: CUDA Graph inter-graph interception.
- FlashAttention shadow replay (selective unfusing).
- FSDP2 support (DTensor-based, hook timing coordination).
- Batch-invariant kernels for deterministic replay.

### Phase 8+ — Future

- Backward-pass tick support (gradient ablation, backward patching).
- Probe DSL bytecode compilation (rbpf/uBPF for zero-cost probes).
- gRPC transport for remote debugging.
- Browser-based UI (WebSocket transport).
- Multi-node debugging.
- SAE integration (feature-level surgery).
- MCP adapter.
- IDE integration (DAP adapter for VS Code).

---

## 19. MVP Definition

### In one sentence

A daemon that, given a path to a single-GPU eager-mode HuggingFace Llama-3-8B and a prompt, lets
a client step through the forward pass at component granularity over JSON-RPC, inspect residual
streams and attention patterns with summary-then-slice semantics, and apply ablation/scaling/addition
interventions — with a session bundle exportable on demand.

### What's in

- Rust daemon, JSON-RPC 2.0 over stdio + Unix socket.
- Schema in JSON-Schema. Versioned. `initialize` capability negotiation.
- Python model host (one process). Loads HF Llama-3-8B in eager mode, bf16, single GPU.
- Tier A hooks only. Refuses to attach to a compiled model.
- Component-granularity tick (canonical 18-name vocabulary for Llama family).
- 8 of 10 verbs: `initialize`, `attach`, `detach`, `step`, `inspect`, `intervene`, `probe`,
  `status`. No `checkpoint`, no `replay`.
- `subscribe` for tick events.
- Content-addressable tensor handles. Summary-then-slice semantics.
- Two built-in views: `residual_stream_norm`, `attention_pattern` (eager SDPA, not FlashAttention).
- Five interventions: `ablate`, `scale`, `add`, `patch`, `clamp`. (All five are trivial to
  implement alongside each other; `route_override` is the only type deferred to Phase 6.)
- Session bundle export.

### What's out

- Multi-GPU, TUI, checkpointing, reverse-step, MoE, FlashAttention, torch.compile, CUDA Graphs,
  FSDP, SAE, gRPC, MCP adapter. All later phases.

### Definition of done

1. JSON-RPC schema frozen at v0.1.0 and published.
2. Daemon starts, attaches to Llama-3-8B in eager mode in <30s.
3. An automated test drives the IOI ablation experiment using only the protocol and produces
   expected score deltas (±5%).
4. Session bundle export reproduces the same result on a different machine of the same GPU class.
5. Overhead with zero probes: <5% on prefill, <2% on decode.
6. Documentation: protocol spec, attach guide, one tutorial (IOI reproduction).

### Why this is the right MVP

- Validates the protocol-first, schema-driven thesis. If a client can drive this MVP from schema
  alone to do interpretability, the concept is proven.
- Exercises every architectural seam — Rust daemon, Python host, PyO3 boundary, protocol schema,
  capability negotiation, content-addressable tensors, session bundles — but only one model, one
  parallelism mode, one execution mode.
- Achievable in ~10 weeks (Phase 0 + 1 + 2) for a 2-person team.

---

## 20. Risk Register

Ranked by expected pain × probability.

### R1: PyTorch internals churn (certain × severe)

Dynamo, AOTAutograd, Inductor, FSDP2, `torch.distributed.pipelining` all evolve on a 6-month
cadence. The surface we build against today will move within 12 months.

**Mitigations**:
- Pin PyTorch versions in CI: last 3 stable + nightly.
- Versioned `BackendAdapter` interface for Tier A/B/C hook adapters. When PyTorch changes,
  replace the adapter, not the daemon.
- Wire-protocol independence from PyTorch versions (the single most important protection).
- Lean on vLLM-Lens, NNsight, Penzai as canaries: if they break, we have time to react.

### R2: HuggingFace model diversity (certain × high)

4000+ models with idiosyncratic forward-pass structures. LlamaAttention ≠ MistralAttention ≠
Qwen2Attention.

**Mitigations**:
- Canonical component vocabulary with per-architecture adapters.
- Model conformance test suite (fixed prompt, assert probes fire in expected order). Run nightly.
- Start with Llama family only. Add one architecture at a time, fully tested.
- "Best-effort module-path mode" for unsupported architectures.

### R3: torch.compile / CUDA Graph incompatibility (certain × severe)

Post-compile hook registration silently ignored. CUDA Graphs skip submodule hooks.

**Mitigations**:
- Tier B/C from §4. Detect at attach time; downgrade granularity gracefully.
- MVP requires eager mode. Refuse compiled models with clear error.
- Phase 7 budget: treat as a research problem, not routine engineering.

### R4: Multi-GPU deadlock (high × catastrophic)

Holding a barrier during NCCL collective → all ranks hang.

**Mitigations**:
- Strict pre/post-collective barriers. Never inside.
- Watchdog with auto-release after T_max.
- Fault injection in CI (deliberately hold barriers, assert watchdog fires).
- Env validation at attach time (refuse if NCCL timeout too low).

### R5: Determinism failures erode trust (high × medium)

Users replay, see different numbers, file bugs.

**Mitigations**:
- Explicit determinism guarantee in capabilities: "ULP-close, not bit-exact."
- Auto-verify on every replay; surface divergence events.
- Three-layer determinism strategy (§10).

### R6: Performance overhead (medium × medium)

If always-on probes cost 30%, nobody uses it except isolated debug runs.

**Mitigations**:
- Zero-cost when off: no Python callback when no probes active.
- Three overhead tiers: passive (~0%), interactive (~3-5%), deep (~20%).
- Publish overhead numbers per tier per model size.

### R7: Scope creep (certain × project failure)

Dense+MoE × DDP+FSDP+TP+PP × compile+CUDA-graph × 6 architectures = 540 cells.

**Mitigations**:
- Explicit support matrix (§17). Public. Honest.
- Refuse unsupported configurations with clear error.
- One model family at a time. Validate everything against Llama before adding others.

### R8: Team size for debugger-grade product (certain × high)

rr took years. NNsight has 5+ continuous contributors.

**Mitigations**:
- Aggressive scope reduction (MVP is the product; everything else is a phase).
- We reimplement but study prior art deeply — we're not starting from zero knowledge.
- Protocol-first means the hardest design work happens early, cheaply.

### R9: Protocol design lock-in (medium × medium)

The protocol shipped in MVP becomes the de facto spec.

**Mitigations**:
- Version every message: `protocol_version: "0.1.0"`.
- Written deprecation policy before v0.2.
- Borrow LSP/DAP versioning conventions.

### R10: Probe DSL safety (medium × medium)

Composable probes with user code can OOM, hang barriers, corrupt model state.

**Mitigations**:
- Two-tier probes: Tier 1 declarative (safe), Tier 2 Python callbacks (marked unsafe).
- Watchdog timer on all callback execution.
- MVP: Tier 1 only.

### R11: Session reproducibility (medium × medium)

"It worked on my machine" for transformer debugging.

**Mitigations**:
- Session bundles (§15): self-contained reproducibility artifacts.
- Env capture + hash verification.
- Replay verification with divergence reporting.

---

## Appendix A: Key Prior Art References

Organized by how they inform specific design decisions:

| Reference | Informs |
|-----------|---------|
| NNsight (Envoy, intervention graph, lazy hooks) | Model adapter, hook manager, intervention engine |
| pyvene (IntervenableConfig, intervention types) | Intervention-as-data pattern |
| nnterp (StandardizedTransformer, RenameConfig) | Canonical component vocabulary |
| baukit (TraceDict, StopForward) | Hook lifecycle, early-stop pattern |
| vLLM-Lens (WorkerExtension, TP-aware capture) | Multi-GPU capture, steering on all ranks |
| vLLM-Hook (config-file-driven hook spec) | Declarative probe definition |
| EasySteer (forward_context extension, conflict resolution) | Intervention composition |
| tokio-console (three-crate arch, delta updates, metadata dedup) | Three-process arch, TUI design |
| trippy (backend/frontend separation, snapshot pattern) | TUI-as-protocol-client |
| ratatui (widget system, immediate mode) | TUI rendering |
| DAP (lifecycle, capability negotiation, reverse-step) | Protocol lifecycle, versioning |
| MCP (tools/resources distinction) | LLM-facing adapter |
| rr (checkpoint/replay, timeline, determinism) | Checkpoint engine, replay strategy |
| transformer-debugger (DerivedScalarType taxonomy) | Built-in view definitions |
| Pernosco (notebook, dataflow view) | Session bundles, bookmarks |
| Garçon (server-per-model, time-to-first-experiment) | Daemon architecture, attach UX |
| Penzai/Treescope (models-as-data, selector system, sharding viz) | Tensor inspection, TUI visualization |
| MAIA (LLM agent driving interpretability tools) | Protocol design for LLM consumers |
| Megatron-LM (Router, MoE dispatch, TP/PP boundaries) | MoE tick design, multi-GPU boundaries |
| FlashAttention (LSE return values, tiling) | Attention inspection strategy |
| NCCL (collective atomicity, ring algorithm) | Pre/post-collective barriers |
| Goodfire Ember / Neuronpedia (API surfaces) | Protocol verb design |
| Perfetto (trace format, SDK) | Observability trace format |
| CockroachDB statement bundles | Session bundle design |
| OpenTelemetry span conventions | Probe namespace design |
| JTAG halt-and-attach | Multi-GPU barrier model |

## Appendix B: Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Daemon | Rust (tokio, tonic, serde_json) | Memory safety, performance, async runtime |
| Protocol | JSON-RPC 2.0 | Simple, LLM-native, DAP-compatible |
| Transport (MVP) | stdio + Unix socket | Works everywhere, debugger convention |
| Transport (future) | gRPC (tonic) + TCP | Bidirectional streaming, remote debugging |
| Model host | Python (asyncio) | PyTorch requires Python; no alternative |
| Rust↔Python | PyO3 | Mature, well-documented, GIL management |
| TUI | Ratatui + crossterm | Immediate mode, sub-ms rendering, Rust-native |
| Tensor format | safetensors | Fast, memory-mapped, Rust + Python support, zero-copy |
| Trace format | Perfetto protobuf | Open, rich schema, existing tooling |
| Tensor hashing | BLAKE3 | Fast (>10 GB/s), cryptographic, Rust-native |
| Config | JSON | LLM-native, no YAML footguns |

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **Tick** | One atomic unit of forward-pass execution. The smallest steppable unit. |
| **Barrier** | A host-side synchronization point that pauses the forward pass at a tick boundary. |
| **Probe** | A named observation/intervention point in the model, with a lifecycle (define → arm → fire → disarm → remove). |
| **Probe point** | A hierarchical address: `model:rank:layer:component:event`. |
| **Intervention recipe** | A JSON-serializable description of a model modification. Data, not code. |
| **Tensor handle** | A content-addressable reference (BLAKE3 hash) to a tensor. Metadata + summary; bytes on demand. |
| **Session bundle** | A self-contained tar.gz artifact for reproducing a debug session. |
| **Bookmark** | A named reference to a tick in the session timeline. |
| **Shadow replay** | Re-executing a subset of the forward pass with different settings (e.g., unfused attention) to inspect otherwise-hidden intermediate values. |
| **ProbeFrame** | The shared-memory record format for tensor handoff between Python host and Rust daemon. |
