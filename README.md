# rocket_surgeon

GDB for neural networks. Step through a transformer forward pass one tick at a time — forward and backward through time — with full surgical intervention between steps. Pause at any layer, component, attention head, or expert boundary. Inspect every tensor with research-grade summary statistics. Ablate, scale, patch, clamp, or reroute anything, then resume. Works on multi-GPU setups. Works on MoE architectures. The protocol is designed so an LLM can pick it up from schema alone with zero system prompts.

Built on 101 papers and 47 reference implementations. Reimplements everything — prior art is reference, never a dependency.

```
llama:*:12:attn.o_proj:output       ← pause here, inspect the tensor, scale it by 0.5, step forward
mixtral:*:8:experts[3]:output       ← same thing, but inside expert 3 of a Mixture-of-Experts layer
gpt2:0:*:residual_post:*            ← fire a probe on every residual stream, all layers, rank 0
```

---

## Why this exists

Every existing interpretability tool makes the same mistake: they treat the forward pass as atomic. You submit an input, you get activations back, you write Python to analyze them. If you want to intervene, you write more Python. If you want to compare two runs, you write more Python. The entire workflow is batch-mode scripting over a black-box execution.

rocket_surgeon treats the forward pass as a program you can step through. The same way GDB lets you set breakpoints, inspect registers, modify memory, and single-step through machine code — rocket_surgeon lets you set probes, inspect tensors, apply interventions, and single-step through transformer internals. The debugger metaphor isn't decorative. It's the architecture.

The wire protocol is the product. The TUI is a client. Python scripts are clients. LLMs are clients. Everything talks JSON-RPC 2.0 over the same 11 verbs and 5 events.

---

## Architecture

Three OS processes, crash-isolated:

```
┌────────────────────────────────────────────────────────────────────────┐
│                             Clients                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ rs-tui   │  │ Python   │  │ LLM (MCP     │  │ IDE (DAP        │   │
│  │ (Rust)   │  │ scripts  │  │  adapter)    │  │  adapter)       │   │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └───────┬─────────┘   │
│       └──────────────┴───────┬───────┴──────────────────┘             │
│                              │                                        │
│                    JSON-RPC 2.0 / Unix socket / TCP                   │
│                              │                                        │
│                 ┌────────────▼─────────────┐                          │
│                 │   Process A: rs-daemon   │                          │
│                 │   • Protocol server      │                          │
│                 │   • Session state machine│                          │
│                 │   • Probe registry       │                          │
│                 │   • Checkpoint index     │                          │
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
│  │ (Python+PyTorch) │  │ (Python+PyTorch) │  │ (Python+PyTorch) │  │
│  │ • Model shard    │  │ • Model shard    │  │ • Model shard    │  │
│  │ • Hook manager   │  │ • Hook manager   │  │ • Hook manager   │  │
│  │ • Barrier gate   │  │ • Barrier gate   │  │ • Barrier gate   │  │
│  │ • Intervention   │  │ • Intervention   │  │ • Intervention   │  │
│  │   engine         │  │   engine         │  │   engine         │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│         GPU 0                 GPU 1                 GPU N            │
└────────────────────────────────────────────────────────────────────────┘
```

**Process A (rs-daemon):** Rust. Owns the protocol server, session state machine, probe registry, checkpoint metadata, tensor handle store (BLAKE3 content-addressed, LRU-evicted, 2 GiB cap), and Perfetto trace sink. Stateless with respect to PyTorch — if the Python host OOMs, the daemon survives with full protocol state intact.

**Process B (rs-host):** Python. One per GPU rank. Owns the PyTorch runtime, model shard, forward-pass hooks, barrier gate (`threading.Event` for tick-boundary pausing), and intervention engine. Captures tensors into a shared-memory ring buffer; computes summary statistics on GPU before CPU transfer.

**Process C (rs-tui):** Rust. Eight independently-complex panels on a Ratatui TEA architecture. Pure protocol client — connect, disconnect, reconnect without affecting model state. Multiple TUI instances can attach to the same daemon.

The daemon-to-host link uses the same JSON-RPC 2.0 schema as the external protocol. Every internal boundary is testable with the same TCK harness.

---

## The Protocol

JSON-RPC 2.0 with Content-Length framing. DAP-inspired semantics. Every response carries the full `SessionState` — session ID, model metadata, current tick position, active probes, available checkpoints, legal actions. No hidden state. Any client can pick up any response cold and know exactly where the debugger is.

### 11 Verbs

| Method | Mutating | What it does |
|--------|----------|--------------|
| `initialize` | No | Capability negotiation. Returns protocol version, feature flags, model metadata. |
| `attach` | Yes | Load model onto GPU(s), spawn worker processes, install hooks on every module. |
| `detach` | Yes | Unload model, release GPU memory, terminate workers. |
| `rocket/step` | Yes | Advance or reverse the forward pass by N ticks. |
| `rocket/inspect` | No | Read tensor summary at a probe point. Slice on explicit request, 64 KB cap. |
| `rocket/intervene` | Yes | Set, clear, or list surgical interventions. Interventions are data, not code. |
| `rocket/probe` | Mixed | Define, enable, disable, delete probes. DTrace-inspired lifecycle. |
| `rocket/checkpoint` | Mixed | Create, restore, list, delete, bookmark checkpoints across three tiers. |
| `rocket/replay` | Yes | Restore checkpoint, replay forward with interventions, check divergence. |
| `rocket/status` | No | Full session state dump with operational metrics. |
| `rocket/subscribe` | No | Subscribe to event notifications with optional filters. |

### 5 Events

| Event | Payload |
|-------|---------|
| `rocket/tick.stopped` | Forward pass paused at tick boundary. Full state included. |
| `rocket/tick.heartbeat` | 1 Hz while stopped. Per-rank GPU utilization, memory, temperature. |
| `rocket/probe.fired` | Probe matched and executed its action. Tensor summary included. |
| `rocket/replay.divergence` | Replayed tensor diverges from original beyond tolerance. |
| `rocket/error` | Unrecoverable: OOM, NCCL hang, host crash. |

### State Machine

```
                    initialize
  [uninitialized] ─────────────► [initialized]
                                   │       ▲
                            attach │       │ detach
                                   ▼       │
                               [attaching] │
                                   │       │
                                   ▼       │
                ┌──────────────► [stopped] ─┘
                │                  │ │ │
                │     step ┌──────┘ │ └──────┐
                │          ▼        │        ▼
                │     [stepping]    │   [replaying]
                │          │        │        │
                │          └──┐     │     ┌──┘
                │             ▼     │     ▼
                ├──── [stopped] ◄───┤───► [stopped]
                │                   │
                │      inspect      │    intervene
                │          ▼        │        ▼
                │     [inspecting]  │   [modifying]
                │          │        │        │
                │          └──┐     │     ┌──┘
                └─────────────┴─────┴─────┘
```

### Error Contract

Every error is structured: machine-readable code, recovery suggestion, severity, current state, valid states for the attempted operation. 18 domain error codes (`-32001` through `-32017`) covering everything from `INVALID_STATE` to `GPU_OOM` to `COMPILED_MODEL`. An LLM can read the error, understand what went wrong, and retry correctly without human guidance.

---

## The Probe System

DTrace-inspired. Five-level hierarchical namespace:

```ebnf
probe_point = model ":" rank ":" layer ":" component ":" event
```

Wildcards at any level. Component paths support indexing for MoE expert addressing:

```
llama:*:12:attn.o_proj:output            ← attention output projection, layer 12, all ranks
mixtral:*:8:router:pre_topk              ← router logits before top-k selection
mixtral:*:8:experts[3].gate_proj:output  ← gate projection of expert 3
llama:0:*:residual_post:*                ← every residual stream, every layer, rank 0
*:*:*:*:*                                ← everything, everywhere, all at once
```

Probes follow the DTrace lifecycle: **register → arm → fire → disarm → deregister**. Zero cost when unarmed — probe points are unconditional branches over empty blocks. Armed probes install hooks. Hooks fire at tick boundaries and feed the tensor into the stats engine, checkpoint system, assertion checker, or intervention pipeline.

---

## Tensor Handling

Tensors are handles, not values. Every tensor is content-addressed via BLAKE3 hash of its raw bytes. The same tensor captured at two different probe points deduplicates to a single store entry.

**Summary-then-slice protocol:** Every inspection returns a ~200-byte summary (shape, dtype, mean, std, min, max, abs_max, L2 norm, sparsity, NaN/Inf counts, 64-bin histogram, top-10 by absolute value). Raw data is only transferred on explicit slice request, capped at 64 KB. An LLM client never accidentally consumes 10 GB.

### Statistics Engine

Two-pass computation with research-grade numerics:

- **Welford online algorithm** for mean and variance — single-pass, numerically stable for arbitrarily long streams
- **LAPACK-style scaled L2 accumulation** (dnrm2) — running-max scale factor prevents overflow/underflow without explicit casting
- **Chan/Golub/LeVeque parallel merge** for combining statistics across GPU ranks
- **64-bin linear histogram** from observed min to max
- **Top-K tracking** via min-heap (O(log K) per element)
- Non-finite values (NaN, Inf) are counted but excluded from all accumulators

Supports f16, bf16, f32, f64 — half-precision types are promoted to f64 for computation via the `half` crate.

### Shared-Memory Data Plane (DOOMRING)

Tensor bytes move through a shared-memory ring buffer, not JSON. The ring (`/dev/shm/rs-<session>-<rank>`) has a fixed header, cache-line-aligned cursors, and power-of-two slots for bitwise wrap. Each slot carries a 128-byte `ProbeFrame` header (rank, layer, component, dtype, shape, tick ID, data offset, size, flags, generation) followed by raw tensor bytes. Notification is a single-byte write on a Unix domain socket auxiliary channel.

---

## Interventions

Interventions are JSON-serializable recipes, not code. Versionable, shareable, diffable, LLM-synthesizable, reproducible.

```json
{
  "id": "ablate-name-mover-9",
  "type": "ablate",
  "target": "gpt2:0:9:attn.o_proj:fwd",
  "params": {},
  "priority": 0
}
```

**Types:** `ablate` (zero out), `scale` (multiply), `add` (additive offset), `patch` (replace), `clamp` (clip to range), `attention_mask`, `embed_swap`, `embed_noise`. MoE `route_override` in Phase 6.

Interventions persist across steps until explicitly cleared. The intervention engine runs inside the Python host at each hook barrier, applies all matching recipes in priority order, and reports which fired back through the protocol response.

---

## Checkpoint & Replay

Three checkpoint tiers:

| Tier | What it stores | Cost |
|------|---------------|------|
| `ProbeLog` | Named bookmark at a tick. No state. | Free |
| `Activation` | Residual streams + RNG state at tick boundary. | Medium |
| `FullSnapshot` | Entire `model.state_dict()`. | Expensive |

**Auto-checkpoint** every √L layer boundaries (Chen 2016 alignment). Reverse stepping is checkpoint-restore + deterministic forward replay: single CUDA stream, cuBLAS deterministic mode, fixed seeds. Replay divergence beyond tolerance fires `rocket/replay.divergence`.

Metadata tier lives in the daemon (zero-latency list/delete/bookmark). State tier lives in worker process memory. Clean separation — daemon never touches PyTorch state.

---

## Capability Negotiation

LSP-inspired. The `initialize` handshake returns boolean, enum, array, and scalar capability flags. Clients adapt to what the server supports — they never assume. Unknown fields are ignored (`additionalProperties: true`). A Phase 2 client talking to a Phase 6 server will see MoE capabilities it doesn't understand and continue operating at its own feature level.

| Phase | Unlocks |
|-------|---------|
| 0–2 (MVP) | Eager execution, single GPU, 5 intervention types, 2 built-in views, component/layer tick granularity |
| 3 | Checkpointing, reverse step |
| 4 | Protobuf wire format, TUI-driven views (head output, logit lens) |
| 5 | Multi-GPU (DDP, FSDP, tensor parallel, pipeline parallel), TCP transport |
| 6 | MoE (router/expert tick granularity, route override intervention) |
| 7 | torch.compile support, head-level tick granularity (FlashAttention shadow replay) |
| 8+ | Backward-pass ticks, SAE feature surgery, WebSocket transport |

---

## Crate Structure

11 crates. ~29K lines of Rust, ~7K lines of Python, ~4K lines of Gherkin.

| Crate | Role |
|-------|------|
| `rocket-surgeon` | Daemon binary — session state machine, JSON-RPC dispatch, tensor store, statistics engine, Perfetto sink |
| `rocket-surgeon-protocol` | Wire protocol types — all verbs, events, errors, capabilities. No I/O, pure types. |
| `rocket-surgeon-probes` | DTrace-inspired probe grammar (winnow parser), registry, pattern matching, assertions |
| `rocket-surgeon-shm` | Shared-memory ring buffer (DOOMRING) — 128-byte ProbeFrame header, zero-copy tensor handoff |
| `rocket-surgeon-transport` | IPC layer — JSON-RPC framing over Unix socket, stdio, TCP |
| `rocket-surgeon-worker` | Rust process embedding Python via PyO3 — runs forward passes, manages hooks, captures tensors |
| `rocket-surgeon-orchestrator` | Worker lifecycle management — spawn, attach, detach, cleanup |
| `rocket-surgeon-python` | PyO3 bridge — BLAKE3 hash (GIL-released), ProbeFrame packing for the Python host |
| `rocket-surgeon-tui` | Terminal UI — Ratatui, 8-panel TEA architecture, Cassowary constraint layout |
| `perfetto-writer` | Standalone Perfetto trace writer — prost protobuf, no C++ FFI, no protoc |
| `xtask` | Build orchestration, CI runner, pre-commit hooks |

---

## Building

One command:

```bash
cargo xtask setup
```

Pins Python via `.python-version`, creates `.venv`, installs dev deps via `uv`, builds the PyO3 extension (maturin), builds the Rust workspace, installs lefthook git hooks, and smoke-checks the result. Idempotent.

**Prerequisites:** Rust 1.88+ (edition 2024), [`uv`](https://docs.astral.sh/uv/), [`lefthook`](https://lefthook.dev/installation/). Everything else — Python interpreter included — is provisioned by `uv` into the project venv.

```bash
source .venv/bin/activate
```

## Testing

```bash
cargo xtask ci              # Full suite: fmt + clippy + ruff + mypy + all tests
cargo test --workspace      # Rust tests only
pytest python/tests/ -v     # Python unit tests
```

End-to-end tests spawn the daemon and drive it over JSON-RPC:

```bash
python tests/test_e2e_lifecycle.py     # Session lifecycle
python tests/test_e2e_stepping.py      # Tick stepping
python tests/test_e2e_inspect.py       # Tensor inspection
python tests/test_e2e_interventions.py # Surgical interventions
python tests/test_e2e_probes.py        # Probe system
python tests/test_e2e_checkpoint.py    # Checkpoint/replay
python tests/test_e2e_subscribe.py     # Event subscriptions
python tests/test_e2e_shm.py           # Shared-memory transport
python tests/test_e2e_perfetto.py      # Perfetto trace integration
python tests/test_e2e_bundle.py        # Session export
```

### Behavioral Specification (TCK)

314 Gherkin scenarios across 37 feature files. The TCK defines the behavioral contract — not just what passes, but what the system *is*.

```
tck/
├── protocol/     25 features — lifecycle, stepping, inspection, intervention,
│                               probes, checkpoints, replay, state envelope,
│                               errors, capabilities, subscriptions, views,
│                               session export, KV cache, sweep, branch
├── model/         5 features — adapter discovery, hook lifecycle, hooks,
│                               mailbox barrier, bridge discovery
├── perfetto/      3 features — daemon lifecycle, track hierarchy, wire format
├── tensor/        2 features — handles, shared memory
├── session/       1 feature  — bundle export
└── moe/           1 feature  — MoE tick granularity
```

---

## Quickstart

```bash
# Start the daemon (reads JSON-RPC on stdin, writes on stdout)
./target/debug/rocket-surgeon

# Initialize a session
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"client_name":"my-client","protocol_version":"0.3.0"}}

# Attach GPT-2 (spawns orchestrator + worker, installs hooks, returns architecture)
{"jsonrpc":"2.0","id":2,"method":"attach","params":{"model_path":"gpt2","model_family":"gpt2","device":"cpu","num_ranks":1}}

# Step 5 ticks into the forward pass
{"jsonrpc":"2.0","id":3,"method":"rocket/step","params":{"direction":"forward","count":5}}

# Inspect the attention output at layer 0
{"jsonrpc":"2.0","id":4,"method":"rocket/inspect","params":{"target":"gpt2:0:0:attn.o_proj:output"}}

# Ablate it and keep stepping
{"jsonrpc":"2.0","id":5,"method":"rocket/intervene","params":{"action":"set","recipe":{"id":"kill-it","type":"ablate","target":"gpt2:0:0:attn.o_proj:fwd","params":{},"priority":0}}}
```

See `docs/tutorial/quickstart.md` for the full walkthrough and `docs/tutorial/ioi.md` for reproducing the IOI circuit analysis (Wang et al. 2023) using protocol commands.

---

## Design Axioms

1. **Protocol is the product.** The wire protocol is the center of gravity. TUI, Python scripts, and LLM clients are equal consumers. If the protocol can't express it, it doesn't exist.
2. **Tensors are handles, not values.** Metadata + summary by default. Explicit materialization required. No client accidentally downloads a GPU's worth of data.
3. **State in every response.** No hidden state. Any client can pick up any response cold and know where the debugger is and what it can do.
4. **Interventions are data.** JSON-serializable recipes. Versionable, shareable, diffable, LLM-synthesizable, reproducible.
5. **Honest about limitations.** Fused kernel hides the attention matrix? Say so. torch.compile skips submodule hooks? Say so. Capability negotiation surfaces what's actually possible.
6. **Zero cost when off.** No probes armed, no client attached → overhead negligible. Unarmed probe points are unconditional branches over empty blocks.

---

## Research Foundation

The architecture was designed after surveying 101 papers and 47 reference implementations:

**Tools studied:** nnsight (Envoy proxy pattern, sentinel hooks), TransformerLens (HookPoint insertion, the "TransformerLens trap"), pyvene (interventions as serializable config), baukit (David Bau's `TraceDict`), OpenAI's transformer-debugger (DerivedScalarType taxonomy, three-phase hooks), SAELens (error term computation), ACDC (circuit discovery edge types), tuned-lens (L-BFGS inversion), rr (checkpoint + replay, diversion for what-if analysis)

**Protocol influences:** DAP (stopped-state inspection waterfall, capability negotiation, StepBack), LSP (initialize handshake, progressive capability), MCP (JSON Schema tool definitions, resource subscriptions, annotation hints for LLM safety)

**Systems studied:** tinygrad (UOp IR, PatternMatcher), vLLM, Megatron-LM, FlashAttention, NCCL, Perfetto, tokio-console (instrumentation/API/TUI as three crates), cuda-checkpoint, safetensors

Literature reviews and source code analyses are in `.context/lit-reviews/`.

---

## Development Methodology

This project follows **JSMNTL** discipline. No shortcuts.

1. Literature review for every component on the hot path (papers, reference impls, community test suites)
2. Written design spec (`docs/specs/`)
3. Written implementation plan
4. TCK-first behavioral specs (Gherkin `.feature` files — write the contract *before* the code)
5. Red → green → subagent code review → fix ALL findings
6. Frequent atomic commits with conventional messages

9 Architectural Decision Records in `docs/adr/`. Research material in `.context/lit-reviews/`. Design decisions in `.context/decisions/`.

---

## Status

Phase 3 in progress. Core protocol, stepping, inspection, interventions, probes, checkpoints, shared memory, Perfetto traces, event subscriptions, and session export are implemented. TUI architecture is designed. Multi-GPU and MoE are architected and specified but not yet on the hot path.

---

## License

MIT OR Apache-2.0
