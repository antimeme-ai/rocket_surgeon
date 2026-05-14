# Debugger & Protocol Pattern Analysis

Reference implementation survey for rocket_surgeon protocol and architecture design.

Sources analyzed:
- `quarantine/transformer-debugger/` -- OpenAI's transformer debugging activation server
- `quarantine/debug-adapter-protocol/` -- Microsoft's Debug Adapter Protocol (DAP) spec
- `quarantine/mcp-spec/` + `quarantine/mcp-servers/` -- Model Context Protocol spec and reference servers
- `quarantine/rr/` -- Mozilla's record-and-replay debugger
- `quarantine/pyo3/` -- PyO3 Rust-Python FFI framework
- `../papers/rocket-surgeon/debuggers/OCallahan2017_Engineering_Record_Replay.pdf`
- `../papers/rocket-surgeon/debuggers/OCallahan2017_rr_Extended_Report.pdf`
- `../papers/rocket-surgeon/debuggers/AOSA_v2_GDB_Chapter.pdf`

---

## 1. Per-Repo Findings

### 1.1 transformer-debugger (OpenAI)

**Architecture.** FastAPI server wrapping an `InteractiveModel` that holds a `Transformer` + `StandardModelContext` + optional `AutoencoderContext`. The activation server exposes REST routes (read_routes, explainer_routes, inference_routes) for a React frontend. Core processing happens in `InteractiveModel.handle_batched_request()` which follows a three-step pipeline:

1. Infer `DerivedScalarType` configs from request specs
2. Run forward pass via `_compute_multi_group_ds_store` with hook-based activation capture
3. Extract response data per processing spec using a registry mapping request types to response types

The system uses an `asyncio.Lock` to prevent concurrent CUDA OOM conditions, and catches `torch.cuda.OutOfMemoryError` by sending SIGKILL to the process.

**Key abstractions.**

- `DerivedScalarType` enum (~100+ members): Exhaustive taxonomy of every possible transformer internal -- LOGITS, MLP_PRE_ACT, MLP_POST_ACT, ATTN_QUERY, ATTN_KEY, ATTN_VALUE, ATTN_QK_LOGITS, ATTN_QK_PROBS, plus derived quantities (write norms, act_times_grad, edge attributions, autoencoder latents). Each member has a `shape_spec_per_token_sequence` mapping to `Dimension` tuples, plus properties like `is_raw_activation_type`, `node_type`, `requires_grad_for_forward_pass`, `location_within_layer`.

- Hook graph system: `Hooks` (sequential callable chain) -> `HookCollection` (named dict) -> `FwdBwdHooks` (fwd + bwd + fwd2 slots) -> component-specific hook types (`MLPHooks` with pre_act/post_act, `AttentionHooks` with q/k/v/qk_logits/qk_probs/v_out, `ResidualStreamHooks` with post_emb/torso/ln_f) -> `TransformerHooks` (mlp + attn + resid + logits per layer). `AutoencoderHooks` add encode/decode/latents/reconstruction/error slots. `HookGraph` abstraction provides activation_location_type-based addressing over the full hook tree.

- Batched request/response: `InferenceRequestSpec` (prompt, ablation_specs, loss_fn_config, trace_config) composed with `DerivedScalarsRequestSpec` or `MultipleTopKDerivedScalarsRequestSpec`. `GroupId` enum (ACT_TIMES_GRAD, ACTIVATION, WRITE_NORM, etc.) controls which derived scalar groups to compute. `REQUEST_RESPONSE_CORRESPONDENCE_REGISTRY` maps request types to response types. All types use `CamelCaseBaseModel` (Pydantic).

**What to steal.**

- The `DerivedScalarType` taxonomy. This is the most thorough enumeration of "things you'd want to look at" in a transformer that exists in any open implementation. Steal the completeness of the taxonomy, not the implementation. rocket_surgeon's `ActivationLocationType` should cover the same semantic space.
- The shape_spec_per_token_sequence pattern -- associating each activation type with its expected shape dimensions. This is essential for validation and for LLM-ergonomic reporting ("this is shape [batch, seq, heads, dim]").
- The hook slot structure (fwd/bwd/fwd2 at every component). The three-phase hook model captures the reality that you need forward hooks, backward hooks, and sometimes a second forward pass for gradient-based attribution.
- The `GroupId` concept for batching related derived scalar requests. When an LLM asks "show me all attention patterns for layer 5", that maps to a group, not individual activations.

**What to avoid.**

- REST-only transport. No streaming, no JSON-RPC, no bidirectional communication. The frontend polls.
- SIGKILL on CUDA OOM. A debugger should recover gracefully, not kill itself.
- Tight coupling to a specific transformer implementation. The `Transformer` class is OpenAI's internal format. rocket_surgeon must work with any HuggingFace model, any custom model, dense or MoE.
- No stepping/tick model. The entire forward pass runs atomically -- you cannot pause between layers, inspect, modify, and continue. This is the single biggest gap that rocket_surgeon fills.
- No multi-GPU awareness. Single-device only.
- Pydantic-based serialization with CamelCase convention. Works for a web app but is wasteful for a high-performance protocol. We want zero-copy where possible.
- asyncio.Lock for GPU serialization. A proper solution uses CUDA streams and events for fine-grained synchronization.

### 1.2 Debug Adapter Protocol (DAP)

**Architecture.** JSON messages over a framed transport (Content-Length header + JSON body, similar to LSP). Three message types:

- `Request`: `{ seq, type: "request", command, arguments }`
- `Response`: `{ seq, request_seq, type: "response", success, command, message?, body }`
- `Event`: `{ seq, type: "event", event, body }`

Sequential message IDs (`seq`) correlate requests to responses. Events are asynchronous server-initiated messages.

**Key abstractions.**

- Lifecycle: `initialize` (capability exchange) -> `launch`/`attach` -> `configurationDone` -> running/stopped event loop -> `terminate`/`disconnect`. The initialization phase establishes which optional features are supported via capability flags.

- State inspection waterfall: When execution stops, the client walks a hierarchy: `threads` -> `stackTrace(threadId)` -> `scopes(frameId)` -> `variables(variablesReference)`. Each level returns references (integer IDs) that are valid only while execution is stopped. Resuming execution invalidates all references.

- Object reference lifetime: Variable references are tied to the stopped state. This is critical -- it means the debugger doesn't need to maintain a persistent object graph, only a snapshot at each stop. When the user says "continue", all references become invalid.

- Capability-based feature negotiation: The `initialize` request includes `supportsXxx` fields. The server's response includes its own `supportsXxx` fields. Both sides only use features that were successfully negotiated. Examples: `supportsConditionalBreakpoints`, `supportsStepBack`, `supportsTerminateRequest`.

**What to steal.**

- The capability negotiation pattern. rocket_surgeon needs this: not every model supports MoE features, not every setup is multi-GPU. The client and server should negotiate what's available at connection time.
- The stopped-state inspection waterfall. Map this to our domain: `stopped` -> `tick_state(tick_id)` -> `layers(tick_id)` -> `activations(layer_id, activation_type)` -> `tensor_data(activation_ref)`. References valid only at the current tick.
- The `seq`/`request_seq` correlation pattern. JSON-RPC 2.0 gives us `id` for free, but DAP's approach of sequential IDs is simpler for debugging the debugger.
- Event-driven state changes. The `stopped` event with a `reason` field (breakpoint, step, exception, pause) maps directly to rocket_surgeon's tick-stop reasons.
- The `StepBack` capability. DAP considers reverse debugging an optional capability. rocket_surgeon should do the same -- checkpoint/replay is powerful but not always available.
- The object reference lifetime model. Tensor data references should be scoped to the current tick state. When the user steps to the next tick, old references become invalid. This prevents stale data bugs and simplifies memory management.

**What to avoid.**

- The Content-Length framing. JSON-RPC 2.0 over stdio or HTTP is cleaner.
- The rigid thread/stack/scope/variable hierarchy. Our domain doesn't map to traditional debugging concepts. We need our own hierarchy: model -> device -> layer -> component -> activation.
- DAP's lack of structured output schemas. Tool results are untyped `body` objects. MCP does this better with `outputSchema`.
- The sequential numbering system. While simple, it leaks ordering assumptions. JSON-RPC 2.0's arbitrary IDs are more flexible.

### 1.3 Model Context Protocol (MCP)

**Architecture.** JSON-RPC 2.0 with three primitive types: resources (read-only data), tools (callable actions), and prompts (templated interactions). Transport: stdio (default), SSE (deprecated), Streamable HTTP.

**Key abstractions.**

- Lifecycle: `initialize` (protocol version + capabilities) -> `initialized` notification -> operation -> shutdown. Version negotiation: client sends version, server responds with same or different. If incompatible, client disconnects.

- Tools: Defined with `name`, `title`, `description`, `inputSchema` (JSON Schema), `outputSchema` (optional), `annotations` (readOnlyHint, idempotentHint, destructiveHint). Tool results contain `content` (unstructured: text, image, audio, resource links, embedded resources) and/or `structuredContent` (JSON matching outputSchema). `isError` flag for tool execution errors vs. protocol errors.

- Resources: URI-addressed read-only data with subscriptions and list-changed notifications. Templates for parameterized resources.

- Prompts: Server-provided prompt templates with arguments. Less relevant for rocket_surgeon.

- Capability negotiation: Client capabilities (roots, sampling, elicitation). Server capabilities (prompts, resources, tools, logging, completions). Sub-capabilities: `listChanged`, `subscribe`.

- List-changed notifications: `notifications/tools/list_changed`, `notifications/resources/list_changed`. Dynamic registration/deregistration of capabilities.

**What to steal.**

- JSON-RPC 2.0 as transport protocol. This is the right choice for rocket_surgeon. It gives us request/response correlation, batch requests, error codes, and a well-understood wire format.
- The tool annotation pattern. `readOnlyHint`, `idempotentHint`, `destructiveHint` map directly to our domain: "inspect activation" is read-only, "modify weight" is destructive. These hints let LLMs make safe decisions about which operations to call autonomously.
- `outputSchema` for structured tool results. Every rocket_surgeon tool should declare its output schema. This lets LLMs parse results reliably without guessing at structure.
- The dual content model: `content` (human-readable text/images) + `structuredContent` (machine-parseable JSON). This is exactly what rocket_surgeon's dual-interface (TUI + LLM) needs. The TUI renders `content`, the LLM consumes `structuredContent`.
- Resource subscriptions for live data. Map this to activation tensors: subscribe to "layer_5.attn.qk_probs" and get notified when it changes (i.e., when the tick advances and that layer re-executes).
- List-changed notifications. When the model architecture is loaded, tools/resources change. When you step into an MoE layer, new expert-specific tools appear. Dynamic capability discovery is essential.
- The `roots` protocol for filesystem scoping. Map to "model roots" -- which model checkpoints, which devices are accessible.

**What to avoid.**

- MCP's stateless tool model. MCP tools are designed to be independently callable. rocket_surgeon needs stateful sessions -- you step, inspect, modify, step again. The tick state IS the session state.
- The lack of a "stopped state" concept. MCP has no notion of execution being paused. We need DAP's stopped-state model on top of MCP's tool/resource primitives.
- SSE transport. Deprecated in MCP itself, and not suitable for high-bandwidth tensor data.

### 1.4 rr (Mozilla Record-and-Replay)

**Architecture.** User-space record-and-replay debugger. Records at the boundary between user-space and kernel: system call results, signal timing, and nondeterministic instruction results. Replays by re-executing the program and injecting recorded nondeterminism at the right points.

Key design decisions from the papers:

- **Recording boundary at user/kernel interface.** CPUs are mostly deterministic. The only nondeterminism comes from system calls, signals, and a few CPU instructions (RDTSC, RDRAND, CPUID core ID). By recording only at this boundary, rr avoids instrumenting application code.

- **Single-thread scheduling.** rr runs only one thread at a time, using ptrace to control scheduling. This eliminates data races as a source of nondeterminism. Trade-off: high-parallelism workloads suffer significant slowdown (~12x for `make -j8`), but low-parallelism workloads see only ~1.5x overhead.

- **In-process system-call interception (seccomp-bpf).** The core optimization. Instead of 4 context switches per syscall (tracee->kernel->rr->kernel->tracee), rr injects an interception library that handles common syscalls (read, gettimeofday) in-process, recording results to a shared buffer. Only blocking or complex syscalls trap to rr via ptrace. This reduced overhead dramatically (Figure 5 in paper shows 2-10x improvement).

- **RCB (Retired Conditional Branches) hardware performance counter.** The only deterministic performance counter on Intel CPUs. Used to measure execution progress for delivering asynchronous events (signals, context switches) at exactly the right point during replay. Paired with general-purpose register state to identify unique execution points.

- **ReplaySession::clone() for checkpointing.** Uses fork() which is (mostly) copy-on-write, making checkpoints cheap (<10ms). This enables efficient reverse-execution debugging: checkpoint periodically, then replay forward from the nearest checkpoint to reach any past state.

- **Trace format.** `TraceFrame = (global_time: FrameTime(i64), tid, Event, Ticks, monotonic_time, Registers, ExtraRegisters)`. Each frame represents a recorded event at a specific point in execution. The `Ticks` field (from RCB counter) provides sub-frame execution position.

**Key abstractions from source.**

- `RecordSession`: `record_step()` returns `RecordResult` (STEP_CONTINUE/STEP_EXITED/STEP_SPAWN_FAILED). Manages TraceWriter, Scheduler, syscall buffering. Chaos mode for non-deterministic scheduling to expose bugs.

- `ReplaySession`: `replay_step(StepConstraints)` with `RunCommand` enum. `ReplayStepKey` for ordering. `clone()` returns partially-initialized clone for efficient checkpointing. `clone_diversion()` for "free execution" (running code that wasn't in the original recording). TraceReader for reading back the recorded trace.

- `ReplayTraceStep` types: TSTEP_ENTER_SYSCALL, TSTEP_EXIT_SYSCALL, TSTEP_DETERMINISTIC_SIGNAL, TSTEP_PROGRAM_ASYNC_SIGNAL_INTERRUPT, etc. These enumerate every kind of event that can occur during replay.

**What to steal.**

- The checkpoint-via-clone pattern. For rocket_surgeon, "checkpoint the forward pass state" means snapshotting tensors on GPU. We can use a similar COW strategy: keep tensors pinned, only copy when modified. The `clone()` API (returns a new session that shares immutable state) is the right abstraction.
- The step-with-constraints model. `replay_step(StepConstraints)` maps to "step one tick, but stop if: you hit this layer, this activation value exceeds threshold, this routing decision is made." StepConstraints is the breakpoint-equivalent for forward passes.
- The global_time monotonic counter. Every tick in rocket_surgeon should have a monotonic global_time. This enables unambiguous references to any point in the forward pass history: "at tick 47, layer 3 attention head 7 had these QK logits."
- The diversion concept. `clone_diversion()` allows running code that diverges from the recording. Map to "what-if" analysis: clone the forward pass state, modify a weight or activation, run the rest of the forward pass, compare results.
- The trace-frame-as-fundamental-unit pattern. Each step through the forward pass should produce a `TickFrame` with: global_time, layer_id, component, all captured activations, routing decisions (for MoE), device placement.

**What to avoid.**

- Everything related to syscall interception, ptrace, seccomp-bpf. Our "nondeterminism boundary" is completely different -- it's the GPU kernel execution boundary, not the user/kernel interface.
- Single-thread scheduling. We're inherently multi-device. We need a different strategy for determinism: controlling CUDA random seeds, deterministic algorithms, synchronization barriers between devices.
- The RCB counter dependency. We have no hardware performance counter equivalent for GPU execution. Our "ticks" are defined at the model architecture level (layer boundaries), not at the instruction level.

### 1.5 PyO3 (Rust-Python FFI)

**Architecture.** PyO3 provides a Rust framework for building Python extension modules and embedding Python in Rust. Key to rocket_surgeon because the debugger engine will be Rust, but the model and tensors live in Python/PyTorch.

**Key patterns from the guides.**

- **GIL management with `Python::detach` / `Python::attach`.** `detach()` releases the GIL so other Python threads (and Rust rayon workers) can run. `attach()` re-acquires it. Pattern: acquire GIL -> extract data from Python objects -> `detach()` -> do Rust work in parallel -> `attach()` -> write results back to Python objects. This is the core pattern for rocket_surgeon's "step" operation: grab the model state from Python, do analysis in Rust, return results.

- **`Py<T>` for GIL-independent storage, `Bound<'py, T>` for GIL-bound operations.** Store `Py<PyAny>` in Rust structs (no lifetime), convert to `Bound<'py, T>` when you need to call Python methods. This is how the debugger engine stores references to the model and tensors.

- **rayon parallelism with GIL release.** `Python::detach(|| instances.par_iter().map(|i| Python::attach(|py| ...)).collect())`. Release GIL in the orchestrator, reacquire per-worker-thread for Python object access. Watch for deadlocks: always `detach()` before spawning workers that will `attach()`.

- **Thread safety for `#[pyclass]`.** Python objects are freely shared between threads, so `#[pyclass]` types must be `Send + Sync`. Options: (a) default interior mutability with runtime borrow checking, (b) `#[pyclass(frozen)]` with atomics, (c) `#[pyclass(frozen)]` with `Mutex<Inner>`. For tensor wrappers, frozen + Mutex is likely best.

- **Cross-extension-module data sharing via PyCapsule.** When rocket_surgeon is a Rust extension module that needs to interact with PyTorch tensors (another native extension), the capsule mechanism is how data crosses the boundary. NumPy's C API pattern: export a `#[repr(C)]` API struct in a capsule, downstream extensions import it. Version checking via ABI version field at the start of the struct.

- **`#[repr(C)]` requirement for cross-boundary types.** Only `#[repr(C)]` types can safely cross cdylib boundaries. Standard Rust types (Vec, String, Box) cannot be shared. The `abi_stable` crate provides equivalents with stable layouts.

**What to steal.**

- The `detach()`/rayon pattern for parallel tensor analysis. When the debugger needs to compute derived scalars across multiple layers simultaneously, release the GIL, use rayon to parallelize across layers, each worker re-acquires the GIL only to read tensor data from PyTorch.
- The `Py<T>` storage pattern for the debugger engine. The engine holds `Py<PyAny>` references to the model, optimizer, tensors. It converts to `Bound<'py, T>` only when executing Python operations (forward pass, hook registration).
- The frozen-pyclass-with-Mutex pattern for debugger state. The `DebugSession` exposed to Python should be `#[pyclass(frozen)]` with a `Mutex<SessionInner>` for the mutable tick state.
- PyCapsule for zero-copy tensor access. If we can get a raw pointer to PyTorch tensor data via the DLPack protocol or NumPy's buffer protocol, we can avoid copies when sending tensor data over the protocol.

**What to avoid.**

- Sharing non-`#[repr(C)]` types across extension boundaries. If rocket_surgeon ever needs to share types with other native extensions, everything at the boundary must be `#[repr(C)]`.
- Holding the GIL while waiting for GPU operations. CUDA kernel launches are asynchronous. If we hold the GIL while waiting for a GPU sync, we block all Python threads. Always `detach()` before `torch.cuda.synchronize()`.
- The `#[pyclass(unsendable)]` escape hatch. This makes concurrent access a runtime error. rocket_surgeon will be used from multiple threads (TUI thread, protocol handler thread, background analysis threads).

### 1.6 GDB (AOSA Chapter -- Shebs)

**Architecture.** GDB is a large, monolithic C codebase (~750K lines at time of writing) organized around several key abstractions. Author Stan Shebs describes the internal architecture shaped by decades of evolution.

**Key architectural observations from the chapter.**

- **Target stack.** GDB abstracts over different execution targets (local process, remote via gdbserver, core files, simulators) through a stack of "target" layers. Each target layer can handle some operations and delegate others down the stack. This is GDB's version of the adapter pattern -- one debugger frontend, many backends.

- **Symbol tables as the core data model.** GDB's internal representation centers on symbol tables: files, functions, types, variables, line numbers. The `struct symbol` is ubiquitous. Symbol lookup is the most performance-critical path -- GDB uses partial symbol tables (psymtabs) that are lazily expanded to full symbol tables (symtabs) on demand to manage memory.

- **The "observer" pattern for event propagation.** GDB uses an observer/notification system for events like breakpoint hits, thread creation/destruction, and inferior process state changes. This decouples the event source from the event consumers (UI, scripting, logging).

- **Expression parser and evaluation.** GDB has a full expression parser that understands C, C++, Fortran, etc. expressions. Evaluation happens on the target's data, not locally. This means evaluating "p x->y->z" requires multiple target memory reads, each of which might fail.

- **Remote protocol (GDB RSP).** A simple, character-based protocol for communicating with remote targets. Packets are `$data#checksum`. Extremely low-level but proven reliable over decades and many transports (serial, TCP, pipes). The simplicity is a feature -- easy to implement a stub.

**What to steal.**

- The target stack / adapter pattern. rocket_surgeon should have a similar layered target abstraction: local GPU target, remote GPU target (for multi-node debugging), recorded-trace target (for replay). Each target implements the same interface; the debugger frontend doesn't care which one is active.
- Lazy expansion of expensive data. GDB's psymtab/symtab split is analogous to our problem: don't materialize all tensor data for all layers eagerly. Have a "summary" level (shapes, norms, top-k values) that's always available, and a "full" level (complete tensor data) that's fetched on demand.
- The observer pattern for event propagation. When a tick-stop event occurs, multiple consumers need to know: the TUI needs to update, the protocol handler needs to send a notification, any subscribed resources need to refresh. Use an observer/event bus, not direct coupling.

**What to avoid.**

- GDB's monolithic C codebase and accumulated technical debt. The chapter is frank about design decisions that were expedient at the time but became constraints. Avoid growing organically without clear module boundaries.
- The text-based remote protocol. GDB RSP is elegant for its simplicity but terrible for high-bandwidth data like tensor contents. JSON-RPC 2.0 gives us structure; we need a binary sideband for bulk tensor data.
- Expression parsing complexity. Our "expressions" are simpler (layer paths, activation types, tensor slicing) but we should define a clear grammar up front rather than growing it ad hoc.

---

## 2. Protocol Design Lessons

### 2.1 DAP Patterns That Map to Our Domain

| DAP Concept | rocket_surgeon Equivalent | Notes |
|---|---|---|
| `initialize` with capabilities | `initialize` with model capabilities | Negotiate: MoE support, multi-GPU, checkpoint/replay, gradient tracking, SAE integration |
| `launch` / `attach` | `attach_model` | Load or connect to an existing model; discover architecture |
| `configurationDone` | `configure_session` | Set breakpoints (layer stops), tick granularity, capture config |
| `stopped` event + reason | `tick_stopped` event + reason | Reasons: tick_boundary, layer_breakpoint, activation_threshold, routing_anomaly, user_pause |
| `threads` request | `devices` request | List GPUs/devices in the model's distributed setup |
| `stackTrace(threadId)` | `layers(device_id)` | List layers assigned to this device (in pipeline/tensor parallel) |
| `scopes(frameId)` | `components(layer_id)` | List components: attention, mlp, residual, experts (MoE) |
| `variables(ref)` | `activations(component_ref)` | Get activation tensors, weights, gradients for a component |
| `evaluate` expression | `evaluate` tensor expression | Slice, reduce, compare tensors: `layer5.attn.qk_probs[:, 3, :, :]` |
| `setVariable` | `set_activation` / `set_weight` | Surgical intervention: modify a value and continue |
| `stepIn` / `stepOut` / `stepOver` | `tick_forward` / `tick_backward` / `skip_layer` | Navigation through the forward pass |
| `StepBack` capability | `checkpoint_replay` capability | Optional reverse debugging via checkpoints |
| Object reference lifetime | Activation reference lifetime | References valid only at current tick; invalidated on step |
| `disconnect` | `detach_model` | Clean up hooks, release resources |

### 2.2 MCP Patterns for LLM Integration

**Tool design for LLM clients.**

Every debugging operation should be exposed as an MCP-style tool with:
- `inputSchema`: JSON Schema for arguments (Zod validation in TS clients, pydantic in Python)
- `outputSchema`: JSON Schema for structured results
- `annotations`: readOnlyHint (most inspection tools), destructiveHint (activation/weight modification), idempotentHint (repeated queries return same data at same tick)

Example tool definitions:

```
tool: inspect_activation
  inputSchema: { layer: int, component: enum, activation_type: enum, slice?: string }
  outputSchema: { shape: int[], dtype: string, summary: { mean, std, min, max, norm }, data?: float[][] }
  annotations: { readOnlyHint: true, idempotentHint: true }

tool: modify_activation
  inputSchema: { layer: int, component: enum, activation_type: enum, slice: string, value: float[][] }
  outputSchema: { previous_summary: {...}, new_summary: {...}, tick_id: int }
  annotations: { destructiveHint: true }

tool: tick_forward
  inputSchema: { count?: int, until_layer?: int, until_condition?: string }
  outputSchema: { new_tick_id: int, stopped_reason: string, layer_id: int }
  annotations: { destructiveHint: true }
```

**Resource design for live data.**

Activations, weights, and routing decisions as MCP-style resources with URI addressing:

```
resource: rs://model/layer/{layer_id}/{component}/{activation_type}
  - Readable: returns tensor summary + optional full data
  - Subscribable: client gets notified when tick advances past this layer

resource: rs://model/architecture
  - Readable: returns full model architecture description
  - Static after attach (no subscription needed)

resource: rs://session/tick/{tick_id}
  - Readable: returns TickFrame with all captured data at that tick
  - Historical: past ticks remain accessible if checkpointed
```

**Dual content model.**

Every tool result should include both:
- `content`: Human-readable text/image for TUI display (formatted tensor summaries, ASCII heatmaps, sparklines)
- `structuredContent`: Machine-parseable JSON matching `outputSchema` for LLM consumption

This is the key insight from MCP: the same tool serves both the TUI user (via `content`) and the LLM user (via `structuredContent`).

**Dynamic capability discovery.**

When the model is loaded, emit `notifications/tools/list_changed` to signal that model-specific tools are now available. When stepping into an MoE layer, new expert-routing tools appear. When a checkpoint is created, replay tools become available. This mirrors MCP's list-changed notification pattern.

### 2.3 Protocol Wire Format Decision

Use JSON-RPC 2.0 as the base protocol, with these extensions:

1. **Capability negotiation** (from DAP): First message is `initialize` with client/server capabilities.
2. **Tool/resource/prompt primitives** (from MCP): Tools for actions, resources for data, prompts for suggested workflows.
3. **Stopped-state model** (from DAP): Tick-stopped events with reason, reference lifetime tied to tick state.
4. **Structured output** (from MCP): `outputSchema` on every tool.
5. **Binary sideband** for bulk tensor data: JSON-RPC for control plane, separate binary channel (shared memory, memory-mapped files, or DLPack) for tensor data. The JSON message references tensor data by ID; the actual bytes are on the sideband.

---

## 3. Checkpoint/Replay Lessons from rr

### 3.1 The Core Insight: Boundary-Based Recording

rr's fundamental insight is that most computation is deterministic; you only need to record what crosses the nondeterminism boundary. For rr, that boundary is the user/kernel interface (syscalls, signals).

For rocket_surgeon, the nondeterminism boundary is:
- **GPU kernel execution**: Floating-point non-associativity means different reduction orders produce different results. CUDA's cuDNN/cuBLAS may choose different algorithms per run.
- **Dropout and random sampling**: Random number generator state.
- **MoE routing decisions**: Expert selection based on gating scores that may vary with numerical precision.
- **Multi-GPU communication**: NCCL all-reduce ordering.

The analogy holds: if we record the outputs at these boundaries (random seeds, routing decisions, reduction results at synchronization points), we can replay deterministically.

### 3.2 Checkpointing Strategy

rr uses `fork()` for cheap COW checkpoints (<10ms). For GPU state, we need a different strategy:

**Approach: Layered checkpoint with lazy materialization.**

1. **Lightweight checkpoint** (always available): Record the random state, routing decisions, and tensor metadata (shapes, devices, dtypes) at each tick. This is tiny -- kilobytes.
2. **Medium checkpoint** (on-demand): Snapshot the activations at layer boundaries to pinned host memory. Use CUDA async memcpy so the GPU pipeline isn't stalled. Cost: ~the size of the activation tensors for one layer. Can be pipelined with computation.
3. **Full checkpoint** (expensive, user-requested): Deep copy all model weights, optimizer state, and activation caches. This is the "save game" operation. Use it sparingly.

**Replay from checkpoints:**

To replay from tick N to tick M:
1. Find the most recent checkpoint before N.
2. Restore model state from checkpoint.
3. Re-execute forward pass from checkpoint tick to tick N, replaying recorded nondeterminism (random seeds, routing decisions).
4. Stop at tick N with full state available.

This mirrors rr's approach: checkpoint periodically, replay forward to reach any past state. The "diversion" concept (rr's `clone_diversion()`) maps to "what-if" analysis: clone the state, make a modification, run forward, compare.

### 3.3 Tick-Frame Format

Inspired by rr's `TraceFrame`:

```
TickFrame {
    global_time: u64,          // monotonic tick counter (rr's FrameTime)
    layer_id: LayerId,          // which layer just executed
    component: ComponentType,   // attention, mlp, residual, expert(N)
    device_id: DeviceId,        // which GPU
    captured_activations: HashMap<ActivationType, TensorRef>,  // activation data by type
    routing_decisions: Option<RoutingInfo>,  // MoE expert selection, if applicable
    random_state: Option<RngState>,         // RNG checkpoint for replay
    parent_tick: u64,           // previous tick (for backward traversal)
    metadata: TickMetadata,     // timing, memory usage, FLOP estimate
}
```

The `global_time` is the unambiguous identifier for any point in the forward pass. All references (activation refs, checkpoint refs) are anchored to a global_time. This is directly from rr's design.

### 3.4 What Doesn't Transfer from rr

- **Instruction-level granularity.** rr records at the CPU instruction level (via RCB counters). We record at the layer/component level. Our "tick" is much coarser than rr's "tick" (retired conditional branches). This is fine -- our users think in layers and heads, not in GPU instructions.
- **Single-thread determinism.** rr achieves determinism by running one thread at a time. We can't serialize GPU execution across devices -- the whole point is parallelism. Instead, we use CUDA synchronization primitives and record the nondeterministic outputs at sync points.
- **Kernel-level syscall modeling.** rr maintains a model of every Linux syscall's memory effects. We need a model of every PyTorch operation's tensor effects, which is essentially the autograd graph. We should hook into PyTorch's autograd rather than building our own op-level model.

---

## 4. PyO3 Patterns for Rust<->Python Tensor Sharing

### 4.1 GIL Management for Debugger Operations

The debugger engine lifecycle for a "tick + inspect" operation:

```
Python::attach(|py| {
    // 1. GIL held: register hooks on the PyTorch model
    register_forward_hooks(py, &model, &hook_config);

    // 2. GIL held: execute one forward pass tick
    execute_tick(py, &model, &inputs);

    // 3. GIL held: extract tensor data from Python hook results
    let raw_tensors: Vec<TensorData> = extract_hook_results(py);

    // 4. GIL released: parallel analysis in Rust
    let analysis_results = py.detach(|| {
        raw_tensors.par_iter().map(|t| {
            compute_derived_scalars(t)  // pure Rust, no GIL needed
        }).collect()
    });

    // 5. GIL held: format results for protocol response
    build_tick_frame(py, analysis_results)
});
```

Critical rules:
- **Never hold the GIL while waiting for GPU sync.** Call `py.detach()` before `torch.cuda.synchronize()`.
- **Minimize GIL-held time.** Extract raw data (via DLPack or buffer protocol) as fast as possible, then release for Rust-side analysis.
- **Use rayon inside `detach()` blocks.** Per-layer analysis, per-head analysis, per-expert analysis all parallelize naturally.

### 4.2 Tensor Data Transfer Strategy

Three tiers, based on data size:

1. **Small data (metadata, summaries, <1KB):** JSON via the protocol. Shape, dtype, statistical summaries (mean, std, min, max, norm, top-k). Always included in `structuredContent`.

2. **Medium data (sliced tensors, heatmaps, 1KB-10MB):** Base64-encoded in JSON protocol messages, or as MCP-style embedded resources. The LLM can request specific slices: "attention head 3, sequence positions 10-20."

3. **Large data (full tensors, >10MB):** Binary sideband channel. Options:
   - **Shared memory** (for local debugger): mmap a file, pass the path + offset + shape + dtype in the JSON message.
   - **DLPack zero-copy** (for in-process Python access): The tensor stays on GPU; the Rust side gets a raw pointer via DLPack. No copy at all.
   - **Streaming binary** (for remote debugger): Chunked transfer with backpressure.

### 4.3 PyO3 Type Design for the Engine

```rust
#[pyclass(frozen)]
struct DebugSession {
    inner: Mutex<SessionInner>,
    model: Py<PyAny>,              // reference to the PyTorch model
    config: SessionConfig,         // immutable after creation
}

struct SessionInner {
    current_tick: u64,
    tick_frames: Vec<TickFrame>,   // recorded history
    checkpoints: BTreeMap<u64, Checkpoint>,
    hook_handles: Vec<Py<PyAny>>,  // PyTorch hook handles for cleanup
    active_subscriptions: HashMap<ResourceUri, Vec<SubscriptionId>>,
}
```

Use `#[pyclass(frozen)]` with `Mutex<SessionInner>` rather than PyO3's default interior mutability:
- The `frozen` attribute means all `#[pymethods]` take `&self` (no `&mut self`).
- The `Mutex` provides explicit, controlled synchronization.
- Multiple Python threads can hold references to the session simultaneously; the Mutex serializes mutations.
- Use `parking_lot::Mutex` via PyO3's `lock_api` feature to avoid GIL deadlocks.

### 4.4 Cross-Extension Patterns

If rocket_surgeon ever needs to share native types with other Rust extensions (e.g., a SAE library, a visualization library):
- Expose a `#[repr(C)]` API struct via PyCapsule.
- Version the API with an ABI version field.
- Share only `#[repr(C)]` types across the boundary. Use `abi_stable` crate equivalents for Vec, String, etc.
- The function-pointer-in-struct pattern (from the PyO3 sharing-types guide) is the correct approach.

However, avoid this complexity unless there's a proven need. The primary interface should be the JSON-RPC protocol. Native sharing is an optimization for tight integration scenarios.

---

## 5. How transformer-debugger Falls Short and What We'd Do Differently

### 5.1 No Stepping Model

**transformer-debugger:** Runs the entire forward pass atomically. You submit a request, it runs inference, you get back results. There is no concept of pausing mid-forward-pass, inspecting, modifying, and continuing.

**rocket_surgeon:** One tick at a time. The forward pass is decomposed into discrete steps (per-layer, per-component, configurable granularity). Between ticks, the user (or LLM) has full access to inspect and modify any activation, weight, routing decision, or gradient. This is the fundamental difference -- transformer-debugger is an "X-ray machine" (observe after the fact), rocket_surgeon is a "surgeon's table" (operate during the procedure).

### 5.2 No Multi-GPU Awareness

**transformer-debugger:** Single-device. The `InteractiveModel` holds one `Transformer` on one GPU. No concept of tensor parallelism, pipeline parallelism, FSDP, or DDP.

**rocket_surgeon:** Multi-GPU from day one. The protocol includes `DeviceId` in every activation reference. The `devices` request (analogous to DAP's `threads`) lists all GPUs. The layer-to-device mapping is exposed. Tensor parallel operations show how data is sharded. Pipeline parallel shows the stage boundaries. FSDP shows which parameters are gathered.

### 5.3 No MoE Support

**transformer-debugger:** Dense transformers only. The `DerivedScalarType` enum has no concept of experts, routing, load balancing, or expert-specific activations.

**rocket_surgeon:** MoE is a first-class citizen. The component hierarchy includes `Expert(n)` as a component type. Routing decisions are captured in every tick frame. New tools appear when stepping into MoE layers: `inspect_routing`, `inspect_expert_activation`, `modify_routing`, `compare_experts`. The `ActivationType` enum includes expert-specific variants: EXPERT_PRE_ACT, EXPERT_POST_ACT, ROUTING_LOGITS, ROUTING_PROBS, EXPERT_LOAD.

### 5.4 No LLM-Ergonomic Interface

**transformer-debugger:** Built for a React web UI. The API returns JSON blobs designed for frontend rendering. An LLM trying to use this API would need to parse nested response objects with no schema guarantees, no structured content, no tool annotations.

**rocket_surgeon:** Dual-interface from the start. Every tool has `inputSchema` + `outputSchema` (so LLMs know exactly what to send and what they'll get back). Every result has `content` (human-readable) + `structuredContent` (machine-parseable). Tool annotations tell the LLM which operations are safe to call autonomously (read-only inspections) vs. which need human approval (destructive modifications). The protocol is MCP-compatible, so any MCP client can drive the debugger.

### 5.5 No Protocol Standard

**transformer-debugger:** Bespoke REST API with Pydantic models. No version negotiation, no capability discovery, no standard error codes, no batch requests, no subscriptions.

**rocket_surgeon:** JSON-RPC 2.0 with DAP-style lifecycle and MCP-style primitives. Capability negotiation at `initialize`. Standard JSON-RPC error codes plus domain-specific error data. Batch requests for efficient multi-query inspection. Resource subscriptions for live-updating views. Version negotiation for forward compatibility.

### 5.6 No Intervention Mechanism

**transformer-debugger:** Read-only. You can observe activations, compute derived scalars, and run ablation studies (by specifying ablation_specs in the request), but ablation is specified up-front before the forward pass, not applied interactively mid-pass.

**rocket_surgeon:** Full surgical intervention between ticks. `set_activation` modifies a tensor value at the current tick. `set_weight` modifies a model parameter. `set_routing` overrides MoE expert selection. `inject_noise` adds perturbation. After modification, `tick_forward` continues the forward pass with the modified state. The modification is recorded in the tick frame for replay and "what-if" comparison.

### 5.7 No Checkpoint/Replay

**transformer-debugger:** Every analysis requires a full forward pass from scratch. Want to look at a different derived scalar? Run the whole forward pass again.

**rocket_surgeon:** Checkpoint at any tick. Replay from any checkpoint. Branch from any checkpoint for "what-if" analysis. Compare two branches side-by-side. This transforms debugging from "run, observe, re-run with different settings" to "navigate freely through the forward pass timeline."

### 5.8 Summary Table

| Capability | transformer-debugger | rocket_surgeon |
|---|---|---|
| Stepping/tick model | No (atomic forward pass) | Yes (configurable granularity) |
| Multi-GPU | No | Yes (first-class) |
| MoE support | No | Yes (first-class) |
| LLM interface | No (REST for React UI) | Yes (MCP-compatible, dual content) |
| Protocol standard | No (bespoke REST) | Yes (JSON-RPC 2.0 + DAP lifecycle + MCP primitives) |
| Intervention | Limited (pre-specified ablation only) | Full (modify any state between ticks) |
| Checkpoint/replay | No | Yes (COW snapshots, timeline navigation) |
| Reverse debugging | No | Yes (via checkpoint replay) |
| Activation taxonomy | Excellent (~100+ DerivedScalarTypes) | Steal + extend with MoE types |
| Hook system | Excellent (fwd/bwd/fwd2 at every component) | Steal pattern, implement in Rust |
