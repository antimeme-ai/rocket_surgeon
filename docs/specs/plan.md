# rocket_surgeon: Implementation Plan

Derived from `design.md`. This document is the bridge between design and execution. Each phase
breaks into numbered work units. Each work unit is TCK-able — the first step of executing it is
writing Gherkin specs, then red tests, then implementation, then review.

Phases 0–2 are planned in detail (they constitute the MVP). Phases 3–7 are planned at task
level; they get detailed plans when execution reaches them.

**How to read this**: Each work unit has a `[depends: ...]` tag listing prerequisite work units.
Work units with no dependencies within a phase can be parallelized. Acceptance criteria define
"done" for each unit. File deliverables name the concrete artifacts produced.

**Scope clarifications** (things the design doc describes that are explicitly deferred past MVP):
- **Protobuf `.proto` definitions**: Design doc §18 lists these as Phase 0 deliverables. Deferred
  to Phase 4 (TUI) when gRPC transport becomes relevant. JSON-Schema is the single source of truth
  for the MVP protocol.
- **Tier 2 Python callback interventions**: Design doc §4 describes unsafe Python callbacks.
  Deferred to Phase 3 (after checkpoint/replay).
- **Multi-session support**: Design doc §3 says "multiple sessions (one per model)." MVP is
  single-session. Multi-session ships in Phase 4 when TUI requires concurrent-client support.
- **Head-level tick granularity**: Requires unfused execution for intervention. MVP exposes head
  data via tensor slicing in inspect (no per-head stepping). Head-level stepping ships in Phase 7
  alongside FlashAttention shadow replay.
- **Qwen2 and Gemma2 adapters**: Design doc lists these in initial support. Deferred to Phase 5
  (multi-GPU) when the model conformance test suite is mature. Adapters are straightforward once
  the Llama adapter validates the pattern.

---

## Critical Path Analysis

The longest sequential dependency chain through Phases 0–2:

```
0.1 ──► 0.2 ──► 0.5 ──► 0.6 ──► 0.7 ──► [Phase 0 done]
                                                │
        1.1 ──► 1.5 ──► 1.6 ──► 1.7 ──► 1.10 ──► 1.11 ──► 1.12 ──► 1.14
                                                                       │
                                                            [Phase 1 done]
                                                                       │
                                                    2.2 ──► 2.3 ──► 2.7
                                                                     │
                                                          [Phase 2 / MVP done]
```

**15 units on the critical path.** Parallelism opportunities that shorten wall-clock time:

| While critical path is at... | These can run in parallel |
|------------------------------|--------------------------|
| 0.1–0.2 | 0.3, 0.8 (no dependencies) |
| 0.5–0.6 | 0.4 (depends only on 0.2) |
| 0.7 | 1.2, 1.3, 1.4, 1.9 (independent Rust crates, need only 0.2) |
| 1.1–1.5 | 1.2, 1.3, 1.4, 1.9 continue in parallel |
| 1.5–1.6–1.7 | 1.8 (needs 1.4 + 1.5, not 1.6) |
| 1.10–1.12 | 1.8 completes if not already done |
| 1.12–1.14 | 1.13, 1.15 are independent of 1.14 |
| 2.2–2.3 | 2.1 can start during Phase 1 (standalone Python); 2.4, 2.5 independent |
| 2.7 | 2.4, 2.5, 2.6 are all independent of 2.7 |

---

## Phase 0 — Protocol Spec + Golden TCK

**Goal**: Freeze the JSON-RPC schema at v0.1.0 and write behavioral specs (Gherkin) for all
protocol verbs. Build the TCK test harness. Red tests fail because no daemon exists. This is the
highest-leverage work — every downstream phase builds on these specs.

**Duration**: Weeks 1–2.
**Prerequisites**: Design doc approved (done).

### 0.1 — Probe-point grammar

Define the five-level hierarchical namespace `model:rank:layer:component:event` as a formal
grammar. Wildcards, escaping, validation rules.

- **Files**:
  - `protocol/probe-grammar.ebnf` — formal grammar
  - `crates/rocket-surgeon-probes/src/grammar.rs` — Rust parser (nom or winnow)
  - `python/rocket_surgeon/probes/grammar.py` — Python parser (mirrors Rust)
- **Acceptance criteria**:
  - Parser accepts all examples from design doc §8
  - Parser rejects malformed points with actionable error messages
  - Wildcard expansion produces correct match sets
  - Round-trip: `parse(format(point)) == point`
- **TCK targets**: probe point parsing, wildcard matching, invalid input rejection
- `[depends: none]`

### 0.2 — JSON-RPC message schema

Define the wire format for all 10 verbs, all events, the session state envelope, and the error
contract as JSON-Schema documents. All 10 verb schemas are defined even though MVP implements only
8 — `checkpoint` and `replay` schemas exist as complete type definitions so the schema is frozen
holistically. Protobuf `.proto` definitions deferred to Phase 4 (see scope clarifications above).

- **Files**:
  - `protocol/schema/v0.1.0/` directory:
    - `initialize.json` — request + response
    - `attach.json` — request + response (includes model config, capabilities)
    - `detach.json` — request + response
    - `step.json` — request + response (direction, count, granularity)
    - `inspect.json` — request + response (target, detail level, tensor summary)
    - `intervene.json` — request + response (recipe, composition mode)
    - `probe.json` — request + response (define, list, enable, disable, remove, set_granularity)
    - `checkpoint.json` — request + response (create, restore, list, delete, bookmark)
    - `replay.json` — request + response (from checkpoint, with interventions)
    - `status.json` — request + response (full state dump)
    - `subscribe.json` — request + response (event filter)
    - `events.json` — tick.stopped, tick.heartbeat, probe.fired, replay.divergence, error
    - `common.json` — shared types: SessionState, TickPosition, TensorSummary, TensorHandle,
      InterventionRecipe, ProbeDefinition, Capabilities, ErrorData
  - `protocol/README.md` — protocol overview for implementers
- **Acceptance criteria**:
  - All 10 verb schemas validate against JSON-Schema Draft 2020-12
  - Every request has a corresponding response schema
  - Every response includes the SessionState envelope with `available_actions` populated
    correctly based on the state machine (design doc §3)
  - Error responses include `error_code` (from the error code registry, 0.5), `valid_states`,
    `suggestion`
  - Capability negotiation schema covers all phase-gated features
  - Schema versioned: `protocol_version: "0.1.0"` in initialize response
  - `checkpoint.json` and `replay.json` are complete type definitions (not stubs) even though
    implementation is Phase 3
- **TCK targets**: message validation, envelope presence, error contract shape,
  `available_actions` correctness per state
- `[depends: 0.1]` (probe grammar needed for target fields in inspect/intervene/probe)

### 0.3 — Canonical component vocabulary

Define the per-architecture adapter mappings from HuggingFace module paths to canonical names.
This is a JSON mapping, independent of probe-point grammar syntax.

- **Files**:
  - `protocol/schema/v0.1.0/components.json` — canonical vocabulary definition
  - `protocol/adapters/llama.json` — Llama family mapping (Llama-2/3, Mistral, CodeLlama)
  - `protocol/adapters/mixtral.json` — Mixtral mapping (MoE schema from day one)
  - `protocol/adapters/gpt-neox.json` — GPT-NeoX mapping
- **Acceptance criteria**:
  - Every canonical name from design doc §4 table is present (18 dense + 8 MoE)
  - MoE components (router, experts, shared_expert) included even though MoE ships Phase 6
  - Llama mapping covers Llama-2, Llama-3, Mistral, CodeLlama (they share structure)
  - Mapping format supports parameterized paths (`layers.{i}.self_attn` → `layers[{i}].attn`)
  - Unknown modules map to a fallback `_raw.<module_path>` canonical name
- **TCK targets**: mapping resolution, parameterized path expansion, unknown module fallback
- `[depends: none]`

### 0.4 — Capability negotiation spec

Define the capability object exchanged at `initialize`. Must accommodate all phases — features
gated by capability flags so clients adapt gracefully.

- **Files**:
  - Part of `protocol/schema/v0.1.0/common.json` (Capabilities type)
  - `protocol/capabilities.md` — narrative doc explaining each capability flag
- **Acceptance criteria**:
  - Every phase-gated feature has a corresponding capability flag
  - Capability flags are boolean or enum (not free-form strings)
  - Client can determine available tick granularities, intervention types, built-in views,
    execution mode, parallelism mode from the capabilities object alone
  - `head_granularity` field explicitly states `"requires_unfused"` (design doc §7 honesty)
  - Unknown capabilities are ignored (forward-compatible)
- **TCK targets**: capability-gated verb rejection, unknown capability handling
- `[depends: 0.2]`

### 0.5 — Error code registry

Define the complete enumeration of structured error codes used across all protocol verbs.
Prevents ad-hoc error code invention during implementation.

- **Files**:
  - `protocol/schema/v0.1.0/errors.json` — error code enum with descriptions
  - Part of `protocol/schema/v0.1.0/common.json` (ErrorData type references the registry)
- **Error codes** (minimum set):
  - `INVALID_STATE` — verb not valid in current state (include `valid_states`)
  - `INVALID_TARGET` — probe point doesn't match any component
  - `INVALID_RECIPE` — intervention recipe malformed or unsupported
  - `MODEL_NOT_ATTACHED` — operation requires attached model
  - `TENSOR_NOT_FOUND` — tensor_id doesn't exist in store
  - `CHECKPOINT_NOT_FOUND` — checkpoint name/id doesn't exist
  - `PROBE_NOT_FOUND` — probe_id doesn't exist
  - `CAPABILITY_NOT_SUPPORTED` — verb requires a capability the session lacks
  - `SLICE_OUT_OF_BOUNDS` — tensor slice indices exceed shape
  - `RESPONSE_TOO_LARGE` — requested data exceeds 64 KB cap
  - `HOST_ERROR` — Python host process error (include traceback summary)
  - `GPU_OOM` — GPU out of memory
  - `NCCL_TIMEOUT` — NCCL collective timed out
  - `REPLAY_DIVERGENCE` — replay exceeded tolerance (informational, not fatal)
  - `UNSUPPORTED_MODEL` — model architecture not in support matrix
  - `COMPILED_MODEL` — model uses torch.compile, Tier A cannot attach
- **Acceptance criteria**:
  - Every error code has a unique string identifier, numeric code, and description
  - Every error includes `suggestion` field with recovery guidance
  - Error codes are referenced by all verb schemas' error response definitions
  - No verb implementation may invent error codes not in this registry
- **TCK targets**: error contract scenarios use registry codes
- `[depends: 0.2]`

### 0.6 — Behavioral specs (Gherkin TCK)

Write Gherkin `.feature` files covering the protocol's behavioral contract. Split by verb group
so work can begin as soon as each verb's schema is drafted. Scenarios must cover every state
transition in the design doc §3 state machine, not just a per-verb count.

- **Files** (in `tck/`):
  - `tck/protocol/lifecycle.feature` — initialize → attach → detach → reinitialize cycle.
    Must cover: double-attach error, detach-without-attach error, reinitialize after detach.
  - `tck/protocol/stepping.feature` — step forward with count=1, count=N, granularity=layer,
    granularity=component. Step at end of forward pass (wraps or stops). Step while not stopped.
  - `tck/protocol/inspection.feature` — inspect at current tick (summary), inspect with
    detail=slice (bounded), inspect nonexistent target, inspect with built-in view names,
    content-addressable tensor_id dedup (same tensor at two probes → same id).
  - `tck/protocol/intervention.feature` — set ablate/scale/add, clear intervention, list
    interventions, composition (two interventions same point, priority ordering, replace mode),
    intervention persists across ticks, intervention on invalid target.
  - `tck/protocol/probes.feature` — define with point + action, list (with filter), enable,
    disable, remove, wildcard matching, priority ordering, probe at same point as intervention.
  - `tck/protocol/checkpoint.feature` — create, restore, list, delete, bookmark (Phase 3 but
    schema is frozen now — scenarios marked `@phase3`).
  - `tck/protocol/replay.feature` — replay from checkpoint, divergence event (marked `@phase3`).
  - `tck/protocol/subscribe.feature` — subscribe to events, filter by layer, multiple
    subscribers, tick.heartbeat during pause.
  - `tck/protocol/errors.feature` — every error code from 0.5 has at least one trigger scenario.
    Invalid state transitions from design doc §3 state machine (step while attaching,
    inspect while stepping, intervene while inspecting).
  - `tck/protocol/state-envelope.feature` — every response includes SessionState,
    `available_actions` matches state machine, tick_id monotonically increases, tick_id unique.
  - `tck/model/adapter.feature` — canonical name resolution for Llama, unknown module fallback.
  - `tck/model/hooks.feature` — tier detection (refuse compiled model), hook registration
    order (prepend=True).
  - `tck/tensor/handles.feature` — BLAKE3 content-addressable id, summary stats accuracy
    (verified against direct computation), slice bounds checking.
  - `tck/moe/tick-granularity.feature` — four MoE tick levels (marked `@phase6`).
  - `tck/session/bundle.feature` — bundle contents validation against design doc §15 manifest.
- **Acceptance criteria**:
  - Every state transition in design doc §3 state machine has at least one scenario
  - Every error code in registry (0.5) has at least one trigger scenario
  - Every response-envelope field from design doc §6 is asserted in at least one scenario
  - `@phase3`, `@phase6` tags mark future-phase scenarios (run but expected to fail/pend)
  - Total scenario count is secondary to transition coverage
- **TCK targets**: this IS the TCK
- `[depends: 0.1, 0.2, 0.3, 0.4, 0.5]`

### 0.7 — TCK test harness

Build the infrastructure that executes Gherkin scenarios against the real daemon+host stack.
This is the runner, fixtures, and step-definition stubs — not the step implementations themselves
(those come in Phase 1 as each feature is built).

- **Files**:
  - `python/tests/tck/__init__.py`
  - `python/tests/tck/conftest.py` — pytest-bdd fixtures: daemon process lifecycle, Unix socket
    connection, JSON-RPC client helper, model fixture (GPT-2-small for CI speed)
  - `python/tests/tck/steps/common.py` — shared step definitions (given/when/then for
    initialize, state assertions, error assertions)
  - `python/tests/tck/steps/` — one step file per feature file (stubs, `@pytest.mark.xfail`)
  - `pyproject.toml` — add `pytest-bdd` to dev dependencies
  - `xtask/src/main.rs` — add `tck` subcommand that runs `pytest python/tests/tck/ -v`
- **Acceptance criteria**:
  - `cargo xtask tck` runs all Gherkin scenarios
  - All scenarios report as `xfail` (expected failure) or `pending` — not as import errors or
    missing step definition errors
  - Fixture starts a real daemon process, connects over Unix socket, and can send/receive
    JSON-RPC messages
  - Fixture tears down daemon process cleanly after each scenario
  - Step definitions for `Given the daemon is initialized` and `Then the response contains
    SessionState` exist (even if the daemon doesn't yet)
- **TCK targets**: this IS the harness
- `[depends: 0.6]`

### 0.8 — ADRs for new architectural decisions

Formalize decisions made in the design doc that supersede or extend existing ADRs.

- **Files**:
  - `docs/adr/ADR-0004-three-process-architecture.md` — why three processes, not two
  - `docs/adr/ADR-0005-tick-model.md` — tick granularities, CUDA events (not cudaDeviceSynchronize),
    MoE four-level ticks, tick scoping
  - `docs/adr/ADR-0006-tensor-handling.md` — content-addressable IDs (BLAKE3),
    summary-then-slice protocol, shared-memory ring buffer
- **Acceptance criteria**:
  - Each ADR follows the existing format (Context, Decision, Consequences)
  - Each references the specific design doc section it formalizes
  - Supersession of architecture.md is noted in ADR-0004
- `[depends: none]`

### Phase 0 exit criteria

- [ ] All 10 verb JSON-Schemas validate against Draft 2020-12 (automated check)
- [ ] Probe-point grammar parser exists in Rust and Python, passes unit tests
- [ ] Error code registry defines all codes; every verb schema references it
- [ ] TCK runner (`cargo xtask tck`) executes all scenarios and reports them as xfail/pending
      (no import errors, no missing step definitions)
- [ ] Every state-machine transition from design doc §3 has a Gherkin scenario
- [ ] Canonical vocabulary covers Llama, Mixtral (MoE), GPT-NeoX
- [ ] Three new ADRs committed
- [ ] Schema frozen: `protocol_version: "0.1.0"`

---

## Phase 1 — Single-GPU Eager Daemon

**Goal**: Rust daemon (Process A) + Python model host (Process B) serving the protocol against
Llama-3-8B on a single GPU. Component-level tick stepping. Tensor inspection with
summary-then-slice. No interventions yet.

**Duration**: Weeks 3–6.
**Prerequisites**: Phase 0 complete.

### 1.1 — Rust daemon skeleton

The protocol server: accept connections, parse JSON-RPC, dispatch to state machine, serialize
responses. Single-session for MVP.

- **Files**:
  - `crates/rocket-surgeon/src/main.rs` — daemon entry point, connection accept loop
  - `crates/rocket-surgeon/src/server.rs` — JSON-RPC server (stdio + Unix socket transport)
  - `crates/rocket-surgeon/src/session.rs` — session state machine (single session)
  - `crates/rocket-surgeon/src/dispatch.rs` — verb dispatch (method string → handler)
  - `crates/rocket-surgeon/src/trace_log.rs` — protocol trace logger: append every JSON-RPC
    request, response, and event to an in-memory JSONL buffer (flushed to session bundle by 2.3)
- **Acceptance criteria**:
  - Daemon starts, listens on stdio and Unix socket
  - Every JSON-RPC message (request, response, event) is logged to the protocol trace buffer
  - `initialize` request returns capabilities
  - Invalid method returns JSON-RPC method-not-found error (-32601)
  - Out-of-state requests return `INVALID_STATE` error with `valid_states` and `suggestion`
  - Every response includes SessionState envelope with correctly populated `available_actions`
  - State machine enforces all transitions from design doc §3
  - Single-session: second `attach` while attached returns `INVALID_STATE`
- **TCK targets**: `lifecycle.feature`, `errors.feature`, `state-envelope.feature`
- `[depends: 0.2, 0.5, 0.7]`

### 1.2 — Protocol types crate

Rust types for all protocol messages. Serializable to/from JSON via serde. Validated against
the JSON-Schema from Phase 0.

- **Files**:
  - `crates/rocket-surgeon-protocol/src/lib.rs` — re-exports
  - `crates/rocket-surgeon-protocol/src/types.rs` — SessionState, TickPosition, Capabilities,
    TensorSummary, TensorHandle, ProbeDefinition, InterventionRecipe, ErrorData
  - `crates/rocket-surgeon-protocol/src/messages.rs` — request/response/event enums for all verbs
  - `crates/rocket-surgeon-protocol/src/jsonrpc.rs` — JSON-RPC 2.0 framing (id, method, params)
  - `crates/rocket-surgeon-protocol/src/errors.rs` — error code enum mirroring registry from 0.5
- **Acceptance criteria**:
  - All types from `protocol/schema/v0.1.0/common.json` have Rust equivalents
  - `serde_json::to_value(msg)` validates against the JSON-Schema (property test)
  - Round-trip: `deserialize(serialize(msg)) == msg` for all message types
  - Error codes enum matches registry from 0.5 exactly (compile-time guarantee)
- **TCK targets**: message validation tests (Rust-side)
- `[depends: 0.2, 0.5]`

### 1.3 — Probe registry (Rust)

In-daemon probe storage: define, list, enable, disable, remove. Wildcard matching against probe
points.

- **Files**:
  - `crates/rocket-surgeon-probes/src/lib.rs` — re-exports
  - `crates/rocket-surgeon-probes/src/grammar.rs` — probe-point parser (from 0.1)
  - `crates/rocket-surgeon-probes/src/registry.rs` — ProbeRegistry: CRUD + wildcard match
  - `crates/rocket-surgeon-probes/src/matcher.rs` — wildcard expansion, point matching
- **Acceptance criteria**:
  - Define probe with point + action + config → returns ProbeId
  - List probes with optional filter (by point pattern, by enabled state)
  - Enable/disable toggles firing without removing definition
  - Remove deletes definition
  - Wildcard `*` matches any value at that level
  - Multiple probes at same point: ordered by priority
  - Probe with invalid point (fails grammar parse) returns `INVALID_TARGET` error
- **TCK targets**: `probes.feature`
- `[depends: 0.1]`

### 1.4 — Tensor handle store (Rust)

Content-addressable tensor storage in the daemon. Receives tensor bytes from shared memory,
computes BLAKE3 hash, stores metadata + summary, serves slices.

- **Files**:
  - `crates/rocket-surgeon/src/tensors.rs` — TensorStore: ingest, lookup by id, slice, evict
  - `crates/rocket-surgeon/src/shm.rs` — shared-memory reader (mmap ring buffer, see 1.8)
- **Acceptance criteria**:
  - Ingest tensor bytes + precomputed summary → compute BLAKE3 → return tensor_id
  - Same bytes at different probe points → same tensor_id (dedup verified)
  - Lookup by tensor_id returns TensorSummary
  - Slice request with index ranges returns bounded byte payload (≤64 KB default cap)
  - Slice with out-of-bounds indices returns `SLICE_OUT_OF_BOUNDS` error
  - Request exceeding cap returns `RESPONSE_TOO_LARGE` error with actual size
  - Eviction policy: LRU with configurable max entries
- **TCK targets**: `handles.feature`
- `[depends: 0.2, 1.2]`

### 1.5 — Python model host skeleton

The Python process that loads the model and runs the forward pass. Communicates with daemon over
Unix socket (JSON-RPC).

- **Files**:
  - `python/rocket_surgeon/host/__init__.py`
  - `python/rocket_surgeon/host/main.py` — entry point, asyncio event loop, JSON-RPC client
  - `python/rocket_surgeon/host/model_loader.py` — load HF model, detect architecture
  - `python/rocket_surgeon/host/rpc.py` — JSON-RPC message handling (receive commands from daemon)
- **Acceptance criteria**:
  - Daemon spawns host as a child process
  - Host connects to daemon Unix socket on startup
  - Receives `_host/attach` command → loads model → reports model_info + capabilities
  - Receives `_host/detach` command → unloads model → reports success
  - Architecture detection correctly identifies Llama family models
  - Refuses compiled models with `COMPILED_MODEL` error (Tier A only for MVP)
  - Reports GPU memory usage after load
  - Host crash is detected by daemon (process exit monitoring), daemon transitions to error state
- **Note**: Internal daemon↔host protocol uses the same JSON-RPC schema as external clients
  (design doc §2). Verbs prefixed with `_host/` for internal dispatch but types are identical.
- **TCK targets**: `lifecycle.feature` (attach/detach scenarios)
- `[depends: 1.1]`

### 1.6 — Model adapter (Python)

Walks the HF model's module tree and builds the canonical name mapping.

- **Files**:
  - `python/rocket_surgeon/host/adapter/__init__.py`
  - `python/rocket_surgeon/host/adapter/base.py` — BaseAdapter: walk module tree, resolve names
  - `python/rocket_surgeon/host/adapter/llama.py` — LlamaAdapter: Llama/Mistral/CodeLlama mapping
  - `python/rocket_surgeon/host/adapter/registry.py` — adapter selection by model type
- **Acceptance criteria**:
  - Given a loaded Llama model, produces a complete mapping of hookable module paths to canonical names
  - Canonical names match the vocabulary from design doc §4 (all 18 dense components)
  - Unknown modules get fallback name `_raw.<original.module.path>`
  - Adapter reports the canonical component list to daemon (for probe-point validation)
  - GPT-2-small works for CI (no large GPU required) — adapter detects GPT2 and maps to
    best-effort canonical names
- **TCK targets**: `adapter.feature`
- `[depends: 1.5, 0.3]`

### 1.7 — Hook manager (Python)

Registers PyTorch forward hooks on mapped modules. Tier A only (eager mode).

- **Files**:
  - `python/rocket_surgeon/host/hooks/__init__.py`
  - `python/rocket_surgeon/host/hooks/manager.py` — HookManager: register, remove, tier detection
  - `python/rocket_surgeon/host/hooks/barrier.py` — BarrierGate: threading.Event-based pause
  - `python/rocket_surgeon/host/hooks/capture.py` — tensor capture callback: snapshot → summary → shm
- **Acceptance criteria**:
  - Hooks registered on all adapter-mapped modules after model load
  - `prepend=True` on all hooks (run before user hooks)
  - Sentinel no-op hook on every module (disables PyTorch fast path)
  - Tier detection: refuse to attach if `OptimizedModule` detected (returns `COMPILED_MODEL`)
  - Hook callback checks active probe set; fast-exit if nothing matches
  - BarrierGate pauses forward pass at tick boundaries
  - Capture callback computes summary stats on GPU (mean, std, min, max, abs_max, l2_norm,
    sparsity, 32-bin histogram, top-8-by-abs), then transfers to CPU
  - Overhead with zero active probes: forward pass within 5% of baseline (measured by 1.16)
- **TCK targets**: `hooks.feature`, `stepping.feature`
- `[depends: 1.5, 1.6]`

### 1.8 — Shared-memory data plane

The ring buffer for tensor handoff between Python host and Rust daemon.

- **Files**:
  - `python/rocket_surgeon/host/shm.py` — Python writer: allocate shared-memory region, write
    ProbeFrames
  - `crates/rocket-surgeon/src/shm.rs` — Rust reader
  - `python/rocket_surgeon/host/hooks/capture.py` — extended: write to shm after GPU→CPU transfer
- **Platform abstraction**: Use Python `multiprocessing.shared_memory` (cross-platform) for the
  shared region. For notification, use a Unix domain socket auxiliary channel (single byte write
  per frame) instead of `eventfd` (Linux-only). macOS and Linux both support Unix domain sockets.
- **Acceptance criteria**:
  - Python allocates shared-memory region at host startup
  - ProbeFrame header format matches design doc §2 (128-byte fixed header + raw bytes)
  - Python writes tensor + header; Rust reads zero-copy via mmap
  - Ring buffer wraps correctly (slot reuse after consumer acknowledges)
  - Notification via Unix domain socket (single byte per frame) works on both Linux and macOS
  - Integration test: tensor captured in Python, readable with correct values in Rust
- **TCK targets**: integration test (not Gherkin — this is internal infrastructure)
- `[depends: 1.4, 1.5]`

### 1.9 — PyO3 thin bridge

Thin Rust-in-Python bridge for hot-path operations that benefit from avoiding JSON-RPC round-trips:
BLAKE3 hashing, ProbeFrame header serialization, summary stat aggregation.

- **Files**:
  - `crates/rocket-surgeon-python/src/lib.rs` — extended: expose hash, header, stats functions
  - `python/rocket_surgeon/host/_bridge.py` — Python wrapper for PyO3 functions with fallback
    to pure-Python if native module not built
- **Acceptance criteria**:
  - `rs.blake3_hash(bytes)` returns same hash as Rust-side computation (consistency test)
  - `rs.serialize_probe_frame_header(...)` produces bytes matching the ProbeFrame spec
  - PyO3 functions release the GIL during computation (`Python::allow_threads`)
  - Pure-Python fallback exists for development without Rust build (slower but functional)
- **TCK targets**: unit tests (not Gherkin)
- `[depends: 0.2]` (needs ProbeFrame spec from schema)

### 1.10 — Step integration

Wire stepping: client sends `rocket/step`, daemon tells host to advance ticks, host runs forward
until next barrier, daemon receives stop notification.

- **Files**:
  - `crates/rocket-surgeon/src/handlers/step.rs` — step handler
  - `crates/rocket-surgeon/src/handlers/status.rs` — status handler
  - `python/rocket_surgeon/host/rpc.py` — extended: handle step commands, drive forward pass
- **Acceptance criteria**:
  - `step` with `count=1, granularity=component` advances exactly one component tick
  - `step` with `count=5` advances five ticks (tick_id increments by 5)
  - `step` with `granularity=layer` steps layer-by-layer (skips inter-component boundaries)
  - Step at end of forward pass: daemon transitions to a terminal state, subsequent step errors
  - Step while not in STOPPED state: returns `INVALID_STATE`
  - `status` returns full SessionState with correct position
  - `tick_id` is monotonically increasing and unique across steps
  - TickPosition (layer, component, event) accurately reflects where the forward pass is paused
- **TCK targets**: `stepping.feature`, `state-envelope.feature`
- `[depends: 1.1, 1.2, 1.5, 1.7]`

### 1.11 — Inspect integration

Wire inspection: client sends `rocket/inspect`, daemon resolves target to a tensor handle,
returns summary or slice.

- **Files**:
  - `crates/rocket-surgeon/src/handlers/inspect.rs` — inspect handler
- **Acceptance criteria**:
  - `inspect` with target matching current tick's component → returns TensorSummary
  - `inspect` with `detail=slice` and valid indices → returns bounded bytes (≤64 KB)
  - `inspect` with nonexistent target → returns `INVALID_TARGET` with suggestion
  - `inspect` with target at a different layer (not current tick) → returns `INVALID_TARGET`
    (must step there first)
  - Content-addressable dedup: same tensor at two probe points returns same tensor_id
  - Summary stats (mean, std, min, max) verified against direct `torch` computation (±1e-5 fp32)
- **TCK targets**: `inspection.feature`, `handles.feature`
- `[depends: 1.4, 1.8, 1.10]`

### 1.12 — Probe event integration

Wire probes to the event system: defined probes fire when their point matches the current tick,
generating `probe.fired` events with tensor summaries.

- **Files**:
  - `crates/rocket-surgeon/src/handlers/probe.rs` — probe handler (CRUD verbs)
  - `crates/rocket-surgeon/src/events.rs` — event generation from probe firings
- **Acceptance criteria**:
  - `probe define` with point matching a component → probe registered
  - On step, if tick matches probe point → `probe.fired` event emitted with TensorSummary
  - Probes with wildcards fire on all matching components
  - Disabled probe does not fire
  - Multiple probes at same point: all fire, in priority order
  - Probe on non-matching component: no event (silent, not error)
- **TCK targets**: `probes.feature` (firing scenarios), `subscribe.feature` (event delivery)
- `[depends: 1.3, 1.10, 1.11]`

### 1.13 — Built-in views: residual_stream_norm + attention_pattern

Two built-in interpretability views as named protocol primitives.

- **Files**:
  - `python/rocket_surgeon/host/views/__init__.py`
  - `python/rocket_surgeon/host/views/residual_norm.py` — L2 norm of residual stream per layer
  - `python/rocket_surgeon/host/views/attention.py` — attention weights (eager SDPA)
  - `crates/rocket-surgeon/src/handlers/inspect.rs` — extended: built-in view dispatch
- **Acceptance criteria**:
  - `inspect` with `view=residual_stream_norm` returns `[f32; num_layers]` L2 norms
  - `inspect` with `view=attention_pattern` and `layer=N` returns per-head attention weights
  - Views registered in capabilities as `built_in_views: ["residual_stream_norm", "attention_pattern"]`
  - Both work against Llama-3-8B with `attn_implementation="eager"` (and GPT-2 for CI)
  - Attention view explicitly requires eager SDPA; returns `CAPABILITY_NOT_SUPPORTED` if model
    uses FlashAttention (MVP does not support FlashAttention inspection)
- **TCK targets**: `inspection.feature` (built-in view scenarios)
- `[depends: 1.11]`

### 1.14 — Subscribe + event delivery

Client subscribes to events. Daemon pushes notifications.

- **Files**:
  - `crates/rocket-surgeon/src/handlers/subscribe.rs` — subscribe handler
  - `crates/rocket-surgeon/src/events.rs` — extended: subscription registry, event routing
- **Acceptance criteria**:
  - `subscribe` with event filter → client registered for matching events
  - `tick.stopped` event delivered when forward pass pauses at barrier
  - `probe.fired` event delivered when probe captures data
  - `tick.heartbeat` sent every 1s while stopped (verify timing ±200ms)
  - Events respect subscription filter (e.g., filter by layer range)
  - Multiple clients receive independent event streams
  - Unsubscribe removes subscription
- **TCK targets**: `subscribe.feature`
- `[depends: 1.12]`

### 1.15 — Perfetto trace sink

Write probe events, tick boundaries, and session metadata to a Perfetto protobuf trace file.

- **Files**:
  - `crates/rocket-surgeon/src/trace.rs` — TraceSink: buffer events, write Perfetto protobuf
  - `crates/rocket-surgeon/Cargo.toml` — add `prost` for protobuf serialization
  - Perfetto trace proto definitions (vendored or generated)
- **Acceptance criteria**:
  - Every tick boundary emits a Perfetto track event (process track per rank, thread track per
    component)
  - Every probe firing emits an instant event on the relevant track
  - Trace file opens in the Perfetto UI (https://ui.perfetto.dev) and shows a coherent timeline
  - Session start/end emits process-level metadata
  - Trace file is flushed on session export and on detach
- **TCK targets**: not Gherkin — validated by Perfetto UI manual verification + automated
  check that the file is valid protobuf
- `[depends: 1.12]`

### 1.16 — End-to-end smoke test + overhead benchmark

Scripted test exercising the full Phase 1 stack, plus performance gating.

- **Files**:
  - `python/tests/test_e2e_phase1.py` — end-to-end test
  - `python/tests/benchmarks/test_overhead.py` — overhead measurement
- **Acceptance criteria**:
  - E2E: start daemon → attach (GPT-2-small for CI, Llama-3-8B for nightly) → step through
    full forward pass → inspect residual norms at every layer → verify summaries against direct
    PyTorch computation → detach → shutdown
  - All Phase 0/1 TCK scenarios that target implemented features pass green (xfail removed)
  - Zero-probe overhead: forward pass within 5% of baseline (100 iterations, report mean ± std)
  - Active-probe overhead (capture all residual streams): within 15% of baseline
  - Results logged to stdout in a parseable format for tracking across commits
  - Test runs on macOS (CI) and Linux (nightly GPU) without platform-specific code paths
- **TCK targets**: all Phase 1 TCK scenarios green
- `[depends: 1.10, 1.11, 1.12, 1.13, 1.14, 1.15]`

### Phase 1 exit criteria

- [ ] Daemon starts and serves protocol on stdio + Unix socket
- [ ] Llama-3-8B loads and attaches in <30s (GPT-2-small in CI)
- [ ] Component-level and layer-level stepping works
- [ ] Tensor inspection returns accurate summaries (verified ±1e-5 against PyTorch)
- [ ] Built-in views work (residual_stream_norm, attention_pattern)
- [ ] Probe lifecycle (define, enable, disable, remove, wildcard) works
- [ ] Event subscription and delivery works (tick.stopped, probe.fired, tick.heartbeat)
- [ ] Perfetto trace opens in Perfetto UI
- [ ] Internal daemon↔host protocol uses same schema as external protocol
- [ ] All Phase 1 TCK scenarios green
- [ ] Overhead within budget (5% zero-probe, 15% active-probe)
- [ ] No regressions: `cargo xtask ci` passes

---

## Phase 2 — Interventions + MVP Completion

**Goal**: Five intervention types (ablate, scale, add, patch, clamp). Session bundle export.
MVP documentation. IOI acceptance test. This completes the MVP.

**Duration**: Weeks 7–8.
**Prerequisites**: Phase 1 complete.

### 2.1 — Intervention engine (Python)

Apply declarative intervention recipes to tensors at barrier points.

**Parallelism note**: This unit has no dependency on Phase 1 — it is standalone Python code with
unit tests against mock tensors. Can be pulled forward if a team member finishes Phase 1 work early.

- **Files**:
  - `python/rocket_surgeon/host/interventions/__init__.py`
  - `python/rocket_surgeon/host/interventions/engine.py` — InterventionEngine: apply recipes
  - `python/rocket_surgeon/host/interventions/recipes.py` — recipe types: Ablate, Scale, Add,
    Patch, Clamp
  - `python/rocket_surgeon/host/interventions/composition.py` — priority ordering, additive/replace
- **Acceptance criteria**:
  - `ablate` zeros the target tensor
  - `scale` multiplies by the given factor
  - `add` adds the given vector (from literal values or tensor_id reference)
  - `patch` replaces the tensor with a previously-captured tensor (by tensor_id)
  - `clamp` clamps all values to `[min, max]` range
  - Multiple interventions at same point compose in priority order (lower = first)
  - Additive composition: `add` + `add` = sum of both vectors
  - `mode: "replace"` overrides all prior interventions at that point
  - All recipe types are JSON-serializable and deserializable
  - Unit tests against mock tensors (no daemon, no model, no GPU required)
- **TCK targets**: `intervention.feature`
- `[depends: none]`

### 2.2 — Intervene verb integration

Wire intervention engine into the daemon↔host protocol.

- **Files**:
  - `crates/rocket-surgeon/src/handlers/intervene.rs` — intervene handler
  - `python/rocket_surgeon/host/hooks/barrier.py` — extended: apply interventions at barrier
  - `python/rocket_surgeon/host/rpc.py` — extended: handle intervene commands
- **Acceptance criteria**:
  - `rocket/intervene` with recipe → sets intervention on host
  - `rocket/intervene` with `action=clear` and intervention id → removes specific intervention
  - `rocket/intervene` with `action=list` → returns all active interventions
  - Intervention applied on next step (not retroactively), visible in subsequent inspect
  - Clearing an intervention takes effect on next step
  - Recipe with invalid target → `INVALID_TARGET`
  - Recipe with unsupported type → `INVALID_RECIPE`
  - `patch` with nonexistent tensor_id → `TENSOR_NOT_FOUND`
- **TCK targets**: `intervention.feature`
- `[depends: 1.10, 1.11, 2.1]`

### 2.3 — Session bundle export

Self-contained tar.gz artifact for reproducibility.

- **Files**:
  - `crates/rocket-surgeon/src/bundle.rs` — BundleExporter: collect session data, write tar.gz
  - `crates/rocket-surgeon/src/handlers/session.rs` — `rocket/session.export` handler
  - `python/rocket_surgeon/host/rpc.py` — extended: collect model info, env for bundle
- **Bundle contents** (matches design doc §15):
  - `manifest.json` — protocol_version, session_id, timestamps, schema version
  - `model-info.json` — model hash, architecture, config, num_layers, hidden_dim
  - `env.json` — GPU model, driver version, CUDA version, NCCL version, torch version, OS,
    Python version, relevant env vars
  - `protocol-trace.jsonl` — complete JSON-RPC request/response/event log
  - `prompt.json` — input tokens and tokenizer info
  - `tensors/` — captured tensors as safetensors files named by tensor_id
  - `interventions.json` — all intervention recipes that were applied during session
  - `trace.perfetto-trace` — Perfetto timeline
  - `bookmarks.json` — empty for MVP (bookmarks ship in Phase 3)
- **Acceptance criteria**:
  - `rocket/session.export` produces tar.gz at specified path
  - All 9 items above are present in the bundle (automated content check)
  - `manifest.json` includes protocol_version matching schema version
  - `protocol-trace.jsonl` contains all JSON-RPC messages exchanged during session
  - `env.json` captures all fields needed for replay verification
  - Bundle size is reasonable (not accidentally including model weights)
- **TCK targets**: `bundle.feature`
- `[depends: 1.14, 1.15, 2.2]`

### 2.4 — Model conformance test suite

Automated test harness that validates probe firings against expected canonical component order
for each supported model family. Runs nightly against latest `transformers` release.

- **Files**:
  - `python/tests/conformance/__init__.py`
  - `python/tests/conformance/test_llama.py` — Llama family conformance
  - `python/tests/conformance/conftest.py` — shared fixtures (load model, attach, step, collect)
  - `xtask/src/main.rs` — add `conformance` subcommand
- **Acceptance criteria**:
  - For Llama-3-8B (nightly) / GPT-2-small (CI): run a fixed prompt, assert probes fire at all
    canonical components in the expected order (embed → layer 0 ln1 → layer 0 attn → ...)
  - Fires on every canonical component listed in the Llama adapter mapping
  - Fails if a component is missing or out of order
  - Fails if `transformers` update changes the module structure (early warning)
  - Test output reports which components matched and which didn't
- `[depends: 1.12, 1.6]`

### 2.5 — MVP documentation

Protocol spec, attach guide, and IOI reproduction tutorial. Required by the MVP Definition
of Done (design doc §19 item 6).

- **Files**:
  - `docs/tutorial/quickstart.md` — how to install, start daemon, attach to a model
  - `docs/tutorial/ioi.md` — step-by-step IOI reproduction tutorial (the worked example)
  - `protocol/README.md` — extended: complete protocol reference with examples for each verb
- **Acceptance criteria**:
  - A new user can follow `quickstart.md` end-to-end on a machine with a GPU
  - `ioi.md` reproduces the IOI ablation experiment using only protocol commands (copy-pasteable)
  - `protocol/README.md` has a usage example for each of the 8 MVP verbs
  - All code examples in docs are tested (extracted and run in CI, or manually verified)
- `[depends: 2.2]` (needs working interventions for the tutorial)

### 2.6 — MCP adapter (stretch)

Wrap the JSON-RPC service as MCP tools and resources.

- **Files**:
  - `python/rocket_surgeon/mcp/__init__.py`
  - `python/rocket_surgeon/mcp/server.py` — MCP server exposing rocket_surgeon verbs as tools
  - `python/rocket_surgeon/mcp/resources.py` — MCP resources for tensors, session state
- **Acceptance criteria**:
  - MCP-capable LLM can connect and call step/inspect/intervene as MCP tools
  - Tensor handles exposed as MCP resources with URI scheme `rocket://session/{id}/tensor/{tid}`
  - Session state available as MCP resource
- **TCK targets**: `tck/mcp/tools.feature`
- `[depends: 2.2]`
- **Note**: Stretch for MVP. Drop if team is <3 people; ships Phase 3 otherwise.

### 2.7 — IOI reproduction acceptance test

The MVP acceptance test: reproduce Indirect Object Identification ablation via the protocol.

- **Files**:
  - `python/tests/test_ioi_acceptance.py` — end-to-end IOI reproduction
  - `python/tests/fixtures/ioi_prompts.json` — IOI test prompts (from Wang et al. 2023,
    "Interpretability in the Wild")
- **Acceptance criteria**:
  - Script connects to daemon, attaches Llama-3-8B (requires GPU; marked `@nightly`)
  - Steps to attention output at layers containing name-mover heads
  - Inspects attention patterns to identify name-mover heads (heads with high attention on
    indirect object position)
  - Sets `ablate` intervention on identified heads
  - Re-runs forward pass with intervention active
  - Measures **logit difference** (logit[IO] - logit[S], the standard IOI metric from
    Wang et al. 2023 §3.1)
  - Ablation of name-mover heads reduces logit difference by ≥50% relative to clean baseline
    (published result: near-complete elimination; we accept ≥50% to accommodate model differences
    between GPT-2 in the original paper and Llama-3-8B)
  - Entire test driven exclusively through the protocol (no direct PyTorch calls)
  - Test produces a session bundle as a side effect (validates 2.3 as well)
- **TCK targets**: this IS the MVP acceptance test
- `[depends: 2.2, 2.3]`

### Phase 2 exit criteria (= MVP gate)

- [ ] Five interventions work (ablate, scale, add, patch, clamp)
- [ ] Intervention composition (priority, additive, replace) works
- [ ] Session bundle export produces valid artifact with all 9 required contents
- [ ] Session bundle includes Perfetto trace that opens in Perfetto UI
- [ ] IOI reproduction acceptance test passes (logit difference reduced ≥50%)
- [ ] Model conformance test suite passes for Llama (nightly) and GPT-2 (CI)
- [ ] MVP documentation exists: quickstart, IOI tutorial, protocol reference
- [ ] No overhead regression from Phase 1 baseline (interventions add ≤2% on top of active-probe overhead)
- [ ] All Phase 0/1/2 TCK scenarios green
- [ ] Protocol schema frozen at v0.1.0

---

## Phases 3–7 — Task-Level Plans

These phases get detailed plans (with numbered work units) when execution reaches them. Below
are task-level breakdowns sufficient for roadmap planning. Findings from the review that were
deferred to these phases are noted.

### Phase 3 — Checkpoint + Reverse Step (weeks 9–11)

| Task | Description |
|------|-------------|
| 3.1 | Checkpoint engine: √N auto-checkpointing at layer boundaries |
| 3.2 | Checkpoint storage: CPU pinned memory with safetensors NVMe spill |
| 3.3 | `rocket/checkpoint` verb: create, restore, list, delete, bookmark |
| 3.4 | Forward replay from checkpoint with probe re-firing |
| 3.5 | `rocket/replay` verb: replay with different interventions |
| 3.6 | Replay divergence detection and `replay.divergence` event |
| 3.7 | Determinism enforcement: seed capture, op-level pinning |
| 3.8 | Reverse step via checkpoint restore + forward replay |
| 3.9 | Bookmark system: named tick references |
| 3.10 | Session bundle extended: include checkpoints and bookmarks |
| 3.11 | Tier 2 interventions: Python callback with watchdog + OOM guard |
| 3.12 | TCK green for all checkpoint/replay/bookmark scenarios |

**Exit criteria**: Reverse-step works. Replay divergence detected and reported. Bookmarks work.
ROME-style locate-then-edit reproduction via reverse-step + intervention.

### Phase 4 — TUI Dogfood (weeks 12–14)

| Task | Description |
|------|-------------|
| 4.1 | Connection layer: JSON-RPC client with reconnection |
| 4.2 | State layer: in-memory session mirror, ingests protocol responses |
| 4.3 | View layer: ratatui rendering, keyboard input |
| 4.4 | Status bar panel |
| 4.5 | Activation summary panel (sparkline history per component) |
| 4.6 | Tensor inspector panel (heatmap, histogram, top-k, slice viewer) |
| 4.7 | Attention pattern panel (per-head heatmap) |
| 4.8 | Intervention panel (active interventions, recipe editor) |
| 4.9 | Timeline panel (tick history, probe firings) |
| 4.10 | Command bar (protocol command input with autocomplete) |
| 4.11 | Multi-session support in daemon (multiple models, multiple clients) |
| 4.12 | Protobuf `.proto` definitions (for future gRPC transport) |
| 4.13 | Dogfood feedback → protocol refinements |

**Exit criteria**: TUI can drive a full debug session against Llama-3-8B. Team dogfoods for 1 week
before starting Phase 5.

### Phase 5 — Multi-GPU (weeks 15–18)

| Task | Description |
|------|-------------|
| 5.1 | Per-rank Python worker management (daemon spawns N host processes) |
| 5.2 | Rank coordination: all-rank barrier synchronization |
| 5.3 | Pre/post-collective barriers (never inside NCCL collectives) |
| 5.4 | DTensor-aware tensor inspection (sharding info in response) |
| 5.5 | TP-aware capture (rank 0 for replicated tensors) |
| 5.6 | `rocket/tensor.gather` command for explicit all-gather |
| 5.7 | NCCL watchdog: env validation, auto-release after T_max |
| 5.8 | Heartbeat with per-rank status |
| 5.9 | Pipeline-parallel support (per-stage barriers) |
| 5.10 | Multi-GPU checkpoint coordination |
| 5.11 | TCP transport for remote debugging |
| 5.12 | Qwen2 adapter |
| 5.13 | Gemma2 adapter |
| 5.14 | TCK green for multi-GPU scenarios |

**Exit criteria**: Same IOI experiment on Llama-3-70B with TP=4. All ranks step in sync.
Pre/post-collective inspection works.

### Phase 6 — MoE (weeks 19–21)

| Task | Description |
|------|-------------|
| 6.1 | Router hook: capture pre-topk logits |
| 6.2 | Routing decision capture: post-topk assignments |
| 6.3 | Per-expert stepping: pause inside specific expert |
| 6.4 | MoE layer tick: post-combine inspection |
| 6.5 | Routing entropy, expert load, dropped token reporting |
| 6.6 | Route override intervention (`route_override` recipe type) |
| 6.7 | Shared expert distinction (DeepSeek-V3 style) |
| 6.8 | Mixtral adapter |
| 6.9 | MoE TUI panel (per-token expert assignment, cluster projection) |
| 6.10 | TCK green for MoE scenarios |

**Exit criteria**: Routing override on Mixtral 8x7B reproducing Geometric Routing paper's
interventions. All four MoE tick granularities work.

### Phase 7 — torch.compile + CUDA Graph + FlashAttention (weeks 22+)

| Task | Description |
|------|-------------|
| 7.1 | Tier B: Dynamo custom backend for FX-graph instrumentation |
| 7.2 | Tier C: CUDA Graph inter-graph interception |
| 7.3 | FlashAttention shadow replay: selective unfusing per-layer |
| 7.4 | Head-level tick granularity (requires unfused execution path) |
| 7.5 | FSDP2 support: DTensor-based, hook timing within unsharded window |
| 7.6 | SimpleFSDP study + integration if applicable |
| 7.7 | Batch-invariant kernels for deterministic replay |
| 7.8 | Tier detection refinement (mixed-mode models) |
| 7.9 | Performance benchmarks across tiers |
| 7.10 | Support matrix expansion |
| 7.11 | TCK green for compiled-model scenarios |

**Exit criteria**: torch.compile model attaches with gracefully degraded granularity. FlashAttention
shadow replay produces accurate attention patterns. FSDP2 model inspectable.

---

## Cross-Cutting Concerns

### CI strategy

`cargo xtask ci` runs the full suite. Extended for Phase 1+:

```
xtask ci:
  1. cargo fmt --check
  2. cargo clippy (deny warnings)
  3. cargo test (Rust unit + integration)
  4. ruff check + format check
  5. mypy (strict)
  6. pytest (Python unit tests)
  7. JSON-Schema validation (schemas validate against Draft 2020-12)
  8. Protocol conformance (message round-trip tests)

xtask tck:
  9. TCK runner (pytest-bdd against real daemon + GPT-2-small)

xtask conformance:
  10. Model conformance suite (nightly, requires GPU)
```

Pre-commit hook runs `xtask ci` (steps 1–8). TCK and conformance run in CI pipeline and nightly,
not on every commit (they start processes and require model fixtures).

**xtask changes needed**: Add `tck` and `conformance` subcommands to `xtask/src/main.rs` (covered
in work units 0.7 and 2.4 respectively).

### Testing strategy

Three test levels:

1. **Unit tests**: Per-crate Rust tests, per-module Python tests. Test individual components
   in isolation. Mock inter-process communication.
2. **TCK tests**: Gherkin scenarios executed against the full daemon+host stack via pytest-bdd.
   Source of truth for protocol correctness. Run with GPT-2-small for CI speed.
3. **Acceptance tests**: End-to-end against real models (Llama-3-8B). Run nightly or pre-release.
   IOI reproduction is the primary acceptance test.

### Documentation strategy

- `protocol/README.md` — protocol overview for implementers
- `protocol/schema/v0.1.0/` — machine-readable schema (the documentation IS the schema)
- `docs/specs/design.md` — comprehensive design spec (exists)
- `docs/specs/plan.md` — this document
- `docs/adr/` — architectural decision records
- `docs/tutorial/quickstart.md` — getting started (ships with MVP, work unit 2.5)
- `docs/tutorial/ioi.md` — IOI reproduction tutorial (ships with MVP, work unit 2.5)
- No separate API docs — the protocol schema + TCK scenarios ARE the API docs

### Dependency management

**Rust**: workspace dependencies in root `Cargo.toml`. `cargo-deny` for license auditing.
Pin major versions.

**Python**: `pyproject.toml` with version ranges. `uv` or `pip-tools` for lockfile.
Dev dependencies separate from runtime. Add `pytest-bdd` to dev deps (Phase 0).

**PyTorch**: CI matrix tests against last 3 stable releases + nightly. `BackendAdapter` interface
(design doc §R1) shields daemon from PyTorch internals churn.

### Git workflow

- Frequent, atomic, descriptive commits
- One commit per work unit completion (minimum)
- Pre-commit hook gates every commit on `cargo xtask ci`
- No upstream remotes
- Beads for all issue tracking (in `.context/beads/`)
