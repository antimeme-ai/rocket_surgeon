# WU 1.5 — Model Host Skeleton

## Purpose

The model host skeleton establishes the process topology through which rocket_surgeon controls PyTorch models. Three Rust binaries connected by a common Transport trait:

- **rs-daemon** (existing) — protocol server, session state machine
- **rs-orchestrator** (new) — manages worker lifecycle, absorbs crashes, fans out commands
- **rs-worker** (new) — embeds Python via PyO3, loads/unloads models through a minimal Python skin

This is the first work unit that brings Python into the runtime. The Python surface is minimal and deliberate: the only Python in the application is what is necessary to reach PyTorch's API and pull its levers. All logic, state management, IPC, and control flow are Rust.

## Preconditions

**RS births the model.** rocket_surgeon is not fit for purpose if not present and configured as the model comes online. The worker loads the model through the Python skin. Hooks are registered before the first forward pass ever runs. There is no "attach to an already-running model" path in the main system.

## Process Topology

```
rs-daemon ──Transport──▶ rs-orchestrator ──Transport──▶ rs-worker[rank 0]
                              │                              │
                              │                        embedded libpython
                              │                        (same process, PyO3)
                              │                              │
                              │                       Python skin → PyTorch
                              │
                              ├──Transport──▶ rs-worker[rank 1]  (Phase 5)
                              ├──Transport──▶ rs-worker[rank 2]  (Phase 5)
                              └──Transport──▶ ...

rs-daemon ──Transport──▶ rs-orchestrator (host 1)        (Phase 5, multi-node)
rs-daemon ──Transport──▶ rs-orchestrator (host 2)        (Phase 5, multi-node)
```

All three are separate OS processes with real IPC from day one. For WU 1.5: one daemon, one orchestrator, one worker. But the binary boundaries and Transport wiring are real — no in-process shortcuts that would need to be torn out for multi-GPU.

### Why three processes

- **Orchestrator absorbs worker crashes.** Workers embed Python + PyTorch + CUDA. They can OOM, segfault in C extensions, hang on NCCL. The orchestrator catches child process death without the daemon ever losing contact.
- **Orchestrator is per-host.** In a multi-node fleet (4 hosts, 32 GPUs), the daemon spawns one orchestrator per host. Each orchestrator spawns workers for its local GPUs. Orchestrating orchestrators is a real operational requirement.
- **Workers are per-rank.** One worker per GPU. Each embeds its own Python interpreter. No GIL contention between ranks.

### Why workers embed Python (not separate Python processes)

PyTorch hooks are in-process Python callbacks — they fire synchronously inside `forward()`. There is no external API to register or trigger hooks. The worker must share an address space with the model. PyO3 links `libpython` into the Rust binary, making model/hooks/Rust control logic cohabit one process. Calls between Rust and Python are function calls, not IPC.

## Transport Trait

Defined in a new `rocket-surgeon-transport` crate (or as a module in `rocket-surgeon-protocol`).

```
Transport
  ├── send(message: &JsonRpcMessage) -> Result<()>
  ├── recv() -> Result<JsonRpcMessage>
  └── close() -> Result<()>
```

Two implementations planned:

| Impl | Wire format | WU 1.5 | Use case |
|------|------------|--------|----------|
| StdioTransport | Content-Length framed JSON-RPC over stdin/stdout | Yes | Parent spawns child, communicates over pipes |
| SocketTransport | Same framing over Unix socket or TCP | Later | Multi-node, independent process lifecycle |

The same Transport trait is used at both boundaries (daemon↔orchestrator and orchestrator↔worker). Same wire format everywhere.

### Content-Length framing

Matches the existing daemon server framing (MCP-compatible):

```
Content-Length: <n>\r\n
\r\n
<n bytes of JSON-RPC>
```

The daemon already implements this in `server.rs`. The Transport trait extracts it so all three binaries share one implementation.

## Crate Layout

| Crate | Type | New? | Purpose |
|-------|------|------|---------|
| `rocket-surgeon-transport` | lib | New | Transport trait, StdioTransport, Content-Length framing |
| `rocket-surgeon-orchestrator` | bin + lib | New | Orchestrator binary — spawn/monitor workers, fan out commands |
| `rocket-surgeon-worker` | bin + lib | New | Worker binary — embed Python, model lifecycle for one rank |
| `rocket-surgeon-python` | lib (cdylib + rlib) | Existing, grows | Python skin functions: load_model, unload_model (adds to existing BLAKE3 + ProbeFrame) |
| `rocket-surgeon` (daemon) | bin + lib | Existing, modified | attach() spawns orchestrator, detach() tears it down |
| `rocket-surgeon-protocol` | lib | Existing, grows | Internal command types for orchestrator↔worker (if distinct from external protocol) |

## rs-orchestrator

### Responsibilities

1. Receive commands from daemon via Transport
2. Spawn worker process(es) as children
3. Monitor worker health (child process exit, heartbeat timeout)
4. Forward commands to appropriate worker(s)
5. Aggregate responses back to daemon
6. Report worker crashes to daemon (never panic, never lose contact)

### WU 1.5 scope

- Spawns exactly one worker on `attach`
- Forwards `attach` and `detach` commands to that worker
- Catches worker exit and reports failure to daemon
- CLI args: `--daemon-transport stdio` (how it talks to daemon), worker binary path

### Command flow (attach)

```
daemon sends:    {"method": "_host/attach", "params": {...}}
orchestrator:    spawns rs-worker child process
orchestrator:    forwards attach to worker via StdioTransport
worker:          initializes Python, loads model, responds success
orchestrator:    relays success to daemon
daemon:          transitions session to STOPPED
```

### Command flow (worker crash)

```
worker process exits unexpectedly (OOM, segfault, etc.)
orchestrator:    detects child exit via waitpid
orchestrator:    sends error notification to daemon
daemon:          transitions session to error state / INITIALIZED
```

## rs-worker

### Responsibilities

1. Receive commands from orchestrator via Transport
2. Initialize embedded Python interpreter on startup (`pyo3::prepare_freethreaded_python`)
3. On attach: call Python skin to load model
4. On detach: call Python skin to unload model, release GPU memory
5. Report errors without panicking (catch Python exceptions via PyO3, report over Transport)

### WU 1.5 scope

- Handles `_host/attach` and `_host/detach` commands
- Loads a model via Python skin on attach (nano model for testing)
- Unloads model on detach
- CLI args: `--orchestrator-transport stdio`

### PyO3 embedding

The worker binary links against `libpython` via PyO3's `auto-initialize` feature. On startup:

```rust
fn main() {
    pyo3::prepare_freethreaded_python();
    // ... Transport setup, command loop
}
```

All Python calls go through `Python::with_gil(|py| { ... })`. The GIL is held only for the duration of Python skin calls — Rust control logic runs without the GIL.

## Python Skin

Minimal functions in the `rocket-surgeon-python` crate (or a Python module it exposes). No logic, no state management, no IPC handling. Just the thinnest possible bridge to PyTorch:

```python
def load_model(source: str, device: str, dtype: str) -> int:
    """Load a model from source (HF hub ID, local path, safetensors dir).
    Returns an opaque handle (integer ID) for subsequent calls."""

def unload_model(handle: int) -> None:
    """Unload model, release GPU memory, delete references."""

def model_metadata(handle: int) -> dict:
    """Return raw model structure: num_layers, num_heads, hidden_dim,
    module tree (raw PyTorch named_modules, not canonical names — 
    canonical mapping is WU 1.6 adapter work)."""
```

For WU 1.5: `load_model`, `unload_model`, and `model_metadata` are all delivered. Model source resolution:
- If source looks like an HF hub ID (contains `/`): `AutoModelForCausalLM.from_pretrained(source)`
- If source is a local path: load from path
- Device and dtype are explicit parameters, no magic

### Model handle registry

The skin maintains a simple dict mapping integer handles to live model objects. This is the only state in Python — an unavoidable minimum since PyTorch model objects must be held somewhere. The Rust worker holds the handle ID and passes it back on subsequent calls.

## Daemon Changes

### attach() modification

Current behavior: validate parameters, transition to STOPPED, return metadata.

New behavior:
1. Validate parameters (model family, execution mode, etc.)
2. Spawn orchestrator as child process with StdioTransport
3. Send `_host/attach` to orchestrator (which forwards to worker)
4. Wait for success/failure response
5. On success: transition to STOPPED, return model metadata
6. On failure: kill orchestrator, remain INITIALIZED, return error

### detach() modification

1. Send `_host/detach` to orchestrator
2. Wait for acknowledgment
3. Terminate orchestrator process
4. Transition to INITIALIZED

### Internal command namespace

Commands between daemon↔orchestrator↔worker use `_host/` prefixed methods to distinguish from the external client protocol:
- `_host/attach` — load model on specified rank(s)
- `_host/detach` — unload model, release resources
- (Future: `_host/step`, `_host/inspect`, `_host/intervene`, etc.)

Same JSON-RPC wire format. The `_host/` prefix is a convention, not a separate protocol.

## Error Handling

All error paths must leave the system in a recoverable state:

| Failure | Handling |
|---------|----------|
| Worker fails to load model | Worker reports error over Transport. Orchestrator relays to daemon. Daemon remains INITIALIZED. |
| Worker crashes during attach | Orchestrator detects child exit. Reports to daemon. Daemon remains INITIALIZED. |
| Orchestrator crashes | Daemon detects child exit. Daemon transitions to INITIALIZED. Client gets error response. |
| Python exception in skin | PyO3 converts to Rust Result::Err. Worker reports over Transport. No panic. |
| Model source not found | Python skin raises, worker catches, reports. |

## Testing Strategy

TCK target: `lifecycle.feature` scenarios covering attach/detach.

Rust unit tests:
- Transport trait: StdioTransport round-trip (send/recv JSON-RPC messages)
- Orchestrator: spawn mock worker, forward command, handle crash
- Worker: embed Python, call load_model/unload_model on nano model

Integration test:
- Full chain: daemon → orchestrator → worker → load nano model → success → detach → cleanup
- Full chain: daemon → orchestrator → worker → load bad model → error propagates back

Python skin tests:
- load_model with `sshleifer/tiny-gpt2` or equivalent nano model
- unload_model releases references
- model_metadata returns expected structure

## Open Questions (to resolve during implementation)

1. **Transport crate vs module**: Does the Transport trait warrant its own crate, or should it live in `rocket-surgeon-protocol`? Likely its own crate since both orchestrator and worker depend on it but not on each other.
2. **Worker binary naming**: `rs-worker` vs `rocket-surgeon-worker` — follow whatever convention the workspace establishes.
3. **Python skin location**: Grow `rocket-surgeon-python` (which already has PyO3 setup) or create a new `rocket-surgeon-model` crate? Likely grow the existing crate since PyO3 setup is already there.
4. **Async in orchestrator**: The orchestrator needs to simultaneously read from daemon Transport and monitor worker child processes. Tokio runtime in orchestrator, or threads? Tokio is already a workspace dep.

## Non-Goals

- Hook registration, barrier gates, tick stepping (WU 1.7 / 1.10)
- Tensor capture, shared memory ring buffer (WU 1.8 / 1.11)
- Multi-rank worker fan-out (Phase 5 — but orchestrator is structurally ready)
- SocketTransport implementation (later WU)
- Late-attach to already-running models (not in main system, ever)
- Model adapter / canonical name mapping (WU 1.6 — but model_metadata returns raw structure)
