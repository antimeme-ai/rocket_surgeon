# ADR-0004: Three-Process Architecture — Rust Daemon + Python Host(s) + Rust TUI

## Status
Proposed

## Context
ADR-0001 established the Rust + Python language split and sketched a two-layer model: Rust extension embedded inside the Python host process, with the TUI as a separate Rust process. The design doc (§2) evolved this into a three-independent-process architecture after deeper study of Garçon, tokio-console, Pernosco, and multi-GPU operational requirements.

The core question: should the Rust daemon live inside the Python host (PyO3 extension module, as ADR-0001 proposed) or as a separate OS process?

Options considered:
1. **Two-process (ADR-0001 model)**: Python host embeds Rust state machine via PyO3. TUI is a separate process. Daemon and model share an address space.
2. **Three-process**: Rust daemon is its own OS process. Python model host(s) are separate child processes. TUI is a separate process. All three communicate over JSON-RPC + shared memory.
3. **Single-process**: everything in Python with Rust called via FFI. Rejected immediately — TUI lifecycle coupled to model, no crash isolation, GIL contention.

Factors driving the decision:
1. Python crashes (GPU OOM, NCCL hang, segfault in a C extension) kill the entire process. If the daemon is in-process, a model crash takes down the debug session, the state machine, and all protocol state.
2. Multi-GPU requires one Python worker per rank (`torch.distributed` mandates one process per rank in most backends). The daemon must fan out to N host processes.
3. Users want the TUI to come and go without affecting the model. Multiple TUI instances should attach simultaneously to the same session.
4. The Garçon pattern (server-per-model, amortized loading) requires the daemon to outlive any single client and potentially manage multiple model hosts.
5. Development iteration speed: changing the Rust daemon should not require rebuilding a PyO3 wheel, and changing the Python host should not require recompiling Rust.

## Decision
**Three independent OS processes, communicating over JSON-RPC 2.0 and shared memory. This supersedes the two-layer architecture in ADR-0001.**

```
Process A: rs-daemon (Rust)
  - Protocol server (Unix socket, stdio, TCP)
  - State machine
  - Probe registry
  - Tensor handle store (content-addressable, BLAKE3)
  - Checkpoint index
  - Session manager
  - Perfetto trace sink

Process B: rs-host (Python) — one per GPU rank
  - PyTorch runtime
  - Model shard
  - Hook manager (Tier A/B/C)
  - Barrier gate (threading.Event)
  - Intervention engine
  - Tensor capture → shared-memory ring buffer

Process C: rs-tui (Rust) — zero or more instances
  - Ratatui rendering
  - Pure protocol client
  - No PyTorch awareness
```

### Inter-process communication

**Control plane**: JSON-RPC 2.0 over Unix domain sockets. The daemon-to-host channel uses the same verb schema as daemon-to-client — internal dispatch uses `_host/` prefixed methods but the type definitions are identical. This means the internal protocol is testable with the same TCK harness.

**Data plane**: Shared-memory ring buffer for tensor handoff. Python writes ProbeFrame records (128-byte fixed header + raw tensor bytes) into a shared region. Rust reads them zero-copy via mmap. Notification of new frames uses a Unix domain socket auxiliary channel (single byte write per frame), not Linux-only `eventfd`, ensuring macOS and Linux both work.

### PyO3 role

PyO3 remains in the architecture but as a **thin bridge** inside the Python host, not the architectural backbone. It accelerates hot-path operations that benefit from avoiding JSON-RPC round-trips: BLAKE3 hashing, ProbeFrame header serialization, summary stat aggregation. A pure-Python fallback exists for development without Rust builds.

## Consequences
- **Good**: Crash isolation. Python OOM or segfault kills only the host process. The daemon preserves session state, protocol trace, and checkpoint index. The daemon can restart the host and re-attach.
- **Good**: Multiple models per daemon. The Garçon pattern enables loading a model once and serving multiple debugging sessions against it, amortizing the 30s+ load time.
- **Good**: TUI lifecycle is fully independent. TUI processes connect and disconnect without affecting the model or daemon. Multiple TUIs (or LLM clients, or scripts) attach concurrently.
- **Good**: Multi-GPU is natural. One host process per rank is the `torch.distributed` execution model. The daemon fans out `step`/`inspect`/`intervene` commands to all rank workers and synchronizes their barriers.
- **Good**: Iteration speed. Changing protocol handling in the daemon requires only recompiling Rust. Changing hook logic in the host requires only restarting the Python process. No wheel rebuilds in the inner development loop.
- **Bad**: IPC overhead. JSON-RPC over Unix socket adds ~50 us per message versus in-process function calls. Mitigated by the shared-memory data plane for tensor bytes (the bulk of the data). Control messages are infrequent relative to forward-pass computation time.
- **Bad**: Deployment complexity. Three binaries to install and coordinate instead of one. Mitigated by the daemon managing host process lifecycle (spawn on `attach`, terminate on `detach`).
- **Bad**: Shared-memory ring buffer is platform-specific infrastructure. Mitigated by using Python `multiprocessing.shared_memory` (cross-platform) and Unix domain sockets for notification (macOS + Linux).
- **Risk**: Daemon-host protocol must be kept in sync with the external protocol. Mitigated by using the same JSON-Schema for both, enforced by the TCK.
