# rocket_surgeon Protocol

JSON-RPC 2.0 wire protocol for controlling the rocket_surgeon daemon.
Three message types: **request** (client to daemon, has `id`), **response**
(daemon to client, references `id`), **notification** (daemon to client,
no `id`, no response expected).

Transport: stdio or Unix socket (MVP), TCP (Phase 5), MCP adapter (Phase 2
stretch). All transports carry the same JSON-RPC 2.0 messages.

## Verbs

11 verbs. Lifecycle verbs use bare names; domain verbs use the `rocket/` namespace.

| Method | Mutating | Description |
|---|---|---|
| `initialize` | No | Capability negotiation. Returns server capabilities and protocol version. |
| `attach` | Yes | Load model onto GPU(s), start host processes, register hooks. |
| `detach` | Yes | Unload model, release GPU memory, terminate host processes. |
| `rocket/step` | Yes | Advance or reverse the forward pass by N ticks. |
| `rocket/inspect` | No | Read tensor data at a probe point. Returns summary by default. |
| `rocket/intervene` | Yes | Set, clear, or list surgical interventions on the forward pass. |
| `rocket/probe` | Mixed | Define, list, enable, disable, or delete probes. |
| `rocket/checkpoint` | Mixed | Create, restore, list, delete, or bookmark checkpoints. |
| `rocket/replay` | Yes | Restore a checkpoint and replay forward with optional interventions. |
| `rocket/status` | No | Full session state dump with operational metrics. |
| `rocket/subscribe` | No | Subscribe to event notifications with optional filters. |

## Events

5 daemon-to-client notifications. Clients must subscribe via `rocket/subscribe`.

| Method | Description |
|---|---|
| `rocket/tick.stopped` | Forward pass paused at a tick boundary. Includes full state. |
| `rocket/tick.heartbeat` | Sent every 1s while stopped. Per-rank GPU status. |
| `rocket/probe.fired` | A probe matched and executed its action. Includes tensor summary. |
| `rocket/replay.divergence` | Replayed tensor diverges from original beyond tolerance. |
| `rocket/error` | Unrecoverable error (OOM, NCCL hang, etc.). |

## Response Envelope

Every response includes `state` (SessionState) + `data`. No hidden state.
Any client can pick up any response cold and know where the debugger is.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "state": {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "model_id": "sha256:abc123def456...",
      "status": "stopped",
      "position": {
        "tick_id": 42,
        "direction": "forward",
        "layer": 3,
        "component": "attn.o_proj",
        "event": "post"
      },
      "tick_id": 42,
      "active_probes": ["p1", "p2"],
      "checkpoints": [],
      "available_actions": ["step", "inspect", "intervene", "probe", "checkpoint", "status"]
    },
    "data": {
      "ticks_executed": 1,
      "stopped_at": { "tick_id": 42, "direction": "forward", "layer": 3, "component": "attn.o_proj", "event": "post" }
    }
  }
}
```

## Error Contract

Errors are structured and actionable. Machine-readable code, recovery suggestion,
severity, current state, and valid states for the attempted operation.

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
      "current_state": "attaching",
      "valid_states": ["stopped"],
      "suggestion": "Wait for attach to complete, then retry",
      "severity": "recoverable"
    }
  }
}
```

### Error Code Registry

All error codes are defined in `schema/v0.1.0/errors.json`. No verb
implementation may invent codes not in this registry.

| Code | Numeric | Severity | Description |
|---|---|---|---|
| `INVALID_STATE` | -32001 | recoverable | Verb not valid in current session state |
| `INVALID_TARGET` | -32002 | recoverable | Probe point pattern doesn't match any component |
| `INVALID_RECIPE` | -32003 | recoverable | Intervention recipe malformed or unsupported |
| `MODEL_NOT_ATTACHED` | -32004 | recoverable | Operation requires an attached model |
| `TENSOR_NOT_FOUND` | -32005 | recoverable | tensor_id doesn't exist in the store |
| `CHECKPOINT_NOT_FOUND` | -32006 | recoverable | Checkpoint name/id doesn't exist |
| `PROBE_NOT_FOUND` | -32007 | recoverable | probe_id doesn't exist |
| `CAPABILITY_NOT_SUPPORTED` | -32008 | recoverable | Verb requires a capability the session lacks |
| `SLICE_OUT_OF_BOUNDS` | -32009 | recoverable | Tensor slice indices exceed shape |
| `RESPONSE_TOO_LARGE` | -32010 | recoverable | Requested data exceeds 64 KB cap |
| `HOST_ERROR` | -32011 | fatal | Python host process error |
| `GPU_OOM` | -32012 | fatal | GPU out of memory |
| `NCCL_TIMEOUT` | -32013 | fatal | NCCL collective timed out |
| `REPLAY_DIVERGENCE` | -32014 | recoverable | Replay exceeded tolerance (informational) |
| `UNSUPPORTED_MODEL` | -32015 | recoverable | Model architecture not in support matrix |
| `COMPILED_MODEL` | -32016 | recoverable | Model uses torch.compile, Tier A cannot attach |
| `MODEL_ALREADY_ATTACHED` | -32017 | recoverable | Attach requested but a model is already loaded |
| `INVALID_PARAMS` | -32602 | recoverable | JSON-RPC standard: malformed request params |

Numeric codes -32001 through -32099 are reserved for rocket_surgeon domain
errors. `INVALID_PARAMS` (-32602) is the standard JSON-RPC code for malformed
params.

## State Machine

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

Valid verbs per state:

| State | Valid verbs |
|---|---|
| `uninitialized` | `initialize` |
| `initialized` | `attach`, `status` |
| `attaching` | `status` (read-only wait) |
| `stopped` | `step`, `inspect`, `intervene`, `probe`, `checkpoint`, `replay`, `detach`, `status`, `subscribe` |
| `stepping` | `status` (read-only wait) |
| `inspecting` | `status` (read-only wait) |
| `modifying` | `status` (read-only wait) |
| `replaying` | `status` (read-only wait) |
| `detaching` | `status` (read-only wait) |

## Schema Files

All schemas live in `schema/v0.1.0/` and use JSON Schema Draft 2020-12.

| File | Description |
|---|---|
| `common.json` | Shared types: SessionState, TickPosition, TensorSummary, ErrorCode, etc. |
| `errors.json` | Authoritative error code registry: all codes, numeric IDs, descriptions, suggestion templates, applicable verbs, severity. |
| `initialize.json` | Request/response for capability negotiation. |
| `attach.json` | Request/response for model loading. |
| `detach.json` | Request/response for model unloading. |
| `step.json` | Request/response for tick advancement. |
| `inspect.json` | Request/response for tensor inspection. |
| `intervene.json` | Request/response for surgical interventions. |
| `probe.json` | Request/response for probe management. |
| `checkpoint.json` | Request/response for checkpoint operations. |
| `replay.json` | Request/response for replay with fidelity verification. |
| `status.json` | Request/response for session diagnostics. |
| `subscribe.json` | Request/response for event subscription. |
| `events.json` | All 5 daemon-to-client notification schemas. |
| `components.json` | Canonical component vocabulary for model adapters. |

## Versioning

Protocol version is declared in the `initialize` response via
`capabilities.protocol_version` (semver string). Clients that request
an unsupported version receive a structured error with the server's
supported range. Wire protocol is independent of PyTorch versions.

## Transports

| Transport | Phase | Notes |
|---|---|---|
| stdio | MVP | Pipes. `echo '{"method":"rocket/status"}' \| rs-daemon`. Maximum composability. |
| Unix socket | MVP | Multi-client. Default for TUI + daemon on same host. |
| TCP | Phase 5 | Remote debugging across machines. |
| MCP adapter | Phase 2 stretch | Wraps the protocol for LLM tool-use via Model Context Protocol. |

All transports carry identical JSON-RPC 2.0 messages. No transport-specific semantics.
