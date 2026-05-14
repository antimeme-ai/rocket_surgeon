# WU 1.1 — Rust Daemon Skeleton

## Goal

Protocol server: accept connections, parse JSON-RPC, dispatch to state machine,
serialize responses. Single-session for MVP. Every response carries a SessionState
envelope. All invalid transitions return structured INVALID_STATE errors.

## Architecture

```
stdin → [framing] → [parse JSON-RPC] → [dispatch] → [handler] → [session]
                                                          ↓
stdout ← [framing] ← [serialize]     ← [envelope]  ← [result]
                                                          ↓
                                                    [trace_log]
```

### Transport

Content-Length framing over stdin/stdout (MCP-compatible):
```
Content-Length: 42\r\n
\r\n
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
```

### Modules

| File | Responsibility |
|------|---------------|
| `main.rs` | CLI args (clap), tokio runtime setup, wire modules |
| `server.rs` | Async stdio read/write loop with content-length framing |
| `session.rs` | SessionState holder, state machine transitions, available_actions |
| `dispatch.rs` | Method string → handler routing, envelope wrapping |
| `trace_log.rs` | In-memory JSONL buffer of all JSON-RPC messages |

### State Machine

```
UNINITIALIZED → [initialize] → INITIALIZED
INITIALIZED   → [attach]     → STOPPED (stub: no real model loading yet)
STOPPED       → [detach]     → INITIALIZED
STOPPED       → [step/inspect/intervene/probe/status/subscribe] → (stub errors or stubs)
```

Valid actions per state:
- `Uninitialized`: (nothing — client must send initialize first)
- `Initialized`: [Attach]
- `Stopped`: [Step, Inspect, Intervene, Probe, Checkpoint, Replay, Detach, Status, Subscribe]
- Transient states (Attaching, Stepping, etc.): (empty — no client actions allowed)

### Handler Stubs

- `initialize`: Set session_id (uuid), return Capabilities with phase1_defaults()
- `attach`: Validate model_family ∈ {llama, mixtral, ...}, set model_id, return AttachResponse
- `detach`: Clear model_id/position, return DetachResponse
- `step/inspect/intervene/probe/checkpoint/replay/subscribe`: Return INVALID_STATE or
  stub responses depending on state
- `status`: Return StatusResponse with uptime + zeroed metrics
- Unknown method: Return JSON-RPC -32601 method-not-found

## Test Plan

### Rust unit tests (session.rs)
1. `new_session_is_uninitialized`
2. `initialize_transitions_to_initialized`
3. `initialize_returns_capabilities`
4. `double_initialize_returns_invalid_state`
5. `attach_from_initialized_transitions_to_stopped`
6. `attach_from_uninitialized_returns_invalid_state`
7. `attach_while_stopped_returns_model_already_attached`
8. `detach_from_stopped_transitions_to_initialized`
9. `detach_from_initialized_returns_model_not_attached`
10. `session_id_is_uuid_and_stable`
11. `model_id_null_before_attach_populated_after`
12. `available_actions_initialized_is_attach_only`
13. `available_actions_stopped_includes_domain_verbs`
14. `unsupported_model_family_returns_error`
15. `re_attach_after_detach_succeeds`

### Rust unit tests (dispatch.rs)
16. `unknown_method_returns_method_not_found`
17. `dispatch_initialize_succeeds`
18. `dispatch_wraps_response_in_envelope`

### Rust unit tests (trace_log.rs)
19. `trace_log_records_messages`
20. `trace_log_is_jsonl_format`

### Rust unit tests (server.rs)
21. `content_length_framing_round_trip`

## Dependencies Needed

- `uuid` (workspace dep, generate session_id)

## Execution Order

1. Add uuid to workspace deps
2. Implement session.rs (state machine + tests)
3. Implement dispatch.rs (routing + tests)
4. Implement trace_log.rs (buffer + tests)
5. Implement server.rs (stdio transport + tests)
6. Wire main.rs
7. Clippy, fmt, CI
8. Code review → fix → commit
