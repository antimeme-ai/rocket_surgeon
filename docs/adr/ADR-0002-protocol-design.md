# ADR-0002: Protocol Design — JSON-RPC 2.0 with DAP-Inspired Semantics

## Status
Proposed

## Context
The machine interface is the primary interface for rocket_surgeon — TUI, Python scripts, and LLM clients all consume it. Needs:
1. Structured, parseable messages (LLM-native)
2. Request-response correlation (concurrent operations)
3. Async event notifications (state changes, probe fires)
4. Capability negotiation (clients discover features)
5. Proven pattern (not a novel protocol)

Options considered:
- **Pure DAP**: standard debugger protocol, VS Code/Neovim support. But assumes call-stack model, variable scoping, breakpoint-line-number paradigm. Poor fit for tensor-based neural net debugging.
- **Custom binary protocol**: fastest, smallest. But LLMs can't parse binary. Development overhead.
- **gRPC**: typed, streaming, multi-language. But heavy dependency, not human-readable, LLMs can't read protobuf wire format.
- **JSON-RPC 2.0**: simple, well-specified, LLM-native. Used by LSP, MCP, many debuggers. With DAP-like semantics (capability negotiation, event notifications, state machine).

## Decision
**JSON-RPC 2.0 with DAP-inspired message semantics and LSP-style capability negotiation.**

Three message types (from DAP):
- **Request**: client → server, has `id` for correlation
- **Response**: server → client, references request `id`
- **Event**: server → client, async notification (no `id`)

Capability negotiation (from LSP): `initialize` response declares what the server supports. Client adapts.

Transport: stdio (default — shell pipes, simplest), TCP (remote debugging), WebSocket (browser/MCP).

NOT a DAP implementation — we use DAP's patterns but define our own domain-specific operations (step through layers, not source lines; inspect tensors, not variables; probe points, not breakpoints).

## Consequences
- **Good**: LLMs parse JSON natively. Any language can implement a client. Tooling exists (JSON Schema validation, protocol testing).
- **Good**: stdio transport means `echo '{"method":"status"}' | rocket_surgeon` works from shell. Maximum composability.
- **Good**: DAP-inspired semantics are proven in debugger UX across multiple IDEs.
- **Bad**: JSON is verbose. Tensor data payloads could be large. Mitigation: summaries by default, full data opt-in with pagination.
- **Bad**: Not a standard DAP server, so existing DAP clients (VS Code debug panel) won't work out of the box. Custom clients required. This is acceptable — the domain is too different from source-level debugging.
- **Risk**: Protocol versioning. Address with semver on protocol version, announced in `initialize`.
