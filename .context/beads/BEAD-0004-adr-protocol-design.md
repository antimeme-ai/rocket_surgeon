---
id: BEAD-0004
title: "ADR: Protocol design — JSON-RPC 2.0 with DAP-inspired semantics"
status: done
priority: high
created: 2026-05-14
completed: 2026-05-14
---

## Description

Decide on the machine interface protocol. Options: pure DAP, gRPC, custom binary, JSON-RPC 2.0.

## Resolution

ADR-0002 written. Decision: JSON-RPC 2.0 with DAP-inspired message semantics (Request/Response/Event) and LSP-style capability negotiation. Transport: stdio default, TCP for remote, WebSocket for browser/MCP. NOT a DAP implementation — domain-specific operations (tensors, layers, probes) not source-level debugging.

See `docs/adr/ADR-0002-protocol-design.md`.
