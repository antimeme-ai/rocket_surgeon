# rocket_surgeon

A proper debugger and in-situ surgery tool for multi-GPU transformer forward passes.

Step through transformer internals — dense and MoE — one tick at a time, forward and backward, with full surgical intervention between steps. Inspect every layer, head, expert, and activation with numerically stable, research-backed summary statistics.

## What it does

- **Timestop debugging**: Pause the forward pass at any tick and inspect the full model state
- **Surgical intervention**: Modify activations, weights, routing decisions, and expert assignments between ticks
- **Multi-GPU native**: Works across DDP, FSDP, tensor parallel, and pipeline parallel setups
- **MoE-aware**: First-class support for Mixture-of-Experts architectures with router and expert-level granularity
- **Dual interface**: TUI for humans, structured JSON-RPC protocol for LLMs and automation
- **Content-addressable tensor store**: BLAKE3-keyed cache with lazy summary statistics (Welford mean/std, Blue's L2 norm, histogram, top-k)

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Client (TUI or LLM)                                    │
│  ↕ JSON-RPC 2.0 over Content-Length framing (stdio)      │
├──────────────────────────────────────────────────────────┤
│  rocket-surgeon daemon                                   │
│  ├── Session state machine (lifecycle, capabilities)     │
│  ├── Dispatch (11 verbs, 5 events)                       │
│  ├── Tensor store (BLAKE3, LRU, stats engine)            │
│  └── Probe registry (capture, assert, aggregate)         │
├──────────────────────────────────────────────────────────┤
│  rocket-surgeon-python (PyO3 bridge)                     │
│  ├── BLAKE3 hash (GIL-released)                          │
│  └── ProbeFrame header (128-byte packed binary)          │
├──────────────────────────────────────────────────────────┤
│  Python model host (hooks into forward pass)             │
└──────────────────────────────────────────────────────────┘
```

## Protocol

The wire protocol is JSON-RPC 2.0 with Content-Length framing (MCP-compatible). Every response carries a `SessionState` envelope with the current session state, available actions, and active probes.

**Verbs:** `initialize`, `attach`, `detach`, `rocket/step`, `rocket/inspect`, `rocket/intervene`, `rocket/probe`, `rocket/checkpoint`, `rocket/replay`, `rocket/status`, `rocket/subscribe`

**Session states:** `uninitialized` → `initialized` → `stopped` ↔ `stepping` / `inspecting` / `modifying` / `replaying`

## Crate structure

| Crate | Description |
|-------|-------------|
| `rocket-surgeon` | Daemon binary — session state machine, JSON-RPC dispatch, tensor store, statistics engine |
| `rocket-surgeon-protocol` | Wire protocol types — all verbs, events, errors, capabilities (no I/O, pure types) |
| `rocket-surgeon-probes` | Probe registry — define, match, enable/disable with priority ordering |
| `rocket-surgeon-python` | PyO3 bridge — BLAKE3 hash + ProbeFrame header for the Python model host |
| `rocket-surgeon-tui` | Terminal UI (planned) |
| `xtask` | Build tooling — CI runner, pre-commit hooks |

## Building

One command, idempotent — pins Python via `.python-version`, creates `.venv`, installs dev deps, builds the PyO3 extension, builds the Rust workspace, and smoke-checks the result:

```bash
cargo xtask setup
```

Prerequisites: Rust 1.85+ (edition 2024) and [`uv`](https://docs.astral.sh/uv/). Everything else (Python interpreter included) is provisioned by `uv` into the project venv.

After bootstrap:

```bash
source .venv/bin/activate
```

## Testing

```bash
# Full CI suite (fmt + clippy + ruff + mypy + all tests)
cargo xtask ci

# Rust tests only
cargo test --workspace

# Python unit tests
pytest python/tests/ -v

# End-to-end tests (spawn the daemon and drive it over JSON-RPC)
python tests/test_e2e_lifecycle.py
```

The project includes a behavioral specification suite (TCK) with 192 Gherkin scenarios across 16 feature files in `tck/`.

## Development methodology

This project follows JSMNTL discipline:

1. Literature review for every component on the hot path
2. Written design spec
3. Implementation plan with TDD
4. TCK-first behavioral specs (Gherkin `.feature` files)
5. Red → green → review → fix ALL findings
6. Frequent atomic commits

Architectural decisions are recorded in `docs/adr/`. Research material lives in `.context/lit-reviews/`.

## Status

**Phase 1** — building the foundation:

- [x] Protocol types crate (all verbs, events, errors, capabilities)
- [x] Probe registry (define, match, enable/disable, priority)
- [x] PyO3 bridge (BLAKE3, ProbeFrame header)
- [x] Daemon skeleton (session state machine, JSON-RPC dispatch, content-length framing)
- [x] Tensor store + statistics engine (BLAKE3 cache, Welford, histogram, top-k)
- [ ] Model host skeleton
- [ ] Adapter layer
- [ ] Shared memory transport
- [ ] Step/inspect/intervene implementation
- [ ] TUI

## License

MIT OR Apache-2.0
