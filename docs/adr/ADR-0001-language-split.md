# ADR-0001: Language Split — Rust Core + Python Hook Layer

## Status
Proposed

## Context
rocket_surgeon needs to:
1. Manage a state machine, protocol server, and TUI with low latency and memory safety
2. Register PyTorch forward hooks on live models (requires Python)
3. Interface with CUDA/CUPTI (C API, callable from either language)
4. Be usable by LLMs via structured protocol (JSON-RPC)
5. Load HuggingFace models (Python ecosystem)

Options considered:
- **Pure Python**: simplest hook integration, but GIL limits concurrency, TUI ecosystem weaker (Textual is async Python, not immediate-mode)
- **Pure Rust**: best performance, but PyTorch hooks REQUIRE Python. Would need to embed Python (complex) or reimplement model loading (TransformerLens trap)
- **Rust + Python via PyO3**: Rust owns state machine, protocol, TUI. Python owns PyTorch hooks, model loading, SAE integration. PyO3 bridges with zero-copy tensor sharing possible via numpy/ndarray interop.

## Decision
**Rust core + Python hook layer, bridged via PyO3.**

Process architecture: Python process is the host (owns PyTorch runtime, GIL, model memory). Rust code compiled as Python extension module (`.so`/`.dylib`). Rust state machine called from Python, returns structured results. TUI runs in a separate Rust process communicating over the protocol.

```
┌──────────────────────────┐     ┌──────────────────┐
│   Python Host Process    │     │  TUI Process     │
│  ┌────────────────────┐  │     │  (Pure Rust)     │
│  │ PyTorch + Hooks    │  │     │  Ratatui         │
│  │ Model Loading      │  │     │                  │
│  │ SAE Integration    │  │     │                  │
│  └────────┬───────────┘  │     └────────┬─────────┘
│           │ PyO3         │              │
│  ┌────────▼───────────┐  │    JSON-RPC  │
│  │ Rust Extension     │◄─┼──────────────┘
│  │ State Machine      │  │  (stdio/TCP)
│  │ Protocol Server    │  │
│  │ Checkpoint Store   │  │
│  │ Probe Registry     │  │
│  └────────────────────┘  │
└──────────────────────────┘
```

## Consequences
- **Good**: Rust state machine is fast and memory-safe. Python hooks work natively. TUI gets immediate-mode rendering. LLM clients talk to protocol server in any language.
- **Good**: PyO3 is mature (used by pydantic, polars, ruff). Zero-copy ndarray possible.
- **Bad**: Build system complexity (maturin for PyO3 builds). Two-language debugging.
- **Bad**: Tensor data crosses Python↔Rust boundary. For inspection (read-only), this is cheap (pointer sharing). For checkpointing (copy to host), overhead is dominated by GPU→CPU transfer regardless.
- **Risk**: PyO3 GIL handling. Rust code must acquire GIL for Python callbacks. Release GIL during long operations (checkpoint I/O, protocol serving).
