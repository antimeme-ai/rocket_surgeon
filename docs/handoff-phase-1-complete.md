# Phase 1 Complete — Handoff Document

**Date:** 2026-05-19
**Branch:** `master` (at `9e54525`)
**Open PR:** #8 (`fix/bead-0009-lint-perimeter`) — lint perimeter expansion, ready for merge
**Status:** Phase 1 complete. Phase 2 ready to start. TUI intermission planned next shift.

---

## What Exists

rocket_surgeon is a debugger and in-situ surgery tool for multi-GPU transformer forward passes. Step through internals one tick at a time, forward and backward, with full surgical intervention between steps.

### The Numbers

| Metric | Count |
|--------|-------|
| Commits on master | 171 |
| Rust crates | 10 |
| Rust LOC | 16,513 |
| Python LOC | 4,055 |
| TCK Gherkin LOC | 3,370 (24 .feature files) |
| E2E test LOC | 2,539 (10 test scripts) |
| Rust unit tests | 499 passing |
| TCK scenarios | 227 (214 xpassed, 13 xfailed) |
| ADRs | 7 |
| Design specs | 12 |
| PRs merged | 7 |

### Crate Map

| Crate | Purpose |
|-------|---------|
| `rocket-surgeon` | Main daemon — session state machine, JSON-RPC dispatch, event loop |
| `rocket-surgeon-protocol` | Wire types — Request/Response/Event, error codes, message schemas |
| `rocket-surgeon-probes` | DTrace-inspired probe system — point grammar, matching, lifecycle |
| `rocket-surgeon-python` | PyO3 extension module — BLAKE3, header serialization |
| `rocket-surgeon-shm` | POSIX shared memory ring buffer for tensor transfer |
| `rocket-surgeon-transport` | Transport trait + Content-Length framing for IPC |
| `rocket-surgeon-orchestrator` | Worker lifecycle management, host attach/detach |
| `rocket-surgeon-worker` | Per-rank model worker — embeds Python via PyO3 |
| `rocket-surgeon-tui` | Terminal UI client (scaffold only) |
| `perfetto-writer` | Standalone Perfetto trace writer (no protoc, no C++ FFI) |

### Python Module Map

| Path | Purpose |
|------|---------|
| `python/rocket_surgeon/bridge.py` | PyO3 thin bridge — called from worker via FFI |
| `python/rocket_surgeon/views.py` | Built-in views (residual_stream_norm, attention_pattern) |
| `python/rocket_surgeon/probes/` | Probe grammar parser + utilities |
| `python/rocket_surgeon/host/` | Model host lifecycle, loader, RPC handler |
| `python/rocket_surgeon/hooks/` | Hook manager, mailbox barrier, capture system |

### Architecture: Three-Process Model

```
Client (TUI / LLM)
    |  JSON-RPC (stdio / TCP / WebSocket)
    v
Daemon (Rust)           ← session state, dispatch, event delivery, Perfetto sink
    |  JSON-RPC (stdin/stdout pipes)
    v
Orchestrator (Rust)     ← worker lifecycle, attach/detach
    |  JSON-RPC (stdin/stdout pipes)
    v
Worker (Rust + PyO3)    ← per-rank, embeds Python, runs forward pass
    |  PyO3 FFI
    v
Python Host             ← model loading, hooks, barriers, views
    |  PyTorch
    v
GPU                     ← actual computation
```

Tensors move via POSIX shared memory ring buffer with BLAKE3 content-addressable IDs. Protocol is JSON-RPC 2.0 with DAP-inspired semantics.

---

## Phase 1 Work Units (All Complete)

| WU | Name | PR | Key Artifact |
|----|------|----|-------------|
| 0.1-0.8 | Protocol spec, TCK harness, ADRs | #1-#3 | 24 .feature files, 7 ADRs |
| 1.1 | Daemon skeleton + session state machine | #3 | `crates/rocket-surgeon/src/main.rs` |
| 1.5 | Python model host skeleton | #4 | `python/rocket_surgeon/host/` |
| 1.6 | Model adapter (canonical name resolution) | #4 | `python/rocket_surgeon/host/adapter.py` |
| 1.7 | Hook manager (PyTorch forward hooks) | #4 | `python/rocket_surgeon/hooks/` |
| 1.8 | Shared memory data plane | #5 | `crates/rocket-surgeon-shm/` |
| 1.9 | PyO3 thin bridge | #4 | `crates/rocket-surgeon-python/` |
| 1.10 | Step integration (barriers + ticks) | #4 | `python/rocket_surgeon/hooks/barrier.py` |
| 1.11 | Inspect integration (summary + slice) | #4 | `rocket/inspect` verb |
| 1.12 | Probe event integration | #4 | `rocket/probe` verb |
| 1.13 | Built-in views | #6 | `rocket/view` verb, passive hooks |
| 1.14 | Subscribe + event delivery | #5 | `rocket/subscribe`, tick.stopped, heartbeat |
| 1.15 | Perfetto trace sink | #7 | `crates/perfetto-writer/`, `.pftrace` output |

**Not started:** WU 1.16 (end-to-end smoke test + overhead benchmark) — deferred, not blocking Phase 2.

---

## ADR Summary

| # | Title | Key Decision |
|---|-------|-------------|
| 0001 | Language Split | Rust core + Python hook layer; PyTorch stays in Python |
| 0002 | Protocol Design | JSON-RPC 2.0, DAP-inspired, unified internal/external schema |
| 0003 | Probe Model | DTrace-inspired naming (`model:rank:layer:component:event`), composable hooks |
| 0004 | Three-Process Architecture | Daemon ↔ Host(s) ↔ TUI, shared-memory tensor transport |
| 0005 | Tick Model | Granularity levels (layer/component/head), CUDA event scoped sync |
| 0006 | Tensor Handling | Content-addressable BLAKE3 IDs, summary-then-slice protocol |
| 0007 | Wire Format Breaking Change | Shared memory ring buffer replaces inline JSON base64 |

---

## Open Beads

| ID | Priority | Title | Status |
|----|----------|-------|--------|
| BEAD-0008 | HIGH | Daemon silent attach failure | **Closed** — fixed in prior work, verified |
| BEAD-0009 | MEDIUM | Lefthook glob excludes tests/ | **Closed** — PR #8 |
| BEAD-0010 | MEDIUM | Perfetto multi-GPU structural issues | Open — deferred to Phase 5 |

BEAD-0010 tracks five issues that are correct for single-GPU but need rework for multi-rank traces: UUID collisions across ranks, global InternTable not rank-partitioned, component_uuids keyed by string not compound key, probe instant hardcodes rank=0, single process track instead of per-rank.

---

## Phase 2: Interventions (Next)

This is where rocket_surgeon becomes a surgery tool. Interventions are data, not code — JSON-serializable recipes that compose declaratively.

### Work Units

| WU | Name | Depends | Summary |
|----|------|---------|---------|
| **2.1** | Intervention Engine | — | Pure Python: ablate, scale, add, patch, clamp recipes; priority-ordered composition |
| **2.2** | Intervene Verb | 2.1 | `rocket/intervene` (set/clear/list) wired through daemon→host→barrier |
| **2.3** | Session Bundle Export | 2.2 | tar.gz reproducibility artifact (9 items: manifest, traces, tensors, interventions) |
| **2.4** | Model Conformance Suite | — | Automated validation of probe firing order per model family |
| **2.5** | MVP Documentation | 2.2 | Quickstart + IOI tutorial + protocol reference |
| **2.6** | MCP Adapter | 2.2 | *(Stretch)* Wrap protocol as MCP tools for LLM-driven sessions |
| **2.7** | IOI Reproduction | 2.2, 2.3 | **MVP acceptance test** — Wang et al. 2023 name-mover ablation, logit diff ≥50% |

**Critical path:** 2.1 → 2.2 → 2.7. WU 2.4 is parallelizable.

### Five Recipe Types

| Type | What It Does | Params |
|------|-------------|--------|
| `ablate` | Zero out (kill a head) | none |
| `scale` | Multiply by factor | `{factor: number}` |
| `add` | Inject vector (activation patching) | `{vector: number[] \| tensor_id}` |
| `patch` | Replace with captured tensor | `{source_tensor_id: hex64}` |
| `clamp` | Bound values | `{min: number, max: number}` |

### How Interventions Work

1. Client sends `rocket/intervene` with recipe → daemon stores in registry
2. Client sends `rocket/step` → forward pass runs to next tick boundary
3. BarrierGate pauses → host applies active interventions in priority order
4. Forward resumes → next tick boundary → repeat

Interventions compose: multiple `add` recipes at the same point sum their vectors (additive mode). `mode: "replace"` overrides all prior recipes at that point. Same probe infrastructure (DTrace naming, wildcard matching) handles both observation and intervention.

### Existing TCK Coverage

`tck/protocol/intervention.feature` — 266 lines, covers:
- All five recipe types with set/clear/list actions
- Composition semantics (priority ordering, additive vs. replace)
- Persistence across steps
- Error handling (INVALID_TARGET, INVALID_RECIPE, INVALID_STATE)

Protocol schema already defined in `protocol/schema/v0.1.0/intervene.json`.

---

## TUI Intermission (Planned Next Shift)

The `rocket-surgeon-tui` crate is a scaffold. Patrick has strong TUI design opinions (per memory) — requires full brainstorming, no assumptions. The TUI is one of two client interfaces (the other being LLM-driven via structured protocol / MCP).

Key design tension: the TUI must serve human operators doing interactive debugging while the protocol serves LLMs doing automated analysis. Same underlying verbs, different UX ergonomics.

### What the TUI Needs to Show

Based on the protocol verbs and Phase 1 capabilities:
- Session state (initialized → stopped → stepping)
- Model architecture tree (layers, components, probes)
- Tensor inspector (summary stats, slice viewer)
- Step controls (forward/backward, granularity selector)
- Probe manager (create/enable/disable/remove)
- Event stream (tick.stopped, probe.fired, heartbeat)
- Perfetto trace viewer integration (or link to ui.perfetto.dev)

Phase 2 will add:
- Intervention editor (recipe builder, active intervention list)
- Session bundle export controls

### Existing TUI Crate

`crates/rocket-surgeon-tui/` — scaffold only, no implementation. Uses `ratatui` per the crate dependencies (verify before starting).

---

## JSMNTL Cycle Reminder

For any new work unit:

1. **Lit review** — papers, reference impls, existing tools
2. **Design spec** → `docs/specs/YYYY-MM-DD-<name>-design.md`
3. **Plan doc** → `docs/superpowers/plans/YYYY-MM-DD-<name>.md`
4. **Execution cycles** (per task in plan):
   - TCK (Gherkin .feature files first)
   - Red (tests compiling, failing)
   - Green (implementation passes tests)
   - Review (adversarial code review, fix all findings)
5. **Commit + push after each task set**

Design doc comes from brainstorming. Plan doc comes from exhaustive study. TCK is the first step of each execution cycle, not after design.

---

## Environment Notes

- Python 3.11 pinned in `.python-version`
- venv at `.venv/` (created by `cargo xtask setup`)
- PyO3 feature unification: workspace `--exclude rocket-surgeon-worker` for tests, worker tested separately with `DYLD_LIBRARY_PATH`
- lefthook for pre-commit hooks (ruff, clippy, fmt, mypy)
- `cargo xtask ci` runs the full suite
- GitHub repo: `antimeme-ai/rocket_surgeon`, `gh` CLI authenticated
- Git identity: `antimemeai` / `hiya@antimeme.ai`
