# Phase 1 Execution Plan — Single-GPU Eager Daemon

Derived from `docs/specs/plan.md` Phase 1 (WU 1.1–1.16). Each work unit follows the JSMNTL
execution cycle: sub-plan → TCK red → implementation → green → code review → fix → commit.

**Scope**: Rust daemon + Python host, single GPU, Llama family, component-level ticks,
tensor summary inspection, probe lifecycle, event subscription. No interventions (Phase 2),
no checkpoints (Phase 3), no MoE (Phase 6), no multi-GPU (Phase 5).

**CI model**: GPT-2-small (fits any machine). Nightly: Llama-3-8B on GPU.

---

## Dependency Graph

```
Batch A (no Phase 1 deps — all unblocked by Phase 0):
  1.2 protocol types     ─┐
  1.3 probe registry      │
  1.9 PyO3 bridge         │
  1.1 daemon skeleton    ─┼─── all can start immediately
                          │
Batch B:                  │
  1.4 tensor store  ◄─── 1.2
  1.5 model host    ◄─── 1.1

Batch C:
  1.6 adapter       ◄─── 1.5
  1.8 shm           ◄─── 1.4 + 1.5

Batch D:
  1.7 hooks         ◄─── 1.5 + 1.6

Batch E:
  1.10 step         ◄─── 1.1 + 1.2 + 1.5 + 1.7

Batch F:
  1.11 inspect      ◄─── 1.4 + 1.8 + 1.10

Batch G:
  1.12 probe events ◄─── 1.3 + 1.10 + 1.11
  1.13 views        ◄─── 1.11

Batch H:
  1.14 subscribe    ◄─── 1.12
  1.15 perfetto     ◄─── 1.12

Batch I:
  1.16 e2e smoke    ◄─── all of the above
```

## Critical Path

```
1.2 → 1.1 → 1.5 → 1.6 → 1.7 → 1.10 → 1.11 → 1.12 → 1.14 → 1.16
```

10 units on the critical path. Parallel opportunities exist at every batch but we execute
sequentially within this conversation, so ordering matters.

---

## Execution Order (Sequential)

The order below optimizes for: (a) unblocking downstream work early, (b) failing fast on
architectural assumptions, (c) keeping each WU self-contained and testable.

### Batch A — Foundations (no Phase 1 deps)

| Order | WU | Rationale |
|-------|-----|-----------|
| 1 | **1.2 Protocol types** | Pure types. Everything downstream needs these. Validates schema→Rust fidelity. |
| 2 | **1.3 Probe registry** | Extends existing probes crate. Self-contained Rust-only. |
| 3 | **1.9 PyO3 bridge** | BLAKE3 + ProbeFrame header. Needed by Python host capture path. |
| 4 | **1.1 Daemon skeleton** | Server + state machine. Consumes 1.2 types. Critical path entry point. |

### Batch B — Storage + Host

| Order | WU | Rationale |
|-------|-----|-----------|
| 5 | **1.4 Tensor store** | Content-addressable store in daemon. Uses 1.2 types. |
| 6 | **1.5 Model host skeleton** | Python host process. Daemon spawns it, connects via JSON-RPC. |

### Batch C — Adapter + Data Plane

| Order | WU | Rationale |
|-------|-----|-----------|
| 7 | **1.6 Model adapter** | Module-tree walker, canonical name mapping. Uses host from 1.5. |
| 8 | **1.8 Shared-memory data plane** | Ring buffer for tensor handoff. Bridges 1.4 (Rust reader) + 1.5 (Python writer). |

### Batch D — Hooks

| Order | WU | Rationale |
|-------|-----|-----------|
| 9 | **1.7 Hook manager** | PyTorch hooks + barrier gate. Needs adapter (1.6) for module mapping. |

### Batch E–F — Integration (Step + Inspect)

| Order | WU | Rationale |
|-------|-----|-----------|
| 10 | **1.10 Step integration** | First end-to-end verb. Client → daemon → host → barrier → stop. |
| 11 | **1.11 Inspect integration** | Second verb. Tensor capture via shm → daemon → client. |

### Batch G — Probes + Views

| Order | WU | Rationale |
|-------|-----|-----------|
| 12 | **1.12 Probe event integration** | Wire probes to firing + events. |
| 13 | **1.13 Built-in views** | residual_stream_norm + attention_pattern. |

### Batch H — Subscribe + Trace

| Order | WU | Rationale |
|-------|-----|-----------|
| 14 | **1.14 Subscribe + event delivery** | Client event streams. |
| 15 | **1.15 Perfetto trace sink** | Structured trace output. |

### Batch I — Validation

| Order | WU | Rationale |
|-------|-----|-----------|
| 16 | **1.16 E2E smoke test + overhead** | Full-stack validation. Flip TCK xfails to green. |

---

## Per-WU JSMNTL Cycle

Each WU follows this cycle exactly:

```
1. Write sub-plan (if WU is complex enough to need one)
2. Identify TCK targets → make step defs real (replace `pass` with assertions)
3. Run TCK → confirm RED (tests fail because implementation doesn't exist)
4. Write implementation code
5. Run TCK → fix until GREEN
6. Run `cargo xtask ci` → fix any lint/type/format issues
7. Spawn subagent code review → fix ALL findings
8. Commit (atomic, descriptive)
9. Push after completing the batch
```

**TCK target mapping** (which feature files each WU turns green):

| WU | Feature files | Scenario count (approx) |
|----|---------------|------------------------|
| 1.1 | lifecycle, errors, state-envelope | ~40 |
| 1.2 | (Rust unit tests, not Gherkin) | — |
| 1.3 | probes (CRUD scenarios only) | ~8 |
| 1.4 | handles | ~14 |
| 1.5 | lifecycle (attach/detach) | ~6 |
| 1.6 | adapter | ~10 |
| 1.7 | hooks, stepping (partially) | ~12 |
| 1.8 | (integration test, not Gherkin) | — |
| 1.9 | (Rust+Python unit tests) | — |
| 1.10 | stepping, state-envelope (position) | ~14 |
| 1.11 | inspection, handles (slice) | ~11 |
| 1.12 | probes (firing scenarios) | ~4 |
| 1.13 | inspection (view scenarios) | ~2 |
| 1.14 | subscribe | ~8 |
| 1.15 | (Perfetto manual validation) | — |
| 1.16 | ALL Phase 1 targets green | ~130 |

---

## New Dependencies to Add

The following workspace dependencies will be needed as we go:

| Crate | Purpose | Added in WU |
|-------|---------|-------------|
| `blake3` | Content-addressable tensor IDs | 1.4 / 1.9 |
| `uuid` | Session IDs | 1.1 |
| `bytes` | Zero-copy buffer management | 1.8 |
| `anyhow` | Already present | — |
| `prost` | Perfetto protobuf | 1.15 |

---

## Risk Register

| Risk | Mitigation | WU |
|------|------------|-----|
| PyTorch hook ordering not deterministic | Test with `prepend=True`, verify in hooks.feature | 1.7 |
| Shared memory portability (macOS vs Linux) | Use `multiprocessing.shared_memory`, test on both | 1.8 |
| CUDA event sync blocks wrong stream | Test on real GPU in nightly; CI uses CPU-only GPT-2 | 1.7 |
| JSON-RPC framing edge cases (large messages, partial reads) | Property tests on message parsing | 1.1 |
| torch.compile detection false positives | Check `isinstance(model, OptimizedModule)` specifically | 1.7 |

---

## Exit Criteria (Phase 1)

From `docs/specs/plan.md`:

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
