---
id: BEAD-0010
title: Perfetto trace writer — multi-GPU structural issues
status: resolved
priority: medium
created: 2026-05-19
resolved: 2026-05-26
resolution: ADR-0010 — process-per-rank, bit-packed UUIDs, per-sequence intern tables
---

## Description

Six-persona adversarial code review of WU 1.15 (perfetto-writer + PerfettoSink)
identified five structural issues that are correct for single-GPU but will need
rework when multi-GPU support lands. Deferred by decision — not blocking PR #7.

## Findings

### M-1: UUID collision across ranks (MEDIUM)

Track UUID scheme (`1000 + rank*1000 + layer`, `10000 + rank*100000 + layer*100 + comp`)
collides when rank >= 10 (layer tracks) or rank >= 1 (component tracks with many
layers). Needs a proper UUID allocation strategy — likely a namespace per rank with
wider spacing or a hash-based scheme.

### M-2: Global InternTable is not rank-partitioned (MEDIUM)

`InternTable` uses a single global `HashMap<String, u64>` and emits interned names
on sequence_id=1. With multiple ranks each needing their own Perfetto sequence,
interned data must be emitted per-sequence. Needs per-sequence intern state.

### M-4: component_uuids keyed by string, not (rank, layer, component) (MEDIUM)

`component_uuids: HashMap<String, u64>` uses component name strings as keys.
Two ranks with the same component name at different layers would alias. Needs
a compound key like `(rank, layer_idx, component_name)`.

### C-2: Probe instant events hardcode rank=0 sequence_id (LOW)

`instant()` always uses `sequence_id(0)` regardless of which rank the probe
belongs to. Fine for single-GPU, wrong for multi-rank traces.

### SP-4: Track hierarchy needs per-rank process tracks (LOW)

Current hierarchy is Session→Process(single)→Ranks→Layers→Components. True
multi-GPU traces need one process track per rank, each with its own subtree.

## Resolution

Closed by ADR-0010 (`docs/adr/ADR-0010-perfetto-multi-rank-tracing.md`) and the
sink rewrite on branch `fix/bead-0010-perfetto-multi-rank`. Findings disposition:

- **M-1 UUID collision** — closed. New scheme is bit-packed
  `(kind:4 | rank:12 | layer:16 | component:32)`. 4096 ranks × 64K layers ×
  4B components; structurally non-colliding within those bounds.
- **M-2 Per-sequence interning** — closed. Sink now holds a
  `HashMap<sequence_id, InternTable>`; `emit_interned_names(rank)` reads that
  rank's table and emits `InternedData` on that rank's sequence with
  `SEQ_INCREMENTAL_STATE_CLEARED` (`TraceWriter::seen_sequences` already
  tracked the first-packet flag). This was an active correctness bug, not
  deferred risk — multi-rank traces would have rendered with empty/garbage
  names past rank 0.
- **M-4 String-keyed component_uuids** — closed. Map is now keyed by
  `(rank, layer, component_name)`. The previous traceconv test triggered
  this bug silently; the rewrite asserts distinctness.
- **C-2 Hardcoded rank=0 in probe instants** — closed. `ProbeFiredEvent`
  gained `rank: u32` (additive, `#[serde(default)]` for back-compat); the
  worker populates it from `WorkerState.rank`; the sink routes to
  `sequence_id(event.rank)`.
- **SP-4 Per-rank process tracks** — closed. Ranks are now ProcessDescriptors
  (with real PIDs surfaced through the orchestrator), with a separate daemon
  ProcessDescriptor for daemon-originated events. The original Thread-per-rank
  model is replaced.

Deferred (called out in ADR-0010 §Deferred): cross-host clock snapshots,
counter tracks for `tick.heartbeat`, MoE expert/router granularity, Tier 2
CUPTI/eBPF/NVML diagnostics.
