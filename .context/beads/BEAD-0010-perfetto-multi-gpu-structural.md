---
id: BEAD-0010
title: Perfetto trace writer — multi-GPU structural issues
status: open
priority: medium
created: 2026-05-19
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

Address as part of the multi-GPU work unit (likely WU 2.x). Each finding should
be a task in that plan. The current single-rank implementation is correct and
complete for its scope.
