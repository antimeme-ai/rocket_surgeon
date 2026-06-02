# ADR-0010: Perfetto multi-rank track and sequence partitioning

## Status
Accepted

## Context

The Perfetto trace sink (WU 1.15, ADR-adjacent spec at
`docs/specs/2026-05-19-perfetto-trace-sink-design.md`) shipped correct for
single-rank execution but was deferred for multi-rank — six-persona adversarial
review filed BEAD-0010 with five structural findings. Re-examining the source
on the fix branch reveals two of them are not "future risk" but active bugs the
single-rank tests fail to catch:

- **M-2 (Per-sequence interning)** — `InternedData` in Perfetto's wire format is
  scoped to a `trusted_packet_sequence_id`. The current sink keeps one global
  `InternTable` and emits it once on a single rank's sequence. The moment a
  second rank emits `SLICE_BEGIN` packets carrying `name_iid=N`, the consumer
  (Perfetto UI, `traceconv`) resolves `N` against whatever it interned on *that*
  sequence — empty, or garbage. Single-rank passes because there is only one
  sequence.
- **M-4 (Component map keyed by string)** — `component_uuids` is
  `HashMap<String, u64>` keyed by `"L{layer}::{component}"`. Two ranks declaring
  the same component name at the same layer (e.g., DDP, FSDP, or any homogeneous
  shard layout) overwrite each other; tick events for the displaced rank route
  to the wrong track. The existing `traceconv_validates_output` test triggers
  this exact case but only asserts text contents, not per-rank routing.

The remaining findings (M-1 UUID collision, C-2 hardcoded rank-0 in probe
instants, SP-4 per-rank process tracks) compound with these two. A piecemeal fix
would force a kludgy intermediate state. This ADR is the load-bearing decision
for how rocket_surgeon represents multi-rank execution in Perfetto traces.

The fix lands as one PR and supersedes §3 (UUID scheme), §5 (Interning), and §6
(PerfettoSink API) of the original design spec.

## Decision

### 1. Track hierarchy: process-per-rank, plus a daemon process

Each `rs-host` worker is a real Linux process with its own PID, GPU, and crash
boundary. The trace models that:

```
[Process] daemon:rs-daemon              pid=daemon_pid   uuid = process(rank=DAEMON)
[Process] rank:0                        pid=worker_0_pid uuid = process(rank=0)
  [Track] L0                                             uuid = layer(rank=0, layer=0)
    [Track] L0::attn::q_proj                             uuid = component(rank=0, L=0, c=0)
    [Track] L0::attn::k_proj                             uuid = component(rank=0, L=0, c=1)
  [Track] L1
    ...
[Process] rank:1                        pid=worker_1_pid uuid = process(rank=1)
  [Track] L0
    [Track] L0::attn::q_proj                             uuid = component(rank=1, L=0, c=0)
  ...
```

A user opening the trace in `ui.perfetto.dev` sees the same process layout they
would see in `htop` — one daemon, one worker per GPU. Crash-isolation maps to
trace-isolation: a worker dying mid-session produces a clean, parseable trace
for the surviving ranks.

The daemon process owns events with no natural rank attribution: replay
divergence, session lifecycle (`attach`/`detach`), `rocket/error`. Worker
processes own tick boundaries and probe firings on their own GPU.

`DAEMON` is encoded as a reserved rank value (`0xFFF` — the all-ones 12-bit
sentinel). It cannot collide with a real worker rank because the rank field is
12 bits.

### 2. UUID scheme: bit-packed and deterministic

```
u64 = (kind:4 | rank:12 | layer:16 | component:32)
```

| Kind | Code | Layer field | Component field |
|------|------|-------------|-----------------|
| `PROCESS` | `0x1` | `0` | `0` |
| `LAYER` | `0x2` | layer index | `0` |
| `COMPONENT` | `0x3` | layer index | component index |
| `COUNTER` | `0x4` | (counter-kind-specific) | (counter-kind-specific) |

Capacities: 16 kinds, 4095 worker ranks + 1 daemon, 65536 layers, ~4.3B
components per layer. The largest released MoE model (Mixtral 8x22B) has 56
layers and well under 200 components per layer; the largest known dense model
has well under 200 layers. This scheme has headroom of several orders of
magnitude over the foreseeable Phase 8 horizon.

Deterministic UUIDs are part of the contract: tests assert by formula
(`process(rank=1) == 0x1_001_0000_00000000`), `traceconv text` output is
grep-able, and future tooling can reverse-derive `(rank, layer, component)`
from any UUID by bit-shift. The cost is a 4-bit type tag in the high nibble,
which Perfetto does not interpret — `track_uuid` is opaque to the consumer.

### 3. Per-sequence interning

Each Perfetto sequence has its own `InternTable`. The sink holds a
`HashMap<sequence_id, InternTable>`, lazily creating one per rank on first use.
Every `slice_begin` interns into the originating rank's table and references
the resulting `name_iid` on that sequence. `emit_interned_names(rank)` reads
that rank's table and emits `InternedData` packets on that rank's sequence,
with `SEQ_INCREMENTAL_STATE_CLEARED` on the first packet per sequence (which
`TraceWriter` already tracks via `seen_sequences`).

Sequence IDs:

```
sequence_id(rank) = 1000 + rank      // worker ranks
sequence_id(daemon) = 999             // daemon
```

The daemon gets a lower number so its sequence is created and emitted before
any worker sequence on a trace opened in `ui.perfetto.dev`.

### 4. Component map keyed by (rank, layer, component_name)

`component_uuids: HashMap<(u32, u32, String), u64>`. M-4 closed by construction.

### 5. Rank in `ProbeFiredEvent`

`ProbeFiredEvent` gains `rank: u32` (additive, `#[serde(default)]` for back-compat
with persisted bundles). The worker populates it from `WorkerState.rank` when
constructing the event. The daemon passes it to `on_probe_fired`. The sink uses
it for both track resolution and sequence routing. C-2 closed.

### 6. Worker PID plumbing

`WorkerHandle` exposes `pid() -> u32` via `child.id()`. The orchestrator
surfaces it through a method on its handle. The daemon, on `attach` success,
calls `PerfettoSink::declare_process(rank, pid, name)` for the daemon (using
`std::process::id()`) and for each worker rank.

## Consequences

### Positive

- The two active correctness bugs (M-2, M-4) are closed. Existing single-rank
  traces remain byte-identical except for the new daemon process track and the
  new 64-bit UUID encoding (which the consumer treats as opaque).
- The trace structure matches OS process structure — debugging multi-GPU runs
  in `ui.perfetto.dev` no longer requires a mental translation from "thread:N"
  to "GPU N's process".
- UUID derivation is closed-form. Tests can assert exact UUIDs. Future tooling
  (e.g., a TUI panel that overlays Perfetto state on the worldline) can decode
  any UUID back to `(kind, rank, layer, component)` with bit-shifts.
- Per-sequence intern tables compose with `SEQ_INCREMENTAL_STATE_CLEARED`
  cleanly: each rank's sequence is independently resumable, which matters
  when (future) we add support for partial trace export on detach-mid-session.

### Negative / accepted

- **Wire format change**: the new UUIDs are not backwards-compatible with traces
  produced by the old sink. Old traces remain readable (the format is
  unchanged), but tooling that assumes the old narrow UUID scheme will break.
  No such tooling exists outside this repo.
- **PR is large** — protocol + worker + orchestrator + daemon + sink + TCK + spec.
  Splitting would force PR #1 to land a kludge for the rank fix that PR #2
  throws away. Accepted; the design hangs together.
- **Reserved rank `0xFFF` (DAEMON)** caps real workers at 4095. Multi-host runs
  with > 4K ranks would hit this; the realistic horizon is 1-2 orders below.
  Re-evaluate at Phase 8 if 8K+ ranks become plausible.

### Deferred

- **Cross-host clock synchronization.** Single-host multi-GPU shares kernel
  `CLOCK_MONOTONIC`, so all sequences are on the same timeline without further
  work. Cross-host requires emitting Perfetto `ClockSnapshot` packets per
  sequence to map between host monotonic clocks. Out of scope for this ADR;
  revisit when Phase 5 introduces multi-host transport.
- **Tier 2 diagnostics** (CUPTI kernel-level spans, eBPF, NVML counters):
  design spec §10 already defers.
- **MoE expert/router-level component tracks**: Phase 6. The UUID scheme
  reserves enough component bits (32) to encode `expert_index << 16 |
  component_index` without further extension.
- **Counter tracks for `tick.heartbeat`** (GPU util, memory, temperature).
  Sink supports `counter_double`, integration deferred to a follow-up — not
  needed to close BEAD-0010.

## References

- BEAD-0010 — original adversarial-review findings
- `docs/specs/2026-05-19-perfetto-trace-sink-design.md` — original design,
  amended by this ADR
- `.context/lit-reviews/perfetto-trace-format.md` — Perfetto wire format study
- ADR-0004 — three-process architecture (daemon, worker, TUI) that this
  decision models
- Perfetto protobuf source vendored at `quarantine/perfetto/`
