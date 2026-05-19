# WU 1.15: Perfetto Trace Sink — Design Spec

## Goal

Write tick boundaries, probe firings, and session metadata to a Perfetto protobuf trace file. The trace opens in `ui.perfetto.dev` and shows a coherent timeline of the debugging session — every component tick as a duration span, every probe firing as an instant event, structured as a process→rank→layer→component track hierarchy.

Build the Perfetto writer as a **generic, standalone crate** (`perfetto-writer`) with no RS-specific concepts. RS-specific track mappings live as a thin integration layer in the daemon crate. The generic crate is an OSS release candidate (like salib).

## Dependencies

- WU 1.12 (probe events): probe firings available in `HostStepResponse.events` — done
- WU 1.14 (subscribe + events): `tick.stopped`, `probe.fired`, `tick.heartbeat` event delivery — done
- Perfetto lit review (`.context/lit-reviews/perfetto-trace-format.md`) — done

## Non-Dependencies

- WU 1.8 (shared memory): orthogonal — shm is a data plane optimization, Perfetto is observability
- CUPTI, eBPF, NVML (Tier 2 diagnostics from design.md §14): deferred to future phases

---

## 1. Wire Format Summary

A `.pftrace` file is a serialized protobuf `Trace` message: `repeated TracePacket packet = 1`. Each packet on disk is `[0x0A][varint(len)][packet_bytes]` — standard protobuf field-1 repeated encoding. This means we can **stream-append** packets to an open file without buffering the whole trace.

Key message types we emit:

| Message | Purpose | When |
|---------|---------|------|
| `TrackDescriptor` | Declare a timeline lane | Session start, attach |
| `TrackEvent` (SLICE_BEGIN/END) | Duration span | Tick start/end |
| `TrackEvent` (INSTANT) | Point event | Probe firing |
| `TrackEvent` (COUNTER) | Numeric sample | (future: loss, grad norm) |
| `DebugAnnotation` | Key-value metadata | Attached to any TrackEvent |
| `InternedData` | String compression | First packet per sequence |

## 2. Protobuf Strategy: Hand-Annotated prost Structs

The monolithic `perfetto_trace.proto` is 19k lines / 686 KB. We need ~10 message types. Two viable approaches:

**Option A: prost-build + vendored proto** — full coverage, requires `protoc` at build time, generates thousands of unused types, adds build complexity.

**Option B: Hand-annotated `#[derive(prost::Message)]` structs** — exact subset, no `protoc`, no `build.rs`, no `.proto` files, still wire-correct via prost encoding. Struct definitions derived from studying the proto source (vendored in `quarantine/perfetto/`).

**Decision: Option B.** Reasons:
1. No build-time proto compilation — the crate builds with `cargo build`, period
2. No `protoc` binary dependency — more portable, simpler CI
3. Only the types we need — small, auditable, documented
4. Still uses `prost` for wire-correct encoding — not hand-rolling varint/tag logic
5. Proto source in quarantine serves as ground truth for field numbers
6. Adding new message types later is trivial (copy field numbers from proto, add struct)

Runtime dependency: `prost` only. Build dependency: none.

## 3. Track Hierarchy

Maps RS concepts to Perfetto tracks per the lit review §3:

```
[Process] session:{session_id}                    uuid=1
  [Thread] rank:0                                 uuid=100
    [Track] L0                                    uuid=1000
      [Track] L0::attn::q_proj                    uuid=10000
      [Track] L0::attn::k_proj                    uuid=10001
      ...
    [Track] L1                                    uuid=1001
      ...
  [Thread] rank:1                                 uuid=101
    ...
```

UUID scheme: deterministic, not random.
- Process: `1`
- Rank N: `100 + N`
- Layer L under rank N: `1000 + N*1000 + L`
- Component C under layer L, rank N: `10000 + N*100000 + L*100 + C`

`trusted_packet_sequence_id`: `1000 + rank`. One sequence per rank.

Track ordering: `sibling_order_rank` set to layer/component index so Perfetto preserves execution order (not alphabetical).

## 4. Event Mapping

### Tick → Duration Event (SLICE_BEGIN / SLICE_END)

Each tick is a component-level forward pass step. When the daemon receives `tick.stopped`:

```
TracePacket {
  timestamp: monotonic_ns,
  trusted_packet_sequence_id: 1000 + rank,
  track_event: TrackEvent {
    type: TYPE_SLICE_BEGIN,  // or TYPE_SLICE_END
    track_uuid: component_track_uuid,
    name_iid: interned component name,
  }
}
```

**Timing**: We get `tick.stopped` after the tick completes. We don't have sub-tick begin/end times from CUPTI (that's Tier 2). For now, emit SLICE_BEGIN at the previous tick's timestamp and SLICE_END at the current tick's timestamp. This gives relative durations between ticks, which is what matters for the timeline view.

Alternative: emit TYPE_INSTANT per tick (simpler, no begin/end tracking). But duration spans give a much better Perfetto UI experience — you see the component as a colored bar, not a dot.

**Decision**: Track last-tick timestamp per component. Emit SLICE_END for previous + SLICE_BEGIN for current on each tick.stopped. Emit final SLICE_END on detach/session end.

### Probe Firing → Instant Event

```
TracePacket {
  timestamp: monotonic_ns,
  trusted_packet_sequence_id: 1000 + rank,
  track_event: TrackEvent {
    type: TYPE_INSTANT,
    track_uuid: component_track_uuid,
    name: "probe:{probe_id}",
    debug_annotations: [
      { name: "shape", string_value: "[32, 16, 128, 128]" },
      { name: "mean", double_value: 0.0312 },
      { name: "std", double_value: 0.0015 },
      { name: "norm", double_value: 1.234 },
      { name: "action", string_value: "capture" },
    ]
  }
}
```

### Session Start → Process Metadata

Emitted once at session creation (or at attach):

```
TracePacket {
  track_descriptor: TrackDescriptor {
    uuid: 1,
    name: "session:{session_id}",
    process: ProcessDescriptor {
      pid: 1,
      process_name: "{model_id}",
    }
  }
}
```

### Session End / Detach → Flush

Close all open SLICE_BEGIN spans with SLICE_END. Flush the BufWriter. The file is complete.

## 5. Interning

Component names repeat every forward pass. With 1600 components × 1000 passes, interning saves ~67 MB (per lit review §3 calculation).

Strategy: emit `InternedData` on the first packet of each sequence (`SEQ_INCREMENTAL_STATE_CLEARED`), mapping `iid → name` for all known component names. Use `name_iid` on all subsequent `TrackEvent` packets.

Interning is managed per-sequence (per-rank). The generic crate provides an `InternTable` that tracks iid assignments. The RS integration layer populates it with component names at attach time.

## 6. Crate Architecture

### `crates/perfetto-writer/` — Generic Crate

```
Cargo.toml          # deps: prost, thiserror
src/
  lib.rs            # pub mod + re-exports
  proto.rs          # Hand-annotated prost structs (TracePacket, TrackEvent, etc.)
  writer.rs         # TraceSink: streaming append to Write
  intern.rs         # InternTable: string → iid mapping
  track.rs          # TrackSet: track declaration + uuid management
  varint.rs         # Protobuf varint + field-1 framing
```

**Public API**:

```rust
// Create a sink that writes to any impl Write
let mut sink = TraceSink::new(BufWriter::new(file));

// Declare tracks
let process = sink.add_process_track(1, "my-session", pid);
let thread = sink.add_thread_track(100, process, "rank:0", tid);
let track = sink.add_track(10000, thread, "L0::attn::q_proj");

// Intern strings
sink.intern(1001, "attn::q_proj", &["component"]);

// Emit events
sink.slice_begin(track, timestamp_ns, name_iid)?;
sink.slice_end(track, timestamp_ns)?;
sink.instant(track, timestamp_ns, "probe:attn_weights", &annotations)?;
sink.counter(counter_track, timestamp_ns, value)?;

// Finalize
sink.flush()?;
```

### `crates/rocket-surgeon/src/perfetto_sink.rs` — RS Integration

Thin layer that:
1. Creates a `TraceSink` pointing at a session-specific `.pftrace` file
2. At attach: declares process/rank/layer/component tracks from the component map
3. At attach: populates intern table with component names
4. On `tick.stopped`: emits SLICE_END for previous component + SLICE_BEGIN for current
5. On `probe.fired`: emits INSTANT event with tensor summary as debug annotations
6. On detach: closes open spans, flushes, returns file path

State: the integration layer owns a `TraceSink` + a mapping from `(rank, layer, component)` → track UUID + last-tick timestamp tracking.

### Integration Point in Daemon

In `main.rs`, after event emission (lines ~600-632), call the perfetto sink:

```rust
// After tick.stopped notification
if let Some(ref mut perfetto) = perfetto_sink {
    perfetto.on_tick_stopped(&position)?;
}

// After probe.fired notification  
if let Some(ref mut perfetto) = perfetto_sink {
    for pe in &hr.events {
        perfetto.on_probe_fired(pe)?;
    }
}
```

The sink is `Option<PerfettoSink>` — None when Perfetto is disabled or no session active.

## 7. File Lifecycle

1. **Create**: On `rocket/attach` success, create `{state_dir}/{session_id}.pftrace`
2. **Write**: Stream-append packets throughout session
3. **Flush**: On `rocket/detach`, close spans, flush BufWriter
4. **Export**: Session bundle (WU 2.3) copies the `.pftrace` file into `trace.perfetto-trace`
5. **Cleanup**: On daemon shutdown, flush any open sinks

The file is always valid — each written packet is self-contained. An interrupted session produces a truncated but parseable trace.

## 8. Clock

`std::time::Instant` for monotonic timestamps. Convert to nanoseconds. Set `timestamp_clock_id = 6` (CLOCK_MONOTONIC) on the first packet of each sequence. All events in a session use the same clock.

## 9. Testing Strategy

The generic crate gets unit tests for:
- Wire format correctness (encode a known trace, compare bytes against reference)
- Roundtrip: encode with our writer, decode with prost (from the same structs)
- Varint encoding edge cases
- Intern table correctness
- Track UUID management

The RS integration gets:
- Unit test: mock tick.stopped → verify TracePacket emitted with correct track/type
- Unit test: mock probe.fired → verify instant event with annotations
- Integration test: run a mini session, open the .pftrace, verify it decodes as valid protobuf
- Manual validation: open in `ui.perfetto.dev` (not automatable, but documented in test plan)

## 10. What This WU Does NOT Do

- **Tier 2 diagnostics** (CUPTI, eBPF, NVML): future phases
- **Counter tracks** for loss/grad norm: future, but the generic crate supports TYPE_COUNTER
- **Multi-rank traces**: the architecture supports it (per-rank sequence IDs, per-rank threads), but we only test single-rank in this WU
- **MoE expert routing tracks**: Phase 6
- **Trace size management / rotation**: future if needed (estimated 100-500 KB/pass is fine)
