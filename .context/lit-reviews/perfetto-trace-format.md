---
topic: Perfetto trace format — wire format, Rust codegen, rocket_surgeon mapping
status: complete
created: 2026-05-18
sources: perfetto.dev docs, github.com/google/perfetto, crates.io, schilk.co
---

# Perfetto Trace Format: Lit Review

Writing Perfetto traces from a Rust application without using the C++ SDK — covering wire
format, Rust codegen strategy, rocket_surgeon concept mapping, and practical sizing.

---

## 1. Wire Format

### Root structure

A `.pftrace` file is a serialized protobuf `Trace` message:

```proto
// protos/perfetto/trace/trace.proto
message Trace {
  repeated TracePacket packet = 1;
}
```

Because `Trace.packet` is field number 1 with LEN wire type, each packet on disk is
preceded by the tag byte `0x0A` (`(1 << 3) | 2`) followed by a varint-encoded length.
This is standard protobuf repeated-field encoding — no framing magic.

**Key insight for streaming**: since the `Trace` message is just a sequence of
length-prefixed `TracePacket` blobs at field 1, you can *append* packets to an open file
one at a time without holding the full trace in memory. The decoder re-reads the file as a
repeated field and reassembles the logical `Trace`. This is how Perfetto's own long-trace
mode works.

### TracePacket

`TracePacket` is the unit of data. All fields except the payload are low-numbered for
wire efficiency. Key fields (from `protos/perfetto/trace/trace_packet.proto`):

| Field | Number | Type | Purpose |
|---|---|---|---|
| `timestamp` | 8 | uint64 | nanoseconds (default CLOCK_BOOTTIME on Android; use clock 6 = CLOCK_MONOTONIC for custom traces) |
| `timestamp_clock_id` | 58 | uint32 | clock source identifier |
| `track_event` | 11 | TrackEvent | timeline event payload |
| `track_descriptor` | 60 | TrackDescriptor | declares a track (written once per track) |
| `trusted_packet_sequence_id` | 10 | uint32 | logical sequence — all packets from one writer share this |
| `sequence_flags` | 13 | uint32 | SEQ_INCREMENTAL_STATE_CLEARED=1, SEQ_NEEDS_INCREMENTAL_STATE=2 |
| `interned_data` | 12 | InternedData | per-sequence string intern table |
| `trace_packet_defaults` | 59 | TracePacketDefaults | applied as defaults to subsequent packets in sequence |
| `first_packet_on_sequence` | 87 | bool | marks first packet from a writer |
| `previous_packet_dropped` | 42 | bool | data-loss indicator |

`trusted_packet_sequence_id` must be non-zero and unique per logical writer. For
hand-crafted traces you assign these yourself (e.g. one per GPU rank).

### TrackDescriptor

Declares a track (a timeline lane). Emitted once before any events on that track.
From `protos/perfetto/trace/track_event/track_descriptor.proto`:

| Field | Type | Purpose |
|---|---|---|
| `uuid` | uint64 | globally unique track ID |
| `parent_uuid` | uint64 | nesting — child tracks appear under parent in UI |
| `name` | string | display label |
| `static_name` | string | compile-time constant variant |
| `process` | ProcessDescriptor | marks this as a process-root track |
| `thread` | ThreadDescriptor | marks this as a thread track |
| `counter` | CounterDescriptor | marks this as a counter track |
| `child_ordering` | enum | LEXICOGRAPHIC / CHRONOLOGICAL / EXPLICIT |
| `sibling_order_rank` | int32 | explicit sort position among siblings |
| `description` | string | tooltip text in Perfetto UI |

**ProcessDescriptor** sub-fields: `pid` (int32), `process_name` (string), `cmdline` (string[]).

**ThreadDescriptor** sub-fields: `pid` (int32), `tid` (int64), `thread_name` (string),
`reference_timestamp_us` (int64) — base for delta timestamps.

### TrackEvent

The payload carried inside `TracePacket.track_event`. From
`protos/perfetto/trace/track_event/track_event.proto`:

| Field | Number | Type | Purpose |
|---|---|---|---|
| `type` | 9 | Type enum | see below |
| `track_uuid` | 11 | uint64 | which track this event belongs to |
| `name` | 23 | string | event display name (non-interned) |
| `name_iid` | 10 | uint64 | interned name reference |
| `categories` | 22 | repeated string | category labels |
| `category_iids` | 3 | repeated uint64 | interned category references |
| `debug_annotations` | 4 | repeated DebugAnnotation | arbitrary key-value metadata |
| `counter_value` | 30 | int64 | for TYPE_COUNTER events |
| `double_counter_value` | 44 | double | floating-point counter |
| `flow_ids` | 47 | repeated fixed64 | connect causally-related events |
| `terminating_flow_ids` | 48 | repeated fixed64 | close a flow |

**Type enum**:
- `TYPE_UNSPECIFIED = 0`
- `TYPE_SLICE_BEGIN = 1` — duration event start
- `TYPE_SLICE_END = 2` — duration event end
- `TYPE_INSTANT = 3` — instant event
- `TYPE_COUNTER = 4` — counter sample

Duration events are open/close pairs: emit `TYPE_SLICE_BEGIN` with a name at t0, emit
`TYPE_SLICE_END` (no name needed) at t1. The UI renders the span between them.

### Minimal valid trace (conceptual proto)

```
Trace {
  // Track 1: process root
  packet {
    track_descriptor {
      uuid: 1
      name: "rocket_surgeon session"
      process { pid: 42  process_name: "rs" }
    }
  }
  // Track 2: rank 0 thread
  packet {
    track_descriptor {
      uuid: 2
      parent_uuid: 1
      name: "rank:0"
      thread { pid: 42  tid: 1  thread_name: "rank0" }
    }
  }
  // Duration event: layer 0 forward tick
  packet {
    timestamp: 1000000        // ns
    trusted_packet_sequence_id: 1001
    sequence_flags: 1         // SEQ_INCREMENTAL_STATE_CLEARED
    first_packet_on_sequence: true
    track_event {
      type: TYPE_SLICE_BEGIN
      track_uuid: 2
      name: "L0::attn::qkv_proj"
    }
  }
  packet {
    timestamp: 1500000
    trusted_packet_sequence_id: 1001
    track_event {
      type: TYPE_SLICE_END
      track_uuid: 2
    }
  }
  // Instant event: probe firing
  packet {
    timestamp: 1200000
    trusted_packet_sequence_id: 1001
    track_event {
      type: TYPE_INSTANT
      track_uuid: 2
      name: "probe:attn_weights"
      debug_annotations { name: "shape"  string_value: "[32, 16, 128, 128]" }
      debug_annotations { name: "norm_mean"  double_value: 0.0312 }
    }
  }
}
```

---

## 2. Writing Perfetto Traces from Rust

### Available crates

| Crate | Approach | Status | Notes |
|---|---|---|---|
| `perfetto_protos` | prost-generated bindings from Perfetto protos | exists on crates.io | unclear maintenance, check before use |
| `tracing-perfetto-sdk-schema` | prost bindings, part of `tracing-perfetto-sdk` workspace | v0.13.1, March 2026 | internal crate; depends on C++ SDK |
| `tracing-perfetto-sdk-layer` | tracing-subscriber layer using C++ SDK | active | FFI to C++, not pure Rust |
| `prost` + vendored protos | DIY: vendor `.proto` files, generate with `prost-build` | prost v0.14.3 | recommended for no-C++-dep path |

**Conclusion**: no mature, standalone, pure-Rust crate for writing Perfetto traces without
the C++ SDK. The correct approach for rocket_surgeon is to vendor the Perfetto `.proto`
files and generate Rust code with `prost-build`. This is straightforward, gives full
control, and avoids a C++ FFI dependency.

### prost vs protobuf crate

**Use prost.**

- `prost` v0.14.3: idiomatic Rust structs, derive macros, `bytes::{Buf,BufMut}`, generates
  clean readable code, widely adopted (tokio ecosystem), passively maintained (bug fixes).
- `protobuf` (rust-protobuf, stepancheg): runtime reflection, larger binary, enum handling
  via `EnumOrUnknown<T>`, its own author suggests preferring prost for most use cases.
- Google's official Rust protobuf (`prost`-based, protocolbuffers repo): still in flux as
  of 2025; prost is what the ecosystem has converged on.

### Vendoring Perfetto .proto files

**Location in the Perfetto repo** (`github.com/google/perfetto`):

```
protos/
  perfetto/
    trace/
      perfetto_trace.proto    # monolithic all-in-one (686 KB, 19k lines) — easiest to vendor
      trace_packet.proto      # TracePacket message
      track_event/
        track_event.proto     # TrackEvent + Type enum
        track_descriptor.proto
        counter_descriptor.proto
        debug_annotation.proto
      common/
        ...
    config/
      ...                     # not needed for writing traces
```

Two strategies:
1. **Monolithic**: vendor only `perfetto_trace.proto` — single file, self-contained, easy.
2. **Selective**: vendor only the subset of protos needed (track_event, track_descriptor,
   trace_packet). Less code generated but requires managing imports.

Recommend strategy 1 (monolithic) to start.

### Rust codegen with prost-build

`Cargo.toml`:
```toml
[build-dependencies]
prost-build = "0.14"

[dependencies]
prost = "0.14"
bytes = "1"
```

`build.rs`:
```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    prost_build::compile_protos(
        &["protos/perfetto_trace.proto"],
        &["protos/"],
    )?;
    Ok(())
}
```

`src/lib.rs`:
```rust
pub mod perfetto {
    include!(concat!(env!("OUT_DIR"), "/perfetto.trace.rs"));
}
```

Prost generates a Rust struct per message, using `Option<T>` for optional fields and
`Vec<T>` for repeated. Enums come as `i32` constants with accessor methods.

**Requires `protoc` on PATH.** Alternative: use `prost-build` with `PROTOC` env var
pointing to a bundled binary (the `protoc-bin-vendored` crate can supply it).

### Streaming append pattern (Rust + prost)

The wire format allows writing `TracePacket`s incrementally. Each packet is:
`[0x0A][varint(len)][packet_bytes]`

```rust
use prost::Message;

fn write_trace_packet(packet: &TracePacket, out: &mut impl std::io::Write)
    -> std::io::Result<()>
{
    let mut buf = Vec::with_capacity(packet.encoded_len() + 16);
    // field tag: field 1, LEN wire type
    encode_varint(0x0A, &mut buf);
    encode_varint(packet.encoded_len() as u64, &mut buf);
    packet.encode(&mut buf).expect("encode failed");
    out.write_all(&buf)
}

fn encode_varint(mut v: u64, buf: &mut Vec<u8>) {
    loop {
        let septet = (v & 0x7F) as u8;
        v >>= 7;
        let cont = if v == 0 { 0u8 } else { 0x80u8 };
        buf.push(septet | cont);
        if v == 0 { break; }
    }
}
```

Use a `BufWriter<File>` as `out` for efficiency. No need for a `Trace` wrapper object in
memory — the file *is* the Trace. Use `O_APPEND` semantics when writing from multiple
threads/ranks.

---

## 3. Mapping rocket_surgeon Concepts to Perfetto

### Proposed mapping

| rocket_surgeon concept | Perfetto concept | Notes |
|---|---|---|
| **Session** | Process track (root) | One `TrackDescriptor` with `process` field; `pid` = session ID; `name` = model name + session timestamp |
| **Rank** | Thread track under session | One `TrackDescriptor` per rank with `thread` field; `tid` = rank; `thread_name` = `"rank:{N}"` |
| **Layer** | Sub-track under rank (no ThreadDescriptor, just `parent_uuid`) | Groups component tracks visually |
| **Component within layer** | Track under layer | `TrackDescriptor` with `parent_uuid` = layer track; `name` = `"L{n}::attn::qkv_proj"` etc. |
| **Tick (forward pass through component)** | Duration event (TYPE_SLICE_BEGIN/END pair) | Begin = component start, End = component end; `name` = component name |
| **Probe firing** | Instant event (TYPE_INSTANT) | On the component's track; carries tensor stats in `debug_annotations` |
| **Tensor summary stats** | `debug_annotations` on probe instant | `mean`, `std`, `norm`, `shape`, `dtype` as key-value pairs |
| **Surgical intervention** | Instant event on component track | Mark with `name: "intervention:clamp_activations"` + `debug_annotations` |
| **MoE expert routing** | Counter track under layer | One counter per expert, value = routing weight or token count |

### Track hierarchy (visual in Perfetto UI)

```
[Process] session:llama3-70b/2026-05-18T14:00Z
  [Thread] rank:0
    [Track] L0
      [Track] L0::attn::q_proj        ← duration events (ticks)
      [Track] L0::attn::k_proj
      [Track] L0::attn::v_proj
      [Track] L0::attn::output
      [Track] L0::mlp::gate
      [Track] L0::mlp::up
      [Track] L0::mlp::down
    [Track] L1
      ...
  [Thread] rank:1
    ...
```

### TrackDescriptor metadata recommendations

- Session (process) track: include `process_name` = model name, `cmdline` entries for
  model size, dtype, parallelism config.
- Rank (thread) track: `thread_name` = `"rank:{N}"`, `pid` = session id, `tid` = rank.
- Component track: `name` = full dotted path (`"L3::mlp::gate_proj"`), `description` =
  human-readable explanation.
- Use `sibling_order_rank` on layer tracks so layers sort numerically (L0 before L10).
- Use `child_ordering: EXPLICIT` + `sibling_order_rank` on component tracks to preserve
  execution order rather than alphabetical.

### trusted_packet_sequence_id assignment

One sequence per rank. Rank N uses sequence ID `1000 + N`. All tracks declared and all
events emitted from rank N's writer use the same sequence ID. This ensures Perfetto can
reconstruct interned string tables correctly per sequence.

### Interning

For a 32-layer model with 50 components, component names repeat once per forward pass.
At 1000 passes, each 50-byte name without interning = 50 × 1600 × 1000 = 80 MB just
for names. With interning: 50 × 1600 = 80 KB (declared once), then 8-byte iids per
reference = 1600 × 1000 × 8 = 12.8 MB. Use `name_iid` + `InternedData` on the first
packet of each sequence (`SEQ_INCREMENTAL_STATE_CLEARED`).

---

## 4. Practical Concerns

### Estimated trace file size

Baseline per tick (no interning, no tensor stats): ~80–120 bytes per TracePacket
(timestamp 8B, sequence_id 4B, track_uuid 8B, type 1B, name ~50B, proto overhead ~20B).

With interning (name as iid): ~30–50 bytes per event packet after the first sequence.

Per forward pass (1600 ticks, 32-layer × 50 components):
- Without interning: 1600 × 100 B = 160 KB/pass
- With interning: 1600 × 40 B = 64 KB/pass

With tensor summary metadata per tick (mean, std, norm, shape as debug_annotations):
Add ~100–200 bytes per probe firing depending on annotation count.
If probes fire on every tick: 1600 × 200 B = 320 KB/pass additional.

Realistic total per forward pass: **100–500 KB/pass**.

At 1000 passes: **100–500 MB total**. Well within Perfetto UI's 2 GB in-browser limit for
moderate session lengths. For long training runs (millions of passes), use the
`trace_processor` server mode for native-performance loading.

### Streaming vs in-memory

**Streaming is viable and recommended.** The `Trace` wire format is just length-delimited
`TracePacket` records at field 1, so you can write packets to an open `BufWriter<File>`
continuously. No need to buffer the whole trace.

For multi-GPU: each rank writes to its own file segment or uses a shared append-mode file
with appropriate locking (or, simpler, one file per rank merged post-session with a merge
tool). Perfetto's `trace_processor` can merge multiple trace files.

### Perfetto UI limits

- Browser in-memory limit: ~2 GB (runtime representation is larger than file size due to
  query-optimized indexing).
- For traces > 2 GB: run `trace_processor` as a local HTTP server
  (`./trace_processor --server /path/to/trace.pftrace`) and open `ui.perfetto.dev` with
  the external server; the UI will auto-detect it at `localhost:9001`.
- For very large sessions: partition traces by time window or number of forward passes.

### File writing performance

`prost`'s `Message::encode` writes into a `bytes::BufMut` (or `Vec<u8>`). Wrapping the
output file in a 64–256 KB `BufWriter` reduces syscall overhead. For multi-rank traces,
consider a background writer thread per rank that drains a crossbeam channel of serialized
packets.

### Clock selection

Use a monotonic clock (CLOCK_MONOTONIC, `timestamp_clock_id = 6`) with nanosecond
resolution. Set `timestamp_clock_id` on the first packet of each sequence. The Perfetto
UI will correlate events across tracks correctly as long as all events in a session use the
same clock.

---

## 5. Proto File Structure Summary

Key files to vendor from `github.com/google/perfetto/protos/perfetto/trace/`:

```
perfetto_trace.proto                   # monolithic all-in-one (recommended for vendoring)

# Or individually:
trace_packet.proto                     # TracePacket root
track_event/track_event.proto          # TrackEvent, Type enum
track_event/track_descriptor.proto     # TrackDescriptor
track_event/counter_descriptor.proto   # CounterDescriptor
track_event/debug_annotation.proto     # DebugAnnotation
common/data_source_descriptor.proto    # if needed
```

The monolithic `perfetto_trace.proto` (19k lines, 686 KB) is an autogenerated merge of
all the above. It's the easiest to vendor: one file, no import path issues.

---

## Decision Recommendation

For rocket_surgeon's trace output:

1. **Vendor** `perfetto_trace.proto` (monolithic) into `protos/`.
2. **Generate** Rust code at build time via `prost-build` in `build.rs`.
3. **Write** trace files using the incremental append pattern (no full-trace in-memory
   buffer).
4. **Map**: Session → Process track, Rank → Thread track, Layer → named sub-track,
   Component → named leaf track, Tick → duration event, Probe → instant event with
   debug_annotations.
5. **Intern** event names and category strings; use `name_iid` for all hot paths.
6. **One sequence ID per rank**; include `SEQ_INCREMENTAL_STATE_CLEARED` on the first
   packet of each sequence when interning state resets.

This gives a pure Rust path with no C++ FFI, full control over the trace structure, and
output that opens directly in `ui.perfetto.dev` or `trace_processor`.
