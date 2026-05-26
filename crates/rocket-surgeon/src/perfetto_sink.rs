//! Multi-rank Perfetto trace sink — per ADR-0010.
//!
//! Maps `rocket_surgeon`'s process topology onto Perfetto's track tree:
//!
//! ```text
//! [Process] daemon:rs-daemon   pid=daemon_pid   uuid=PROCESS|DAEMON
//! [Process] rank:N             pid=worker_N_pid uuid=PROCESS|N
//!   [Track] L{l}                                uuid=LAYER|N|l
//!     [Track] L{l}::{component}                 uuid=COMPONENT|N|l|c
//! ```
//!
//! Each worker rank gets its own Perfetto sequence (`sequence_id = 1000 + rank`)
//! with its own `InternTable`; the daemon has its own sequence (`999`). Track
//! UUIDs are bit-packed (`kind:4 | rank:12 | layer:16 | component:32`) so they
//! are deterministic, debuggable in hex, and structurally non-colliding within
//! 4096 ranks × 64K layers × 4B components.

#![allow(dead_code)]
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use perfetto_writer::intern::InternTable;
use perfetto_writer::proto::DebugAnnotation;
use perfetto_writer::writer::{TraceWriter, WriteError};
use rocket_surgeon_protocol::messages::ProbeFiredEvent;
use rocket_surgeon_protocol::types::TickPosition;

// ── UUID + sequence scheme (ADR-0010 §1, §2, §3) ──────────────────────────

/// Reserved rank value identifying the daemon process. The rank field is 12
/// bits, so the all-ones sentinel cannot collide with any real worker rank
/// (max 4094 workers).
pub const DAEMON_RANK: u32 = 0xFFF;

/// Sequence id for daemon-originated events (replay divergence, session
/// lifecycle, errors). Sits just below worker sequences so the daemon's
/// `InternedData` packet appears first in the trace.
pub const DAEMON_SEQUENCE: u32 = 999;

/// Worker sequence ids start at this base.
pub const WORKER_SEQUENCE_BASE: u32 = 1000;

/// Maximum worker rank value (12-bit field minus the DAEMON sentinel).
pub const MAX_WORKER_RANK: u32 = 0xFFE;

#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    Process = 0x1,
    Layer = 0x2,
    Component = 0x3,
    Counter = 0x4,
}

/// Bit-pack a track UUID. `kind` occupies the top 4 bits, `rank` the next 12,
/// `layer` the next 16, `component` the bottom 32.
///
/// Panics if any field exceeds its allotted bits — these are invariants of the
/// scheme, not runtime conditions.
#[inline]
#[must_use]
pub fn track_uuid(kind: TrackKind, rank: u32, layer: u32, component: u32) -> u64 {
    assert!(rank <= 0xFFF, "rank overflow: {rank} exceeds 12 bits");
    assert!(layer <= 0xFFFF, "layer overflow: {layer} exceeds 16 bits");
    // component is u32, fits by construction
    ((kind as u64) << 60)
        | (u64::from(rank) << 48)
        | (u64::from(layer) << 32)
        | u64::from(component)
}

#[inline]
#[must_use]
pub fn process_uuid(rank: u32) -> u64 {
    track_uuid(TrackKind::Process, rank, 0, 0)
}

#[inline]
#[must_use]
pub fn layer_uuid(rank: u32, layer: u32) -> u64 {
    track_uuid(TrackKind::Layer, rank, layer, 0)
}

#[inline]
#[must_use]
pub fn component_uuid(rank: u32, layer: u32, component: u32) -> u64 {
    track_uuid(TrackKind::Component, rank, layer, component)
}

/// Sequence id for a given rank. `DAEMON_RANK` maps to `DAEMON_SEQUENCE`;
/// worker ranks map to `WORKER_SEQUENCE_BASE + rank`.
#[inline]
#[must_use]
pub fn sequence_id(rank: u32) -> u32 {
    if rank == DAEMON_RANK {
        DAEMON_SEQUENCE
    } else {
        WORKER_SEQUENCE_BASE + rank
    }
}

// ── Sink ──────────────────────────────────────────────────────────────────

struct OpenSlice {
    sequence: u32,
}

/// Per-rank state: a Perfetto sequence and its own intern table.
struct RankState {
    sequence: u32,
    intern: InternTable,
    /// Tracks whether `emit_interned_names` has been called for this rank's
    /// sequence yet. Used to choose between the initial `*_CLEARED` flag and
    /// subsequent incremental updates (the latter currently unused but
    /// scaffolded for when we start emitting late-bound names).
    declared: bool,
}

pub struct PerfettoSink {
    writer: TraceWriter<BufWriter<File>>,
    epoch: Instant,
    path: PathBuf,
    /// Component track UUID lookup, keyed by `(rank, layer, component_name)`.
    /// Closing M-4: distinct ranks declaring the same component name no longer
    /// collide.
    component_uuids: HashMap<(u32, u32, String), u64>,
    /// Per-rank component index counter (for `sibling_order_rank` and the
    /// component bit-field). Incremented on each new component declared under
    /// (rank, layer).
    component_index: HashMap<(u32, u32), u32>,
    /// Per-sequence intern tables (closing M-2). Daemon and each worker rank
    /// have their own iid namespace.
    ranks: HashMap<u32, RankState>,
    /// Open slices keyed by track UUID — value carries the originating
    /// sequence so `close()` can route `SLICE_END` to the right sequence.
    open_slices: HashMap<u64, OpenSlice>,
}

#[derive(Debug, thiserror::Error)]
pub enum PerfettoError {
    #[error("write error: {0}")]
    Write(#[from] WriteError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl PerfettoSink {
    /// Create a sink writing `{dir}/{session_id}.pftrace`, declaring the daemon
    /// `ProcessDescriptor` up front. Worker ranks must be declared separately via
    /// `declare_process(rank, pid, ...)` once the orchestrator returns them.
    pub fn create(
        dir: &Path,
        session_id: &str,
        model_name: &str,
        daemon_pid: u32,
        epoch: Instant,
    ) -> Result<Self, PerfettoError> {
        let path = dir.join(format!("{session_id}.pftrace"));
        let file = File::create(&path)?;
        let buf_writer = BufWriter::with_capacity(256 * 1024, file);
        let mut writer = TraceWriter::new(buf_writer);

        // Daemon ProcessDescriptor. The session name is the descriptor name so
        // the Perfetto UI groups everything under a single recognizable label;
        // the process_name carries the model id (mirroring the original spec).
        writer.write_process_track(
            process_uuid(DAEMON_RANK),
            &format!("daemon:{session_id}"),
            daemon_pid_as_i32(daemon_pid),
            model_name,
        )?;

        Ok(Self {
            writer,
            epoch,
            path,
            component_uuids: HashMap::new(),
            component_index: HashMap::new(),
            ranks: HashMap::new(),
            open_slices: HashMap::new(),
        })
    }

    /// Declare a worker rank as its own `ProcessDescriptor`. Idempotent — calling
    /// twice on the same rank is a no-op (after the first declaration).
    pub fn declare_process(
        &mut self,
        rank: u32,
        worker_pid: u32,
        display_name: &str,
    ) -> Result<(), PerfettoError> {
        assert!(
            rank <= MAX_WORKER_RANK,
            "worker rank {rank} exceeds MAX_WORKER_RANK ({MAX_WORKER_RANK})"
        );
        if self.ranks.contains_key(&sequence_id(rank)) {
            return Ok(());
        }
        self.writer.write_process_track(
            process_uuid(rank),
            display_name,
            daemon_pid_as_i32(worker_pid),
            display_name,
        )?;
        self.ranks.insert(
            sequence_id(rank),
            RankState {
                sequence: sequence_id(rank),
                intern: InternTable::new(),
                declared: false,
            },
        );
        Ok(())
    }

    pub fn declare_layer(&mut self, rank: u32, layer: u32) -> Result<(), PerfettoError> {
        let name = format!("L{layer}");
        self.writer.write_track(
            layer_uuid(rank, layer),
            process_uuid(rank),
            &name,
            saturating_i32(layer),
        )?;
        Ok(())
    }

    pub fn declare_component(
        &mut self,
        rank: u32,
        layer: u32,
        component_name: &str,
    ) -> Result<u64, PerfettoError> {
        let key = (rank, layer, component_name.to_owned());
        if let Some(&uuid) = self.component_uuids.get(&key) {
            return Ok(uuid);
        }
        let component_index = *self
            .component_index
            .entry((rank, layer))
            .and_modify(|i| *i += 1)
            .or_insert(0);
        let uuid = component_uuid(rank, layer, component_index);
        let display_name = format!("L{layer}::{component_name}");
        self.writer.write_track(
            uuid,
            layer_uuid(rank, layer),
            &display_name,
            saturating_i32(component_index),
        )?;
        // Pre-intern the display name on this rank's sequence so subsequent
        // slice events can reference the iid without an extra declare step.
        self.ensure_rank(rank).intern.intern(&display_name);
        self.component_uuids.insert(key, uuid);
        Ok(uuid)
    }

    /// Emit `InternedData` for every name interned on `rank`'s sequence. Should
    /// be called once per rank after all component declarations and before any
    /// `SLICE_BEGIN` packets that reference iids. Per ADR-0010 §3, every sequence
    /// has its own intern table.
    pub fn emit_interned_names(&mut self, rank: u32) -> Result<(), PerfettoError> {
        let seq = sequence_id(rank);
        // Snapshot the entries into owned strings so the borrow on `self.ranks`
        // ends before we touch `self.writer`. The intern table is small
        // (component names per rank) so the clone is negligible.
        let pairs: Vec<(u64, String)> = {
            let state = self.ensure_rank(rank);
            state
                .intern
                .entries()
                .map(|(iid, name)| (iid, name.to_owned()))
                .collect()
        };
        if pairs.is_empty() {
            return Ok(());
        }
        let refs: Vec<(u64, &str)> = pairs
            .iter()
            .map(|(iid, name)| (*iid, name.as_str()))
            .collect();
        self.writer.write_interned_names(seq, &refs)?;
        if let Some(state) = self.ranks.get_mut(&seq) {
            state.declared = true;
        }
        Ok(())
    }

    pub fn on_tick_stopped(&mut self, position: &TickPosition) -> Result<(), PerfettoError> {
        let now_ns = self.epoch.elapsed().as_nanos() as u64;
        let rank = position.rank.unwrap_or(0);
        let track = self.resolve_or_create_component(rank, position.layer, &position.component)?;
        let display_name = format!("L{}::{}", position.layer, position.component);

        let seq = sequence_id(rank);

        if let Some(open) = self.open_slices.remove(&track) {
            self.writer.slice_end(open.sequence, track, now_ns)?;
        }

        let name_iid = self.ensure_rank(rank).intern.intern(&display_name);
        self.writer.slice_begin(seq, track, now_ns, name_iid)?;
        self.open_slices.insert(track, OpenSlice { sequence: seq });

        Ok(())
    }

    pub fn on_probe_fired(&mut self, event: &ProbeFiredEvent) -> Result<(), PerfettoError> {
        let now_ns = self.epoch.elapsed().as_nanos() as u64;
        let rank = event.rank;
        let seq = sequence_id(rank);
        // Lazily materialize the rank state if a probe fires on a rank we
        // never explicitly declared. This keeps single-rank traces working
        // when no `declare_process` was made for rank 0.
        let _ = self.ensure_rank(rank);

        let mut annotations = Vec::new();
        if let Some(ref summary) = event.tensor_summary {
            annotations.push(DebugAnnotation {
                name: Some("shape".into()),
                string_value: Some(format!("{:?}", summary.shape)),
                ..DebugAnnotation::default()
            });
            annotations.push(DebugAnnotation {
                name: Some("mean".into()),
                double_value: Some(summary.stats.mean),
                ..DebugAnnotation::default()
            });
            annotations.push(DebugAnnotation {
                name: Some("std".into()),
                double_value: Some(summary.stats.std),
                ..DebugAnnotation::default()
            });
            annotations.push(DebugAnnotation {
                name: Some("l2_norm".into()),
                double_value: Some(summary.stats.l2_norm),
                ..DebugAnnotation::default()
            });
        }

        let probe_name = format!("probe:{}", event.probe_id);
        let track = self
            .find_track_for_point(rank, &event.point)
            .unwrap_or_else(|| process_uuid(rank));
        self.writer
            .instant(seq, track, now_ns, &probe_name, &annotations)?;
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), PerfettoError> {
        let now_ns = self.epoch.elapsed().as_nanos() as u64;
        let open: Vec<(u64, u32)> = self
            .open_slices
            .drain()
            .map(|(track, slice)| (track, slice.sequence))
            .collect();
        for (track, seq) in open {
            self.writer.slice_end(seq, track, now_ns)?;
        }
        self.writer.flush()?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn ensure_rank(&mut self, rank: u32) -> &mut RankState {
        let seq = sequence_id(rank);
        self.ranks.entry(seq).or_insert_with(|| RankState {
            sequence: seq,
            intern: InternTable::new(),
            declared: false,
        })
    }

    fn resolve_or_create_component(
        &mut self,
        rank: u32,
        layer: u32,
        component_name: &str,
    ) -> Result<u64, PerfettoError> {
        if let Some(&uuid) = self
            .component_uuids
            .get(&(rank, layer, component_name.to_owned()))
        {
            return Ok(uuid);
        }
        self.declare_component(rank, layer, component_name)
    }

    /// Probe points arrive as model-qualified strings like
    /// `"llama:0:12:attn.o_proj:output"`. We need to find the closest declared
    /// component track on the originating rank. Exact key match if the caller
    /// passes `"L{layer}::{component}"`; otherwise we look for the rank-scoped
    /// component whose display name appears in the probe point. This is a
    /// best-effort routing — unknown points fall back to the rank's process
    /// track in `on_probe_fired`.
    fn find_track_for_point(&self, rank: u32, point: &str) -> Option<u64> {
        let mut best: Option<(usize, u64)> = None;
        for ((r, _l, name), &uuid) in &self.component_uuids {
            if *r != rank {
                continue;
            }
            let display = format!("::{name}");
            if point.contains(&display) || point.contains(name) {
                let score = name.len();
                if best.is_none_or(|(prev, _)| score > prev) {
                    best = Some((score, uuid));
                }
            }
        }
        best.map(|(_, uuid)| uuid)
    }
}

#[inline]
fn saturating_i32(v: u32) -> i32 {
    v.min(i32::MAX as u32) as i32
}

#[inline]
fn daemon_pid_as_i32(pid: u32) -> i32 {
    saturating_i32(pid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use perfetto_writer::proto::{self, TracePacket};
    use prost::Message;
    use rocket_surgeon_protocol::types::{
        DType, Histogram, Phase, ProbeAction, StepDirection, TensorStats, TensorSummary, TickEvent,
    };
    use tempfile::TempDir;

    const DAEMON_TEST_PID: u32 = 1;

    fn decode_varint(buf: &[u8]) -> (u64, usize) {
        let mut value: u64 = 0;
        let mut shift = 0;
        for (i, &byte) in buf.iter().enumerate() {
            value |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return (value, i + 1);
            }
            shift += 7;
        }
        panic!("truncated varint");
    }

    fn decode_trace_packets(data: &[u8]) -> Vec<TracePacket> {
        let mut packets = Vec::new();
        let mut offset = 0;
        while offset < data.len() {
            assert_eq!(data[offset], 0x0A);
            offset += 1;
            let (len, consumed) = decode_varint(&data[offset..]);
            offset += consumed;
            let packet = TracePacket::decode(&data[offset..offset + len as usize])
                .expect("valid TracePacket");
            packets.push(packet);
            offset += len as usize;
        }
        packets
    }

    fn make_position(rank: u32, layer: u32, component: &str) -> TickPosition {
        TickPosition {
            tick_id: 1,
            direction: StepDirection::Forward,
            rank: Some(rank),
            layer,
            component: component.to_owned(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        }
    }

    fn make_probe_event(rank: u32, probe_id: &str, point: &str) -> ProbeFiredEvent {
        ProbeFiredEvent {
            probe_id: probe_id.to_owned(),
            point: point.to_owned(),
            tick_id: 1,
            tensor_summary: Some(TensorSummary {
                tensor_id: "abc".to_owned(),
                shape: vec![32, 16, 128],
                dtype: DType::Float32,
                device: "cpu".to_owned(),
                sharding: None,
                stats: TensorStats {
                    mean: 0.5,
                    std: 0.1,
                    min: -1.0,
                    max: 2.0,
                    abs_max: 2.0,
                    sparsity: 0.0,
                    l2_norm: 1.234,
                    histogram: Histogram {
                        bins: 0,
                        edges: vec![],
                        counts: vec![],
                    },
                },
                top_k: vec![],
            }),
            action: ProbeAction::Capture,
            timestamp: "2026-05-19T00:00:00Z".to_owned(),
            rank,
        }
    }

    // ── UUID scheme invariants ────────────────────────────────────────

    #[test]
    fn process_uuids_distinct_per_rank() {
        assert_ne!(process_uuid(0), process_uuid(1));
        assert_ne!(process_uuid(0), process_uuid(DAEMON_RANK));
    }

    #[test]
    fn same_component_on_different_ranks_yields_distinct_uuids() {
        // The bug M-4 was reported as fixing exactly this: rank 0 and rank 1
        // declaring `L0::attn::q_proj` at component index 0 must not alias.
        assert_ne!(component_uuid(0, 0, 0), component_uuid(1, 0, 0));
        assert_ne!(component_uuid(0, 5, 12), component_uuid(1, 5, 12));
    }

    #[test]
    fn track_uuid_bit_packing_is_lossless() {
        let uuid = component_uuid(0xABC, 0x1234, 0xCAFE_BABE);
        let kind = (uuid >> 60) & 0xF;
        let rank = (uuid >> 48) & 0xFFF;
        let layer = (uuid >> 32) & 0xFFFF;
        let comp = uuid & 0xFFFF_FFFF;
        assert_eq!(kind, TrackKind::Component as u64);
        assert_eq!(rank, 0xABC);
        assert_eq!(layer, 0x1234);
        assert_eq!(comp, 0xCAFE_BABE);
    }

    #[test]
    fn sequence_ids_isolate_daemon_from_workers() {
        assert_eq!(sequence_id(DAEMON_RANK), DAEMON_SEQUENCE);
        assert_eq!(sequence_id(0), WORKER_SEQUENCE_BASE);
        assert_eq!(sequence_id(7), WORKER_SEQUENCE_BASE + 7);
        assert_ne!(sequence_id(DAEMON_RANK), sequence_id(0));
    }

    #[test]
    #[should_panic(expected = "rank overflow")]
    fn track_uuid_panics_on_rank_overflow() {
        let _ = track_uuid(TrackKind::Process, 0x1000, 0, 0);
    }

    // ── Single-rank baseline (compat with original behaviour) ─────────

    #[test]
    fn create_writes_daemon_process_track() {
        let dir = TempDir::new().unwrap();
        let sink = PerfettoSink::create(
            dir.path(),
            "test-session",
            "gpt2",
            DAEMON_TEST_PID,
            Instant::now(),
        )
        .unwrap();
        let path = sink.path().to_owned();
        drop(sink);

        let data = std::fs::read(&path).unwrap();
        let packets = decode_trace_packets(&data);
        assert_eq!(packets.len(), 1);
        let td = packets[0].track_descriptor.as_ref().unwrap();
        assert_eq!(td.uuid, Some(process_uuid(DAEMON_RANK)));
        assert!(td.process.is_some());
        assert_eq!(td.name.as_deref(), Some("daemon:test-session"));
    }

    #[test]
    fn on_tick_stopped_emits_slice_events() {
        let dir = TempDir::new().unwrap();
        let mut sink = PerfettoSink::create(
            dir.path(),
            "test-session",
            "gpt2",
            DAEMON_TEST_PID,
            Instant::now(),
        )
        .unwrap();

        let pos = make_position(0, 0, "attn::q_proj");
        sink.on_tick_stopped(&pos).unwrap();
        sink.on_tick_stopped(&pos).unwrap();
        let path = sink.path().to_owned();
        sink.close().unwrap();

        let data = std::fs::read(&path).unwrap();
        let packets = decode_trace_packets(&data);
        let event_count = packets.iter().filter(|p| p.track_event.is_some()).count();
        assert!(event_count >= 3);
    }

    #[test]
    fn on_probe_fired_emits_instant_on_correct_sequence() {
        let dir = TempDir::new().unwrap();
        let mut sink = PerfettoSink::create(
            dir.path(),
            "test-session",
            "gpt2",
            DAEMON_TEST_PID,
            Instant::now(),
        )
        .unwrap();
        sink.declare_process(0, 100, "rank:0").unwrap();
        sink.declare_layer(0, 0).unwrap();
        sink.declare_component(0, 0, "attn::q_proj").unwrap();

        let probe = make_probe_event(0, "probe1", "L0::attn::q_proj");
        sink.on_probe_fired(&probe).unwrap();
        let path = sink.path().to_owned();
        sink.close().unwrap();

        let data = std::fs::read(&path).unwrap();
        let packets = decode_trace_packets(&data);
        let instant = packets
            .iter()
            .find(|p| {
                p.track_event
                    .as_ref()
                    .is_some_and(|ev| ev.r#type == Some(proto::TYPE_INSTANT))
            })
            .expect("expected an instant event");
        let ev = instant.track_event.as_ref().unwrap();
        assert_eq!(ev.name.as_deref(), Some("probe:probe1"));
        assert_eq!(instant.trusted_packet_sequence_id, Some(sequence_id(0)));
        assert!(!ev.debug_annotation.is_empty());
    }

    #[test]
    fn close_ends_all_open_slices() {
        let dir = TempDir::new().unwrap();
        let mut sink = PerfettoSink::create(
            dir.path(),
            "test-session",
            "gpt2",
            DAEMON_TEST_PID,
            Instant::now(),
        )
        .unwrap();

        let pos = make_position(0, 0, "attn::q_proj");
        sink.on_tick_stopped(&pos).unwrap();
        let path = sink.path().to_owned();
        sink.close().unwrap();

        let data = std::fs::read(&path).unwrap();
        let packets = decode_trace_packets(&data);
        let slice_end_count = packets
            .iter()
            .filter(|p| {
                p.track_event
                    .as_ref()
                    .is_some_and(|ev| ev.r#type == Some(proto::TYPE_SLICE_END))
            })
            .count();
        assert_eq!(slice_end_count, 1);
    }

    // ── Multi-rank correctness (the BEAD-0010 fix) ────────────────────

    #[test]
    fn same_component_name_on_two_ranks_emits_distinct_tracks() {
        let dir = TempDir::new().unwrap();
        let mut sink = PerfettoSink::create(
            dir.path(),
            "test-session",
            "gpt2",
            DAEMON_TEST_PID,
            Instant::now(),
        )
        .unwrap();
        sink.declare_process(0, 100, "rank:0").unwrap();
        sink.declare_process(1, 101, "rank:1").unwrap();
        sink.declare_layer(0, 0).unwrap();
        sink.declare_layer(1, 0).unwrap();
        let r0 = sink.declare_component(0, 0, "attn::q_proj").unwrap();
        let r1 = sink.declare_component(1, 0, "attn::q_proj").unwrap();
        assert_ne!(
            r0, r1,
            "M-4: same component name on different ranks must get distinct UUIDs"
        );
    }

    #[test]
    fn tick_events_route_to_their_originating_ranks_sequence() {
        let dir = TempDir::new().unwrap();
        let mut sink = PerfettoSink::create(
            dir.path(),
            "test-session",
            "gpt2",
            DAEMON_TEST_PID,
            Instant::now(),
        )
        .unwrap();
        sink.declare_process(0, 100, "rank:0").unwrap();
        sink.declare_process(1, 101, "rank:1").unwrap();
        sink.declare_layer(0, 0).unwrap();
        sink.declare_layer(1, 0).unwrap();
        sink.declare_component(0, 0, "attn::q_proj").unwrap();
        sink.declare_component(1, 0, "attn::q_proj").unwrap();

        sink.on_tick_stopped(&make_position(0, 0, "attn::q_proj"))
            .unwrap();
        sink.on_tick_stopped(&make_position(1, 0, "attn::q_proj"))
            .unwrap();
        let path = sink.path().to_owned();
        sink.close().unwrap();

        let data = std::fs::read(&path).unwrap();
        let packets = decode_trace_packets(&data);

        // Find SLICE_BEGIN packets and group by sequence.
        let mut by_sequence: HashMap<u32, Vec<&TracePacket>> = HashMap::new();
        for p in &packets {
            if let Some(ev) = p.track_event.as_ref()
                && ev.r#type == Some(proto::TYPE_SLICE_BEGIN)
                && let Some(seq) = p.trusted_packet_sequence_id
            {
                by_sequence.entry(seq).or_default().push(p);
            }
        }
        let rank0_begins = by_sequence.get(&sequence_id(0)).map_or(0, Vec::len);
        let rank1_begins = by_sequence.get(&sequence_id(1)).map_or(0, Vec::len);
        assert!(
            rank0_begins >= 1,
            "rank-0 SLICE_BEGIN must land on sequence {} (got {} on it)",
            sequence_id(0),
            rank0_begins,
        );
        assert!(
            rank1_begins >= 1,
            "rank-1 SLICE_BEGIN must land on sequence {} (got {} on it)",
            sequence_id(1),
            rank1_begins,
        );
    }

    #[test]
    fn intern_tables_are_partitioned_per_sequence() {
        // M-2: each sequence's InternedData packet is independent. Two ranks
        // declaring the same component name end up with iid=1 *each*, but the
        // mapping iid→name is scoped to its own sequence.
        let dir = TempDir::new().unwrap();
        let mut sink = PerfettoSink::create(
            dir.path(),
            "test-session",
            "gpt2",
            DAEMON_TEST_PID,
            Instant::now(),
        )
        .unwrap();
        sink.declare_process(0, 100, "rank:0").unwrap();
        sink.declare_process(1, 101, "rank:1").unwrap();
        sink.declare_layer(0, 0).unwrap();
        sink.declare_layer(1, 0).unwrap();
        sink.declare_component(0, 0, "attn::q_proj").unwrap();
        sink.declare_component(0, 0, "attn::k_proj").unwrap();
        sink.declare_component(1, 0, "mlp::gate").unwrap();
        sink.emit_interned_names(0).unwrap();
        sink.emit_interned_names(1).unwrap();
        let path = sink.path().to_owned();
        sink.close().unwrap();

        let data = std::fs::read(&path).unwrap();
        let packets = decode_trace_packets(&data);

        let mut interned_by_sequence: HashMap<u32, Vec<String>> = HashMap::new();
        for p in &packets {
            if let Some(ref id) = p.interned_data
                && let Some(seq) = p.trusted_packet_sequence_id
            {
                for ev_name in &id.event_names {
                    if let Some(ref name) = ev_name.name {
                        interned_by_sequence
                            .entry(seq)
                            .or_default()
                            .push(name.clone());
                    }
                }
            }
        }

        let r0 = interned_by_sequence
            .get(&sequence_id(0))
            .expect("rank-0 sequence must have interned names");
        let r1 = interned_by_sequence
            .get(&sequence_id(1))
            .expect("rank-1 sequence must have interned names");
        assert!(
            r0.iter().any(|n| n == "L0::attn::q_proj"),
            "rank-0 sequence missing L0::attn::q_proj, got {r0:?}",
        );
        assert!(
            r0.iter().any(|n| n == "L0::attn::k_proj"),
            "rank-0 sequence missing L0::attn::k_proj, got {r0:?}",
        );
        assert!(
            r1.iter().any(|n| n == "L0::mlp::gate"),
            "rank-1 sequence missing L0::mlp::gate, got {r1:?}",
        );
        // Crucially: rank-0's names should NOT appear on rank-1's sequence,
        // and vice versa. That's the per-sequence partition.
        assert!(
            !r1.iter().any(|n| n == "L0::attn::q_proj"),
            "rank-1 sequence leaked rank-0 names: {r1:?}",
        );
    }

    #[test]
    fn probe_event_lands_on_its_event_rank() {
        // C-2: probe.fired now carries `rank`, and the sink uses it for sequence routing.
        let dir = TempDir::new().unwrap();
        let mut sink = PerfettoSink::create(
            dir.path(),
            "test-session",
            "gpt2",
            DAEMON_TEST_PID,
            Instant::now(),
        )
        .unwrap();
        sink.declare_process(2, 102, "rank:2").unwrap();

        let probe = make_probe_event(2, "p1", "L0::attn::q_proj");
        sink.on_probe_fired(&probe).unwrap();
        let path = sink.path().to_owned();
        sink.close().unwrap();

        let data = std::fs::read(&path).unwrap();
        let packets = decode_trace_packets(&data);
        let instant = packets
            .iter()
            .find(|p| {
                p.track_event
                    .as_ref()
                    .is_some_and(|ev| ev.r#type == Some(proto::TYPE_INSTANT))
            })
            .expect("expected an instant event");
        assert_eq!(instant.trusted_packet_sequence_id, Some(sequence_id(2)));
    }

    #[test]
    fn output_file_is_valid_perfetto_trace() {
        let dir = TempDir::new().unwrap();
        let mut sink = PerfettoSink::create(
            dir.path(),
            "test-session",
            "gpt2",
            DAEMON_TEST_PID,
            Instant::now(),
        )
        .unwrap();

        sink.declare_process(0, 100, "rank:0").unwrap();
        sink.declare_layer(0, 0).unwrap();
        sink.declare_component(0, 0, "attn::q_proj").unwrap();
        sink.emit_interned_names(0).unwrap();

        let pos = make_position(0, 0, "attn::q_proj");
        sink.on_tick_stopped(&pos).unwrap();
        sink.on_tick_stopped(&pos).unwrap();

        let probe = make_probe_event(0, "p1", "L0::attn::q_proj");
        sink.on_probe_fired(&probe).unwrap();

        let path = sink.path().to_owned();
        sink.close().unwrap();

        let data = std::fs::read(&path).unwrap();
        let packets = decode_trace_packets(&data);
        assert!(packets.len() >= 7);

        for packet in &packets {
            let mut rebuf = Vec::new();
            packet.encode(&mut rebuf).unwrap();
            let redecoded = TracePacket::decode(rebuf.as_slice()).unwrap();
            assert_eq!(&redecoded, packet);
        }
    }

    fn find_traceconv() -> Option<std::path::PathBuf> {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
        let wrapper = std::path::Path::new(&manifest)
            .parent()?
            .parent()?
            .join("quarantine/perfetto/tools/traceconv");
        wrapper.is_file().then_some(wrapper)
    }

    #[test]
    fn traceconv_validates_multi_rank_output() {
        let Some(traceconv) = find_traceconv() else {
            eprintln!("skipping: quarantine/perfetto/tools/traceconv not found");
            return;
        };

        let dir = TempDir::new().unwrap();
        let mut sink = PerfettoSink::create(
            dir.path(),
            "traceconv-test",
            "gpt2",
            DAEMON_TEST_PID,
            Instant::now(),
        )
        .unwrap();

        sink.declare_process(0, 100, "rank:0").unwrap();
        sink.declare_process(1, 101, "rank:1").unwrap();
        sink.declare_layer(0, 0).unwrap();
        sink.declare_layer(0, 1).unwrap();
        sink.declare_layer(1, 0).unwrap();
        let r0_q = sink.declare_component(0, 0, "attn::q_proj").unwrap();
        sink.declare_component(0, 0, "attn::k_proj").unwrap();
        sink.declare_component(0, 1, "mlp::gate").unwrap();
        let r1_q = sink.declare_component(1, 0, "attn::q_proj").unwrap();
        assert_ne!(
            r0_q, r1_q,
            "M-4 regression: same name on different ranks must not alias"
        );
        sink.emit_interned_names(0).unwrap();
        sink.emit_interned_names(1).unwrap();

        for _ in 0..3 {
            sink.on_tick_stopped(&make_position(0, 0, "attn::q_proj"))
                .unwrap();
            sink.on_tick_stopped(&make_position(1, 0, "attn::q_proj"))
                .unwrap();
        }

        sink.on_probe_fired(&make_probe_event(0, "p1", "L0::attn::q_proj"))
            .unwrap();
        sink.on_probe_fired(&make_probe_event(1, "p2", "L0::attn::q_proj"))
            .unwrap();

        let path = sink.path().to_owned();
        sink.close().unwrap();

        let text_out = dir.path().join("trace.textproto");
        let output = std::process::Command::new("python3")
            .arg(&traceconv)
            .arg("text")
            .arg(&path)
            .arg(&text_out)
            .output()
            .expect("failed to run traceconv");

        assert!(
            output.status.success(),
            "traceconv failed (exit {}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );

        let text = std::fs::read_to_string(&text_out).expect("traceconv output file missing");
        assert!(!text.is_empty(), "traceconv produced empty output");

        assert!(
            text.contains("track_descriptor"),
            "missing track_descriptor"
        );
        assert!(text.contains("track_event"), "missing track_event");

        assert!(
            text.contains("daemon:traceconv-test"),
            "missing daemon process track"
        );
        assert!(text.contains("rank:0"), "missing rank:0 process track");
        assert!(text.contains("rank:1"), "missing rank:1 process track");
        assert!(
            text.contains("L0::attn::q_proj"),
            "missing component L0::attn::q_proj"
        );
        assert!(
            text.contains("L0::attn::k_proj"),
            "missing component L0::attn::k_proj"
        );
        assert!(
            text.contains("L1::mlp::gate"),
            "missing component L1::mlp::gate"
        );

        assert!(
            text.contains("type: TYPE_SLICE_BEGIN"),
            "missing SLICE_BEGIN events"
        );
        assert!(
            text.contains("type: TYPE_SLICE_END"),
            "missing SLICE_END events"
        );
        assert!(
            text.contains("type: TYPE_INSTANT"),
            "missing INSTANT events"
        );
        assert!(text.contains("probe:p1"), "missing probe:p1 instant");
        assert!(text.contains("probe:p2"), "missing probe:p2 instant");

        assert!(
            text.contains("debug_annotation"),
            "missing debug_annotation on probe instants"
        );
    }
}
