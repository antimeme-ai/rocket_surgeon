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

const PROCESS_UUID: u64 = 1;
const SEQUENCE_BASE: u32 = 1000;

fn rank_uuid(rank: u32) -> u64 {
    100 + u64::from(rank)
}

fn layer_uuid(rank: u32, layer: u32) -> u64 {
    1000 + u64::from(rank) * 1000 + u64::from(layer)
}

fn component_uuid(rank: u32, layer: u32, component_index: u32) -> u64 {
    10000 + u64::from(rank) * 100_000 + u64::from(layer) * 100 + u64::from(component_index)
}

fn sequence_id(rank: u32) -> u32 {
    SEQUENCE_BASE + rank
}

struct OpenSlice {
    rank: u32,
}

pub struct PerfettoSink {
    writer: TraceWriter<BufWriter<File>>,
    intern: InternTable,
    epoch: Instant,
    path: PathBuf,
    component_uuids: HashMap<String, u64>,
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
    pub fn create(
        dir: &Path,
        session_id: &str,
        model_name: &str,
        epoch: Instant,
    ) -> Result<Self, PerfettoError> {
        let path = dir.join(format!("{session_id}.pftrace"));
        let file = File::create(&path)?;
        let buf_writer = BufWriter::with_capacity(256 * 1024, file);
        let mut writer = TraceWriter::new(buf_writer);

        writer.write_process_track(PROCESS_UUID, session_id, 1, model_name)?;

        Ok(Self {
            writer,
            intern: InternTable::new(),
            epoch,
            path,
            component_uuids: HashMap::new(),
            open_slices: HashMap::new(),
        })
    }

    pub fn declare_rank(&mut self, rank: u32) -> Result<(), PerfettoError> {
        let name = format!("rank:{rank}");
        self.writer
            .write_thread_track(rank_uuid(rank), PROCESS_UUID, &name, 1, i64::from(rank))?;
        Ok(())
    }

    pub fn declare_layer(&mut self, rank: u32, layer: u32) -> Result<(), PerfettoError> {
        let name = format!("L{layer}");
        self.writer.write_track(
            layer_uuid(rank, layer),
            rank_uuid(rank),
            &name,
            layer.min(i32::MAX as u32) as i32,
        )?;
        Ok(())
    }

    pub fn declare_component(
        &mut self,
        rank: u32,
        layer: u32,
        component_index: u32,
        component_name: &str,
    ) -> Result<(), PerfettoError> {
        let uuid = component_uuid(rank, layer, component_index);
        let display_name = format!("L{layer}::{component_name}");
        self.writer.write_track(
            uuid,
            layer_uuid(rank, layer),
            &display_name,
            component_index.min(i32::MAX as u32) as i32,
        )?;
        self.intern.intern(&display_name);
        self.component_uuids.insert(display_name, uuid);
        Ok(())
    }

    pub fn emit_interned_names(&mut self, rank: u32) -> Result<(), PerfettoError> {
        let refs: Vec<(u64, &str)> = self.intern.entries().collect();
        if !refs.is_empty() {
            self.writer.write_interned_names(sequence_id(rank), &refs)?;
        }
        Ok(())
    }

    pub fn on_tick_stopped(&mut self, position: &TickPosition) -> Result<(), PerfettoError> {
        let now_ns = self.epoch.elapsed().as_nanos() as u64;
        let rank = position.rank.unwrap_or(0);
        let display_name = format!("L{}::{}", position.layer, position.component);
        let track = self.resolve_or_create_track(rank, position.layer, &display_name)?;
        let seq = sequence_id(rank);

        if self.open_slices.remove(&track).is_some() {
            self.writer.slice_end(seq, track, now_ns)?;
        }

        let name_iid = self.intern.intern(&display_name);
        self.writer.slice_begin(seq, track, now_ns, name_iid)?;
        self.open_slices.insert(track, OpenSlice { rank });

        Ok(())
    }

    pub fn on_probe_fired(&mut self, event: &ProbeFiredEvent) -> Result<(), PerfettoError> {
        let now_ns = self.epoch.elapsed().as_nanos() as u64;
        let rank = 0u32;
        let seq = sequence_id(rank);

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

        let track = self.find_track_for_point(&event.point);
        self.writer
            .instant(seq, track, now_ns, &probe_name, &annotations)?;

        Ok(())
    }

    pub fn close(&mut self) -> Result<(), PerfettoError> {
        let now_ns = self.epoch.elapsed().as_nanos() as u64;
        let open: Vec<(u64, u32)> = self
            .open_slices
            .drain()
            .map(|(track, slice)| (track, slice.rank))
            .collect();
        for (track, rank) in open {
            self.writer.slice_end(sequence_id(rank), track, now_ns)?;
        }
        self.writer.flush()?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn resolve_or_create_track(
        &mut self,
        rank: u32,
        layer: u32,
        display_name: &str,
    ) -> Result<u64, PerfettoError> {
        if let Some(&uuid) = self.component_uuids.get(display_name) {
            return Ok(uuid);
        }
        let component_index = self.component_uuids.len() as u32;
        let uuid = component_uuid(rank, layer, component_index);
        self.writer.write_track(
            uuid,
            layer_uuid(rank, layer),
            display_name,
            component_index.min(i32::MAX as u32) as i32,
        )?;
        self.intern.intern(display_name);
        self.component_uuids.insert(display_name.to_owned(), uuid);
        Ok(uuid)
    }

    fn find_track_for_point(&self, point: &str) -> u64 {
        if let Some(&uuid) = self.component_uuids.get(point) {
            return uuid;
        }
        let mut best: Option<(&str, u64)> = None;
        for (name, &uuid) in &self.component_uuids {
            if point.contains(name.as_str()) || name.contains(point) {
                let is_longer = best.is_none_or(|(prev, _)| name.len() > prev.len());
                if is_longer {
                    best = Some((name, uuid));
                }
            }
        }
        best.map_or_else(|| rank_uuid(0), |(_, uuid)| uuid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use perfetto_writer::proto::{self, TracePacket};
    use prost::Message;
    use rocket_surgeon_protocol::types::{
        DType, Histogram, ProbeAction, StepDirection, TensorStats, TensorSummary, TickEvent,
    };
    use tempfile::TempDir;

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

    fn make_position(layer: u32, component: &str) -> TickPosition {
        TickPosition {
            tick_id: 1,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer,
            component: component.to_owned(),
            event: TickEvent::Output,
            replay_of: None,
        }
    }

    fn make_probe_event(probe_id: &str) -> ProbeFiredEvent {
        ProbeFiredEvent {
            probe_id: probe_id.to_owned(),
            point: "L0::attn::q_proj".to_owned(),
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
        }
    }

    #[test]
    fn create_writes_process_track() {
        let dir = TempDir::new().unwrap();
        let sink =
            PerfettoSink::create(dir.path(), "test-session", "gpt2", Instant::now()).unwrap();
        let path = sink.path().to_owned();
        drop(sink);

        let data = std::fs::read(&path).unwrap();
        let packets = decode_trace_packets(&data);
        assert_eq!(packets.len(), 1);
        let td = packets[0].track_descriptor.as_ref().unwrap();
        assert_eq!(td.uuid, Some(PROCESS_UUID));
        assert!(td.process.is_some());
    }

    #[test]
    fn on_tick_stopped_emits_slice_events() {
        let dir = TempDir::new().unwrap();
        let mut sink =
            PerfettoSink::create(dir.path(), "test-session", "gpt2", Instant::now()).unwrap();

        let pos = make_position(0, "attn::q_proj");
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
    fn on_probe_fired_emits_instant() {
        let dir = TempDir::new().unwrap();
        let mut sink =
            PerfettoSink::create(dir.path(), "test-session", "gpt2", Instant::now()).unwrap();

        let pos = make_position(0, "attn::q_proj");
        sink.on_tick_stopped(&pos).unwrap();

        let probe = make_probe_event("probe1");
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
        assert!(!ev.debug_annotation.is_empty());
    }

    #[test]
    fn close_ends_all_open_slices() {
        let dir = TempDir::new().unwrap();
        let mut sink =
            PerfettoSink::create(dir.path(), "test-session", "gpt2", Instant::now()).unwrap();

        let pos = make_position(0, "attn::q_proj");
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

    #[test]
    fn output_file_is_valid_perfetto_trace() {
        let dir = TempDir::new().unwrap();
        let mut sink =
            PerfettoSink::create(dir.path(), "test-session", "gpt2", Instant::now()).unwrap();

        sink.declare_rank(0).unwrap();
        sink.declare_layer(0, 0).unwrap();
        sink.declare_component(0, 0, 0, "attn::q_proj").unwrap();
        sink.emit_interned_names(0).unwrap();

        let pos = make_position(0, "attn::q_proj");
        sink.on_tick_stopped(&pos).unwrap();
        sink.on_tick_stopped(&pos).unwrap();

        let probe = make_probe_event("p1");
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
    fn traceconv_validates_output() {
        let Some(traceconv) = find_traceconv() else {
            eprintln!("skipping: quarantine/perfetto/tools/traceconv not found");
            return;
        };

        let dir = TempDir::new().unwrap();
        let mut sink =
            PerfettoSink::create(dir.path(), "traceconv-test", "gpt2", Instant::now()).unwrap();

        sink.declare_rank(0).unwrap();
        sink.declare_rank(1).unwrap();
        sink.declare_layer(0, 0).unwrap();
        sink.declare_layer(0, 1).unwrap();
        sink.declare_layer(1, 0).unwrap();
        sink.declare_component(0, 0, 0, "attn::q_proj").unwrap();
        sink.declare_component(0, 0, 1, "attn::k_proj").unwrap();
        sink.declare_component(0, 1, 0, "mlp::gate").unwrap();
        sink.declare_component(1, 0, 0, "attn::q_proj").unwrap();
        sink.emit_interned_names(0).unwrap();
        sink.emit_interned_names(1).unwrap();

        for _ in 0..3 {
            sink.on_tick_stopped(&make_position(0, "attn::q_proj"))
                .unwrap();
            sink.on_tick_stopped(&make_position(1, "mlp::gate"))
                .unwrap();
        }

        sink.on_probe_fired(&make_probe_event("p1")).unwrap();
        sink.on_probe_fired(&make_probe_event("p2")).unwrap();

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

        assert!(text.contains("traceconv-test"), "missing session name");
        assert!(text.contains("rank:0"), "missing rank:0 track");
        assert!(text.contains("rank:1"), "missing rank:1 track");
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
