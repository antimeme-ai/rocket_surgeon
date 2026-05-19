use std::collections::HashSet;
use std::io::Write;

use prost::Message;

use crate::proto::{
    self, CounterDescriptor, DebugAnnotation, EventName, InternedData, ProcessDescriptor,
    ThreadDescriptor, TracePacket, TrackDescriptor, TrackEvent,
};
use crate::varint;

#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode error: {0}")]
    Encode(#[from] prost::EncodeError),
}

pub struct TraceWriter<W: Write> {
    writer: W,
    buf: Vec<u8>,
    seen_sequences: HashSet<u32>,
}

impl<W: Write> TraceWriter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            buf: Vec::with_capacity(4096),
            seen_sequences: HashSet::new(),
        }
    }

    pub fn write_packet(&mut self, packet: &TracePacket) -> Result<(), WriteError> {
        self.buf.clear();
        let payload_len = packet.encoded_len();
        varint::field1_tag_and_length(payload_len, &mut self.buf);
        packet.encode(&mut self.buf)?;
        self.writer.write_all(&self.buf)?;
        Ok(())
    }

    pub fn write_process_track(
        &mut self,
        uuid: u64,
        name: &str,
        pid: i32,
        process_name: &str,
    ) -> Result<(), WriteError> {
        self.write_packet(&TracePacket {
            track_descriptor: Some(TrackDescriptor {
                uuid: Some(uuid),
                name: Some(name.to_owned()),
                process: Some(ProcessDescriptor {
                    pid: Some(pid),
                    process_name: Some(process_name.to_owned()),
                    ..ProcessDescriptor::default()
                }),
                ..TrackDescriptor::default()
            }),
            ..TracePacket::default()
        })
    }

    pub fn write_thread_track(
        &mut self,
        uuid: u64,
        parent_uuid: u64,
        name: &str,
        pid: i32,
        tid: i64,
    ) -> Result<(), WriteError> {
        self.write_packet(&TracePacket {
            track_descriptor: Some(TrackDescriptor {
                uuid: Some(uuid),
                parent_uuid: Some(parent_uuid),
                name: Some(name.to_owned()),
                thread: Some(ThreadDescriptor {
                    pid: Some(pid),
                    tid: Some(tid),
                    thread_name: Some(name.to_owned()),
                }),
                ..TrackDescriptor::default()
            }),
            ..TracePacket::default()
        })
    }

    pub fn write_track(
        &mut self,
        uuid: u64,
        parent_uuid: u64,
        name: &str,
        order: i32,
    ) -> Result<(), WriteError> {
        self.write_packet(&TracePacket {
            track_descriptor: Some(TrackDescriptor {
                uuid: Some(uuid),
                parent_uuid: Some(parent_uuid),
                name: Some(name.to_owned()),
                child_ordering: Some(proto::CHILD_ORDERING_EXPLICIT),
                sibling_order_rank: Some(order),
                ..TrackDescriptor::default()
            }),
            ..TracePacket::default()
        })
    }

    pub fn write_counter_track(
        &mut self,
        uuid: u64,
        parent_uuid: u64,
        name: &str,
        unit_name: &str,
    ) -> Result<(), WriteError> {
        self.write_packet(&TracePacket {
            track_descriptor: Some(TrackDescriptor {
                uuid: Some(uuid),
                parent_uuid: Some(parent_uuid),
                name: Some(name.to_owned()),
                counter: Some(CounterDescriptor {
                    unit_name: Some(unit_name.to_owned()),
                    ..CounterDescriptor::default()
                }),
                ..TrackDescriptor::default()
            }),
            ..TracePacket::default()
        })
    }

    pub fn write_interned_names(
        &mut self,
        sequence_id: u32,
        names: &[(u64, &str)],
    ) -> Result<(), WriteError> {
        let first = self.seen_sequences.insert(sequence_id);
        self.write_packet(&TracePacket {
            timestamp_clock_id: Some(proto::CLOCK_MONOTONIC),
            trusted_packet_sequence_id: Some(sequence_id),
            sequence_flags: Some(proto::SEQ_INCREMENTAL_STATE_CLEARED),
            first_packet_on_sequence: Some(first),
            interned_data: Some(InternedData {
                event_names: names
                    .iter()
                    .map(|&(iid, name)| EventName {
                        iid: Some(iid),
                        name: Some(name.to_owned()),
                    })
                    .collect(),
                ..InternedData::default()
            }),
            ..TracePacket::default()
        })
    }

    pub fn slice_begin(
        &mut self,
        sequence_id: u32,
        track_uuid: u64,
        timestamp_ns: u64,
        name_iid: u64,
    ) -> Result<(), WriteError> {
        self.write_packet(&TracePacket {
            timestamp: Some(timestamp_ns),
            timestamp_clock_id: Some(proto::CLOCK_MONOTONIC),
            trusted_packet_sequence_id: Some(sequence_id),
            sequence_flags: Some(proto::SEQ_NEEDS_INCREMENTAL_STATE),
            track_event: Some(TrackEvent {
                r#type: Some(proto::TYPE_SLICE_BEGIN),
                track_uuid: Some(track_uuid),
                name_iid: Some(name_iid),
                ..TrackEvent::default()
            }),
            ..TracePacket::default()
        })
    }

    pub fn slice_end(
        &mut self,
        sequence_id: u32,
        track_uuid: u64,
        timestamp_ns: u64,
    ) -> Result<(), WriteError> {
        self.write_packet(&TracePacket {
            timestamp: Some(timestamp_ns),
            timestamp_clock_id: Some(proto::CLOCK_MONOTONIC),
            trusted_packet_sequence_id: Some(sequence_id),
            track_event: Some(TrackEvent {
                r#type: Some(proto::TYPE_SLICE_END),
                track_uuid: Some(track_uuid),
                ..TrackEvent::default()
            }),
            ..TracePacket::default()
        })
    }

    pub fn instant(
        &mut self,
        sequence_id: u32,
        track_uuid: u64,
        timestamp_ns: u64,
        name: &str,
        annotations: &[DebugAnnotation],
    ) -> Result<(), WriteError> {
        self.write_packet(&TracePacket {
            timestamp: Some(timestamp_ns),
            timestamp_clock_id: Some(proto::CLOCK_MONOTONIC),
            trusted_packet_sequence_id: Some(sequence_id),
            track_event: Some(TrackEvent {
                r#type: Some(proto::TYPE_INSTANT),
                track_uuid: Some(track_uuid),
                name: Some(name.to_owned()),
                debug_annotation: annotations.to_vec(),
                ..TrackEvent::default()
            }),
            ..TracePacket::default()
        })
    }

    pub fn counter_double(
        &mut self,
        sequence_id: u32,
        track_uuid: u64,
        timestamp_ns: u64,
        value: f64,
    ) -> Result<(), WriteError> {
        self.write_packet(&TracePacket {
            timestamp: Some(timestamp_ns),
            timestamp_clock_id: Some(proto::CLOCK_MONOTONIC),
            trusted_packet_sequence_id: Some(sequence_id),
            track_event: Some(TrackEvent {
                r#type: Some(proto::TYPE_COUNTER),
                track_uuid: Some(track_uuid),
                double_counter_value: Some(value),
                ..TrackEvent::default()
            }),
            ..TracePacket::default()
        })
    }

    pub fn flush(&mut self) -> Result<(), WriteError> {
        self.writer.flush()?;
        Ok(())
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert_eq!(data[offset], 0x0A, "expected field-1 tag");
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

    #[test]
    fn write_packet_produces_field1_framed_output() {
        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        let packet = TracePacket {
            timestamp: Some(42),
            ..TracePacket::default()
        };
        w.write_packet(&packet).unwrap();
        drop(w);

        assert_eq!(out[0], 0x0A);
        let (len, consumed) = decode_varint(&out[1..]);
        let decoded = TracePacket::decode(&out[1 + consumed..1 + consumed + len as usize]).unwrap();
        assert_eq!(decoded.timestamp, Some(42));
    }

    #[test]
    fn write_process_track_creates_descriptor() {
        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.write_process_track(1, "test-session", 42, "llama-3-8b")
            .unwrap();
        drop(w);

        let packets = decode_trace_packets(&out);
        assert_eq!(packets.len(), 1);
        let td = packets[0].track_descriptor.as_ref().unwrap();
        assert_eq!(td.uuid, Some(1));
        assert_eq!(td.name.as_deref(), Some("test-session"));
        assert_eq!(td.child_ordering, None);
        let proc = td.process.as_ref().unwrap();
        assert_eq!(proc.pid, Some(42));
        assert_eq!(proc.process_name.as_deref(), Some("llama-3-8b"));
    }

    #[test]
    fn write_thread_track_creates_descriptor_with_parent() {
        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.write_thread_track(100, 1, "rank:0", 1, 0).unwrap();
        drop(w);

        let packets = decode_trace_packets(&out);
        let td = packets[0].track_descriptor.as_ref().unwrap();
        assert_eq!(td.uuid, Some(100));
        assert_eq!(td.parent_uuid, Some(1));
        assert_eq!(td.child_ordering, None);
        let thread = td.thread.as_ref().unwrap();
        assert_eq!(thread.tid, Some(0));
        assert_eq!(thread.thread_name.as_deref(), Some("rank:0"));
    }

    #[test]
    fn write_track_creates_ordered_child() {
        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.write_track(1000, 100, "L0", 0).unwrap();
        drop(w);

        let packets = decode_trace_packets(&out);
        let td = packets[0].track_descriptor.as_ref().unwrap();
        assert_eq!(td.uuid, Some(1000));
        assert_eq!(td.parent_uuid, Some(100));
        assert_eq!(td.name.as_deref(), Some("L0"));
        assert_eq!(td.child_ordering, Some(proto::CHILD_ORDERING_EXPLICIT));
        assert_eq!(td.sibling_order_rank, Some(0));
    }

    #[test]
    fn slice_begin_end_pair() {
        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.slice_begin(1001, 10000, 1_000_000, 1).unwrap();
        w.slice_end(1001, 10000, 2_000_000).unwrap();
        drop(w);

        let packets = decode_trace_packets(&out);
        assert_eq!(packets.len(), 2);

        let begin = &packets[0];
        assert_eq!(begin.timestamp, Some(1_000_000));
        assert_eq!(begin.timestamp_clock_id, Some(proto::CLOCK_MONOTONIC));
        assert_eq!(begin.trusted_packet_sequence_id, Some(1001));
        assert_eq!(
            begin.sequence_flags,
            Some(proto::SEQ_NEEDS_INCREMENTAL_STATE)
        );
        let ev = begin.track_event.as_ref().unwrap();
        assert_eq!(ev.r#type, Some(proto::TYPE_SLICE_BEGIN));
        assert_eq!(ev.track_uuid, Some(10000));
        assert_eq!(ev.name_iid, Some(1));

        let end = &packets[1];
        assert_eq!(end.timestamp, Some(2_000_000));
        assert_eq!(end.timestamp_clock_id, Some(proto::CLOCK_MONOTONIC));
        let ev = end.track_event.as_ref().unwrap();
        assert_eq!(ev.r#type, Some(proto::TYPE_SLICE_END));
        assert_eq!(ev.track_uuid, Some(10000));
    }

    #[test]
    fn instant_event_with_annotations() {
        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.instant(
            1001,
            10000,
            5_000_000,
            "probe:attn_weights",
            &[DebugAnnotation {
                name: Some("mean".into()),
                double_value: Some(0.0312),
                ..DebugAnnotation::default()
            }],
        )
        .unwrap();
        drop(w);

        let packets = decode_trace_packets(&out);
        assert_eq!(packets[0].timestamp_clock_id, Some(proto::CLOCK_MONOTONIC));
        let ev = packets[0].track_event.as_ref().unwrap();
        assert_eq!(ev.r#type, Some(proto::TYPE_INSTANT));
        assert_eq!(ev.name.as_deref(), Some("probe:attn_weights"));
        assert_eq!(ev.debug_annotation.len(), 1);
        assert_eq!(ev.debug_annotation[0].double_value, Some(0.0312));
    }

    #[test]
    fn write_interned_names_emits_seq_cleared() {
        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.write_interned_names(1001, &[(1, "L0::attn::q_proj"), (2, "L0::attn::k_proj")])
            .unwrap();
        drop(w);

        let packets = decode_trace_packets(&out);
        let p = &packets[0];
        assert_eq!(p.trusted_packet_sequence_id, Some(1001));
        assert_eq!(p.sequence_flags, Some(proto::SEQ_INCREMENTAL_STATE_CLEARED));
        assert_eq!(p.first_packet_on_sequence, Some(true));
        assert_eq!(p.timestamp_clock_id, Some(proto::CLOCK_MONOTONIC));
        let interned = p.interned_data.as_ref().unwrap();
        assert_eq!(interned.event_names.len(), 2);
        assert_eq!(interned.event_names[0].iid, Some(1));
        assert_eq!(
            interned.event_names[0].name.as_deref(),
            Some("L0::attn::q_proj")
        );
    }

    #[test]
    fn first_packet_on_sequence_only_on_first_call() {
        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.write_interned_names(1001, &[(1, "a")]).unwrap();
        w.write_interned_names(1001, &[(1, "a"), (2, "b")]).unwrap();
        drop(w);

        let packets = decode_trace_packets(&out);
        assert_eq!(packets[0].first_packet_on_sequence, Some(true));
        assert_eq!(packets[1].first_packet_on_sequence, Some(false));
    }

    #[test]
    fn counter_double_event() {
        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.counter_double(1001, 20000, 7_000_000, 42.5).unwrap();
        drop(w);

        let packets = decode_trace_packets(&out);
        assert_eq!(packets[0].timestamp_clock_id, Some(proto::CLOCK_MONOTONIC));
        let ev = packets[0].track_event.as_ref().unwrap();
        assert_eq!(ev.r#type, Some(proto::TYPE_COUNTER));
        assert_eq!(ev.double_counter_value, Some(42.5));
    }

    #[test]
    fn multiple_packets_decode_as_valid_trace() {
        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.write_process_track(1, "sess", 1, "model").unwrap();
        w.write_thread_track(100, 1, "rank:0", 1, 0).unwrap();
        w.write_track(1000, 100, "L0", 0).unwrap();
        w.write_interned_names(1001, &[(1, "L0::attn")]).unwrap();
        w.slice_begin(1001, 1000, 100, 1).unwrap();
        w.slice_end(1001, 1000, 200).unwrap();
        w.instant(1001, 1000, 150, "probe", &[]).unwrap();
        w.flush().unwrap();
        drop(w);

        let packets = decode_trace_packets(&out);
        assert_eq!(packets.len(), 7);
    }
}
