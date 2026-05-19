use prost::Message;

// Field numbers verified against quarantine/perfetto/protos/perfetto/trace/perfetto_trace.proto

#[derive(Clone, PartialEq, Message)]
pub struct TracePacket {
    #[prost(uint64, optional, tag = "8")]
    pub timestamp: Option<u64>,
    #[prost(uint32, optional, tag = "58")]
    pub timestamp_clock_id: Option<u32>,
    #[prost(message, optional, tag = "11")]
    pub track_event: Option<TrackEvent>,
    #[prost(message, optional, tag = "60")]
    pub track_descriptor: Option<TrackDescriptor>,
    #[prost(uint32, optional, tag = "10")]
    pub trusted_packet_sequence_id: Option<u32>,
    #[prost(uint32, optional, tag = "13")]
    pub sequence_flags: Option<u32>,
    #[prost(message, optional, tag = "12")]
    pub interned_data: Option<InternedData>,
    #[prost(bool, optional, tag = "87")]
    pub first_packet_on_sequence: Option<bool>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TrackEvent {
    #[prost(int32, optional, tag = "9")]
    pub r#type: Option<i32>,
    #[prost(uint64, optional, tag = "11")]
    pub track_uuid: Option<u64>,
    #[prost(string, optional, tag = "23")]
    pub name: Option<String>,
    #[prost(uint64, optional, tag = "10")]
    pub name_iid: Option<u64>,
    #[prost(message, repeated, tag = "4")]
    pub debug_annotation: Vec<DebugAnnotation>,
    #[prost(int64, optional, tag = "30")]
    pub counter_value: Option<i64>,
    #[prost(double, optional, tag = "44")]
    pub double_counter_value: Option<f64>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct TrackDescriptor {
    #[prost(uint64, optional, tag = "1")]
    pub uuid: Option<u64>,
    #[prost(uint64, optional, tag = "5")]
    pub parent_uuid: Option<u64>,
    #[prost(string, optional, tag = "2")]
    pub name: Option<String>,
    #[prost(message, optional, tag = "3")]
    pub process: Option<ProcessDescriptor>,
    #[prost(message, optional, tag = "4")]
    pub thread: Option<ThreadDescriptor>,
    #[prost(message, optional, tag = "8")]
    pub counter: Option<CounterDescriptor>,
    #[prost(int32, optional, tag = "11")]
    pub child_ordering: Option<i32>,
    #[prost(int32, optional, tag = "12")]
    pub sibling_order_rank: Option<i32>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct ProcessDescriptor {
    #[prost(int32, optional, tag = "1")]
    pub pid: Option<i32>,
    #[prost(string, repeated, tag = "2")]
    pub cmdline: Vec<String>,
    #[prost(string, optional, tag = "6")]
    pub process_name: Option<String>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct ThreadDescriptor {
    #[prost(int32, optional, tag = "1")]
    pub pid: Option<i32>,
    #[prost(int64, optional, tag = "2")]
    pub tid: Option<i64>,
    #[prost(string, optional, tag = "5")]
    pub thread_name: Option<String>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct CounterDescriptor {
    #[prost(string, repeated, tag = "2")]
    pub categories: Vec<String>,
    #[prost(string, optional, tag = "6")]
    pub unit_name: Option<String>,
    #[prost(bool, optional, tag = "5")]
    pub is_incremental: Option<bool>,
}

#[derive(Clone, PartialEq, Message)]
pub struct DebugAnnotation {
    #[prost(string, optional, tag = "10")]
    pub name: Option<String>,
    #[prost(uint64, optional, tag = "1")]
    pub name_iid: Option<u64>,
    #[prost(bool, optional, tag = "2")]
    pub bool_value: Option<bool>,
    #[prost(uint64, optional, tag = "3")]
    pub uint_value: Option<u64>,
    #[prost(int64, optional, tag = "4")]
    pub int_value: Option<i64>,
    #[prost(double, optional, tag = "5")]
    pub double_value: Option<f64>,
    #[prost(string, optional, tag = "6")]
    pub string_value: Option<String>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct InternedData {
    #[prost(message, repeated, tag = "1")]
    pub event_categories: Vec<EventCategory>,
    #[prost(message, repeated, tag = "2")]
    pub event_names: Vec<EventName>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct EventCategory {
    #[prost(uint64, optional, tag = "1")]
    pub iid: Option<u64>,
    #[prost(string, optional, tag = "2")]
    pub name: Option<String>,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct EventName {
    #[prost(uint64, optional, tag = "1")]
    pub iid: Option<u64>,
    #[prost(string, optional, tag = "2")]
    pub name: Option<String>,
}

pub const TYPE_SLICE_BEGIN: i32 = 1;
pub const TYPE_SLICE_END: i32 = 2;
pub const TYPE_INSTANT: i32 = 3;
pub const TYPE_COUNTER: i32 = 4;

pub const SEQ_INCREMENTAL_STATE_CLEARED: u32 = 1;
pub const SEQ_NEEDS_INCREMENTAL_STATE: u32 = 2;

pub const CLOCK_MONOTONIC: u32 = 3;

pub const CHILD_ORDERING_EXPLICIT: i32 = 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_packet_roundtrip() {
        let packet = TracePacket {
            timestamp: Some(1_000_000),
            timestamp_clock_id: Some(CLOCK_MONOTONIC),
            trusted_packet_sequence_id: Some(1001),
            sequence_flags: Some(SEQ_INCREMENTAL_STATE_CLEARED),
            first_packet_on_sequence: Some(true),
            track_event: Some(TrackEvent {
                r#type: Some(TYPE_SLICE_BEGIN),
                track_uuid: Some(100),
                name: Some("L0::attn::q_proj".into()),
                ..TrackEvent::default()
            }),
            ..TracePacket::default()
        };

        let mut buf = Vec::new();
        packet.encode(&mut buf).unwrap();
        assert!(!buf.is_empty());

        let decoded = TracePacket::decode(buf.as_slice()).unwrap();
        assert_eq!(decoded, packet);
    }

    #[test]
    fn track_descriptor_roundtrip() {
        let packet = TracePacket {
            track_descriptor: Some(TrackDescriptor {
                uuid: Some(1),
                name: Some("session:abc".into()),
                process: Some(ProcessDescriptor {
                    pid: Some(42),
                    process_name: Some("llama-3-8b".into()),
                    ..ProcessDescriptor::default()
                }),
                child_ordering: Some(CHILD_ORDERING_EXPLICIT),
                ..TrackDescriptor::default()
            }),
            ..TracePacket::default()
        };

        let mut buf = Vec::new();
        packet.encode(&mut buf).unwrap();
        let decoded = TracePacket::decode(buf.as_slice()).unwrap();
        assert_eq!(decoded, packet);
    }

    #[test]
    fn instant_event_with_annotations_roundtrip() {
        let packet = TracePacket {
            timestamp: Some(5_000_000),
            trusted_packet_sequence_id: Some(1001),
            track_event: Some(TrackEvent {
                r#type: Some(TYPE_INSTANT),
                track_uuid: Some(10000),
                name: Some("probe:attn_weights".into()),
                debug_annotation: vec![
                    DebugAnnotation {
                        name: Some("mean".into()),
                        double_value: Some(0.0312),
                        ..DebugAnnotation::default()
                    },
                    DebugAnnotation {
                        name: Some("shape".into()),
                        string_value: Some("[32, 16, 128, 128]".into()),
                        ..DebugAnnotation::default()
                    },
                ],
                ..TrackEvent::default()
            }),
            ..TracePacket::default()
        };

        let mut buf = Vec::new();
        packet.encode(&mut buf).unwrap();
        let decoded = TracePacket::decode(buf.as_slice()).unwrap();
        assert_eq!(decoded, packet);
    }

    #[test]
    fn interned_data_roundtrip() {
        let packet = TracePacket {
            trusted_packet_sequence_id: Some(1001),
            sequence_flags: Some(SEQ_INCREMENTAL_STATE_CLEARED),
            first_packet_on_sequence: Some(true),
            interned_data: Some(InternedData {
                event_names: vec![
                    EventName {
                        iid: Some(1),
                        name: Some("L0::attn::q_proj".into()),
                    },
                    EventName {
                        iid: Some(2),
                        name: Some("L0::attn::k_proj".into()),
                    },
                ],
                event_categories: vec![EventCategory {
                    iid: Some(1),
                    name: Some("component".into()),
                }],
            }),
            ..TracePacket::default()
        };

        let mut buf = Vec::new();
        packet.encode(&mut buf).unwrap();
        let decoded = TracePacket::decode(buf.as_slice()).unwrap();
        assert_eq!(decoded, packet);
    }

    #[test]
    fn interned_name_iid_event_roundtrip() {
        let packet = TracePacket {
            timestamp: Some(2_000_000),
            trusted_packet_sequence_id: Some(1001),
            track_event: Some(TrackEvent {
                r#type: Some(TYPE_SLICE_BEGIN),
                track_uuid: Some(10000),
                name_iid: Some(1),
                ..TrackEvent::default()
            }),
            ..TracePacket::default()
        };

        let mut buf = Vec::new();
        packet.encode(&mut buf).unwrap();
        let decoded = TracePacket::decode(buf.as_slice()).unwrap();
        assert_eq!(decoded.track_event.unwrap().name_iid, Some(1));
    }

    #[test]
    fn counter_event_roundtrip() {
        let packet = TracePacket {
            timestamp: Some(3_000_000),
            trusted_packet_sequence_id: Some(1001),
            track_event: Some(TrackEvent {
                r#type: Some(TYPE_COUNTER),
                track_uuid: Some(20000),
                double_counter_value: Some(1.234),
                ..TrackEvent::default()
            }),
            ..TracePacket::default()
        };

        let mut buf = Vec::new();
        packet.encode(&mut buf).unwrap();
        let decoded = TracePacket::decode(buf.as_slice()).unwrap();
        assert_eq!(
            decoded.track_event.unwrap().double_counter_value,
            Some(1.234)
        );
    }

    #[test]
    fn thread_track_roundtrip() {
        let packet = TracePacket {
            track_descriptor: Some(TrackDescriptor {
                uuid: Some(100),
                parent_uuid: Some(1),
                name: Some("rank:0".into()),
                thread: Some(ThreadDescriptor {
                    pid: Some(1),
                    tid: Some(0),
                    thread_name: Some("rank:0".into()),
                }),
                ..TrackDescriptor::default()
            }),
            ..TracePacket::default()
        };

        let mut buf = Vec::new();
        packet.encode(&mut buf).unwrap();
        let decoded = TracePacket::decode(buf.as_slice()).unwrap();
        assert_eq!(decoded, packet);
    }
}
