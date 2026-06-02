//! Round-trip + field-fidelity properties for the prost message definitions.
//!
//! Oracle: prost's own decoder is the model. A `TracePacket` encoded then
//! decoded must equal the original (tier 4 roundtrip). Because we generate
//! packets with many fields set at once, any two fields that accidentally shared
//! a proto tag number — the classic hand-written-`#[prost]` bug — would corrupt
//! each other and fail equality. This is the property that guards the schema.

use perfetto_writer::proto::*;
use proptest::prelude::*;
use prost::Message;

fn opt<T: std::fmt::Debug + Clone, S: Strategy<Value = T>>(
    s: S,
) -> impl Strategy<Value = Option<T>> {
    prop_oneof![1 => Just(None), 3 => s.prop_map(Some)]
}

/// Finite doubles only: NaN breaks struct `PartialEq` (NaN != NaN) even though the
/// bits round-trip faithfully — that case is covered separately at the bit level
/// in `nan_double_bits_preserved`. ±inf compare equal, so they stay in.
fn finite_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        8 => -1e12f64..1e12f64,
        1 => Just(f64::INFINITY),
        1 => Just(f64::NEG_INFINITY),
    ]
}

fn small_string() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => "[a-zA-Z0-9:_ \\[\\],.-]{0,24}",
        1 => any::<String>(),
    ]
}

fn debug_annotation() -> impl Strategy<Value = DebugAnnotation> {
    (
        opt(small_string()),
        opt(any::<u64>()),
        opt(any::<bool>()),
        opt(any::<u64>()),
        opt(any::<i64>()),
        opt(finite_f64()),
        opt(small_string()),
    )
        .prop_map(
            |(name, name_iid, bool_value, uint_value, int_value, double_value, string_value)| {
                DebugAnnotation {
                    name,
                    name_iid,
                    bool_value,
                    uint_value,
                    int_value,
                    double_value,
                    string_value,
                }
            },
        )
}

fn track_event() -> impl Strategy<Value = TrackEvent> {
    (
        opt(any::<i32>()),
        opt(any::<u64>()),
        opt(small_string()),
        opt(any::<u64>()),
        prop::collection::vec(debug_annotation(), 0..4),
        opt(any::<i64>()),
        opt(finite_f64()),
    )
        .prop_map(
            |(
                r#type,
                track_uuid,
                name,
                name_iid,
                debug_annotation,
                counter_value,
                double_counter_value,
            )| {
                TrackEvent {
                    r#type,
                    track_uuid,
                    name,
                    name_iid,
                    debug_annotation,
                    counter_value,
                    double_counter_value,
                }
            },
        )
}

fn track_descriptor() -> impl Strategy<Value = TrackDescriptor> {
    (
        opt(any::<u64>()),
        opt(any::<u64>()),
        opt(small_string()),
        opt((
            opt(any::<i32>()),
            prop::collection::vec(small_string(), 0..3),
            opt(small_string()),
        )
            .prop_map(|(pid, cmdline, process_name)| ProcessDescriptor {
                pid,
                cmdline,
                process_name,
            })),
        opt(
            (opt(any::<i32>()), opt(any::<i64>()), opt(small_string())).prop_map(
                |(pid, tid, thread_name)| ThreadDescriptor {
                    pid,
                    tid,
                    thread_name,
                },
            ),
        ),
        opt((
            prop::collection::vec(small_string(), 0..3),
            opt(small_string()),
            opt(any::<bool>()),
        )
            .prop_map(
                |(categories, unit_name, is_incremental)| CounterDescriptor {
                    categories,
                    unit_name,
                    is_incremental,
                },
            )),
        opt(any::<i32>()),
        opt(any::<i32>()),
    )
        .prop_map(
            |(
                uuid,
                parent_uuid,
                name,
                process,
                thread,
                counter,
                child_ordering,
                sibling_order_rank,
            )| {
                TrackDescriptor {
                    uuid,
                    parent_uuid,
                    name,
                    process,
                    thread,
                    counter,
                    child_ordering,
                    sibling_order_rank,
                }
            },
        )
}

fn interned_data() -> impl Strategy<Value = InternedData> {
    (
        prop::collection::vec(
            (opt(any::<u64>()), opt(small_string()))
                .prop_map(|(iid, name)| EventCategory { iid, name }),
            0..4,
        ),
        prop::collection::vec(
            (opt(any::<u64>()), opt(small_string()))
                .prop_map(|(iid, name)| EventName { iid, name }),
            0..4,
        ),
    )
        .prop_map(|(event_categories, event_names)| InternedData {
            event_categories,
            event_names,
        })
}

fn trace_packet() -> impl Strategy<Value = TracePacket> {
    (
        opt(any::<u64>()),
        opt(any::<u32>()),
        opt(track_event()),
        opt(track_descriptor()),
        opt(any::<u32>()),
        opt(any::<u32>()),
        opt(interned_data()),
        opt(any::<bool>()),
    )
        .prop_map(
            |(
                timestamp,
                timestamp_clock_id,
                track_event,
                track_descriptor,
                trusted_packet_sequence_id,
                sequence_flags,
                interned_data,
                first_packet_on_sequence,
            )| TracePacket {
                timestamp,
                timestamp_clock_id,
                track_event,
                track_descriptor,
                trusted_packet_sequence_id,
                sequence_flags,
                interned_data,
                first_packet_on_sequence,
            },
        )
}

proptest! {
    /// tier 4/6 — arbitrary packet survives encode→decode unchanged.
    #[test]
    fn packet_roundtrips(packet in trace_packet()) {
        let mut buf = Vec::new();
        packet.encode(&mut buf).unwrap();
        let decoded = TracePacket::decode(buf.as_slice()).unwrap();
        prop_assert_eq!(decoded, packet);
    }

    /// `encoded_len()` must exactly predict the bytes written — the writer relies
    /// on this for its length-prefix frame. A mismatch would desync the stream.
    #[test]
    fn encoded_len_matches_bytes(packet in trace_packet()) {
        let predicted = packet.encoded_len();
        let mut buf = Vec::new();
        packet.encode(&mut buf).unwrap();
        prop_assert_eq!(predicted, buf.len());
    }

    /// Metamorphic: re-encoding a decoded packet is a fixed point (proto encoding
    /// is canonical for this schema), and decoding is idempotent.
    #[test]
    fn reencode_is_fixed_point(packet in trace_packet()) {
        let mut a = Vec::new();
        packet.encode(&mut a).unwrap();
        let decoded = TracePacket::decode(a.as_slice()).unwrap();
        let mut b = Vec::new();
        decoded.encode(&mut b).unwrap();
        prop_assert_eq!(a, b);
    }
}

/// NaN doubles: `PartialEq` says NaN != NaN, but the *bits* must survive transit.
/// Metamorphic bit-level oracle for the double-valued fields.
#[test]
fn nan_double_bits_preserved() {
    for bits in [
        0x7ff8_0000_0000_0001u64, // quiet NaN
        0xfff8_0000_0000_0000,    // negative NaN
        0x7ff0_0000_0000_0001,    // signaling NaN
    ] {
        let nan = f64::from_bits(bits);
        let packet = TracePacket {
            track_event: Some(TrackEvent {
                double_counter_value: Some(nan),
                ..TrackEvent::default()
            }),
            ..TracePacket::default()
        };
        let mut buf = Vec::new();
        packet.encode(&mut buf).unwrap();
        let decoded = TracePacket::decode(buf.as_slice()).unwrap();
        let got = decoded.track_event.unwrap().double_counter_value.unwrap();
        assert_eq!(got.to_bits(), bits, "NaN payload bits not preserved");
    }
}
