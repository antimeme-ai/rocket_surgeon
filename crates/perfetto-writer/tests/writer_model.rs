//! Stateful model-based tests for `TraceWriter`, plus track-hierarchy forest
//! invariants and an exception-raising IO property.
//!
//! Model: the writer turns a sequence of high-level calls into a stream of
//! length-delimited field-1 `TracePacket` records. We maintain, in parallel, an
//! independent description of (a) how many records must appear, (b) the salient
//! fields each record must carry — re-derived from Perfetto semantics, not copied
//! from the writer — and (c) the `first_packet_on_sequence` flag, which is the
//! writer's only piece of mutable state (a per-sequence "have I seen this id"
//! set). We decode the real stream and assert it matches the model.

use std::io::Write;

use perfetto_writer::proto::{self, DebugAnnotation, TracePacket};
use perfetto_writer::writer::{TraceWriter, WriteError};
use proptest::collection::vec;
use proptest::prelude::*;
use prost::Message;

/// Frame decoder: walks the field-1 LEN-delimited stream the writer produces.
/// Returns the packets, or `Err` describing a framing defect (bad tag / leftover
/// bytes / truncation) — the stream must tile exactly.
fn decode_stream(data: &[u8]) -> Result<Vec<TracePacket>, String> {
    let mut packets = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        if data[offset] != 0x0A {
            return Err(format!(
                "expected field-1 tag at {offset}, got {:#x}",
                data[offset]
            ));
        }
        offset += 1;
        let mut len: u64 = 0;
        let mut shift = 0;
        let mut consumed = 0;
        loop {
            if offset + consumed >= data.len() {
                return Err("truncated length varint".into());
            }
            let b = data[offset + consumed];
            len |= u64::from(b & 0x7F) << shift;
            consumed += 1;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        offset += consumed;
        let end = offset + len as usize;
        if end > data.len() {
            return Err("packet length exceeds buffer".into());
        }
        let packet = TracePacket::decode(&data[offset..end]).map_err(|e| e.to_string())?;
        packets.push(packet);
        offset = end;
    }
    Ok(packets)
}

fn small_name() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9:_]{0,16}"
}

#[derive(Clone, Debug)]
enum Op {
    ProcessTrack {
        uuid: u64,
        name: String,
        pid: i32,
        process_name: String,
    },
    ThreadTrack {
        uuid: u64,
        parent_uuid: u64,
        name: String,
        pid: i32,
        tid: i64,
    },
    Track {
        uuid: u64,
        parent_uuid: u64,
        name: String,
        order: i32,
    },
    CounterTrack {
        uuid: u64,
        parent_uuid: u64,
        name: String,
        unit: String,
    },
    InternedNames {
        seq: u32,
        names: Vec<(u64, String)>,
    },
    SliceBegin {
        seq: u32,
        track: u64,
        ts: u64,
        name_iid: u64,
    },
    SliceEnd {
        seq: u32,
        track: u64,
        ts: u64,
    },
    Instant {
        seq: u32,
        track: u64,
        ts: u64,
        name: String,
    },
    CounterDouble {
        seq: u32,
        track: u64,
        ts: u64,
        value: f64,
    },
}

fn op() -> impl Strategy<Value = Op> {
    // Sequence ids drawn from a tiny pool (1..=3) so the first-packet-on-sequence
    // state machine is exercised with real repeats, not all-unique ids.
    let seq = 1u32..=3;
    prop_oneof![
        (any::<u64>(), small_name(), any::<i32>(), small_name()).prop_map(
            |(uuid, name, pid, process_name)| Op::ProcessTrack {
                uuid,
                name,
                pid,
                process_name
            }
        ),
        (
            any::<u64>(),
            any::<u64>(),
            small_name(),
            any::<i32>(),
            any::<i64>()
        )
            .prop_map(|(uuid, parent_uuid, name, pid, tid)| Op::ThreadTrack {
                uuid,
                parent_uuid,
                name,
                pid,
                tid
            }),
        (any::<u64>(), any::<u64>(), small_name(), any::<i32>()).prop_map(
            |(uuid, parent_uuid, name, order)| Op::Track {
                uuid,
                parent_uuid,
                name,
                order
            }
        ),
        (any::<u64>(), any::<u64>(), small_name(), small_name()).prop_map(
            |(uuid, parent_uuid, name, unit)| Op::CounterTrack {
                uuid,
                parent_uuid,
                name,
                unit
            }
        ),
        (seq.clone(), vec((any::<u64>(), small_name()), 0..4))
            .prop_map(|(seq, names)| Op::InternedNames { seq, names }),
        (seq.clone(), any::<u64>(), any::<u64>(), any::<u64>()).prop_map(
            |(seq, track, ts, name_iid)| Op::SliceBegin {
                seq,
                track,
                ts,
                name_iid
            }
        ),
        (seq.clone(), any::<u64>(), any::<u64>()).prop_map(|(seq, track, ts)| Op::SliceEnd {
            seq,
            track,
            ts
        }),
        (seq.clone(), any::<u64>(), any::<u64>(), small_name()).prop_map(
            |(seq, track, ts, name)| Op::Instant {
                seq,
                track,
                ts,
                name
            }
        ),
        (seq, any::<u64>(), any::<u64>(), -1e9f64..1e9f64).prop_map(|(seq, track, ts, value)| {
            Op::CounterDouble {
                seq,
                track,
                ts,
                value,
            }
        }),
    ]
}

fn apply(w: &mut TraceWriter<Vec<u8>>, op: &Op) {
    match op {
        Op::ProcessTrack {
            uuid,
            name,
            pid,
            process_name,
        } => w
            .write_process_track(*uuid, name, *pid, process_name)
            .unwrap(),
        Op::ThreadTrack {
            uuid,
            parent_uuid,
            name,
            pid,
            tid,
        } => w
            .write_thread_track(*uuid, *parent_uuid, name, *pid, *tid)
            .unwrap(),
        Op::Track {
            uuid,
            parent_uuid,
            name,
            order,
        } => w.write_track(*uuid, *parent_uuid, name, *order).unwrap(),
        Op::CounterTrack {
            uuid,
            parent_uuid,
            name,
            unit,
        } => w
            .write_counter_track(*uuid, *parent_uuid, name, unit)
            .unwrap(),
        Op::InternedNames { seq, names } => {
            let refs: Vec<(u64, &str)> = names.iter().map(|(i, n)| (*i, n.as_str())).collect();
            w.write_interned_names(*seq, &refs).unwrap();
        }
        Op::SliceBegin {
            seq,
            track,
            ts,
            name_iid,
        } => w.slice_begin(*seq, *track, *ts, *name_iid).unwrap(),
        Op::SliceEnd { seq, track, ts } => w.slice_end(*seq, *track, *ts).unwrap(),
        Op::Instant {
            seq,
            track,
            ts,
            name,
        } => w
            .instant(*seq, *track, *ts, name, &[] as &[DebugAnnotation])
            .unwrap(),
        Op::CounterDouble {
            seq,
            track,
            ts,
            value,
        } => w.counter_double(*seq, *track, *ts, *value).unwrap(),
    }
}

proptest! {
    /// The stateful model property: every op produces exactly one record whose
    /// fields match the independently-derived expectation, and the
    /// first-packet-on-sequence flag follows the seen-set model.
    #[test]
    fn stream_matches_model(ops in vec(op(), 0..120)) {
        let mut w = TraceWriter::new(Vec::new());
        for op in &ops { apply(&mut w, op); }
        let bytes = w.into_inner();

        let packets = decode_stream(&bytes).expect("stream must frame cleanly");
        prop_assert_eq!(packets.len(), ops.len(), "one record per op");

        let mut seen_seq = std::collections::HashSet::new();
        for (op, p) in ops.iter().zip(&packets) {
            match op {
                Op::ProcessTrack { uuid, name, pid, process_name } => {
                    let td = p.track_descriptor.as_ref().expect("track descriptor");
                    prop_assert_eq!(td.uuid, Some(*uuid));
                    prop_assert_eq!(td.name.as_deref(), Some(name.as_str()));
                    prop_assert_eq!(td.child_ordering, None);
                    let proc = td.process.as_ref().expect("process");
                    prop_assert_eq!(proc.pid, Some(*pid));
                    prop_assert_eq!(proc.process_name.as_deref(), Some(process_name.as_str()));
                }
                Op::ThreadTrack { uuid, parent_uuid, name, pid, tid } => {
                    let td = p.track_descriptor.as_ref().unwrap();
                    prop_assert_eq!(td.uuid, Some(*uuid));
                    prop_assert_eq!(td.parent_uuid, Some(*parent_uuid));
                    let th = td.thread.as_ref().expect("thread");
                    prop_assert_eq!(th.pid, Some(*pid));
                    prop_assert_eq!(th.tid, Some(*tid));
                    prop_assert_eq!(th.thread_name.as_deref(), Some(name.as_str()));
                }
                Op::Track { uuid, parent_uuid, name, order } => {
                    let td = p.track_descriptor.as_ref().unwrap();
                    prop_assert_eq!(td.uuid, Some(*uuid));
                    prop_assert_eq!(td.parent_uuid, Some(*parent_uuid));
                    prop_assert_eq!(td.name.as_deref(), Some(name.as_str()));
                    prop_assert_eq!(td.child_ordering, Some(proto::CHILD_ORDERING_EXPLICIT));
                    prop_assert_eq!(td.sibling_order_rank, Some(*order));
                }
                Op::CounterTrack { uuid, parent_uuid, name, unit } => {
                    let td = p.track_descriptor.as_ref().unwrap();
                    prop_assert_eq!(td.uuid, Some(*uuid));
                    prop_assert_eq!(td.parent_uuid, Some(*parent_uuid));
                    prop_assert_eq!(td.name.as_deref(), Some(name.as_str()));
                    let c = td.counter.as_ref().expect("counter");
                    prop_assert_eq!(c.unit_name.as_deref(), Some(unit.as_str()));
                }
                Op::InternedNames { seq, names } => {
                    let expect_first = seen_seq.insert(*seq);
                    prop_assert_eq!(p.trusted_packet_sequence_id, Some(*seq));
                    prop_assert_eq!(p.sequence_flags, Some(proto::SEQ_INCREMENTAL_STATE_CLEARED));
                    prop_assert_eq!(p.first_packet_on_sequence, Some(expect_first),
                        "first_packet_on_sequence state-machine diverged for seq {}", seq);
                    let interned = p.interned_data.as_ref().expect("interned data");
                    prop_assert_eq!(interned.event_names.len(), names.len());
                    for (en, (iid, name)) in interned.event_names.iter().zip(names) {
                        prop_assert_eq!(en.iid, Some(*iid));
                        prop_assert_eq!(en.name.as_deref(), Some(name.as_str()));
                    }
                }
                Op::SliceBegin { seq, track, ts, name_iid } => {
                    prop_assert_eq!(p.timestamp, Some(*ts));
                    prop_assert_eq!(p.timestamp_clock_id, Some(proto::CLOCK_MONOTONIC));
                    prop_assert_eq!(p.trusted_packet_sequence_id, Some(*seq));
                    prop_assert_eq!(p.sequence_flags, Some(proto::SEQ_NEEDS_INCREMENTAL_STATE));
                    let ev = p.track_event.as_ref().expect("track event");
                    prop_assert_eq!(ev.r#type, Some(proto::TYPE_SLICE_BEGIN));
                    prop_assert_eq!(ev.track_uuid, Some(*track));
                    prop_assert_eq!(ev.name_iid, Some(*name_iid));
                }
                Op::SliceEnd { seq, track, ts } => {
                    prop_assert_eq!(p.timestamp, Some(*ts));
                    prop_assert_eq!(p.trusted_packet_sequence_id, Some(*seq));
                    let ev = p.track_event.as_ref().unwrap();
                    prop_assert_eq!(ev.r#type, Some(proto::TYPE_SLICE_END));
                    prop_assert_eq!(ev.track_uuid, Some(*track));
                }
                Op::Instant { seq, track, ts, name } => {
                    prop_assert_eq!(p.timestamp, Some(*ts));
                    prop_assert_eq!(p.trusted_packet_sequence_id, Some(*seq));
                    let ev = p.track_event.as_ref().unwrap();
                    prop_assert_eq!(ev.r#type, Some(proto::TYPE_INSTANT));
                    prop_assert_eq!(ev.track_uuid, Some(*track));
                    prop_assert_eq!(ev.name.as_deref(), Some(name.as_str()));
                }
                Op::CounterDouble { seq, track, ts, value } => {
                    prop_assert_eq!(p.timestamp, Some(*ts));
                    prop_assert_eq!(p.trusted_packet_sequence_id, Some(*seq));
                    let ev = p.track_event.as_ref().unwrap();
                    prop_assert_eq!(ev.r#type, Some(proto::TYPE_COUNTER));
                    prop_assert_eq!(ev.track_uuid, Some(*track));
                    prop_assert_eq!(ev.double_counter_value, Some(*value));
                }
            }
        }
    }
}

// ---- Track-hierarchy forest invariants ------------------------------------

/// A generated, valid track tree: node 0 is the process root (no parent); every
/// other node picks an already-declared node as its parent, so the result is a
/// connected tree by construction. uuids are distinct (index + 1).
#[derive(Clone, Debug)]
struct TreeSpec {
    /// parent[i] = index of node i's parent, or `usize::MAX` for the root (i == 0).
    parent: Vec<usize>,
}

fn tree_spec() -> impl Strategy<Value = TreeSpec> {
    (1usize..16).prop_flat_map(|n| {
        // For node i (i>=1), parent index is in 0..i — guarantees acyclicity.
        let parents = (1..n).map(|i| 0..i).collect::<Vec<_>>();
        parents.prop_map(move |ps| {
            let mut parent = vec![usize::MAX];
            parent.extend(ps);
            TreeSpec { parent }
        })
    })
}

fn uuid_of(i: usize) -> u64 {
    i as u64 + 1
}

proptest! {
    /// Emit a valid track forest through the writer, decode it, and assert the
    /// reconstructed hierarchy is a valid rooted tree matching the spec:
    ///   - all uuids distinct,
    ///   - exactly one root (no parent_uuid),
    ///   - every non-root parent_uuid refers to a declared uuid (no dangling),
    ///   - the parent relation equals the input spec (no corruption in transit).
    #[test]
    fn track_hierarchy_is_a_faithful_forest(spec in tree_spec()) {
        let mut w = TraceWriter::new(Vec::new());
        w.write_process_track(uuid_of(0), "root", 1, "model").unwrap();
        for i in 1..spec.parent.len() {
            w.write_track(uuid_of(i), uuid_of(spec.parent[i]), "node", i as i32).unwrap();
        }
        let bytes = w.into_inner();
        let packets = decode_stream(&bytes).expect("frames cleanly");
        prop_assert_eq!(packets.len(), spec.parent.len());

        let mut declared = std::collections::HashSet::new();
        let mut roots = 0;
        let mut reconstructed: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        for p in &packets {
            let td = p.track_descriptor.as_ref().expect("descriptor");
            let uuid = td.uuid.expect("uuid present");
            prop_assert!(declared.insert(uuid), "duplicate uuid {} in stream", uuid);
            match td.parent_uuid {
                None => roots += 1,
                Some(pu) => { reconstructed.insert(uuid, pu); }
            }
        }
        prop_assert_eq!(roots, 1, "a forest emitted as a tree must have exactly one root");

        // No dangling parents: every referenced parent uuid was declared.
        for (&child, &pu) in &reconstructed {
            prop_assert!(declared.contains(&pu),
                "child {} references undeclared parent {}", child, pu);
        }
        // Parent relation survived transit unchanged.
        for i in 1..spec.parent.len() {
            prop_assert_eq!(reconstructed.get(&uuid_of(i)).copied(),
                Some(uuid_of(spec.parent[i])), "parent edge corrupted for node {}", i);
        }
    }
}

// ---- Exception-raising property -------------------------------------------

/// A writer whose `write_all` fails after `ok_writes` successful calls.
struct FlakyWriter {
    ok_writes: usize,
    count: usize,
}

impl Write for FlakyWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.count >= self.ok_writes {
            return Err(std::io::Error::other("disk full"));
        }
        self.count += 1;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

proptest! {
    /// Exception-raising oracle (MATERIA: 113x more effective, almost nobody
    /// writes these): a failing sink must surface `WriteError::Io`, never panic
    /// and never silently report success.
    #[test]
    fn io_failure_surfaces_as_write_error(ok_writes in 0usize..5) {
        let w = FlakyWriter { ok_writes, count: 0 };
        let mut tw = TraceWriter::new(w);
        // Issue more writes than will succeed.
        let mut saw_err = false;
        for ts in 0..(ok_writes as u64 + 3) {
            match tw.slice_end(1, 1, ts) {
                Ok(()) => {}
                Err(WriteError::Io(_)) => { saw_err = true; break; }
                Err(other) => prop_assert!(false, "unexpected error variant: {other:?}"),
            }
        }
        prop_assert!(saw_err, "failing sink never produced a WriteError::Io");
    }
}
