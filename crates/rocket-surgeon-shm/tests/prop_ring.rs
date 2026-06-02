//! Property-based tests for the DOOMRING SPSC shared-memory ring buffer.
//!
//! MATERIA oracle tiers exercised here:
//!   - Tier 4 (metamorphic): the header is the single source of truth for the
//!     consumed length, independent of how much the producer actually wrote.
//!   - Tier 4 (roundtrip):   bytes published == bytes consumed, for arbitrary
//!     payloads spanning the full boundary set (empty .. slot capacity).
//!   - Tier 2-relative exception-raising: oversized payloads, stale generations,
//!     and corrupt header-size fields each produce the *specific* typed error,
//!     never a panic and never a silent truncation.
//!
//! Every ring lives in its own POSIX shm region with a unique name and is
//! unlinked at the end of the case, so cases are independent and leak nothing.

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config as PtConfig, TestRunner};
use rocket_surgeon_shm::region::ShmRegion;
use rocket_surgeon_shm::ring::{DoomRingConsumer, DoomRingProducer};
use rocket_surgeon_shm::{
    FRAME_OFFSET_SIZE, PROBE_FRAME_HEADER_SIZE, RingConfig, ShmError, serialize_probe_frame,
};
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Unique, short shm name (macOS caps POSIX shm names at 30 bytes).
fn unique_name(tag: char) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("/rs{tag}{}-{n}", std::process::id())
}

/// A 128-byte probe header whose `size` and `generation` fields are set; all
/// other fields zero. Built via the production serializer so this also pins
/// `serialize_probe_frame`'s size/generation offsets.
fn header_with(size: u64, generation: u32) -> [u8; PROBE_FRAME_HEADER_SIZE] {
    serialize_probe_frame(0, 0, 0, 0, 0, &[0u32; 8], 0, 0, size, 0, generation)
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Power-of-two slot counts that keep regions tiny.
fn backuptics() -> impl Strategy<Value = u32> {
    prop::sample::select(vec![1u32, 2, 4, 8, 16])
}

/// `slot_size` in a range that keeps the region small but spans a few orders.
fn slot_size() -> impl Strategy<Value = u64> {
    PROBE_FRAME_HEADER_SIZE as u64..=8192
}

fn ring_config() -> impl Strategy<Value = RingConfig> {
    (backuptics(), slot_size()).prop_map(|(b, s)| RingConfig::new(b, s).expect("valid config"))
}

/// A config plus a payload whose length is sampled to hit the boundary set:
/// empty, 1, mid, cap-1, cap. We bias toward the boundaries because that is
/// where off-by-one slot-arithmetic bugs live.
fn config_and_payload() -> impl Strategy<Value = (RingConfig, Vec<u8>)> {
    ring_config().prop_flat_map(|cfg| {
        let cap = cfg.slot_data_capacity();
        // A length strategy that deliberately weights the edges.
        let len = prop_oneof![
            2 => Just(0usize),
            2 => Just(1usize.min(cap)),
            2 => Just(cap),
            1 => Just(cap.saturating_sub(1)),
            3 => 0..=cap,
        ];
        len.prop_flat_map(move |l| (Just(cfg), prop::collection::vec(any::<u8>(), l)))
    })
}

// ---------------------------------------------------------------------------
// Generator distribution evidence (MATERIA: measure what you generate)
// ---------------------------------------------------------------------------

/// Sample the payload generator and assert the boundary buckets are all hit.
/// Printed counts are the durable evidence that the generator is not trivial.
#[test]
fn generator_distribution_covers_boundaries() {
    const N: u32 = 3000;
    let mut runner = TestRunner::new(PtConfig::default());
    let strat = config_and_payload();

    let (mut empty, mut one, mut mid, mut near_cap, mut at_cap) = (0u32, 0, 0, 0, 0);
    for _ in 0..N {
        let (cfg, payload) = strat.new_tree(&mut runner).unwrap().current();
        let cap = cfg.slot_data_capacity();
        let l = payload.len();
        if l == 0 {
            empty += 1;
        } else if l == 1 {
            one += 1;
        } else if l == cap {
            at_cap += 1;
        } else if l + 1 == cap {
            near_cap += 1;
        } else {
            mid += 1;
        }
        assert!(l <= cap, "generator must never exceed capacity");
    }
    eprintln!(
        "payload-len distribution over {N}: empty={empty} one={one} mid={mid} \
         near_cap={near_cap} at_cap={at_cap}"
    );
    // Each interesting bucket must be non-trivially represented.
    assert!(empty > 50, "empty payloads underrepresented: {empty}");
    assert!(one > 50, "1-byte payloads underrepresented: {one}");
    assert!(
        at_cap > 50,
        "at-capacity payloads underrepresented: {at_cap}"
    );
    assert!(mid > 200, "mid-range payloads underrepresented: {mid}");
}

// ---------------------------------------------------------------------------
// Roundtrip property (tier 4)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    /// publish(header, data) then try_consume() yields byte-identical header
    /// and data, for any payload length in [0, capacity]. The first frame on a
    /// fresh ring is at tick 0, so generation 0 matches the consumer's cursor.
    #[test]
    fn publish_consume_roundtrip((cfg, payload) in config_and_payload()) {
        let name = unique_name('r');
        let mut producer = DoomRingProducer::create(&name, cfg).unwrap();
        let mut consumer = DoomRingConsumer::open(&name).unwrap();

        let header = header_with(payload.len() as u64, 0);
        let tick = producer.publish(&header, &payload).unwrap();
        prop_assert_eq!(tick, 0);

        let frame = consumer.try_consume().unwrap().expect("a frame is available");
        prop_assert_eq!(&frame.header[..], &header[..]);
        prop_assert_eq!(&frame.data[..], &payload[..]);
        consumer.advance().unwrap();
        // Drained: nothing more to read.
        prop_assert!(consumer.try_consume().unwrap().is_none());

        drop(consumer);
        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Metamorphic: the header SIZE field, not the producer's write length, governs
// how many bytes the consumer returns.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 150, ..ProptestConfig::default() })]

    /// Publish `full` bytes but stamp the header with a smaller `claimed` size.
    /// The consumer must return exactly `claimed` bytes, and they must be the
    /// prefix of what was written.
    #[test]
    fn header_size_is_source_of_truth(
        // slot_size > header guarantees capacity >= 1 so a non-empty payload
        // exists (the `1..=cap` range below is only valid for cap >= 1).
        (cfg, full) in (backuptics(), (PROBE_FRAME_HEADER_SIZE as u64 + 1)..=8192)
            .prop_map(|(b, s)| RingConfig::new(b, s).expect("valid config"))
            .prop_flat_map(|cfg| {
                let cap = cfg.slot_data_capacity();
                (Just(cfg), prop::collection::vec(any::<u8>(), 1..=cap))
            }),
        frac in 0.0f64..1.0,
    ) {
        let claimed = ((full.len() as f64) * frac) as u64;
        let name = unique_name('m');
        let mut producer = DoomRingProducer::create(&name, cfg).unwrap();
        let consumer = DoomRingConsumer::open(&name).unwrap();

        let header = header_with(claimed, 0);
        producer.publish(&header, &full).unwrap();

        let frame = consumer.try_consume().unwrap().expect("frame available");
        prop_assert_eq!(frame.data.len() as u64, claimed);
        prop_assert_eq!(&frame.data[..], &full[..claimed as usize]);

        drop(consumer);
        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Exception-raising properties (113x more effective; almost nobody writes them)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 150, ..ProptestConfig::default() })]

    /// Any payload strictly larger than the slot's data capacity is rejected
    /// with TensorTooLarge — never written, never truncated.
    #[test]
    fn oversized_payload_rejected(
        (cfg, overflow) in ring_config().prop_flat_map(|cfg| (Just(cfg), 1usize..=4096))
    ) {
        let cap = cfg.slot_data_capacity();
        let payload = vec![0xAB_u8; cap + overflow];
        let name = unique_name('o');
        let mut producer = DoomRingProducer::create(&name, cfg).unwrap();

        let header = header_with(payload.len() as u64, 0);
        let err = producer.publish(&header, &payload).unwrap_err();
        prop_assert!(
            matches!(err, ShmError::TensorTooLarge { tensor_size, slot_capacity }
                if tensor_size == payload.len() && slot_capacity == cap),
            "expected TensorTooLarge, got {:?}", err
        );
        // The producer cursor must not have advanced on a rejected publish.
        prop_assert_eq!(producer.maketic(), 0);

        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 150, ..ProptestConfig::default() })]

    /// A header whose generation does not match the consumer's slot cursor is a
    /// stale/torn read and must surface as StaleSlot, not as bogus data.
    #[test]
    fn stale_generation_rejected(
        (cfg, payload) in config_and_payload(),
        bad_gen in 1u32..=u32::MAX,
    ) {
        // payload is always <= capacity, so publish succeeds; the size check
        // passes (size <= cap) and we reach the generation guard regardless of
        // whether the payload is empty.
        let name = unique_name('g');
        let mut producer = DoomRingProducer::create(&name, cfg).unwrap();
        let consumer = DoomRingConsumer::open(&name).unwrap();

        // Consumer at tick 0 expects generation 0; we stamp a different one.
        let header = header_with(payload.len() as u64, bad_gen);
        producer.publish(&header, &payload).unwrap();

        let res = consumer.try_consume();
        prop_assert!(res.is_err(), "expected StaleSlot error, got Ok");
        let err = res.err().unwrap();
        prop_assert!(
            matches!(err, ShmError::StaleSlot { expected: 0, actual } if actual == bad_gen),
            "expected StaleSlot{{expected:0, actual:{}}}, got {:?}", bad_gen, err
        );

        drop(consumer);
        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 150, ..ProptestConfig::default() })]

    /// A corrupt header claiming a size beyond the slot capacity must be caught
    /// as ReadOutOfBounds before any read is attempted — defense against a torn
    /// or malicious header driving an over-read.
    #[test]
    fn corrupt_oversize_header_rejected(
        cfg in ring_config(),
        excess in 1u64..=1_000_000,
    ) {
        let cap = cfg.slot_data_capacity();
        let name = unique_name('c');
        let mut producer = DoomRingProducer::create(&name, cfg).unwrap();
        let consumer = DoomRingConsumer::open(&name).unwrap();

        // Publish an empty payload (always valid), but stamp the header's size
        // field to claim more than the slot can hold — a torn/corrupt header.
        let payload: Vec<u8> = Vec::new();
        let mut header = header_with(0, 0);
        let bogus = cap as u64 + excess;
        header[FRAME_OFFSET_SIZE..FRAME_OFFSET_SIZE + 8].copy_from_slice(&bogus.to_le_bytes());
        producer.publish(&header, &payload).unwrap();

        let res = consumer.try_consume();
        prop_assert!(res.is_err(), "expected ReadOutOfBounds error, got Ok");
        let err = res.err().unwrap();
        prop_assert!(
            matches!(err, ShmError::ReadOutOfBounds { length, capacity, .. }
                if length == bogus as usize && capacity == cap),
            "expected ReadOutOfBounds, got {:?}", err
        );

        drop(consumer);
        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }
}
