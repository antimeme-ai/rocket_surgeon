//! Stateful, model-based test for the DOOMRING SPSC ring buffer.
//!
//! MATERIA tier 6 (model oracle). We generate a random sequence of
//! `Publish` / `Consume` operations and drive the real ring (a POSIX shm
//! region) in lockstep with an abstract model — a `VecDeque<Vec<u8>>` holding
//! the in-flight frames. After *every* operation the real system must agree
//! with the model:
//!   - Publish succeeds iff the model has spare capacity (< backuptics
//!     in-flight); on success the payload is enqueued, on a full ring it is
//!     rejected with `RingFull` and the model is unchanged.
//!   - Consume returns `Some(front)` iff the model is non-empty, and the
//!     returned bytes equal the model's front; `None` iff empty.
//!
//! This is the property that actually exercises slot wrap-around, the
//! generation guard, and the never-read-an-overwritten-slot invariant all at
//! once. proptest shrinks any failing op sequence to a minimal counterexample.

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config as PtConfig, TestRunner};
use rocket_surgeon_shm::region::ShmRegion;
use rocket_surgeon_shm::ring::{DoomRingConsumer, DoomRingProducer};
use rocket_surgeon_shm::{PROBE_FRAME_HEADER_SIZE, RingConfig, ShmError, serialize_probe_frame};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_name() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("/rss{}-{n}", std::process::id())
}

fn header_with(size: u64, generation: u32) -> [u8; PROBE_FRAME_HEADER_SIZE] {
    serialize_probe_frame(0, 0, 0, 0, 0, &[0u32; 8], 0, 0, size, 0, generation)
}

#[derive(Debug, Clone)]
enum Op {
    Publish(Vec<u8>),
    Consume,
}

/// Small payloads (<= 32 bytes) so any `slot_size` >= 256 holds them.
fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        // Publish-biased so the ring actually fills and wraps.
        3 => prop::collection::vec(any::<u8>(), 0..=32).prop_map(Op::Publish),
        2 => Just(Op::Consume),
    ]
}

fn config() -> impl Strategy<Value = RingConfig> {
    // backuptics small => the ring fills quickly; slot_size >= 256 => cap >= 128.
    (prop::sample::select(vec![1u32, 2, 4, 8]), 256u64..=2048)
        .prop_map(|(b, s)| RingConfig::new(b, s).expect("valid config"))
}

/// Pure simulation of the abstract model, used only to *measure* how often a
/// generated sequence drives the ring into its full state.
fn simulate_reaches_full(cap: u32, ops: &[Op]) -> bool {
    let mut inflight = 0u32;
    for op in ops {
        match op {
            Op::Publish(_) => {
                if inflight >= cap {
                    return true;
                }
                inflight += 1;
            }
            Op::Consume => {
                inflight = inflight.saturating_sub(1);
            }
        }
    }
    false
}

#[test]
fn op_sequences_actually_fill_the_ring() {
    // Evidence that the generator stresses the backpressure path, not just the
    // happy path. If almost no sequence fills the ring, the RingFull branch is
    // untested and this number tells us so.
    const N: u32 = 1500;
    let mut runner = TestRunner::new(PtConfig::default());
    let strat = (config(), prop::collection::vec(op(), 1..=40));
    let (mut full_hits, mut total_ops) = (0u32, 0u64);
    for _ in 0..N {
        let (cfg, ops) = strat.new_tree(&mut runner).unwrap().current();
        total_ops += ops.len() as u64;
        if simulate_reaches_full(cfg.backuptics, &ops) {
            full_hits += 1;
        }
    }
    eprintln!(
        "stateful gen over {N}: full-reaching sequences={full_hits} \
         ({:.1}%), avg ops/seq={:.1}",
        100.0 * f64::from(full_hits) / f64::from(N),
        total_ops as f64 / f64::from(N),
    );
    assert!(
        full_hits > N / 5,
        "backpressure path underexercised: only {full_hits}/{N} sequences fill the ring"
    );
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 250, ..ProptestConfig::default() })]

    #[test]
    fn ring_matches_vecdeque_model(
        cfg in config(),
        ops in prop::collection::vec(op(), 1..=60),
    ) {
        let name = unique_name();
        let mut producer = DoomRingProducer::create(&name, cfg).unwrap();
        let mut consumer = DoomRingConsumer::open(&name).unwrap();
        let mut model: VecDeque<Vec<u8>> = VecDeque::new();
        let backuptics = u64::from(cfg.backuptics);

        for (i, op) in ops.iter().enumerate() {
            match op {
                Op::Publish(payload) => {
                    let tick = producer.maketic();
                    let generation = (tick & 0xFFFF_FFFF) as u32;
                    let header = header_with(payload.len() as u64, generation);
                    let result = producer.publish(&header, payload);

                    if (model.len() as u64) < backuptics {
                        prop_assert!(
                            result.is_ok(),
                            "step {}: publish into a non-full ring (model {} < cap {}) failed: {:?}",
                            i, model.len(), backuptics, result
                        );
                        prop_assert_eq!(result.unwrap(), tick);
                        model.push_back(payload.clone());
                    } else {
                        prop_assert!(
                            matches!(result, Err(ShmError::RingFull { .. })),
                            "step {}: publish into a full ring should be RingFull, got {:?}",
                            i, result
                        );
                        // Model unchanged on rejection.
                    }
                }
                Op::Consume => {
                    let got = consumer.try_consume().unwrap();
                    match model.front() {
                        Some(expected) => {
                            let frame = got.expect("model non-empty => a frame must be available");
                            prop_assert_eq!(
                                &frame.data, expected,
                                "step {}: FIFO order violated", i
                            );
                            consumer.advance().unwrap();
                            model.pop_front();
                        }
                        None => {
                            prop_assert!(
                                got.is_none(),
                                "step {}: model empty but ring yielded a frame", i
                            );
                        }
                    }
                }
            }
            // Cross-check the cursors against the model after every step.
            prop_assert_eq!(
                producer.maketic() - consumer.nettics(),
                model.len() as u64,
                "step {}: in-flight count diverged from model", i
            );
        }

        drop(consumer);
        drop(producer);
        ShmRegion::unlink(&name).unwrap();
    }
}
