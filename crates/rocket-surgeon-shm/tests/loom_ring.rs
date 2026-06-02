//! Exhaustive-interleaving test of the DOOMRING SPSC ordering protocol.
//!
//! WHAT THIS TESTS — AND WHAT IT DOES NOT.
//! The production ring (`ring.rs`) synchronizes a producer and a consumer that
//! live in *different processes* through two `AtomicU64` cursors (`maketic`,
//! `nettics`) written with `Release` and read with `Acquire`, layered over a
//! raw `mmap`'d region. loom cannot instrument that raw pointer or the kernel
//! mapping, so this test re-expresses the *algorithm* — the cursor protocol and
//! the slot read/write discipline — over `loom`'s model atomics and lets loom
//! explore every legal interleaving and memory-ordering outcome.
//!
//! Oracle (model / linearizability flavored): the consumer must observe exactly
//! the values the producer published, in FIFO order, with NO stale or torn slot
//! read — including across slot *reuse* (wrap-around), which is the classic
//! place a missing Acquire/Release lets a consumer's read race a producer's
//! overwrite. If you weaken any `Release`/`Acquire` here to `Relaxed`, loom
//! finds a counterexample where the slot assertion fails. That is the proof the
//! ordering in the real code is load-bearing, not decorative.
//!
//! Run with:  RUSTFLAGS="--cfg loom" cargo test -p rocket-surgeon-shm --test `loom_ring`
//! (A normal `cargo test` / clippy build compiles this file to nothing.)

#![cfg(loom)]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicU64, Ordering};

/// Distinct, non-zero payload for a given tick so a stale read (which would see
/// 0 or a neighbouring tick's value) is detectable.
fn payload(tick: u64) -> u64 {
    tick.wrapping_mul(0x9E37_79B9).wrapping_add(0x1234)
}

/// One producer publishing `n` ticks and one consumer draining `n` ticks over a
/// ring of `cap` slots (cap must be a power of two). Asserts FIFO + no stale
/// reads under every interleaving loom explores.
fn run_spsc(cap: u64, n: u64) {
    loom::model(move || {
        let mask = cap - 1;
        let maketic = Arc::new(AtomicU64::new(0));
        let nettics = Arc::new(AtomicU64::new(0));
        let slots: Arc<Vec<AtomicU64>> = Arc::new((0..cap).map(|_| AtomicU64::new(0)).collect());

        let producer = {
            let maketic = maketic.clone();
            let nettics = nettics.clone();
            let slots = slots.clone();
            loom::thread::spawn(move || {
                for tick in 0..n {
                    // Backpressure: wait until the slot (tick & mask), last used
                    // at tick-cap, has been consumed (nettics moved past it).
                    loop {
                        let nt = nettics.load(Ordering::Acquire);
                        if tick - nt < cap {
                            break;
                        }
                        loom::thread::yield_now();
                    }
                    let slot = (tick & mask) as usize;
                    // Data write, then the Release store that publishes it.
                    slots[slot].store(payload(tick), Ordering::Relaxed);
                    maketic.store(tick + 1, Ordering::Release);
                }
            })
        };

        let consumer = {
            loom::thread::spawn(move || {
                let mut nt = 0u64;
                while nt < n {
                    let mk = maketic.load(Ordering::Acquire);
                    if mk > nt {
                        let slot = (nt & mask) as usize;
                        let v = slots[slot].load(Ordering::Relaxed);
                        assert_eq!(
                            v,
                            payload(nt),
                            "stale/torn slot read at tick {nt}: got {v:#x}, want {:#x}",
                            payload(nt)
                        );
                        nt += 1;
                        // Release so the producer's Acquire-load of nettics sees
                        // that this slot is free to overwrite.
                        nettics.store(nt, Ordering::Release);
                    } else {
                        loom::thread::yield_now();
                    }
                }
            })
        };

        producer.join().unwrap();
        consumer.join().unwrap();
    });
}

/// No wrap: 2 publishes into a 2-slot ring. Smallest proof of the publish
/// happens-before consume ordering.
#[test]
fn spsc_no_wrap() {
    run_spsc(2, 2);
}

/// With wrap and backpressure: 3 publishes into a 2-slot ring forces slot 0 to
/// be reused, so the producer must wait for the consumer's `nettics` Release
/// before overwriting. This is the interleaving that catches a missing
/// Acquire/Release on the cursors.
#[test]
fn spsc_wrap_reuse() {
    run_spsc(2, 3);
}
