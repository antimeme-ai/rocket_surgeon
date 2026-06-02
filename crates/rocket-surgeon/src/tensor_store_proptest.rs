//! Stateful model-based & exception-raising property tests for the
//! content-addressed tensor store (B004 — callsign INDIA).
//!
//! ## What this tests
//!
//! [`crate::tensor_store::TensorStore`] is a BLAKE3 content-addressed cache with
//! LRU eviction bounded by both an entry count and a byte budget. The existing
//! example suite pins it at oracle tiers 2-3. This module climbs to:
//!
//!   * tier 6 — STATEFUL MODEL-BASED. We generate sequences of store operations
//!     (insert / get / raw-data / summarize / slice) over a small payload pool,
//!     drive them against the real store and an independently-written abstract
//!     [`Model`] (a `HashMap` of contents + an explicit recency `Vec`) in
//!     lockstep, and after EVERY operation assert the real store agrees with the
//!     model on: the returned handle / presence / bytes, the live id-set,
//!     `len()` and `bytes_used()`. The model encodes the store's exact LRU
//!     contract — *which operations count as an access* (every hit touches
//!     recency; an out-of-bounds slice on a present id still touches; a miss
//!     never touches) and the eviction order (least-recently-touched first).
//!     This is the Hughes (2016) "abstract model in parallel" pattern, where
//!     the real bugs in caches live.
//!
//!   * tier 6 — CONTENT-ADDRESSING. Identical bytes always hash to one entry
//!     (dedup, no byte growth); distinct bytes hash to distinct entries.
//!
//!   * tier 5 — EXCEPTION-RAISING. `slice` is a total function of its bounds:
//!     in-range ⇒ exactly the right bytes; out-of-range or `offset+len`
//!     overflow ⇒ `SliceOutOfBounds`; a missing id ⇒ `NotFound`. Never a panic.
//!
//!   * a generator-distribution test so we know the op corpus actually triggers
//!     eviction, dedup, hits, misses, and all three slice outcomes.
//!
//! ## Modeling choices (mirrors `tensor_store.rs` exactly)
//!
//!   * Recency is a monotonic counter in the impl (`last_access_gen`); the model
//!     reproduces the induced order with a move-to-back `Vec`, which yields the
//!     identical least-recently-touched victim.
//!   * On a NEW insert the impl evicts *before* the new entry receives a
//!     generation, so the new entry is never its own eviction candidate; the
//!     model evicts from its order vec before pushing the new id.
//!   * A dedup insert (id already present) touches recency and returns without
//!     evicting; an oversized single payload evicts everything then inserts
//!     anyway (so the store may exceed `max_bytes` at `len() == 1`).

use std::collections::{HashMap, HashSet};

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

use rocket_surgeon_protocol::types::DType;

use crate::tensor_store::{StoreError, TensorStore};

fn hex_id(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

// ── abstract reference model ────────────────────────────────────────────────

struct Model {
    /// id → contents (the model keeps full bytes so it can answer `slice`).
    data: HashMap<String, Vec<u8>>,
    /// Recency order: index 0 is the least-recently-touched (next victim),
    /// the last element is the most-recently-touched.
    order: Vec<String>,
    max_entries: usize,
    max_bytes: usize,
    bytes: usize,
}

impl Model {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            data: HashMap::new(),
            order: Vec::new(),
            max_entries,
            max_bytes,
            bytes: 0,
        }
    }

    /// Move an existing id to most-recently-used. No-op if absent.
    fn touch(&mut self, id: &str) {
        if let Some(pos) = self.order.iter().position(|x| x == id) {
            let s = self.order.remove(pos);
            self.order.push(s);
        }
    }

    fn evict_oldest(&mut self) {
        if self.order.is_empty() {
            return;
        }
        let id = self.order.remove(0);
        if let Some(d) = self.data.remove(&id) {
            self.bytes -= d.len();
        }
    }

    fn insert(&mut self, data: Vec<u8>) -> String {
        let id = hex_id(&data);
        if self.data.contains_key(&id) {
            self.touch(&id);
            return id;
        }
        let len = data.len();
        while !self.data.is_empty()
            && (self.data.len() >= self.max_entries || self.bytes + len > self.max_bytes)
        {
            self.evict_oldest();
        }
        self.data.insert(id.clone(), data);
        self.bytes += len;
        self.order.push(id.clone());
        id
    }

    /// Returns whether the id was present (and touches it if so).
    fn access(&mut self, id: &str) -> bool {
        if self.data.contains_key(id) {
            self.touch(id);
            true
        } else {
            false
        }
    }

    /// Mirrors `TensorStore::slice`: a missing id is `NotFound` and does NOT
    /// touch recency; a present id touches recency *then* bounds-checks.
    fn slice(&mut self, id: &str, offset: u64, len: u64) -> Result<Vec<u8>, SliceKind> {
        if !self.data.contains_key(id) {
            return Err(SliceKind::NotFound);
        }
        self.touch(id);
        let d = &self.data[id];
        let data_len = d.len() as u64;
        match offset.checked_add(len) {
            Some(end) if end <= data_len => Ok(d[offset as usize..end as usize].to_vec()),
            _ => Err(SliceKind::OutOfBounds),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SliceKind {
    NotFound,
    OutOfBounds,
}

// ── operation alphabet ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Op {
    Insert(usize),
    Get(usize),
    RawData(usize),
    Summarize(usize),
    Slice(usize, u64, u64),
}

/// Aggregate counters used both as a liveness sanity check and by the
/// distribution test.
#[derive(Default, Debug)]
struct Counters {
    inserts: usize,
    dedup_hits: usize,
    evictions_observed: usize,
    oversized_inserts: usize,
    get_hits: usize,
    get_misses: usize,
    slice_ok: usize,
    slice_oob: usize,
    slice_not_found: usize,
}

fn op_strategy(pool_len: usize) -> impl Strategy<Value = Op> {
    // Bias offset/len small (payloads are <40 bytes) so in-range slices are
    // actually hit, while still reaching huge offsets for the overflow path.
    let offset = prop_oneof![4 => 0u64..40, 1 => any::<u64>()];
    let len = prop_oneof![4 => 0u64..40, 1 => any::<u64>()];
    prop_oneof![
        (0..pool_len).prop_map(Op::Insert),
        (0..pool_len).prop_map(Op::Get),
        (0..pool_len).prop_map(Op::RawData),
        (0..pool_len).prop_map(Op::Summarize),
        (0..pool_len, offset, len).prop_map(|(i, o, l)| Op::Slice(i, o, l)),
    ]
}

type Scenario = (Vec<Vec<u8>>, usize, usize, Vec<Op>);

fn scenario() -> impl Strategy<Value = Scenario> {
    let pool = prop::collection::vec(prop::collection::vec(any::<u8>(), 0..40), 2..6);
    (pool, 1usize..5, 1usize..120).prop_flat_map(|(pool, max_entries, max_bytes)| {
        let pool_len = pool.len();
        (
            Just(pool),
            Just(max_entries),
            Just(max_bytes),
            prop::collection::vec(op_strategy(pool_len), 0..64),
        )
    })
}

/// Drive a scenario against store + model in lockstep, asserting agreement
/// after every operation. Returns the aggregate counters on success.
fn run_scenario(
    pool: &[Vec<u8>],
    max_entries: usize,
    max_bytes: usize,
    ops: &[Op],
) -> Result<Counters, TestCaseError> {
    let mut store = TensorStore::with_limits(max_entries, max_bytes);
    let mut model = Model::new(max_entries, max_bytes);
    let mut c = Counters::default();

    for op in ops {
        match op {
            Op::Insert(i) => {
                let data = pool[*i].clone();
                let len_before = model.data.len();
                let was_present = model.data.contains_key(&hex_id(&data));

                let real = store
                    .insert(
                        data.clone(),
                        vec![data.len() as u64],
                        DType::Uint8,
                        "cpu".into(),
                    )
                    .tensor_id;
                let modeled = model.insert(data.clone());

                // Content-addressing: the handle id is the BLAKE3 of the bytes,
                // and the two implementations agree.
                prop_assert_eq!(&real, &modeled);
                prop_assert_eq!(&real, &hex_id(&data));

                c.inserts += 1;
                if was_present {
                    c.dedup_hits += 1;
                } else if model.data.len() <= len_before {
                    c.evictions_observed += 1;
                }
                if data.len() > max_bytes {
                    c.oversized_inserts += 1;
                }
            }
            Op::Get(i) => {
                let id = hex_id(&pool[*i]);
                let real = store.get(&id).is_some();
                let modeled = model.access(&id);
                prop_assert_eq!(real, modeled);
                if real {
                    c.get_hits += 1;
                } else {
                    c.get_misses += 1;
                }
            }
            Op::RawData(i) => {
                let id = hex_id(&pool[*i]);
                let real = store.raw_data(&id).map(<[u8]>::to_vec);
                let modeled = model.access(&id);
                prop_assert_eq!(real.is_some(), modeled);
                if let Some(bytes) = real {
                    // The bytes behind an id are exactly the content that hashes
                    // to it.
                    prop_assert_eq!(&bytes, &pool[*i]);
                }
            }
            Op::Summarize(i) => {
                let id = hex_id(&pool[*i]);
                let real = store.summarize(&id).is_some();
                let modeled = model.access(&id);
                prop_assert_eq!(real, modeled);
            }
            Op::Slice(i, offset, len) => {
                let id = hex_id(&pool[*i]);
                let real = store.slice(&id, *offset, *len);
                let modeled = model.slice(&id, *offset, *len);
                match (&real, &modeled) {
                    (Ok(rb), Ok(mb)) => {
                        prop_assert_eq!(rb, mb);
                        c.slice_ok += 1;
                    }
                    (Err(StoreError::NotFound(_)), Err(SliceKind::NotFound)) => {
                        c.slice_not_found += 1;
                    }
                    (Err(StoreError::SliceOutOfBounds { .. }), Err(SliceKind::OutOfBounds)) => {
                        c.slice_oob += 1;
                    }
                    _ => {
                        return Err(TestCaseError::fail(format!(
                            "slice outcome mismatch: real={real:?} model={modeled:?}"
                        )));
                    }
                }
            }
        }

        // ── post-operation invariants: real store == abstract model ──────────
        prop_assert_eq!(store.len(), model.data.len());
        prop_assert_eq!(store.bytes_used(), model.bytes);

        let real_ids: HashSet<String> = store.ids().map(str::to_owned).collect();
        let model_ids: HashSet<String> = model.data.keys().cloned().collect();
        prop_assert_eq!(&real_ids, &model_ids);
        for id in &model_ids {
            prop_assert!(store.contains(id));
        }

        // Bound sanity: never more than max_entries entries; never over the
        // byte budget except the deliberate single-oversized-entry case.
        prop_assert!(store.len() <= max_entries);
        prop_assert!(
            store.bytes_used() <= max_bytes || store.len() <= 1,
            "bytes {} over budget {} with {} entries",
            store.bytes_used(),
            max_bytes,
            store.len()
        );
    }

    Ok(c)
}

// ── tier-6 stateful model-based ──────────────────────────────────────────────

proptest! {
    #[test]
    fn store_matches_lru_model((pool, max_entries, max_bytes, ops) in scenario()) {
        run_scenario(&pool, max_entries, max_bytes, &ops)?;
    }
}

// ── tier-6 content-addressing ────────────────────────────────────────────────

proptest! {
    /// Re-inserting identical bytes dedups to a single entry with the BLAKE3 id
    /// and no byte growth.
    #[test]
    fn identical_bytes_dedup(data in prop::collection::vec(any::<u8>(), 0..64), reps in 1usize..6) {
        let mut store = TensorStore::new();
        let expected = hex_id(&data);
        for _ in 0..reps {
            let h = store.insert(data.clone(), vec![data.len() as u64], DType::Uint8, "cpu".into());
            prop_assert_eq!(&h.tensor_id, &expected);
        }
        prop_assert_eq!(store.len(), 1);
        prop_assert_eq!(store.bytes_used(), data.len());
    }

    /// Distinct contents hash to distinct ids; identical contents collapse.
    #[test]
    fn content_determines_identity(
        a in prop::collection::vec(any::<u8>(), 0..64),
        b in prop::collection::vec(any::<u8>(), 0..64),
    ) {
        let mut store = TensorStore::new();
        let ha = store.insert(a.clone(), vec![a.len() as u64], DType::Uint8, "cpu".into()).tensor_id;
        let hb = store.insert(b.clone(), vec![b.len() as u64], DType::Uint8, "cpu".into()).tensor_id;
        if a == b {
            prop_assert_eq!(&ha, &hb);
            prop_assert_eq!(store.len(), 1);
        } else {
            // BLAKE3 collisions on distinct inputs are cryptographically
            // impossible, so distinct content ⇒ distinct id ⇒ two entries.
            prop_assert_ne!(&ha, &hb);
            prop_assert_eq!(store.len(), 2);
        }
    }
}

// ── tier-5 exception-raising: slice as a total bounds function ────────────────

proptest! {
    #[test]
    fn slice_bounds_oracle(
        data in prop::collection::vec(any::<u8>(), 1..64),
        offset in prop_oneof![0u64..80, any::<u64>()],
        len in prop_oneof![0u64..80, any::<u64>()],
    ) {
        let mut store = TensorStore::new();
        let h = store.insert(data.clone(), vec![data.len() as u64], DType::Uint8, "cpu".into());
        let data_len = data.len() as u64;
        let res = store.slice(&h.tensor_id, offset, len);
        match offset.checked_add(len) {
            Some(end) if end <= data_len => {
                let s = res.expect("in-range slice must succeed");
                prop_assert_eq!(s, data[offset as usize..end as usize].to_vec());
            }
            _ => {
                let is_oob = matches!(res, Err(StoreError::SliceOutOfBounds { .. }));
                prop_assert!(is_oob, "expected SliceOutOfBounds, got {res:?}");
            }
        }
    }

    /// Any operation against an id that was never inserted reports absence —
    /// never a panic.
    #[test]
    fn missing_id_reports_absence(
        id in "[0-9a-f]{64}",
        offset in any::<u64>(),
        len in 0u64..64,
    ) {
        let mut store = TensorStore::new();
        prop_assert!(matches!(store.slice(&id, offset, len), Err(StoreError::NotFound(_))));
        prop_assert!(store.summarize(&id).is_none());
        prop_assert!(store.raw_data(&id).is_none());
        prop_assert!(store.get(&id).is_none());
        prop_assert!(!store.contains(&id));
    }
}

// ── generator-distribution measurement ───────────────────────────────────────

#[test]
fn op_generator_distribution_is_non_trivial() {
    let mut runner = TestRunner::deterministic();
    let strat = scenario();
    let n = 400usize;

    let mut agg = Counters::default();
    let mut with_eviction = 0usize;
    let mut with_dedup = 0usize;

    for _ in 0..n {
        let tree = strat
            .new_tree(&mut runner)
            .expect("strategy produces a value");
        let (pool, me, mb, ops) = tree.current();
        let c = run_scenario(&pool, me, mb, &ops).expect("sampled scenario stays in lockstep");
        if c.evictions_observed > 0 {
            with_eviction += 1;
        }
        if c.dedup_hits > 0 {
            with_dedup += 1;
        }
        agg.inserts += c.inserts;
        agg.dedup_hits += c.dedup_hits;
        agg.evictions_observed += c.evictions_observed;
        agg.oversized_inserts += c.oversized_inserts;
        agg.get_hits += c.get_hits;
        agg.get_misses += c.get_misses;
        agg.slice_ok += c.slice_ok;
        agg.slice_oob += c.slice_oob;
        agg.slice_not_found += c.slice_not_found;
    }

    let pct = |k: usize| k as f64 / n as f64 * 100.0;
    eprintln!(
        "tensor_store op generator over {n} scenarios:\n  \
         scenarios w/ eviction: {:.1}%   w/ dedup: {:.1}%\n  \
         totals — inserts: {}  dedup-hits: {}  evictions: {}  oversized: {}\n  \
         get hit/miss: {}/{}   slice ok/oob/not-found: {}/{}/{}",
        pct(with_eviction),
        pct(with_dedup),
        agg.inserts,
        agg.dedup_hits,
        agg.evictions_observed,
        agg.oversized_inserts,
        agg.get_hits,
        agg.get_misses,
        agg.slice_ok,
        agg.slice_oob,
        agg.slice_not_found,
    );

    // The corpus must exercise the interesting paths, not just trivial inserts.
    assert!(with_eviction > n / 5, "too few scenarios trigger eviction");
    assert!(with_dedup > n / 10, "too few scenarios trigger dedup");
    assert!(
        agg.oversized_inserts > 0,
        "never exercised oversized insert"
    );
    assert!(
        agg.get_hits > 0 && agg.get_misses > 0,
        "get hit/miss imbalance"
    );
    assert!(
        agg.slice_ok > 0 && agg.slice_oob > 0 && agg.slice_not_found > 0,
        "slice outcomes not all covered: ok={} oob={} not_found={}",
        agg.slice_ok,
        agg.slice_oob,
        agg.slice_not_found
    );
}
