//! Stateful model-based + metamorphic tests for the string-interning table.
//!
//! Abstraction function: an `InternTable` abstracts to an ordered map
//! `name -> iid` where iids are assigned 1, 2, 3, … in first-occurrence order.
//! We run a generated sequence of operations against the real table and a
//! reference model in lockstep and assert they agree after every step. This is
//! the gold-standard oracle (MATERIA tier 6, stateful).

use std::collections::HashMap;

use perfetto_writer::intern::InternTable;
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// Reference model: dict + monotone counter starting at 1.
#[derive(Default)]
struct Model {
    map: HashMap<String, u64>,
    next: u64,
}

impl Model {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            next: 1,
        }
    }
    fn intern(&mut self, name: &str) -> u64 {
        if let Some(&iid) = self.map.get(name) {
            return iid;
        }
        let iid = if self.next == 0 { 1 } else { self.next };
        self.map.insert(name.to_owned(), iid);
        self.next = iid + 1;
        iid
    }
    fn get(&self, name: &str) -> Option<u64> {
        self.map.get(name).copied()
    }
}

#[derive(Clone, Debug)]
enum Op {
    Intern(String),
    Get(String),
}

/// A small alphabet of names so collisions (re-interning) are common, plus the
/// occasional novel name. Without a small pool, every `Intern` would be unique
/// and the idempotence path would never be exercised.
fn name() -> impl Strategy<Value = String> {
    prop_oneof![
        6 => prop::sample::select(vec![
            "L0::attn::q_proj".to_string(),
            "L0::attn::k_proj".to_string(),
            "L1::mlp::up".to_string(),
            "component".to_string(),
            String::new(), // empty name is a legal, distinct key
        ]),
        2 => "[a-zA-Z0-9:_]{0,12}",
        1 => any::<String>(), // arbitrary unicode, including control chars
    ]
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        3 => name().prop_map(Op::Intern),
        1 => name().prop_map(Op::Get),
    ]
}

proptest! {
    /// The lockstep stateful property: real table tracks the model exactly, and
    /// every structural invariant holds after each operation.
    #[test]
    fn table_tracks_model(ops in vec(op(), 0..200)) {
        let mut table = InternTable::new();
        let mut model = Model::new();

        for op in &ops {
            match op {
                Op::Intern(s) => {
                    let real = table.intern(s);
                    let expected = model.intern(s);
                    prop_assert_eq!(real, expected, "intern({:?}) diverged", s);
                }
                Op::Get(s) => {
                    prop_assert_eq!(table.get(s), model.get(s), "get({:?}) diverged", s);
                }
            }

            // Invariants checked after every step:
            prop_assert_eq!(table.len(), model.map.len());
            prop_assert_eq!(table.is_empty(), model.map.is_empty());

            // entries() is the exact inverse of intern/get and forms a bijection.
            let entries: Vec<(u64, String)> =
                table.entries().map(|(iid, n)| (iid, n.to_owned())).collect();
            prop_assert_eq!(entries.len(), model.map.len());
            let mut iids: Vec<u64> = entries.iter().map(|(iid, _)| *iid).collect();
            iids.sort_unstable();
            iids.dedup();
            prop_assert_eq!(iids.len(), entries.len(), "duplicate iid in table — not injective");
            for (iid, n) in &entries {
                prop_assert_eq!(table.get(n), Some(*iid), "entries/get inconsistent for {:?}", n);
                prop_assert_eq!(model.get(n), Some(*iid), "table has name model lacks");
            }
            // Density: iids are exactly the contiguous range 1..=len.
            let expected_ids: Vec<u64> = (1..=model.map.len() as u64).collect();
            prop_assert_eq!(iids, expected_ids, "iids must be dense and start at 1");
        }
    }

    /// Idempotence (metamorphic): re-interning an already-seen name returns the
    /// same iid and never grows the table.
    #[test]
    fn reinterning_is_stable(names in vec(name(), 1..50)) {
        let mut table = InternTable::new();
        let first: Vec<u64> = names.iter().map(|n| table.intern(n)).collect();
        let len_after_first = table.len();
        // Replay the whole sequence again.
        for (n, &id) in names.iter().zip(&first) {
            prop_assert_eq!(table.intern(n), id, "second pass changed iid for {:?}", n);
        }
        prop_assert_eq!(table.len(), len_after_first, "replay grew the table");
    }

    /// Injectivity: distinct names always get distinct iids.
    #[test]
    fn distinct_names_distinct_iids(names in prop::collection::hash_set(name(), 0..60)) {
        let mut table = InternTable::new();
        let mut ids = std::collections::HashSet::new();
        for n in &names {
            let id = table.intern(n);
            prop_assert!(ids.insert(id), "iid {} reused for distinct name {:?}", id, n);
        }
        prop_assert_eq!(table.len(), names.len());
    }

    /// Metamorphic order-independence: the SET of names and the table size are
    /// invariant under any permutation of the input order. (The id *assignment*
    /// is order-dependent — that's first-occurrence rank — but the structure is
    /// not.) We feed the same multiset in two orders and compare the name-sets.
    #[test]
    fn order_independent_membership(
        names in vec(name(), 0..40),
        seed in any::<u64>(),
    ) {
        let mut a = InternTable::new();
        for n in &names { a.intern(n); }

        // A cheap deterministic permutation driven by `seed`: rotate.
        let mut permuted = names.clone();
        if !permuted.is_empty() {
            let k = (seed % permuted.len() as u64) as usize;
            permuted.rotate_left(k);
        }
        let mut b = InternTable::new();
        for n in &permuted { b.intern(n); }

        prop_assert_eq!(a.len(), b.len());
        let names_a: std::collections::HashSet<String> =
            a.entries().map(|(_, n)| n.to_owned()).collect();
        let names_b: std::collections::HashSet<String> =
            b.entries().map(|(_, n)| n.to_owned()).collect();
        prop_assert_eq!(names_a, names_b, "permutation changed the interned name-set");
    }
}

/// Generator-distribution evidence: classify the op stream and confirm that
/// re-interning (idempotence path) and novel interning are both well exercised.
#[test]
fn generator_exercises_reintern_and_novel() {
    const N: usize = 2_000;
    let mut runner = TestRunner::deterministic();
    let strat = vec(op(), 0..200);
    let (mut reintern, mut novel, mut gets, mut total_intern) = (0u64, 0u64, 0u64, 0u64);
    for _ in 0..N {
        let ops = strat.new_tree(&mut runner).unwrap().current();
        let mut seen = std::collections::HashSet::new();
        for op in ops {
            match op {
                Op::Intern(s) => {
                    total_intern += 1;
                    if seen.insert(s) {
                        novel += 1;
                    } else {
                        reintern += 1;
                    }
                }
                Op::Get(_) => gets += 1,
            }
        }
    }
    eprintln!(
        "intern op distribution over {N} sequences: novel={novel} reintern={reintern} gets={gets} (total intern={total_intern})"
    );
    assert!(
        reintern > 0,
        "re-intern path never exercised — name pool too wide"
    );
    assert!(novel > 0, "novel-intern path never exercised");
    assert!(gets > 0, "get path never exercised");
}
