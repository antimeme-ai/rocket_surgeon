//! Stateful model-based test for `ProbeRegistry` (MATERIA tier 6, stateful).
//!
//! Per Hughes: generate sequences of API calls, maintain an abstract model in
//! parallel, and assert the real registry agrees with the model after *every*
//! step. proptest shrinks a failing sequence to a minimal reproducer.
//!
//! Implementation relation: the registry is an associative store keyed by probe
//! id, with insertion-sequence tiebreaking and a `(priority, seq)` total order on
//! listing/matching. The model is a `BTreeMap<id, (def, seq)>` plus a monotonic
//! `next_seq` counter — the simplest structure that captures storage, dedup,
//! enable/disable, removal, ordering, and match-filtering.
//!
//! Fault model: dropped/duplicated inserts, wrong dedup precedence (the real
//! `define` checks duplicate-id *before* parsing the point — a subtle ordering
//! the model must mirror), mis-sorted listings, seq-counter drift on rejected
//! inserts, and enable/disable/remove on absent ids.

use std::collections::BTreeMap;

use proptest::prelude::*;

use rocket_surgeon_probes::grammar::ProbePoint;
use rocket_surgeon_probes::registry::{ProbeRegistry, RegistryError};
use rocket_surgeon_protocol::types::{ProbeAction, ProbeDefinition};

// A small, fixed alphabet of ids forces collisions (so dedup is exercised) and
// keeps the model legible.
const IDS: [&str; 4] = ["p0", "p1", "p2", "p3"];

// Valid points, deliberately overlapping under wildcard matching so that
// `matching`/`matching_enabled` filter non-trivially.
const VALID_POINTS: [&str; 6] = [
    "llama:0:12:attn.o_proj:0:output",
    "llama:*:*:mlp:*:output",
    "mixtral:*:8:experts[3]:0:output",
    "*:*:*:*:*:*",
    "llama:0:13:mlp:0:input",
    "mixtral:0:8:router:0:pre_topk",
];

// Strings the grammar must reject — define() should surface InvalidPoint for
// these (unless the id already exists, in which case DuplicateId wins first).
const INVALID_POINTS: [&str; 3] = ["bad", "llama:0:12:mlp:output", ":0:12:mlp:0:output"];

// Concrete targets for matching queries.
const TARGETS: [&str; 4] = [
    "llama:0:12:attn.o_proj:0:output",
    "llama:0:13:mlp:0:input",
    "mixtral:0:8:experts[3]:0:output",
    "mixtral:0:8:router:0:pre_topk",
];

#[derive(Debug, Clone)]
enum Op {
    Define {
        id: usize,
        point: PointChoice,
        priority: i32,
        enabled: bool,
    },
    Enable(usize),
    Disable(usize),
    Remove(usize),
    Get(usize),
    Matching(usize),
    MatchingEnabled(usize),
    ListAndLen,
    ActiveIds,
}

#[derive(Debug, Clone)]
enum PointChoice {
    Valid(usize),
    Invalid(usize),
}

fn point_str(choice: &PointChoice) -> &'static str {
    match *choice {
        PointChoice::Valid(i) => VALID_POINTS[i],
        PointChoice::Invalid(i) => INVALID_POINTS[i],
    }
}

fn arb_point_choice() -> impl Strategy<Value = PointChoice> {
    prop_oneof![
        4 => (0..VALID_POINTS.len()).prop_map(PointChoice::Valid),
        1 => (0..INVALID_POINTS.len()).prop_map(PointChoice::Invalid),
    ]
}

fn arb_op() -> impl Strategy<Value = Op> {
    prop_oneof![
        // Weight Define heavily so the store actually fills.
        5 => (0..IDS.len(), arb_point_choice(), -2i32..=2, any::<bool>())
            .prop_map(|(id, point, priority, enabled)| Op::Define { id, point, priority, enabled }),
        2 => (0..IDS.len()).prop_map(Op::Enable),
        2 => (0..IDS.len()).prop_map(Op::Disable),
        2 => (0..IDS.len()).prop_map(Op::Remove),
        1 => (0..IDS.len()).prop_map(Op::Get),
        2 => (0..TARGETS.len()).prop_map(Op::Matching),
        2 => (0..TARGETS.len()).prop_map(Op::MatchingEnabled),
        1 => Just(Op::ListAndLen),
        1 => Just(Op::ActiveIds),
    ]
}

// --- The abstract model ---

struct Model {
    // id -> (definition, insertion sequence)
    entries: BTreeMap<String, (ProbeDefinition, u64)>,
    next_seq: u64,
}

impl Model {
    fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            next_seq: 0,
        }
    }

    /// Listing order: sort by (priority, seq). seq is unique so the order is total.
    fn sorted(&self) -> Vec<ProbeDefinition> {
        let mut v: Vec<_> = self.entries.values().collect();
        v.sort_by_key(|(def, seq)| (def.priority, *seq));
        v.into_iter().map(|(def, _)| def.clone()).collect()
    }
}

fn make_def(id: &str, point: &str, priority: i32, enabled: bool) -> ProbeDefinition {
    ProbeDefinition {
        id: id.to_owned(),
        point: point.to_owned(),
        action: ProbeAction::Capture,
        config: None,
        enabled,
        priority,
    }
}

#[allow(clippy::too_many_lines)] // one match arm per registry operation; splitting hurts readability
fn run_sequence(ops: &[Op]) -> Result<(), TestCaseError> {
    let mut reg = ProbeRegistry::new();
    let mut model = Model::new();

    for (step, op) in ops.iter().enumerate() {
        match op {
            Op::Define {
                id,
                point,
                priority,
                enabled,
            } => {
                let id_str = IDS[*id];
                let point_s = point_str(point);
                let def = make_def(id_str, point_s, *priority, *enabled);

                let real = reg.define(def.clone());

                // Model: duplicate-id is checked BEFORE the point is parsed.
                if model.entries.contains_key(id_str) {
                    prop_assert!(
                        matches!(real, Err(RegistryError::DuplicateId { .. })),
                        "step {}: expected DuplicateId for {}, got {:?}",
                        step,
                        id_str,
                        real,
                    );
                } else if ProbePoint::parse(point_s).is_err() {
                    prop_assert!(
                        matches!(real, Err(RegistryError::InvalidPoint(_))),
                        "step {}: expected InvalidPoint for {:?}, got {:?}",
                        step,
                        point_s,
                        real,
                    );
                    // Rejected insert must NOT advance the sequence counter.
                } else {
                    prop_assert!(
                        real.as_ref().ok().map(String::as_str) == Some(id_str),
                        "step {}: define should have returned id {}, got {:?}",
                        step,
                        id_str,
                        real,
                    );
                    model
                        .entries
                        .insert(id_str.to_owned(), (def, model.next_seq));
                    model.next_seq += 1;
                }
            }
            Op::Enable(id) => {
                let id_str = IDS[*id];
                let real = reg.enable(id_str);
                match model.entries.get_mut(id_str) {
                    Some((def, _)) => {
                        def.enabled = true;
                        prop_assert_eq!(real.ok(), Some(def.clone()), "enable mismatch");
                    }
                    None => prop_assert!(
                        matches!(real, Err(RegistryError::NotFound { .. })),
                        "step {}: enable absent id should be NotFound",
                        step,
                    ),
                }
            }
            Op::Disable(id) => {
                let id_str = IDS[*id];
                let real = reg.disable(id_str);
                match model.entries.get_mut(id_str) {
                    Some((def, _)) => {
                        def.enabled = false;
                        prop_assert_eq!(real.ok(), Some(def.clone()), "disable mismatch");
                    }
                    None => prop_assert!(
                        matches!(real, Err(RegistryError::NotFound { .. })),
                        "step {}: disable absent id should be NotFound",
                        step,
                    ),
                }
            }
            Op::Remove(id) => {
                let id_str = IDS[*id];
                let real = reg.remove(id_str);
                match model.entries.remove(id_str) {
                    Some((def, _)) => {
                        prop_assert_eq!(real.ok(), Some(def), "remove mismatch");
                    }
                    None => prop_assert!(
                        matches!(real, Err(RegistryError::NotFound { .. })),
                        "step {}: remove absent id should be NotFound",
                        step,
                    ),
                }
            }
            Op::Get(id) => {
                let id_str = IDS[*id];
                let real = reg.get(id_str);
                let expected = model.entries.get(id_str).map(|(def, _)| def);
                prop_assert_eq!(real, expected, "get mismatch");
            }
            Op::Matching(t) => {
                let target = ProbePoint::parse(TARGETS[*t]).unwrap();
                let real = reg.matching(&target);
                let expected: Vec<ProbeDefinition> = model
                    .sorted()
                    .into_iter()
                    .filter(|def| ProbePoint::parse(&def.point).unwrap().matches(&target))
                    .collect();
                prop_assert_eq!(real, expected, "matching mismatch");
            }
            Op::MatchingEnabled(t) => {
                let target = ProbePoint::parse(TARGETS[*t]).unwrap();
                let real = reg.matching_enabled(&target);
                let expected: Vec<ProbeDefinition> = model
                    .sorted()
                    .into_iter()
                    .filter(|def| {
                        def.enabled && ProbePoint::parse(&def.point).unwrap().matches(&target)
                    })
                    .collect();
                prop_assert_eq!(real, expected, "matching_enabled mismatch");
            }
            Op::ListAndLen => {
                prop_assert_eq!(reg.list(), model.sorted(), "list mismatch");
                prop_assert_eq!(reg.len(), model.entries.len(), "len mismatch");
                prop_assert_eq!(
                    reg.is_empty(),
                    model.entries.is_empty(),
                    "is_empty mismatch",
                );
            }
            Op::ActiveIds => {
                // active_probe_ids order is unspecified (HashMap iteration) — compare as sets.
                let mut real = reg.active_probe_ids();
                real.sort();
                let mut expected: Vec<String> = model
                    .entries
                    .values()
                    .filter(|(def, _)| def.enabled)
                    .map(|(def, _)| def.id.clone())
                    .collect();
                expected.sort();
                prop_assert_eq!(real, expected, "active_probe_ids mismatch");
            }
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// The real registry tracks the abstract model across an arbitrary op
    /// sequence. Any divergence is shrunk to a minimal failing sequence.
    #[test]
    fn registry_tracks_model(ops in prop::collection::vec(arb_op(), 0..40)) {
        run_sequence(&ops)?;
    }
}
