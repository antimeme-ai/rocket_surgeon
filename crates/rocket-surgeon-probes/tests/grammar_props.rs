//! Property / metamorphic / model-based tests for the probe-point grammar.
//!
//! MATERIA oracle tiers exercised here:
//!
//! - tier 4 (roundtrip): `parse ∘ render == id` over generated ASTs.
//! - tier 6 (model): `matches` against an `Option`-per-field reference.
//! - exception-raising: structurally-broken strings yield the specific
//!   `ParseError`, never a panic, never a wrong-accept.
//!
//! The fault model: the parser is a hand-written winnow recursive-descent over a
//! 6-field colon-delimited grammar. The faults we hunt are (a) render/parse
//! disagreement (non-injective render, dropped fields, mis-ordered fields),
//! (b) over-acceptance (a malformed string parsed as some valid point), and
//! (c) panics on adversarial input (overflow, unclosed brackets, empties).
//!
//! Generator distribution is asserted explicitly in `generator_distribution` so
//! that "we tested wildcards / `MoE` indexing" is evidence, not hope.

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

use rocket_surgeon_probes::grammar::{
    ComponentOrWild, ComponentSeg, NameOrWild, NumOrWild, ProbePoint,
};

// ---------------------------------------------------------------------------
// Generators — restricted to the exact AST space the parser can *produce*.
//
// The parser never yields an empty component path (it uses `separated(1.., …)`)
// and identifiers are `[A-Za-z][A-Za-z0-9_-]*`. The generator mirrors those
// invariants exactly, so any roundtrip failure is a real parser/render bug, not
// a generator that strayed outside the language.
// ---------------------------------------------------------------------------

/// Matches the parser's `identifier`: one leading ASCII alpha, then
/// alphanumerics / `_` / `-`.
fn arb_ident() -> impl Strategy<Value = String> {
    "[A-Za-z][A-Za-z0-9_-]{0,6}"
}

fn arb_name_or_wild() -> impl Strategy<Value = NameOrWild> {
    prop_oneof![
        1 => Just(NameOrWild::Wildcard),
        4 => arb_ident().prop_map(NameOrWild::Name),
    ]
}

fn arb_num_or_wild() -> impl Strategy<Value = NumOrWild> {
    prop_oneof![
        1 => Just(NumOrWild::Wildcard),
        4 => any::<u32>().prop_map(NumOrWild::Num),
    ]
}

fn arb_component_seg() -> impl Strategy<Value = ComponentSeg> {
    prop_oneof![
        // Plain named segment (e.g. `attn`, `o_proj`).
        3 => arb_ident().prop_map(ComponentSeg::Named),
        // Indexed segment — the MoE expert / head form (e.g. `experts[3]`).
        2 => (arb_ident(), any::<u32>())
            .prop_map(|(name, index)| ComponentSeg::Indexed { name, index }),
    ]
}

fn arb_component_or_wild() -> impl Strategy<Value = ComponentOrWild> {
    prop_oneof![
        1 => Just(ComponentOrWild::Wildcard),
        // 1..=4 segment dotted path; covers single, MoE-indexed, and deep paths.
        4 => prop::collection::vec(arb_component_seg(), 1..=4).prop_map(ComponentOrWild::Path),
    ]
}

fn arb_probe_point() -> impl Strategy<Value = ProbePoint> {
    (
        arb_name_or_wild(),
        arb_num_or_wild(),
        arb_num_or_wild(),
        arb_component_or_wild(),
        arb_num_or_wild(),
        arb_name_or_wild(),
    )
        .prop_map(
            |(model, rank, layer, component, call_index, event)| ProbePoint {
                model,
                rank,
                layer,
                component,
                call_index,
                event,
            },
        )
}

// ---------------------------------------------------------------------------
// Reference model for `matches` (tier 6).
//
// Per field, collapse to `Option<Repr>`: `None` == wildcard. Two fields are
// compatible iff either is a wildcard or both are equal. A point matches iff
// every field is compatible. This is a structurally independent restatement of
// the production logic; if the two ever disagree, one is wrong.
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq)]
enum FieldRepr {
    Name(Option<String>),
    Num(Option<u32>),
    Comp(Option<Vec<ComponentSeg>>),
}

fn name_repr(n: &NameOrWild) -> FieldRepr {
    FieldRepr::Name(match n {
        NameOrWild::Wildcard => None,
        NameOrWild::Name(s) => Some(s.clone()),
    })
}

fn num_repr(n: &NumOrWild) -> FieldRepr {
    FieldRepr::Num(match n {
        NumOrWild::Wildcard => None,
        NumOrWild::Num(v) => Some(*v),
    })
}

fn comp_repr(c: &ComponentOrWild) -> FieldRepr {
    FieldRepr::Comp(match c {
        ComponentOrWild::Wildcard => None,
        ComponentOrWild::Path(p) => Some(p.clone()),
    })
}

fn field_compatible(a: &FieldRepr, b: &FieldRepr) -> bool {
    match (a, b) {
        (FieldRepr::Name(x), FieldRepr::Name(y)) => x.is_none() || y.is_none() || x == y,
        (FieldRepr::Num(x), FieldRepr::Num(y)) => x.is_none() || y.is_none() || x == y,
        (FieldRepr::Comp(x), FieldRepr::Comp(y)) => x.is_none() || y.is_none() || x == y,
        _ => unreachable!("field reprs compared across kinds (positions are kind-aligned)"),
    }
}

fn model_matches(a: &ProbePoint, b: &ProbePoint) -> bool {
    let fields_a = [
        name_repr(&a.model),
        num_repr(&a.rank),
        num_repr(&a.layer),
        comp_repr(&a.component),
        num_repr(&a.call_index),
        name_repr(&a.event),
    ];
    let fields_b = [
        name_repr(&b.model),
        num_repr(&b.rank),
        num_repr(&b.layer),
        comp_repr(&b.component),
        num_repr(&b.call_index),
        name_repr(&b.event),
    ];
    fields_a
        .iter()
        .zip(fields_b.iter())
        .all(|(x, y)| field_compatible(x, y))
}

// ---------------------------------------------------------------------------
// Roundtrip (tier 4): parse ∘ render == id.
// ---------------------------------------------------------------------------

proptest! {
    /// The central grammar law. Rendering an AST and parsing it back must yield
    /// the identical AST — proves render is injective enough and parse recovers
    /// every field with no over- or under-acceptance on the valid language.
    #[test]
    fn parse_render_roundtrip(p in arb_probe_point()) {
        let rendered = p.to_string();
        let reparsed = ProbePoint::parse(&rendered)
            .unwrap_or_else(|e| panic!("rendered point {rendered:?} failed to parse: {e}"));
        prop_assert_eq!(reparsed, p);
    }

    /// Render is idempotent through a parse: render == render∘parse∘render. Guards
    /// against a parse that silently normalises (e.g. drops a leading zero) — that
    /// would surface as a string mismatch even when the prior test's AST matched.
    #[test]
    fn render_is_stable_through_parse(p in arb_probe_point()) {
        let once = p.to_string();
        let twice = ProbePoint::parse(&once).unwrap().to_string();
        prop_assert_eq!(once, twice);
    }
}

// ---------------------------------------------------------------------------
// Matching (tier 6 model + metamorphic).
// ---------------------------------------------------------------------------

proptest! {
    /// Model oracle: production `matches` agrees with the Option-per-field
    /// reference for every generated pair.
    #[test]
    fn matches_agrees_with_model(a in arb_probe_point(), b in arb_probe_point()) {
        prop_assert_eq!(a.matches(&b), model_matches(&a, &b));
    }

    /// Metamorphic: wildcard-matching is symmetric (every field matcher treats a
    /// wildcard on either side identically).
    #[test]
    fn matches_is_symmetric(a in arb_probe_point(), b in arb_probe_point()) {
        prop_assert_eq!(a.matches(&b), b.matches(&a));
    }

    /// Property: every point matches itself (reflexivity holds even with
    /// wildcards, since `Wildcard` matches `Wildcard`).
    #[test]
    fn matches_is_reflexive(p in arb_probe_point()) {
        prop_assert!(p.matches(&p));
    }

    /// Metamorphic monotonicity: replacing any concrete field of a *matching*
    /// pattern with a wildcard can only preserve the match, never destroy it.
    ///
    /// The pattern is built to match `target` by construction (each field is
    /// either `target`'s value or already a wildcard), so there is nothing to
    /// reject — then we widen three more fields and require the match to survive.
    #[test]
    fn wildcarding_a_field_preserves_match(
        target in arb_probe_point(),
        wild_model in any::<bool>(),
        wild_event in any::<bool>(),
    ) {
        let mut pattern = target.clone();
        if wild_model {
            pattern.model = NameOrWild::Wildcard;
        }
        if wild_event {
            pattern.event = NameOrWild::Wildcard;
        }
        // Sanity: constructed pattern matches its target.
        prop_assert!(pattern.matches(&target));

        let mut widened = pattern;
        widened.rank = NumOrWild::Wildcard;
        widened.layer = NumOrWild::Wildcard;
        widened.component = ComponentOrWild::Wildcard;
        prop_assert!(widened.matches(&target), "widening destroyed a match");
    }

    /// The all-wildcard point matches anything.
    #[test]
    fn full_wildcard_matches_everything(p in arb_probe_point()) {
        let star = ProbePoint::parse("*:*:*:*:*:*").unwrap();
        prop_assert!(star.matches(&p));
        prop_assert!(p.matches(&star));
    }
}

// ---------------------------------------------------------------------------
// Exception-raising properties.
//
// MATERIA: exception-raising tests have a 113x mutation-killing odds ratio and
// almost nobody writes them. Each strategy below produces a string that is
// *guaranteed* malformed for a specific reason; the oracle is "parse returns the
// ParseError type" — and, implicitly, "does not panic" (a panic fails the test).
// ---------------------------------------------------------------------------

/// Split a valid rendered point into its six fields so a malformation can target
/// one of them precisely.
fn arb_valid_fields() -> impl Strategy<Value = ProbePoint> {
    arb_probe_point()
}

proptest! {
    /// Wrong arity: a valid point with one field (and its colon) removed has only
    /// five fields and must be rejected.
    #[test]
    fn five_fields_rejected(p in arb_valid_fields(), drop_idx in 0usize..6) {
        let rendered = p.to_string();
        let fields: Vec<&str> = rendered.split(':').collect();
        prop_assume!(fields.len() == 6);
        let kept: Vec<&str> = fields
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != drop_idx)
            .map(|(_, s)| *s)
            .collect();
        let broken = kept.join(":");
        prop_assert!(
            ProbePoint::parse(&broken).is_err(),
            "5-field input {broken:?} was wrongly accepted",
        );
    }

    /// Wrong arity: a seventh field appended must be rejected (trailing input is
    /// not consumed by the 6-field grammar).
    #[test]
    fn seven_fields_rejected(p in arb_valid_fields(), extra in arb_ident()) {
        let broken = format!("{p}:{extra}");
        prop_assert!(
            ProbePoint::parse(&broken).is_err(),
            "7-field input {broken:?} was wrongly accepted",
        );
    }

    /// A numeric field fed an alphabetic token must be rejected (no field accepts
    /// both a name and a number except via the explicit `*`).
    #[test]
    fn alpha_in_num_field_rejected(layer_name in "[a-z]{1,4}") {
        // rank, layer, call_index are numeric. Inject an identifier into layer.
        let broken = format!("llama:0:{layer_name}:mlp:0:output");
        prop_assert!(ProbePoint::parse(&broken).is_err());
    }

    /// A name field fed a leading digit must be rejected (identifier requires a
    /// leading alpha).
    #[test]
    fn leading_digit_name_rejected(n in 0u32..1000) {
        let broken = format!("{n}x:0:0:mlp:0:output");
        prop_assert!(ProbePoint::parse(&broken).is_err());
    }

    /// Numbers past u32::MAX overflow and must be a clean error, not a panic or a
    /// silent truncation.
    #[test]
    fn u32_overflow_rejected(extra_digits in "[0-9]{1,4}") {
        let broken = format!("llama:0:{}{}:mlp:0:output", u64::from(u32::MAX), extra_digits);
        prop_assert!(ProbePoint::parse(&broken).is_err());
    }

    /// Empty component bracket `name[]` must be rejected — the index is a
    /// mandatory non-negative integer.
    #[test]
    fn empty_index_bracket_rejected(name in arb_ident()) {
        let broken = format!("llama:0:0:{name}[]:0:output");
        prop_assert!(ProbePoint::parse(&broken).is_err());
    }

    /// Unclosed component bracket must be rejected, never panic.
    #[test]
    fn unclosed_bracket_rejected(name in arb_ident(), idx in any::<u32>()) {
        let broken = format!("llama:0:0:{name}[{idx}:0:output");
        prop_assert!(ProbePoint::parse(&broken).is_err());
    }

    /// A leading dot (empty component segment) must be rejected.
    #[test]
    fn leading_dot_component_rejected(name in arb_ident()) {
        let broken = format!("llama:0:0:.{name}:0:output");
        prop_assert!(ProbePoint::parse(&broken).is_err());
    }

    /// Robustness sweep: arbitrary bytes never panic the parser. The oracle is
    /// "returns a Result" (i.e. control returns at all). Miller-style: if the
    /// stupid thing crashes it, we haven't earned the smart thing.
    #[test]
    fn arbitrary_input_never_panics(s in ".*") {
        let _ = ProbePoint::parse(&s);
    }
}

// ---------------------------------------------------------------------------
// Generator distribution evidence (proptest `cover` equivalent).
//
// Samples the strategy directly and asserts each interesting category is
// actually exercised. Without this, a roundtrip suite can be 95% trivial and we
// would not know. Run with `--nocapture` to see the printed histogram.
// ---------------------------------------------------------------------------

#[test]
fn generator_distribution() {
    const N: usize = 4000;
    let mut runner = TestRunner::deterministic();
    let strat = arb_probe_point();

    let mut model_wild = 0usize;
    let mut any_num_wild = 0usize;
    let mut comp_wild = 0usize;
    let mut has_indexed = 0usize; // MoE / head indexing exercised
    let mut multi_seg = 0usize; // dotted paths exercised
    let mut all_concrete = 0usize;
    let mut full_wild = 0usize;

    for _ in 0..N {
        let p = strat
            .new_tree(&mut runner)
            .expect("strategy produced a value")
            .current();

        let mw = matches!(p.model, NameOrWild::Wildcard);
        let nw = matches!(p.rank, NumOrWild::Wildcard)
            || matches!(p.layer, NumOrWild::Wildcard)
            || matches!(p.call_index, NumOrWild::Wildcard);
        let cw = matches!(p.component, ComponentOrWild::Wildcard);
        let ew = matches!(p.event, NameOrWild::Wildcard);

        if mw {
            model_wild += 1;
        }
        if nw {
            any_num_wild += 1;
        }
        if cw {
            comp_wild += 1;
        }
        if let ComponentOrWild::Path(segs) = &p.component {
            if segs
                .iter()
                .any(|s| matches!(s, ComponentSeg::Indexed { .. }))
            {
                has_indexed += 1;
            }
            if segs.len() > 1 {
                multi_seg += 1;
            }
        }
        if !mw && !nw && !cw && !ew {
            all_concrete += 1;
        }
        if mw && nw && cw && ew {
            full_wild += 1;
        }
    }

    let pct = |c: usize| 100.0 * c as f64 / N as f64;
    eprintln!("--- arb_probe_point distribution over {N} samples ---");
    eprintln!("  model wildcard:        {:.1}%", pct(model_wild));
    eprintln!("  any numeric wildcard:  {:.1}%", pct(any_num_wild));
    eprintln!("  component wildcard:    {:.1}%", pct(comp_wild));
    eprintln!("  has indexed seg (MoE): {:.1}%", pct(has_indexed));
    eprintln!("  multi-segment path:    {:.1}%", pct(multi_seg));
    eprintln!("  all-concrete point:    {:.1}%", pct(all_concrete));
    eprintln!("  full wildcard point:   {:.1}%", pct(full_wild));

    // Every interesting form must be meaningfully represented. Thresholds are
    // deliberately loose (we assert presence, not a precise mix) but high enough
    // that a generator regression that stops producing a form is caught.
    assert!(model_wild > N / 50, "model wildcards under-generated");
    assert!(any_num_wild > N / 10, "numeric wildcards under-generated");
    assert!(comp_wild > N / 50, "component wildcards under-generated");
    assert!(has_indexed > N / 10, "MoE indexed segments under-generated");
    assert!(multi_seg > N / 20, "multi-segment paths under-generated");
    assert!(
        all_concrete > N / 50,
        "fully-concrete points under-generated"
    );
}
