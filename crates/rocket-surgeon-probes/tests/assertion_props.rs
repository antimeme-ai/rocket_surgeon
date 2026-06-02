//! Property / metamorphic / model tests for the assertion mini-language.
//!
//! Grammar: `field cmp_op float` (space0 permitted before each token).
//! Fields: `mean` `std` `min` `max` `abs_max` `sparsity` `l2_norm`|`norm`.
//! Ops: `<` `>` `<=` `>=` `==` `!=`.
//! Float: `-? digit+ (. digit+)? ([eE][+-]? digit+)?`.
//!
//! Oracle tiers:
//!
//! - tier 6 (model): `evaluate` matches a direct `f64` comparison against the
//!   known field value; the float parser matches Rust std `str::parse` (catches
//!   reconstruction bugs).
//! - tier 4 (metamorphic): complementary operators (`<` vs `>=`, `>` vs `<=`,
//!   `==` vs `!=`) negate each other on finite inputs; whitespace at the three
//!   legal positions is inert.
//! - exception-raising: bad field / missing op / bad number / trailing junk all
//!   yield `AssertionParseError`, never a panic.

use proptest::prelude::*;

use rocket_surgeon_probes::assertion::{Assertion, CmpOp, StatsField};
use rocket_surgeon_protocol::types::{Histogram, TensorStats};

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/// (token-as-typed, canonical field). `norm` is an alias for `l2_norm`.
fn arb_field() -> impl Strategy<Value = (&'static str, StatsField)> {
    prop_oneof![
        Just(("mean", StatsField::Mean)),
        Just(("std", StatsField::Std)),
        Just(("min", StatsField::Min)),
        Just(("max", StatsField::Max)),
        Just(("abs_max", StatsField::AbsMax)),
        Just(("sparsity", StatsField::Sparsity)),
        Just(("l2_norm", StatsField::L2Norm)),
        Just(("norm", StatsField::L2Norm)),
    ]
}

fn arb_op() -> impl Strategy<Value = (&'static str, CmpOp)> {
    prop_oneof![
        Just(("<", CmpOp::Lt)),
        Just((">", CmpOp::Gt)),
        Just(("<=", CmpOp::Le)),
        Just((">=", CmpOp::Ge)),
        Just(("==", CmpOp::Eq)),
        Just(("!=", CmpOp::Ne)),
    ]
}

/// A string that matches the float grammar exactly, plus the f64 it denotes.
fn arb_float_str() -> impl Strategy<Value = String> {
    (
        any::<bool>(),                                       // sign
        "[0-9]{1,6}",                                        // int part (required)
        proptest::option::of("[0-9]{1,6}"),                  // optional fraction
        proptest::option::of((any::<bool>(), "[0-9]{1,3}")), // optional exponent
    )
        .prop_map(|(neg, int, frac, exp)| {
            let mut s = String::new();
            if neg {
                s.push('-');
            }
            s.push_str(&int);
            if let Some(f) = frac {
                s.push('.');
                s.push_str(&f);
            }
            if let Some((esign, edigits)) = exp {
                s.push('e');
                if esign {
                    s.push('-');
                }
                s.push_str(&edigits);
            }
            s
        })
}

/// Finite f64 field value for evaluate/complementarity tests.
fn arb_finite() -> impl Strategy<Value = f64> {
    prop_oneof![
        (-1e6f64..1e6f64),
        Just(0.0f64),
        Just(-0.0f64),
        Just(1.0f64),
        Just(f64::MIN_POSITIVE),
    ]
}

fn stats_with(field: StatsField, v: f64) -> TensorStats {
    let mut s = TensorStats {
        mean: 0.0,
        std: 0.0,
        min: 0.0,
        max: 0.0,
        abs_max: 0.0,
        sparsity: 0.0,
        l2_norm: 0.0,
        histogram: Histogram {
            bins: 0,
            edges: vec![],
            counts: vec![],
        },
    };
    match field {
        StatsField::Mean => s.mean = v,
        StatsField::Std => s.std = v,
        StatsField::Min => s.min = v,
        StatsField::Max => s.max = v,
        StatsField::AbsMax => s.abs_max = v,
        StatsField::Sparsity => s.sparsity = v,
        StatsField::L2Norm => s.l2_norm = v,
    }
    s
}

// ---------------------------------------------------------------------------
// Parse model: field/op recovered, float matches std parse.
// ---------------------------------------------------------------------------

proptest! {
    /// The parser recovers exactly the field and operator typed, and the numeric
    /// value equals Rust std `f64` parse of the same literal (the model oracle —
    /// catches any reconstruction bug in `float_literal`).
    #[test]
    fn parse_recovers_field_op_value(
        (ftok, fexp) in arb_field(),
        (otok, oexp) in arb_op(),
        num in arb_float_str(),
    ) {
        let src = format!("{ftok} {otok} {num}");
        let a = Assertion::parse(&src)
            .unwrap_or_else(|e| panic!("valid assertion {src:?} failed: {e}"));
        prop_assert_eq!(a.field, fexp);
        prop_assert_eq!(a.op, oexp);
        let oracle: f64 = num.parse().unwrap();
        // Bit-exact: both come from parsing the identical literal, so the parser's
        // reconstruction must reproduce std's value to the bit (the grammar cannot
        // produce NaN, so payload ambiguity is not a concern).
        prop_assert!(
            a.value.to_bits() == oracle.to_bits(),
            "value {} != std-parse {} for {:?}", a.value, oracle, num,
        );
    }

    /// Metamorphic: whitespace at the three legal positions (before field, op,
    /// value) does not change the parsed assertion.
    #[test]
    fn leading_and_inter_token_space_is_inert(
        (ftok, _f) in arb_field(),
        (otok, _o) in arb_op(),
        num in arb_float_str(),
    ) {
        let tight = format!("{ftok}{otok}{num}");
        let spaced = format!("  {ftok}   {otok}    {num}");
        let a = Assertion::parse(&tight);
        let b = Assertion::parse(&spaced);
        prop_assert_eq!(a.ok(), b.ok());
    }
}

// ---------------------------------------------------------------------------
// Evaluate model (tier 6).
// ---------------------------------------------------------------------------

proptest! {
    /// `evaluate` equals a direct comparison of the known field value against the
    /// threshold, for `<`, `>`, `<=`, `>=`. (`==`/`!=` use an epsilon band and are
    /// covered by the complementarity test below.)
    #[test]
    fn evaluate_matches_direct_comparison(
        (ftok, fexp) in arb_field(),
        lhs in arb_finite(),
        rhs in arb_finite(),
    ) {
        let stats = stats_with(fexp, lhs);
        for (otok, expected) in [
            ("<", lhs < rhs),
            (">", lhs > rhs),
            ("<=", lhs <= rhs),
            (">=", lhs >= rhs),
        ] {
            let src = format!("{ftok} {otok} {rhs}");
            let a = Assertion::parse(&src).unwrap();
            prop_assert_eq!(a.evaluate(&stats), expected, "op {} on {} vs {}", otok, lhs, rhs);
        }
    }

    /// Metamorphic complementarity on finite inputs: `<` negates `>=`, `>` negates
    /// `<=`, and `==` negates `!=`. (NaN breaks this — see the documented finding
    /// test `nan_breaks_operator_complementarity`.)
    #[test]
    fn operators_are_complementary_on_finite_inputs(
        (ftok, fexp) in arb_field(),
        lhs in arb_finite(),
        rhs in arb_finite(),
    ) {
        let stats = stats_with(fexp, lhs);
        let eval = |op: &str| {
            Assertion::parse(&format!("{ftok} {op} {rhs}"))
                .unwrap()
                .evaluate(&stats)
        };
        prop_assert_ne!(eval("<"), eval(">="), "Lt/Ge not complementary");
        prop_assert_ne!(eval(">"), eval("<="), "Gt/Le not complementary");
        prop_assert_ne!(eval("=="), eval("!="), "Eq/Ne not complementary");
    }
}

// ---------------------------------------------------------------------------
// Exception-raising properties.
// ---------------------------------------------------------------------------

proptest! {
    /// Any field token not in the known set is rejected.
    #[test]
    fn unknown_field_rejected(name in "[a-z]{1,8}") {
        const KNOWN: [&str; 8] =
            ["mean", "std", "min", "max", "abs_max", "sparsity", "l2_norm", "norm"];
        prop_assume!(!KNOWN.contains(&name.as_str()));
        let src = format!("{name} < 1.0");
        prop_assert!(Assertion::parse(&src).is_err(), "accepted {:?}", src);
    }

    /// A field with no operator (just a value) is rejected.
    #[test]
    fn missing_operator_rejected((ftok, _f) in arb_field(), num in arb_float_str()) {
        let src = format!("{ftok} {num}");
        prop_assert!(Assertion::parse(&src).is_err(), "accepted {:?}", src);
    }

    /// A non-numeric right-hand side is rejected.
    #[test]
    fn non_numeric_value_rejected((ftok, _f) in arb_field(), junk in "[a-z]{1,4}") {
        let src = format!("{ftok} < {junk}");
        prop_assert!(Assertion::parse(&src).is_err(), "accepted {:?}", src);
    }

    /// Trailing garbage after a complete assertion is rejected (parse must consume
    /// all input — no silent prefix-accept).
    #[test]
    fn trailing_garbage_rejected(
        (ftok, _f) in arb_field(),
        (otok, _o) in arb_op(),
        num in arb_float_str(),
        junk in "[a-z]{1,4}",
    ) {
        let src = format!("{ftok} {otok} {num} {junk}");
        prop_assert!(Assertion::parse(&src).is_err(), "accepted {:?}", src);
    }

    /// Bare field with nothing else is rejected.
    #[test]
    fn bare_field_rejected((ftok, _f) in arb_field()) {
        prop_assert!(Assertion::parse(ftok).is_err(), "accepted bare field {:?}", ftok);
    }

    /// Robustness: arbitrary input never panics the parser.
    #[test]
    fn arbitrary_input_never_panics(s in ".*") {
        let _ = Assertion::parse(&s);
    }
}

// ---------------------------------------------------------------------------
// Documented finding: NaN breaks operator complementarity.
//
// This is IEEE-754-correct, not a code defect — but it is a latent semantic
// surprise: a TensorStats field of NaN (e.g. stats of an all-NaN activation)
// makes BOTH `x < t` and `x >= t` false, and BOTH `x == t` and `x != t` false.
// An assertion-based probe firing on NaN therefore reports "pass" for every
// comparison, silently. Pinned here so the behavior is intentional and visible,
// and flagged in PLATOON-FINDINGS.md for the protocol owner.
// ---------------------------------------------------------------------------

#[test]
fn nan_breaks_operator_complementarity() {
    let stats = stats_with(StatsField::Mean, f64::NAN);

    let lt = Assertion::parse("mean < 1.0").unwrap().evaluate(&stats);
    let ge = Assertion::parse("mean >= 1.0").unwrap().evaluate(&stats);
    // Complementarity would require lt != ge; with NaN, both are false.
    assert!(!lt && !ge, "expected NaN to make both < and >= false");

    let eq = Assertion::parse("mean == 1.0").unwrap().evaluate(&stats);
    let ne = Assertion::parse("mean != 1.0").unwrap().evaluate(&stats);
    // Eq is `(NaN - v).abs() < EPSILON` = false; Ne is `>= EPSILON` = false.
    assert!(!eq && !ne, "expected NaN to make both == and != false");
}
