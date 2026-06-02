use rocket_surgeon_protocol::types::TickGranularity;

pub struct StepPlan {
    pub ticks_to_drain: u32,
    pub granularity: TickGranularity,
}

pub fn plan_step(count: u32, granularity: Option<TickGranularity>) -> StepPlan {
    StepPlan {
        ticks_to_drain: count,
        granularity: granularity.unwrap_or(TickGranularity::Component),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainState {
    Counting,
    Draining,
}

pub fn is_layer_boundary(current_layer: Option<u32>, new_layer: u32) -> bool {
    match current_layer {
        None => false,
        Some(prev) => new_layer != prev,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_step_defaults_to_component() {
        let plan = plan_step(3, None);
        assert_eq!(plan.ticks_to_drain, 3);
        assert_eq!(plan.granularity, TickGranularity::Component);
    }

    #[test]
    fn plan_step_respects_explicit_granularity() {
        let plan = plan_step(1, Some(TickGranularity::Layer));
        assert_eq!(plan.granularity, TickGranularity::Layer);
    }

    #[test]
    fn is_layer_boundary_detects_change() {
        assert!(is_layer_boundary(Some(0), 1));
        assert!(!is_layer_boundary(Some(0), 0));
        assert!(!is_layer_boundary(None, 0));
    }
}

// ===========================================================================
// Property-based tests for the step driver's pure planning functions
// (B004 platoon, MIKE lane).
//
// `step_driver` had only three example-based tests (oracle tier 2/3) and zero
// property coverage; it is squarely in NOVEMBER's mutation-re-audit crosshairs
// this wave. The two pure functions here have exact model oracles, so we pin
// them at tier 6 (model-based) plus metamorphic relations, which is precisely
// what kills the `unwrap_or` / `!=`-vs-`==` / arm-deletion mutants.
// ===========================================================================
#[cfg(test)]
mod prop_tests {
    use proptest::prelude::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    use super::{is_layer_boundary, plan_step};
    use rocket_surgeon_protocol::types::TickGranularity;

    fn granularity_variants() -> Vec<TickGranularity> {
        vec![
            TickGranularity::Layer,
            TickGranularity::Component,
            TickGranularity::Head,
            TickGranularity::Expert,
        ]
    }

    proptest! {
        /// Model-based: `plan_step` is the identity on `count` (it never
        /// rescales or clamps the drain count) and applies the documented
        /// `Component` default exactly when granularity is absent. Covers the
        /// full u32 range incl. 0 and u32::MAX. Oracle tier 6.
        #[test]
        fn plan_step_is_identity_on_count_and_defaults_component(
            count in any::<u32>(),
            gran_idx in proptest::option::of(0usize..4),
        ) {
            let variants = granularity_variants();
            let gran = gran_idx.map(|i| variants[i]);
            let plan = plan_step(count, gran);

            prop_assert_eq!(plan.ticks_to_drain, count, "drain count must pass through verbatim");
            let expected = gran.unwrap_or(TickGranularity::Component);
            prop_assert_eq!(plan.granularity, expected, "granularity default/passthrough wrong");
        }

        /// Metamorphic: the drain count is independent of the granularity knob —
        /// changing granularity must never change how many ticks we plan to
        /// drain. Catches a mutant that crosses the two fields.
        #[test]
        fn plan_step_count_independent_of_granularity(
            count in any::<u32>(),
            a in 0usize..4,
            b in 0usize..4,
        ) {
            let variants = granularity_variants();
            let pa = plan_step(count, Some(variants[a]));
            let pb = plan_step(count, Some(variants[b]));
            prop_assert_eq!(pa.ticks_to_drain, pb.ticks_to_drain);
            prop_assert_eq!(pa.ticks_to_drain, count);
        }

        /// Model-based: `is_layer_boundary` is exactly `matches!(cur, Some(p) if
        /// p != new)`. An independent reference predicate is the oracle. The
        /// `None` (start-of-pass) case is never a boundary for ANY new layer.
        /// Oracle tier 6.
        #[test]
        fn is_layer_boundary_matches_reference(
            cur in proptest::option::of(0u32..64),
            new in 0u32..64,
        ) {
            let expected = matches!(cur, Some(prev) if prev != new);
            prop_assert_eq!(is_layer_boundary(cur, new), expected);
        }

        /// Property: staying on the same layer is never a boundary (reflexive),
        /// and crossing to any different layer always is. Directly pins both
        /// arms of the `Some` branch against the `==`/`!=` mutant.
        #[test]
        fn is_layer_boundary_reflexive_and_strict(prev in 0u32..64, other in 0u32..64) {
            prop_assert!(!is_layer_boundary(Some(prev), prev), "same layer flagged as boundary");
            prop_assume!(other != prev);
            prop_assert!(is_layer_boundary(Some(prev), other), "layer change not flagged");
        }

        /// Property: with no current layer (`None`), there is no boundary,
        /// universally over the new layer.
        #[test]
        fn is_layer_boundary_none_is_never_boundary(new in any::<u32>()) {
            prop_assert!(!is_layer_boundary(None, new));
        }
    }

    /// Generator audit (proptest analogue of Hypothesis `cover`): sample the
    /// layer-boundary input space and assert both outcomes are exercised in
    /// meaningful proportion, so the predicate tests above are not silently
    /// degenerate (e.g. all `None`, or all same-layer). If the generator drifts
    /// to mostly-trivial inputs this fails loudly.
    #[test]
    fn generator_audit_layer_boundary_distribution() {
        const N: u32 = 4000;
        let mut runner = TestRunner::default();
        let strat = (proptest::option::of(0u32..64), 0u32..64);
        let (mut none_cur, mut boundary, mut same) = (0u32, 0u32, 0u32);
        for _ in 0..N {
            let (cur, new) = strat.new_tree(&mut runner).unwrap().current();
            match cur {
                None => none_cur += 1,
                Some(p) if p != new => boundary += 1,
                Some(_) => same += 1,
            }
        }
        // None-current and boundary dominate the natural distribution.
        assert!(
            none_cur > N / 10,
            "too few None-current samples: {none_cur}/{N}"
        );
        assert!(
            boundary > N / 10,
            "too few boundary samples: {boundary}/{N}"
        );
        // EVIDENCE: the `Some(p) if p == new` (same-layer, no-boundary) branch is
        // hit only ~Pr(Some)·(1/64) ≈ 0.6% of the time by chance. The natural
        // generator barely exercises it — which is exactly why
        // `is_layer_boundary_reflexive_and_strict` CONSTRUCTS `Some(prev), prev`
        // rather than waiting for a random hit. This bound documents that floor.
        assert!(same > 5, "same-layer branch under-sampled: {same}/{N}");
    }
}
