use rocket_surgeon_protocol::messages::{Divergence, HostReplayRequest, ReplayStopAt};
use rocket_surgeon_protocol::types::InterventionRecipe;

pub struct ReplayContext {
    pub checkpoint_id: String,
    pub verify: bool,
    pub deterministic: bool,
    pub cosine_threshold: f64,
    pub mre_threshold: f64,
    pub stop_at: Option<ReplayStopAt>,
    pub interventions: Vec<InterventionRecipe>,
    pub divergences: Vec<Divergence>,
    pub ticks_replayed: u32,
}

impl ReplayContext {
    pub fn from_request(req: &HostReplayRequest) -> Self {
        Self {
            checkpoint_id: req.checkpoint_id.clone(),
            verify: req.verify,
            deterministic: req.deterministic,
            cosine_threshold: req.cosine_threshold,
            mre_threshold: req.mre_threshold,
            stop_at: req.stop_at.clone(),
            interventions: req.interventions.clone(),
            divergences: Vec::new(),
            ticks_replayed: 0,
        }
    }

    pub fn should_stop(&self, layer: u32, component: &str) -> bool {
        if let Some(ref stop) = self.stop_at {
            layer == stop.layer && component == stop.component
        } else {
            false
        }
    }
}

// ===========================================================================
// Property-based tests for replay control (B002 platoon, BRAVO lane).
//
// The numerical divergence math (cosine / max-relative-error) lives in
// `bridge::compare_activations_from_ptr` (PyO3) and Python `replay.py`
// (ECHO's lane). What is pure-Rust and testable here is the replay *control*:
// `from_request` field propagation and the `should_stop` predicate. See
// PLATOON-FINDINGS.md "Gaps" for the divergence-metamorphic relations deferred
// to ECHO.
// ===========================================================================
#[cfg(test)]
mod prop_tests {
    use proptest::prelude::*;
    use rocket_surgeon_protocol::messages::{HostReplayRequest, ReplayStopAt};

    use super::ReplayContext;

    fn request_strategy() -> impl Strategy<Value = HostReplayRequest> {
        (
            any::<u64>(),
            "[a-z]{0,8}",
            proptest::option::of((0u32..16, "[a-z_]{0,8}")),
            any::<bool>(),
            any::<bool>(),
            -1.0f64..2.0,
            0.0f64..1.0,
        )
            .prop_map(
                |(model_handle, checkpoint_id, stop, verify, deterministic, cosine, mre)| {
                    HostReplayRequest {
                        model_handle,
                        checkpoint_id,
                        stop_at: stop.map(|(layer, component)| ReplayStopAt { layer, component }),
                        interventions: Vec::new(),
                        verify,
                        deterministic,
                        cosine_threshold: cosine,
                        mre_threshold: mre,
                    }
                },
            )
    }

    proptest! {
        /// Model-based: `from_request` copies every control field verbatim and
        /// starts with an empty divergence log and zero replayed ticks. Oracle
        /// tier 6 (the request is the model). Just had analogues of this class
        /// of bug elsewhere in the wire format, so we pin every field.
        #[test]
        fn from_request_propagates_fields(req in request_strategy()) {
            let ctx = ReplayContext::from_request(&req);
            prop_assert_eq!(&ctx.checkpoint_id, &req.checkpoint_id);
            prop_assert_eq!(ctx.verify, req.verify);
            prop_assert_eq!(ctx.deterministic, req.deterministic);
            // Verbatim copy: compare bit patterns (exact equality is the spec).
            prop_assert_eq!(ctx.cosine_threshold.to_bits(), req.cosine_threshold.to_bits());
            prop_assert_eq!(ctx.mre_threshold.to_bits(), req.mre_threshold.to_bits());
            prop_assert_eq!(ctx.stop_at.is_some(), req.stop_at.is_some());
            if let (Some(a), Some(b)) = (&ctx.stop_at, &req.stop_at) {
                prop_assert_eq!(a.layer, b.layer);
                prop_assert_eq!(&a.component, &b.component);
            }
            // Fresh context starts clean.
            prop_assert!(ctx.divergences.is_empty(), "fresh ctx has divergences");
            prop_assert_eq!(ctx.ticks_replayed, 0, "fresh ctx has replayed ticks");
        }

        /// Model-based + exception/metamorphic: `should_stop` is true for
        /// EXACTLY the configured (layer, component) and nothing else; with no
        /// stop point it is universally false. Oracle tier 6.
        #[test]
        fn should_stop_matches_predicate(
            req in request_strategy(),
            probe_layer in 0u32..16,
            probe_comp in "[a-z_]{0,8}",
        ) {
            let ctx = ReplayContext::from_request(&req);
            let expected = req
                .stop_at
                .as_ref()
                .is_some_and(|s| s.layer == probe_layer && s.component == probe_comp);
            prop_assert_eq!(ctx.should_stop(probe_layer, &probe_comp), expected);
        }

        /// Metamorphic: with no stop point configured, replay never stops early
        /// for ANY probe coordinate.
        #[test]
        fn no_stop_point_never_stops(
            mut req in request_strategy(),
            probe_layer in 0u32..1000,
            probe_comp in ".{0,16}",
        ) {
            req.stop_at = None;
            let ctx = ReplayContext::from_request(&req);
            prop_assert!(!ctx.should_stop(probe_layer, &probe_comp));
        }

        /// Metamorphic: when a stop point IS configured, `should_stop` fires at
        /// that exact coordinate and is false at any coordinate that differs in
        /// either dimension.
        #[test]
        fn stop_point_fires_only_at_target(
            mut req in request_strategy(),
            layer in 0u32..16,
            comp in "[a-z_]{1,8}",
            other_layer in 0u32..16,
            other_comp in "[a-z_]{1,8}",
        ) {
            req.stop_at = Some(ReplayStopAt { layer, component: comp.clone() });
            let ctx = ReplayContext::from_request(&req);
            prop_assert!(ctx.should_stop(layer, &comp), "did not fire at target");
            prop_assume!(other_layer != layer || other_comp != comp);
            prop_assert!(
                !ctx.should_stop(other_layer, &other_comp),
                "fired at a non-target coordinate"
            );
        }
    }
}
