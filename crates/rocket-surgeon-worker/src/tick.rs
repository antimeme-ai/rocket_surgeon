#![allow(dead_code)]

use std::time::Instant;

use rocket_surgeon_protocol::types::{Phase, StepDirection, TickClock, TickEvent, TickPosition};

/// Per-session tick cursor carrying the three-clock model.
///
/// The tick model has three incommensurable clocks:
/// - `token` — sequence position; increments once per token processed.
/// - `operator` — within-token traversal index; resets to 0 at each new token.
/// - `wall` — nanosecond real time since the session started.
///
/// `tick_id` is an **alias** for the operator clock — it is NOT a global
/// monotonic counter. It resets to 0 at each new token alongside `operator`.
pub struct TickState {
    rank: u32,
    layer: u32,
    component: String,
    call_index: u32,
    /// Total operators traversed across the whole session — diagnostics only,
    /// never surfaced as `tick_id`.
    step_count: u64,
    phase: Phase,
    /// Token sequence clock — increments once per token.
    token: u64,
    /// Within-token traversal clock; `tick_id` is an alias for this value.
    operator: u64,
    /// Reportable token position; `None` until the first operator advances.
    token_position: Option<u64>,
    session_start: Instant,
}

impl TickState {
    pub fn new(rank: u32) -> Self {
        Self {
            rank,
            layer: 0,
            component: String::new(),
            call_index: 0,
            step_count: 0,
            // The initial forward pass of a prompt is a prefill.
            phase: Phase::Prefill,
            token: 0,
            operator: 0,
            token_position: None,
            session_start: Instant::now(),
        }
    }

    /// Advance the operator clock by one traversal step within the current
    /// token. Records the component coordinate and makes `token_position`
    /// present (it tracks the token clock once stepping has begun).
    pub fn advance(&mut self, component: &str, layer: u32, call_index: u32) {
        self.operator += 1;
        self.layer = layer;
        component.clone_into(&mut self.component);
        self.call_index = call_index;
        self.step_count += 1;
        self.token_position = Some(self.token);
    }

    /// Advance to the next token: increment the token clock and reset the
    /// operator clock to 0. A fresh token after prefill is a decode step, so
    /// the phase transitions to [`Phase::Decode`] unless it is already a
    /// chunked-prefill phase (which manages its own progression).
    pub fn advance_token(&mut self) {
        // `set_token_position` accepts any u64, so `token` can sit at the
        // ceiling on entry; a bare `+= 1` would panic (debug) or wrap to 0
        // (release), the latter silently violating token monotonicity. Saturate
        // instead — matching the `wall_ns` saturating idiom below. Only `token`
        // is reachable at the ceiling (one `set_token_position` call); `operator`
        // and `step_count` would need 2^64 advances, so they stay bare adds.
        self.token = self.token.saturating_add(1);
        self.operator = 0;
        self.token_position = Some(self.token);
        if matches!(self.phase, Phase::Prefill) {
            self.phase = Phase::Decode;
        }
    }

    pub fn set_phase(&mut self, phase: Phase) {
        self.phase = phase;
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn set_token_position(&mut self, pos: u64) {
        self.token = pos;
        self.token_position = Some(pos);
    }

    /// `tick_id` is an alias for the operator clock.
    pub fn tick_id(&self) -> u64 {
        self.operator
    }

    pub fn token(&self) -> u64 {
        self.token
    }

    pub fn operator(&self) -> u64 {
        self.operator
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    pub fn component(&self) -> &str {
        &self.component
    }

    pub fn call_index(&self) -> u32 {
        self.call_index
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn to_tick_position(&self) -> TickPosition {
        TickPosition {
            tick_id: self.operator,
            direction: StepDirection::Forward,
            rank: Some(self.rank),
            layer: self.layer,
            component: self.component.clone(),
            event: TickEvent::Output,
            replay_of: None,
            phase: self.phase,
            token_position: self.token_position,
            clock: Some(TickClock {
                token: self.token,
                operator: self.operator,
                // `wall_ns` must be a non-zero nanosecond timestamp. Truncate
                // the u128 nanos to u64 first (a u64 nanosecond counter spans
                // ~584 years, far beyond any session), then guarantee non-zero
                // with a saturating add so the invariant holds post-cast.
                wall_ns: (self.session_start.elapsed().as_nanos() as u64).saturating_add(1),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tick_state_starts_at_zero() {
        let state = TickState::new(0);
        assert_eq!(state.tick_id(), 0);
        assert_eq!(state.token(), 0);
        assert_eq!(state.operator(), 0);
        assert_eq!(state.layer(), 0);
        assert_eq!(state.component(), "");
        assert_eq!(state.call_index(), 0);
    }

    #[test]
    fn new_tick_state_starts_in_prefill() {
        let state = TickState::new(0);
        assert_eq!(state.phase(), Phase::Prefill);
        // token_position is absent until the first operator advances.
        assert_eq!(state.to_tick_position().token_position, None);
    }

    #[test]
    fn tick_id_aliases_operator_clock() {
        let mut state = TickState::new(0);
        state.advance("q_proj", 0, 0);
        assert_eq!(state.tick_id(), state.operator());
        assert_eq!(state.tick_id(), 1);
        state.advance("k_proj", 0, 0);
        assert_eq!(state.tick_id(), state.operator());
        assert_eq!(state.tick_id(), 2);
    }

    #[test]
    fn advance_increments_operator_clock() {
        let mut state = TickState::new(0);
        state.advance("q_proj", 0, 0);
        assert_eq!(state.operator(), 1);
        state.advance("k_proj", 0, 0);
        assert_eq!(state.operator(), 2);
    }

    #[test]
    fn advance_updates_position() {
        let mut state = TickState::new(0);
        state.advance("q_proj", 3, 0);
        assert_eq!(state.layer(), 3);
        assert_eq!(state.component(), "q_proj");
        assert_eq!(state.call_index(), 0);
    }

    #[test]
    fn advance_tracks_call_index() {
        let mut state = TickState::new(0);
        state.advance("embed", 0, 0);
        assert_eq!(state.operator(), 1);
        state.advance("embed", 0, 1);
        assert_eq!(state.operator(), 2);
        assert_eq!(state.call_index(), 1);
    }

    #[test]
    fn advance_makes_token_position_present() {
        let mut state = TickState::new(0);
        assert_eq!(state.to_tick_position().token_position, None);
        state.advance("q_proj", 0, 0);
        assert_eq!(state.to_tick_position().token_position, Some(0));
    }

    #[test]
    fn advance_token_increments_token_and_resets_operator() {
        let mut state = TickState::new(0);
        state.advance("q_proj", 0, 0);
        state.advance("k_proj", 0, 0);
        assert_eq!(state.operator(), 2);
        assert_eq!(state.token(), 0);

        state.advance_token();
        assert_eq!(state.token(), 1);
        assert_eq!(state.operator(), 0);
        // tick_id, being the operator alias, resets too.
        assert_eq!(state.tick_id(), 0);
    }

    #[test]
    fn advance_token_transitions_prefill_to_decode() {
        let mut state = TickState::new(0);
        assert_eq!(state.phase(), Phase::Prefill);
        state.advance_token();
        assert_eq!(state.phase(), Phase::Decode);
    }

    #[test]
    fn advance_token_preserves_chunked_prefill_phase() {
        let mut state = TickState::new(0);
        let chunked = Phase::PrefillChunked {
            chunk_size: 512,
            chunk_index: 0,
            total_chunks: 4,
        };
        state.set_phase(chunked);
        state.advance_token();
        assert_eq!(state.phase(), chunked);
    }

    #[test]
    fn operator_clock_resets_each_token() {
        let mut state = TickState::new(0);
        for _ in 0..3 {
            state.advance("comp", 0, 0);
        }
        assert_eq!(state.operator(), 3);
        state.advance_token();
        assert_eq!(state.operator(), 0);
        for _ in 0..2 {
            state.advance("comp", 0, 0);
        }
        assert_eq!(state.operator(), 2);
        // token clock keeps climbing across tokens.
        assert_eq!(state.token(), 1);
    }

    #[test]
    fn set_token_position_updates_token_clock() {
        let mut state = TickState::new(0);
        state.set_token_position(7);
        assert_eq!(state.token(), 7);
        let pos = state.to_tick_position();
        assert_eq!(pos.token_position, Some(7));
        assert_eq!(pos.clock.unwrap().token, 7);
    }

    #[test]
    fn to_tick_position_has_all_three_clocks() {
        let mut state = TickState::new(0);
        state.advance("q_proj", 5, 0);
        let pos = state.to_tick_position();
        assert_eq!(pos.tick_id, 1);
        assert_eq!(pos.layer, 5);
        assert_eq!(pos.component, "q_proj");
        assert_eq!(pos.rank, Some(0));

        let clock = pos.clock.expect("clock must be present");
        assert_eq!(clock.token, 0);
        assert_eq!(clock.operator, 1);
        // wall_ns must be a non-zero nanosecond timestamp.
        assert!(clock.wall_ns > 0, "wall_ns must be non-zero");
    }

    #[test]
    fn to_tick_position_tick_id_equals_clock_operator() {
        let mut state = TickState::new(0);
        for _ in 0..5 {
            state.advance("comp", 0, 0);
        }
        let pos = state.to_tick_position();
        assert_eq!(pos.tick_id, pos.clock.unwrap().operator);
    }

    #[test]
    fn step_count_tracks_total_steps_across_tokens() {
        let mut state = TickState::new(0);
        assert_eq!(state.step_count(), 0);
        state.advance("a", 0, 0);
        assert_eq!(state.step_count(), 1);
        state.advance_token();
        state.advance("b", 0, 0);
        // step_count counts every operator regardless of token resets.
        assert_eq!(state.step_count(), 2);
    }

    #[test]
    fn serialized_json_satisfies_tick_id_equals_clock_operator() {
        let mut state = TickState::new(0);
        state.advance("q_proj", 0, 0);
        state.advance("k_proj", 0, 0);
        let pos = state.to_tick_position();
        let json = serde_json::to_value(&pos).unwrap();
        assert_eq!(json["tick_id"], json["clock"]["operator"]);
        assert!(json["clock"]["wall_ns"].as_u64().unwrap() > 0);
    }
}

// ===========================================================================
// Property-based / stateful model-based tests for the step-driver tick cursor
// (B002 platoon, BRAVO lane).
//
// The TickState is the step-driver FSM: a three-clock cursor advanced one
// operator (or token) at a time. We model it with an independent abstract
// state and assert the real cursor matches after EVERY operation in an
// arbitrary op sequence (oracle tier 6, model-based / stateful).
// ===========================================================================
#[cfg(test)]
mod prop_tests {
    use proptest::prelude::*;

    use super::{Phase, TickState};

    /// Phases the driver can be placed in. Chunked-prefill carries fields, but
    /// for transition purposes only the variant tag matters to `advance_token`.
    fn phase_variants() -> Vec<Phase> {
        vec![
            Phase::Prefill,
            Phase::Decode,
            Phase::PrefillChunked {
                chunk_size: 512,
                chunk_index: 0,
                total_chunks: 4,
            },
        ]
    }

    #[derive(Clone, Debug)]
    enum TickOp {
        Advance {
            component: u8,
            layer: u32,
            call_index: u32,
        },
        AdvanceToken,
        SetPhase(usize),
        SetTokenPosition(u64),
    }

    fn tick_op_strategy() -> impl Strategy<Value = TickOp> {
        prop_oneof![
            3 => (0u8..4, 0u32..8, 0u32..4).prop_map(|(component, layer, call_index)| {
                TickOp::Advance { component, layer, call_index }
            }),
            2 => Just(TickOp::AdvanceToken),
            1 => (0usize..3).prop_map(TickOp::SetPhase),
            1 => (0u64..1000).prop_map(TickOp::SetTokenPosition),
        ]
    }

    /// Independent reference model of the cursor. Mirrors the documented
    /// three-clock semantics without sharing code with `TickState`.
    struct TickModel {
        token: u64,
        operator: u64,
        step_count: u64,
        layer: u32,
        component: String,
        call_index: u32,
        phase: Phase,
        token_position: Option<u64>,
    }

    impl TickModel {
        fn new() -> Self {
            Self {
                token: 0,
                operator: 0,
                step_count: 0,
                layer: 0,
                component: String::new(),
                call_index: 0,
                phase: Phase::Prefill,
                token_position: None,
            }
        }

        fn apply(&mut self, op: &TickOp, phases: &[Phase]) {
            match *op {
                TickOp::Advance {
                    component,
                    layer,
                    call_index,
                } => {
                    self.operator += 1;
                    self.layer = layer;
                    self.component = format!("c{component}");
                    self.call_index = call_index;
                    self.step_count += 1;
                    self.token_position = Some(self.token);
                }
                TickOp::AdvanceToken => {
                    self.token += 1;
                    self.operator = 0;
                    self.token_position = Some(self.token);
                    if matches!(self.phase, Phase::Prefill) {
                        self.phase = Phase::Decode;
                    }
                }
                TickOp::SetPhase(i) => {
                    self.phase = phases[i];
                }
                TickOp::SetTokenPosition(pos) => {
                    self.token = pos;
                    self.token_position = Some(pos);
                }
            }
        }
    }

    fn apply_real(state: &mut TickState, op: &TickOp, phases: &[Phase]) {
        match *op {
            TickOp::Advance {
                component,
                layer,
                call_index,
            } => state.advance(&format!("c{component}"), layer, call_index),
            TickOp::AdvanceToken => state.advance_token(),
            TickOp::SetPhase(i) => state.set_phase(phases[i]),
            TickOp::SetTokenPosition(pos) => state.set_token_position(pos),
        }
    }

    proptest! {
        /// Stateful model-based: the real cursor matches the abstract model on
        /// every getter after every op, and the load-bearing invariants hold
        /// throughout (tick_id aliases operator; to_tick_position is consistent;
        /// wall_ns is always non-zero). Oracle tier 6.
        #[test]
        fn tick_state_matches_model(
            ops in proptest::collection::vec(tick_op_strategy(), 0..80),
        ) {
            let phases = phase_variants();
            let mut state = TickState::new(7);
            let mut model = TickModel::new();

            for op in &ops {
                model.apply(op, &phases);
                apply_real(&mut state, op, &phases);

                prop_assert_eq!(state.token(), model.token, "token clock");
                prop_assert_eq!(state.operator(), model.operator, "operator clock");
                prop_assert_eq!(state.step_count(), model.step_count, "step_count");
                prop_assert_eq!(state.layer(), model.layer, "layer cursor");
                prop_assert_eq!(state.component(), model.component.as_str(), "component");
                prop_assert_eq!(state.call_index(), model.call_index, "call_index");
                prop_assert_eq!(state.phase(), model.phase, "phase");

                // Invariant: tick_id is exactly the operator clock alias.
                prop_assert_eq!(state.tick_id(), state.operator(), "tick_id != operator");

                let pos = state.to_tick_position();
                prop_assert_eq!(pos.tick_id, model.operator, "position tick_id");
                prop_assert_eq!(pos.layer, model.layer, "position layer");
                prop_assert_eq!(pos.component.as_str(), model.component.as_str(), "position comp");
                prop_assert_eq!(pos.phase, model.phase, "position phase");
                prop_assert_eq!(pos.token_position, model.token_position, "token_position");
                prop_assert_eq!(pos.rank, Some(7), "rank");
                let clock = pos.clock.expect("clock must be present");
                prop_assert_eq!(clock.token, model.token, "clock token");
                prop_assert_eq!(clock.operator, model.operator, "clock operator");
                prop_assert_eq!(clock.operator, pos.tick_id, "clock operator != tick_id");
                prop_assert!(clock.wall_ns > 0, "wall_ns must be non-zero");
            }
        }

        /// Metamorphic: advancing the operator clock NEVER changes the token
        /// clock. Token motion is the sole province of advance_token /
        /// set_token_position.
        #[test]
        fn advance_does_not_touch_token(
            n in 0u32..50,
            layer in 0u32..8,
        ) {
            let mut state = TickState::new(0);
            let token_before = state.token();
            for _ in 0..n {
                state.advance("comp", layer, 0);
            }
            prop_assert_eq!(state.token(), token_before, "advance moved the token clock");
            prop_assert_eq!(state.operator(), u64::from(n), "operator did not count advances");
            prop_assert_eq!(state.step_count(), u64::from(n), "step_count did not count advances");
        }

        /// Metamorphic: step_count is invariant under token boundaries — it
        /// counts every operator regardless of how many advance_token resets
        /// occur between them.
        #[test]
        fn step_count_survives_token_resets(
            steps in proptest::collection::vec(0u32..5, 0..20),
        ) {
            let mut state = TickState::new(0);
            let mut expected = 0u64;
            for &k in &steps {
                for _ in 0..k {
                    state.advance("c", 0, 0);
                    expected += 1;
                }
                state.advance_token();
                prop_assert_eq!(state.operator(), 0, "operator not reset by advance_token");
            }
            prop_assert_eq!(state.step_count(), expected, "step_count miscounted");
        }

        /// Property: advance_token drives Prefill -> Decode but is a fixed point
        /// for every other phase (Decode stays Decode; chunked-prefill is
        /// self-managed). Exercises the phase transition rule directly.
        #[test]
        fn advance_token_phase_rule(start in 0usize..3) {
            let phases = phase_variants();
            let mut state = TickState::new(0);
            state.set_phase(phases[start]);
            state.advance_token();
            let expected = if matches!(phases[start], Phase::Prefill) {
                Phase::Decode
            } else {
                phases[start]
            };
            prop_assert_eq!(state.phase(), expected);
        }
    }
}

// ===========================================================================
// tick_id contract + identity tests (B004 platoon, MIKE lane).
//
// MANDATE: pin the tick_id invariant the impl ACTUALLY maintains across
// generated forward/reverse step sequences, and FLAG (not silently encode) the
// ADR/impl contradiction.
//
//   *** ADR/IMPL CONTRADICTION — for the protocol owner, do not "fix" here ***
//   ADR-0005-tick-model.md:83 — "`tick_id` is a monotonic `u64`, never reused,
//     never reset within a session. It is the primary key for checkpoints,
//     probe firings, intervention attachment, and session bundle references."
//   tick.rs:90-92 — `tick_id()` returns `self.operator`.
//   tick.rs:67-74 — `advance_token()` resets `self.operator = 0`.
//   => The impl's tick_id RESETS to 0 at every token and therefore COLLIDES
//      across tokens. It is NOT the unique, monotonic, never-reset primary key
//      the ADR promises. dispatch.rs:461 / :1641 surface this `tick_id` as the
//      protocol identity, so the collision is observable on the wire.
//   The field that DOES satisfy the ADR contract is `step_count` (tick.rs:21-22,
//      "Total operators traversed across the whole session"), which is marked
//      "diagnostics only, never surfaced as tick_id".
//
// These tests PIN the implemented (reset-per-token) semantics so a future
// reconciliation toward the ADR shows up as a deliberate, test-breaking change.
// They are the evidence behind PLATOON-FINDINGS.md "ADR contradiction".
// ===========================================================================
#[cfg(test)]
mod tick_id_contract_tests {
    use proptest::prelude::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;
    use std::collections::{HashMap, HashSet};

    use super::{StepDirection, TickState};

    /// A navigation step. `Forward` advances one operator within the token;
    /// `NextToken` advances the token clock (resetting operator); `SeekToken`
    /// is the ONLY backward motion the impl offers (true reverse stepping is
    /// deferred to Phase 8+ per ADR-0005:89) — it repositions the token clock to
    /// an arbitrary (possibly earlier) value via `set_token_position`.
    #[derive(Clone, Debug)]
    enum Nav {
        Forward { component: u8 },
        NextToken,
        SeekToken(u64),
    }

    fn nav_strategy() -> impl Strategy<Value = Nav> {
        prop_oneof![
            4 => (0u8..6).prop_map(|component| Nav::Forward { component }),
            2 => Just(Nav::NextToken),
            // Include the FULL u64 range here, unlike the BRAVO generator which
            // caps SetTokenPosition at 0..1000 (tick.rs:374) and so can never
            // reach the advance_token overflow boundary. See findings.
            2 => any::<u64>().prop_map(Nav::SeekToken),
        ]
    }

    proptest! {
        /// Invariant the impl maintains: across ANY forward/reverse navigation,
        /// `tick_id` is byte-for-byte the operator clock, and the emitted
        /// position always reports `direction: Forward` (TickState never emits
        /// Backward — finding: the schema has StepDirection::Backward but the
        /// cursor hardcodes Forward at tick.rs:121). Oracle tier 5 (property).
        #[test]
        fn tick_id_aliases_operator_and_direction_is_always_forward(
            navs in proptest::collection::vec(nav_strategy(), 0..120),
        ) {
            let mut state = TickState::new(3);
            for nav in &navs {
                match *nav {
                    Nav::Forward { component } => state.advance(&format!("c{component}"), 0, 0),
                    Nav::NextToken => state.advance_token(),
                    Nav::SeekToken(pos) => state.set_token_position(pos),
                }
                prop_assert_eq!(state.tick_id(), state.operator(), "tick_id drifted from operator");
                let p = state.to_tick_position();
                prop_assert_eq!(p.tick_id, state.operator(), "position tick_id != operator");
                prop_assert_eq!(p.direction, StepDirection::Forward, "cursor emitted non-Forward");
            }
        }

        /// CONTRADICTION WITNESS (pins impl behavior, documents ADR violation):
        /// over a forward pass that crosses >=1 token boundary with operators on
        /// each side, `tick_id` is NOT injective over the session — at least one
        /// value recurs — whereas the full `(token, operator)` clock pair and
        /// `step_count` ARE unique per visited tick. This is the concrete
        /// counterexample to ADR-0005:83's "never reset / primary key" claim.
        /// Oracle tier 6 (model: identity-as-set).
        #[test]
        fn tick_id_is_not_a_unique_session_key_but_clock_and_step_count_are(
            // per-token operator counts; >=2 tokens, each with >=1 operator,
            // guarantees a reset that forces a tick_id collision.
            counts in proptest::collection::vec(1u32..6, 2..8),
        ) {
            let mut state = TickState::new(0);
            let mut tick_ids: Vec<u64> = Vec::new();
            let mut clock_pairs: HashSet<(u64, u64)> = HashSet::new();
            let mut step_counts: HashSet<u64> = HashSet::new();

            for (i, &k) in counts.iter().enumerate() {
                if i > 0 {
                    state.advance_token();
                }
                for _ in 0..k {
                    state.advance("c", 0, 0);
                    tick_ids.push(state.tick_id());
                    clock_pairs.insert((state.token(), state.operator()));
                    step_counts.insert(state.step_count());
                }
            }

            let total: usize = counts.iter().map(|&k| k as usize).sum();
            // tick_id collides: fewer distinct ids than visited ticks.
            let distinct_ids: HashSet<u64> = tick_ids.iter().copied().collect();
            prop_assert!(
                distinct_ids.len() < total,
                "tick_id was unexpectedly unique across {} ticks (ADR contract would hold!) — \
                 distinct={}; if this fires, the impl was reconciled to the ADR",
                total, distinct_ids.len()
            );
            // The actual unique keys: full clock pair and step_count.
            prop_assert_eq!(clock_pairs.len(), total, "(token, operator) pair not unique");
            prop_assert_eq!(step_counts.len(), total, "step_count not unique");
        }

        /// Model-based: `step_count` is the field that satisfies the ADR's
        /// "monotonic u64, never reset within a session" contract. Over any
        /// forward/reverse op sequence it is non-decreasing, increments by
        /// exactly 1 on each operator advance, and is untouched by token motion
        /// (NextToken / SeekToken). Oracle tier 6.
        #[test]
        fn step_count_is_the_monotonic_never_reset_clock(
            navs in proptest::collection::vec(nav_strategy(), 0..120),
        ) {
            let mut state = TickState::new(0);
            let mut prev = state.step_count();
            let mut model = 0u64;
            for nav in &navs {
                match *nav {
                    Nav::Forward { component } => { state.advance(&format!("c{component}"), 0, 0); model += 1; }
                    Nav::NextToken => state.advance_token(),
                    Nav::SeekToken(pos) => state.set_token_position(pos),
                }
                let now = state.step_count();
                prop_assert!(now >= prev, "step_count decreased: {} -> {}", prev, now);
                prop_assert_eq!(now, model, "step_count diverged from operator-advance count");
                prev = now;
            }
        }

        /// Metamorphic (reverse motion): `set_token_position` (the only backward
        /// motion) moves ONLY the token clock — it must never disturb the
        /// operator clock / tick_id. Pins the subtle semantic that seeking back
        /// to an earlier token leaves the within-token operator index intact.
        #[test]
        fn seek_token_preserves_operator(
            pre in 0u32..8,
            target in any::<u64>(),
        ) {
            let mut state = TickState::new(0);
            for _ in 0..pre {
                state.advance("c", 0, 0);
            }
            let op_before = state.operator();
            state.set_token_position(target);
            prop_assert_eq!(state.operator(), op_before, "seek disturbed the operator clock");
            prop_assert_eq!(state.tick_id(), op_before, "seek disturbed tick_id");
            prop_assert_eq!(state.token(), target, "seek did not set the token clock");
        }

        /// Exception/robustness (tier 5): the token clock must never panic or
        /// silently wrap to a SMALLER value under `advance_token`, even from the
        /// u64 ceiling. advance_token must be monotonic-or-saturating on `token`.
        /// REGRESSION for the overflow bug recorded in PLATOON-FINDINGS.md:
        /// pre-fix, `set_token_position(u64::MAX); advance_token()` panicked with
        /// "attempt to add with overflow" (debug) / wrapped to 0 (release).
        ///
        /// NOTE the generator: `any::<u64>()` alone never samples the exact
        /// ceiling, so the boundary is injected explicitly — the bug is invisible
        /// without it (a generator-coverage lesson recorded in findings).
        #[test]
        fn advance_token_never_overflows_or_regresses_token(
            seed in prop_oneof![
                Just(u64::MAX),
                Just(u64::MAX - 1),
                Just(u64::MAX - 2),
                any::<u64>(),
            ],
        ) {
            let mut state = TickState::new(0);
            state.set_token_position(seed);
            let before = state.token();
            state.advance_token(); // must not panic
            prop_assert!(state.token() >= before, "advance_token decreased the token clock");
        }
    }

    /// Deterministic witness of the overflow boundary (companion to the
    /// property above so the boundary is ALWAYS exercised, not just biased):
    /// seeking to the u64 ceiling and advancing the token must saturate, never
    /// panic and never wrap below the ceiling.
    #[test]
    fn advance_token_saturates_at_ceiling() {
        let mut state = TickState::new(0);
        state.set_token_position(u64::MAX);
        state.advance_token();
        assert_eq!(
            state.token(),
            u64::MAX,
            "token must saturate, not wrap/panic"
        );
        // operator still resets — token saturation does not change that.
        assert_eq!(state.operator(), 0);
    }

    /// Generator audit (cover-style): the navigation generator must actually
    /// exercise (a) sequences that cross >=1 token boundary, (b) operator
    /// advances, and (c) `SeekToken` values in the high u64 range that can reach
    /// the overflow boundary the BRAVO generator (capped at 0..1000) could not.
    #[test]
    fn generator_audit_nav_distribution() {
        const N: u32 = 2000;
        const HIGH: u64 = u64::MAX / 2;
        let mut runner = TestRunner::default();
        let strat = proptest::collection::vec(nav_strategy(), 0..120);
        let (mut with_reset, mut with_forward, mut high_seek, mut total_navs) =
            (0u32, 0u32, 0u32, 0u64);
        for _ in 0..N {
            let navs = strat.new_tree(&mut runner).unwrap().current();
            let mut saw_reset = false;
            let mut saw_forward = false;
            for nav in &navs {
                total_navs += 1;
                match *nav {
                    Nav::NextToken => saw_reset = true,
                    Nav::Forward { .. } => saw_forward = true,
                    Nav::SeekToken(p) if p >= HIGH => high_seek += 1,
                    Nav::SeekToken(_) => {}
                }
            }
            if saw_reset {
                with_reset += 1;
            }
            if saw_forward {
                with_forward += 1;
            }
        }
        assert!(
            with_reset > N / 2,
            "too few token-reset sequences: {with_reset}/{N}"
        );
        assert!(
            with_forward > N / 2,
            "too few operator-advance sequences: {with_forward}/{N}"
        );
        // High-half SeekToken values must appear, proving we reach the region
        // around the overflow boundary that the capped BRAVO generator misses.
        assert!(
            high_seek > 0,
            "no high-range SeekToken samples in {total_navs} navs"
        );
    }

    /// Explicit witness (not a property) of the cross-token `tick_id` collision,
    /// kept as a readable regression for the ADR finding: two distinct ticks in
    /// two different tokens share `tick_id == 1`, yet differ in their full clock.
    #[test]
    fn cross_token_tick_id_collision_witness() {
        let mut state = TickState::new(0);
        state.advance("a", 0, 0);
        let first = state.to_tick_position(); // token 0, op 1, tick_id 1
        state.advance_token();
        state.advance("b", 0, 0);
        let second = state.to_tick_position(); // token 1, op 1, tick_id 1

        assert_eq!(first.tick_id, 1);
        assert_eq!(
            second.tick_id, 1,
            "tick_id collides across tokens (pins impl, violates ADR-0005:83)"
        );
        // They are genuinely different ticks — only the full clock disambiguates.
        let c1 = first.clock.unwrap();
        let c2 = second.clock.unwrap();
        assert_ne!((c1.token, c1.operator), (c2.token, c2.operator));
        assert_eq!((c1.token, c1.operator), (0, 1));
        assert_eq!((c2.token, c2.operator), (1, 1));
    }

    /// Map a few `tick_id`s to the ticks that claim them, demonstrating `tick_id`
    /// is many-to-one across a session (the precise sense in which it is not a
    /// primary key). Belt-and-braces evidence for the finding.
    #[test]
    fn tick_id_is_many_to_one_across_tokens() {
        let mut state = TickState::new(0);
        let mut by_id: HashMap<u64, usize> = HashMap::new();
        for token in 0..4u64 {
            if token > 0 {
                state.advance_token();
            }
            for _ in 0..3 {
                state.advance("c", 0, 0);
                *by_id.entry(state.tick_id()).or_default() += 1;
            }
        }
        // tick_ids 1,2,3 each claimed once per token => 4 ticks per id.
        for id in 1..=3u64 {
            assert_eq!(by_id[&id], 4, "tick_id {id} should map to 4 distinct ticks");
        }
    }
}
