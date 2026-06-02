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
        self.token += 1;
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
