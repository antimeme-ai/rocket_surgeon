#![allow(dead_code)]

use std::time::Instant;

use rocket_surgeon_protocol::types::{Phase, StepDirection, TickClock, TickEvent, TickPosition};

pub struct TickState {
    tick_id: u64,
    rank: u32,
    layer: u32,
    component: String,
    call_index: u32,
    step_count: u64,
    phase: Phase,
    token_position: Option<u64>,
    operator_within_token: u64,
    session_start: Instant,
}

impl TickState {
    pub fn new(rank: u32) -> Self {
        Self {
            tick_id: 0,
            rank,
            layer: 0,
            component: String::new(),
            call_index: 0,
            step_count: 0,
            phase: Phase::Decode,
            token_position: None,
            operator_within_token: 0,
            session_start: Instant::now(),
        }
    }

    pub fn advance(&mut self, component: &str, layer: u32, call_index: u32) {
        self.tick_id += 1;
        self.operator_within_token += 1;
        self.layer = layer;
        component.clone_into(&mut self.component);
        self.call_index = call_index;
        self.step_count += 1;
    }

    pub fn advance_token(&mut self) {
        self.operator_within_token = 0;
    }

    pub fn set_phase(&mut self, phase: Phase) {
        self.phase = phase;
    }

    pub fn set_token_position(&mut self, pos: u64) {
        self.token_position = Some(pos);
    }

    pub fn tick_id(&self) -> u64 {
        self.tick_id
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
            tick_id: self.tick_id,
            direction: StepDirection::Forward,
            rank: Some(self.rank),
            layer: self.layer,
            component: self.component.clone(),
            event: TickEvent::Output,
            replay_of: None,
            phase: self.phase,
            token_position: self.token_position,
            clock: Some(TickClock {
                token: self.token_position.unwrap_or(0),
                operator: self.operator_within_token,
                wall_ns: self.session_start.elapsed().as_nanos() as u64,
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
        assert_eq!(state.layer(), 0);
        assert_eq!(state.component(), "");
        assert_eq!(state.call_index(), 0);
    }

    #[test]
    fn advance_increments_tick_id() {
        let mut state = TickState::new(0);
        state.advance("q_proj", 0, 0);
        assert_eq!(state.tick_id(), 1);
        state.advance("k_proj", 0, 0);
        assert_eq!(state.tick_id(), 2);
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
        assert_eq!(state.tick_id(), 1);
        state.advance("embed", 0, 1);
        assert_eq!(state.tick_id(), 2);
        assert_eq!(state.call_index(), 1);
    }

    #[test]
    fn tick_id_is_monotonic() {
        let mut state = TickState::new(0);
        let mut prev = state.tick_id();
        for i in 0..100 {
            state.advance("comp", i % 4, 0);
            assert!(state.tick_id() > prev);
            prev = state.tick_id();
        }
    }

    #[test]
    fn to_tick_position() {
        let mut state = TickState::new(0);
        state.advance("q_proj", 5, 0);
        let pos = state.to_tick_position();
        assert_eq!(pos.tick_id, 1);
        assert_eq!(pos.layer, 5);
        assert_eq!(pos.component, "q_proj");
        assert_eq!(pos.rank, Some(0));
    }

    #[test]
    fn step_count_tracks_total_steps() {
        let mut state = TickState::new(0);
        assert_eq!(state.step_count(), 0);
        state.advance("a", 0, 0);
        assert_eq!(state.step_count(), 1);
        state.advance("b", 0, 0);
        assert_eq!(state.step_count(), 2);
    }
}
