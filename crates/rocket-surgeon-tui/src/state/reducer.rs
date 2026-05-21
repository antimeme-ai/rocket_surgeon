//! Pure input reduction: applies a decoded [`InputEvent`] to [`UiState`].
//!
//! No I/O, no rendering, no dirty tracking — the event loop redraws every
//! frame (immediate mode). `App` owns the state and calls [`apply_input`].
//! Daemon-driven state changes arrive as their own `Action` variants once the
//! daemon link is wired (BEAD-0015 slice 2).

use crate::input::events::{CommandEvent, InputEvent, ModeEvent, NavigationEvent};
use crate::input::mode::Mode;
use crate::state::UiState;

/// Apply a decoded input event to the UI state.
///
/// `InputEvent::Quit` ends the loop and is handled by the caller; `Resize`
/// needs no state change (the next draw re-reads the terminal size). Both are
/// no-ops here.
pub fn apply_input(state: &mut UiState, event: &InputEvent) {
    match event {
        InputEvent::Navigation(nav) => reduce_navigation(state, nav),
        InputEvent::Mode(mode_event) => reduce_mode(state, *mode_event),
        InputEvent::Command(cmd) => reduce_command(state, cmd),
        InputEvent::Resize { .. } | InputEvent::Quit => {}
    }
}

fn reduce_navigation(state: &mut UiState, nav: &NavigationEvent) {
    let cursor = &mut state.cursor;
    match nav {
        NavigationEvent::Up => cursor.layer = cursor.layer.saturating_sub(1),
        NavigationEvent::Down => cursor.layer = cursor.layer.saturating_add(1),
        NavigationEvent::Left => {
            cursor.token_position = cursor.token_position.saturating_sub(1);
        }
        NavigationEvent::Right => {
            cursor.token_position = cursor.token_position.saturating_add(1);
        }
        NavigationEvent::PageUp => cursor.layer = cursor.layer.saturating_sub(10),
        NavigationEvent::PageDown => cursor.layer = cursor.layer.saturating_add(10),
        NavigationEvent::Home => {
            cursor.layer = 0;
            cursor.token_position = 0;
        }
        NavigationEvent::End => {
            // `clamp_cursor` bounds `layer` to the model's depth; the token
            // position is bounded by the rendering view (slice 5).
            cursor.layer = u32::MAX;
            cursor.token_position = u64::MAX;
        }
        NavigationEvent::ZoomIn
        | NavigationEvent::ZoomOut
        | NavigationEvent::JumpTo(_)
        | NavigationEvent::ContinuousAdjust { .. } => {}
    }
    clamp_cursor(state);
}

fn clamp_cursor(state: &mut UiState) {
    if let Some(caps) = &state.session.capabilities
        && let Some(num_layers) = caps.num_layers
        && num_layers > 0
    {
        state.cursor.layer = state.cursor.layer.min(num_layers - 1);
    }
}

fn reduce_mode(state: &mut UiState, event: ModeEvent) {
    let target = match event {
        ModeEvent::EnterCommand => Mode::Command,
        ModeEvent::EnterInspect => Mode::Inspect,
        ModeEvent::EnterIntervene => Mode::Intervene,
        ModeEvent::ExitToNormal => Mode::Normal,
    };
    if let Some(new_mode) = state.mode.transition(target) {
        if new_mode == Mode::Normal {
            state.command_buffer.clear();
        }
        state.mode = new_mode;
    }
}

fn reduce_command(state: &mut UiState, cmd: &CommandEvent) {
    match cmd {
        CommandEvent::Char(c) => state.command_buffer.push(*c),
        CommandEvent::Backspace => {
            state.command_buffer.pop();
        }
        CommandEvent::Execute => {
            state.status_line = format!("executed: {}", state.command_buffer);
            state.command_buffer.clear();
        }
        CommandEvent::Cancel => state.command_buffer.clear(),
        CommandEvent::TabComplete | CommandEvent::HistoryPrev | CommandEvent::HistoryNext => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::initial_ui_state;

    #[test]
    fn nav_down_increments_layer() {
        let mut state = initial_ui_state();
        apply_input(&mut state, &InputEvent::Navigation(NavigationEvent::Down));
        assert_eq!(state.cursor.layer, 1);
    }

    #[test]
    fn nav_up_at_zero_stays_zero() {
        let mut state = initial_ui_state();
        apply_input(&mut state, &InputEvent::Navigation(NavigationEvent::Up));
        assert_eq!(state.cursor.layer, 0);
    }

    #[test]
    fn nav_right_increments_token() {
        let mut state = initial_ui_state();
        apply_input(&mut state, &InputEvent::Navigation(NavigationEvent::Right));
        assert_eq!(state.cursor.token_position, 1);
    }

    #[test]
    fn page_down_jumps_10_layers() {
        let mut state = initial_ui_state();
        apply_input(
            &mut state,
            &InputEvent::Navigation(NavigationEvent::PageDown),
        );
        assert_eq!(state.cursor.layer, 10);
    }

    #[test]
    fn home_resets_cursor() {
        let mut state = initial_ui_state();
        state.cursor.layer = 15;
        state.cursor.token_position = 42;
        apply_input(&mut state, &InputEvent::Navigation(NavigationEvent::Home));
        assert_eq!(state.cursor.layer, 0);
        assert_eq!(state.cursor.token_position, 0);
    }

    fn capabilities(num_layers: u32) -> rocket_surgeon_protocol::types::Capabilities {
        rocket_surgeon_protocol::types::Capabilities {
            protocol_version: "0.3.0".into(),
            supports_reverse_step: false,
            supports_checkpointing: false,
            supports_moe: false,
            supports_backward: false,
            supports_sae: false,
            execution_mode: rocket_surgeon_protocol::types::ExecutionMode::Eager,
            parallelism: rocket_surgeon_protocol::types::Parallelism::SingleGpu,
            tick_granularities: vec![],
            intervention_types: vec![],
            built_in_views: vec![],
            head_granularity: rocket_surgeon_protocol::types::HeadGranularity::Native,
            transports: vec![],
            wire_formats: vec![],
            max_response_bytes: 0,
            model_family: None,
            model_id: None,
            num_layers: Some(num_layers),
            num_heads: None,
            hidden_dim: None,
            num_ranks: None,
            num_experts: None,
            top_k_experts: None,
            shared_memory_supported: false,
        }
    }

    #[test]
    fn nav_down_clamps_to_max_layer() {
        let mut state = initial_ui_state();
        state.session.capabilities = Some(capabilities(4));
        state.cursor.layer = 3;
        apply_input(&mut state, &InputEvent::Navigation(NavigationEvent::Down));
        assert_eq!(state.cursor.layer, 3, "clamped to num_layers - 1");
    }

    #[test]
    fn mode_change_transitions() {
        let mut state = initial_ui_state();
        apply_input(&mut state, &InputEvent::Mode(ModeEvent::EnterCommand));
        assert_eq!(state.mode, Mode::Command);
    }

    #[test]
    fn invalid_mode_transition_is_no_op() {
        let mut state = initial_ui_state();
        state.mode = Mode::Command;
        apply_input(&mut state, &InputEvent::Mode(ModeEvent::EnterInspect));
        assert_eq!(state.mode, Mode::Command, "Command -> Inspect is rejected");
    }

    #[test]
    fn command_char_appends_to_buffer() {
        let mut state = initial_ui_state();
        state.mode = Mode::Command;
        apply_input(&mut state, &InputEvent::Command(CommandEvent::Char('h')));
        assert_eq!(state.command_buffer, "h");
    }

    #[test]
    fn command_backspace_removes_last_char() {
        let mut state = initial_ui_state();
        state.mode = Mode::Command;
        state.command_buffer = "hel".into();
        apply_input(&mut state, &InputEvent::Command(CommandEvent::Backspace));
        assert_eq!(state.command_buffer, "he");
    }

    #[test]
    fn command_execute_clears_buffer_and_sets_status() {
        let mut state = initial_ui_state();
        state.mode = Mode::Command;
        state.command_buffer = "step".into();
        apply_input(&mut state, &InputEvent::Command(CommandEvent::Execute));
        assert!(state.command_buffer.is_empty());
        assert!(state.status_line.contains("step"));
    }

    #[test]
    fn exit_command_mode_clears_buffer() {
        let mut state = initial_ui_state();
        state.mode = Mode::Command;
        state.command_buffer = "hello".into();
        apply_input(&mut state, &InputEvent::Mode(ModeEvent::ExitToNormal));
        assert!(state.command_buffer.is_empty());
    }
}
