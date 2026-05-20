use rocket_surgeon_protocol::types::{Status, TickPosition};

use crate::input::events::{CommandEvent, InputEvent, ModeEvent, NavigationEvent};
use crate::input::mode::Mode;

use super::{DataDep, SessionSnapshot, UiState};

#[derive(Debug, Clone)]
pub enum UiEvent {
    Input(InputEvent),
    Daemon(DaemonEvent),
    Internal(InternalEvent),
}

#[derive(Debug, Clone)]
pub enum DaemonEvent {
    Connected { protocol_version: String },
    Disconnected,
    StatusChanged(Status),
    PositionChanged(TickPosition),
    SessionUpdated(SessionSnapshot),
}

#[derive(Debug, Clone)]
pub enum InternalEvent {
    RequestStarted,
    RequestFinished,
    StatusMessage(String),
}

pub fn reduce(state: UiState, event: UiEvent) -> UiState {
    match event {
        UiEvent::Input(input) => reduce_input(state, input),
        UiEvent::Daemon(daemon) => reduce_daemon(state, daemon),
        UiEvent::Internal(internal) => reduce_internal(state, internal),
    }
}

fn reduce_input(mut state: UiState, event: InputEvent) -> UiState {
    match event {
        InputEvent::Navigation(nav) => {
            reduce_navigation(&mut state, nav);
            state
        }
        InputEvent::Mode(mode_event) => {
            reduce_mode(&mut state, mode_event);
            state
        }
        InputEvent::Command(cmd) => {
            reduce_command(&mut state, cmd);
            state
        }
        InputEvent::Quit => state,
        InputEvent::Resize { .. } => {
            mark_all_dirty(&mut state);
            state
        }
    }
}

fn reduce_navigation(state: &mut UiState, nav: NavigationEvent) {
    match nav {
        NavigationEvent::Up => {
            state.cursor.layer = state.cursor.layer.saturating_sub(1);
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
        NavigationEvent::Down => {
            state.cursor.layer = state.cursor.layer.saturating_add(1);
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
        NavigationEvent::Left => {
            state.cursor.token_position = state.cursor.token_position.saturating_sub(1);
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
        NavigationEvent::Right => {
            state.cursor.token_position = state.cursor.token_position.saturating_add(1);
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
        NavigationEvent::PageUp => {
            state.cursor.layer = state.cursor.layer.saturating_sub(10);
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
        NavigationEvent::PageDown => {
            state.cursor.layer = state.cursor.layer.saturating_add(10);
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
        NavigationEvent::Home => {
            state.cursor.layer = 0;
            state.cursor.token_position = 0;
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
        NavigationEvent::End => {
            // Actual max layer/token will be clamped by views
            state.cursor.layer = u32::MAX;
            state.cursor.token_position = u64::MAX;
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
        NavigationEvent::ZoomIn | NavigationEvent::ZoomOut => {
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
        NavigationEvent::JumpTo(_) => {
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
        NavigationEvent::ContinuousAdjust { .. } => {
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
    }
    clamp_cursor(state);
}

fn clamp_cursor(state: &mut UiState) {
    if let Some(caps) = &state.session.capabilities {
        if let Some(num_layers) = caps.num_layers {
            if num_layers > 0 {
                state.cursor.layer = state.cursor.layer.min(num_layers - 1);
            }
        }
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
        mark_dep_dirty(state, &DataDep::Mode);
    }
}

fn reduce_command(state: &mut UiState, cmd: CommandEvent) {
    match cmd {
        CommandEvent::Char(c) => {
            state.command_buffer.push(c);
        }
        CommandEvent::Backspace => {
            state.command_buffer.pop();
        }
        CommandEvent::Execute => {
            state.status_line = format!("executed: {}", state.command_buffer);
            state.command_buffer.clear();
        }
        CommandEvent::Cancel => {
            state.command_buffer.clear();
        }
        _ => {}
    }
    mark_dep_dirty(state, &DataDep::Mode);
}

fn reduce_daemon(mut state: UiState, event: DaemonEvent) -> UiState {
    match event {
        DaemonEvent::Connected { protocol_version } => {
            state.session.protocol_version = protocol_version;
            state.session.status = Status::Initialized;
            mark_dep_dirty(&mut state, &DataDep::SessionStatus);
        }
        DaemonEvent::Disconnected => {
            state.session.status = Status::Uninitialized;
            mark_dep_dirty(&mut state, &DataDep::SessionStatus);
        }
        DaemonEvent::StatusChanged(status) => {
            state.session.status = status;
            mark_dep_dirty(&mut state, &DataDep::SessionStatus);
        }
        DaemonEvent::PositionChanged(pos) => {
            state.session.position = Some(pos);
            mark_dep_dirty(&mut state, &DataDep::CursorPosition);
        }
        DaemonEvent::SessionUpdated(snapshot) => {
            state.session = snapshot;
            mark_all_dirty(&mut state);
        }
    }
    state
}

fn reduce_internal(mut state: UiState, event: InternalEvent) -> UiState {
    match event {
        InternalEvent::RequestStarted => {
            state.pending_requests = state.pending_requests.saturating_add(1);
        }
        InternalEvent::RequestFinished => {
            state.pending_requests = state.pending_requests.saturating_sub(1);
        }
        InternalEvent::StatusMessage(msg) => {
            state.status_line = msg;
        }
    }
    state
}

fn mark_dep_dirty(state: &mut UiState, dep: &DataDep) {
    for view in &state.views {
        if view.data_deps.contains(dep) {
            state.dirty.insert(view.id.clone());
        }
    }
}

fn mark_all_dirty(state: &mut UiState) {
    for view in &state.views {
        state.dirty.insert(view.id.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ViewId, ViewKind, ViewSlot, initial_ui_state};

    fn state_with_views() -> UiState {
        let mut state = initial_ui_state();
        state.views = vec![
            ViewSlot {
                id: ViewId(0),
                kind: ViewKind::LayerStack,
                data_deps: vec![DataDep::CursorPosition, DataDep::SessionStatus],
            },
            ViewSlot {
                id: ViewId(1),
                kind: ViewKind::StatusBar,
                data_deps: vec![DataDep::SessionStatus, DataDep::Mode],
            },
            ViewSlot {
                id: ViewId(2),
                kind: ViewKind::TensorDetail,
                data_deps: vec![DataDep::CursorPosition],
            },
        ];
        state
    }

    #[test]
    fn nav_down_increments_layer() {
        let state = state_with_views();
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Navigation(NavigationEvent::Down)),
        );
        assert_eq!(new.cursor.layer, 1);
    }

    #[test]
    fn nav_up_at_zero_stays_zero() {
        let state = state_with_views();
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Navigation(NavigationEvent::Up)),
        );
        assert_eq!(new.cursor.layer, 0);
    }

    #[test]
    fn nav_right_increments_token() {
        let state = state_with_views();
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Navigation(NavigationEvent::Right)),
        );
        assert_eq!(new.cursor.token_position, 1);
    }

    #[test]
    fn nav_dirties_cursor_dependent_views() {
        let state = state_with_views();
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Navigation(NavigationEvent::Down)),
        );
        assert!(new.dirty.contains(&ViewId(0)));
        assert!(new.dirty.contains(&ViewId(2)));
        assert!(!new.dirty.contains(&ViewId(1)));
    }

    #[test]
    fn mode_change_dirties_mode_dependent_views() {
        let state = state_with_views();
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Mode(ModeEvent::EnterCommand)),
        );
        assert_eq!(new.mode, Mode::Command);
        assert!(new.dirty.contains(&ViewId(1)));
        assert!(!new.dirty.contains(&ViewId(2)));
    }

    #[test]
    fn invalid_mode_transition_is_no_op() {
        let mut state = state_with_views();
        state.mode = Mode::Command;
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Mode(ModeEvent::EnterInspect)),
        );
        assert_eq!(new.mode, Mode::Command);
        assert!(new.dirty.is_empty());
    }

    #[test]
    fn daemon_connected_sets_status() {
        let state = state_with_views();
        let new = reduce(
            state,
            UiEvent::Daemon(DaemonEvent::Connected {
                protocol_version: "0.3.0".into(),
            }),
        );
        assert_eq!(new.session.status, Status::Initialized);
        assert_eq!(new.session.protocol_version, "0.3.0");
        assert!(new.dirty.contains(&ViewId(0)));
        assert!(new.dirty.contains(&ViewId(1)));
    }

    #[test]
    fn daemon_disconnected_resets_status() {
        let mut state = state_with_views();
        state.session.status = Status::Stopped;
        let new = reduce(state, UiEvent::Daemon(DaemonEvent::Disconnected));
        assert_eq!(new.session.status, Status::Uninitialized);
    }

    #[test]
    fn session_updated_dirties_all() {
        let state = state_with_views();
        let snapshot = state.session.clone();
        let new = reduce(
            state,
            UiEvent::Daemon(DaemonEvent::SessionUpdated(snapshot)),
        );
        assert_eq!(new.dirty.len(), 3);
    }

    #[test]
    fn request_counting() {
        let state = initial_ui_state();
        let s1 = reduce(state, UiEvent::Internal(InternalEvent::RequestStarted));
        assert_eq!(s1.pending_requests, 1);
        let s2 = reduce(s1, UiEvent::Internal(InternalEvent::RequestStarted));
        assert_eq!(s2.pending_requests, 2);
        let s3 = reduce(s2, UiEvent::Internal(InternalEvent::RequestFinished));
        assert_eq!(s3.pending_requests, 1);
    }

    #[test]
    fn resize_dirties_all_views() {
        let state = state_with_views();
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Resize {
                width: 200,
                height: 50,
            }),
        );
        assert_eq!(new.dirty.len(), 3);
    }

    #[test]
    fn page_down_jumps_10_layers() {
        let state = state_with_views();
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Navigation(NavigationEvent::PageDown)),
        );
        assert_eq!(new.cursor.layer, 10);
    }

    #[test]
    fn home_resets_cursor() {
        let mut state = state_with_views();
        state.cursor.layer = 15;
        state.cursor.token_position = 42;
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Navigation(NavigationEvent::Home)),
        );
        assert_eq!(new.cursor.layer, 0);
        assert_eq!(new.cursor.token_position, 0);
    }

    fn test_capabilities(num_layers: u32) -> rocket_surgeon_protocol::types::Capabilities {
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
        let mut state = state_with_views();
        state.session.capabilities = Some(test_capabilities(4));
        state.cursor.layer = 3;
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Navigation(NavigationEvent::Down)),
        );
        assert_eq!(new.cursor.layer, 3);
    }

    #[test]
    fn initial_state_has_empty_command_buffer() {
        let state = initial_ui_state();
        assert!(state.command_buffer.is_empty());
    }

    #[test]
    fn command_char_appends_to_buffer() {
        let mut state = state_with_views();
        state.mode = Mode::Command;
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Command(CommandEvent::Char('h'))),
        );
        assert_eq!(new.command_buffer, "h");
    }

    #[test]
    fn command_backspace_removes_last_char() {
        let mut state = state_with_views();
        state.mode = Mode::Command;
        state.command_buffer = "hel".into();
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Command(CommandEvent::Backspace)),
        );
        assert_eq!(new.command_buffer, "he");
    }

    #[test]
    fn exit_command_mode_clears_buffer() {
        let mut state = state_with_views();
        state.mode = Mode::Command;
        state.command_buffer = "hello".into();
        let new = reduce(
            state,
            UiEvent::Input(InputEvent::Mode(ModeEvent::ExitToNormal)),
        );
        assert!(new.command_buffer.is_empty());
    }
}
