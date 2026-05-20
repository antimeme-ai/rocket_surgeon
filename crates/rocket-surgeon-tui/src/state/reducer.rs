use rocket_surgeon_protocol::types::{Status, TickPosition};

use crate::input::events::{
    CommandEvent, InputEvent, ModeEvent, NavigationEvent,
};
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
            state.cursor.token_position =
                state.cursor.token_position.saturating_sub(1);
            mark_dep_dirty(state, &DataDep::CursorPosition);
        }
        NavigationEvent::Right => {
            state.cursor.token_position =
                state.cursor.token_position.saturating_add(1);
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
}

fn reduce_mode(state: &mut UiState, event: ModeEvent) {
    let target = match event {
        ModeEvent::EnterCommand => Mode::Command,
        ModeEvent::EnterInspect => Mode::Inspect,
        ModeEvent::EnterIntervene => Mode::Intervene,
        ModeEvent::ExitToNormal => Mode::Normal,
    };

    if let Some(new_mode) = state.mode.transition(target) {
        state.mode = new_mode;
        mark_dep_dirty(state, &DataDep::Mode);
    }
}

fn reduce_command(state: &mut UiState, _cmd: CommandEvent) {
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
    use crate::state::{ViewId, ViewKind, ViewSlot};

    fn state_with_views() -> UiState {
        let mut state = UiState::initial();
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
        let state = UiState::initial();
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
}
