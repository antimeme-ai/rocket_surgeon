use std::collections::HashSet;

use super::{DataDep, UiState, ViewId};

// Built and unit-tested ahead of reducer-driven layout updates.
#[allow(dead_code)]
pub fn compute_dirty(old: &UiState, new: &UiState) -> HashSet<ViewId> {
    let mut dirty = HashSet::new();

    let changed_deps = changed_data_deps(old, new);
    if changed_deps.is_empty() {
        return dirty;
    }

    for view in &new.views {
        for dep in &view.data_deps {
            if changed_deps.contains(dep) {
                dirty.insert(view.id.clone());
                break;
            }
        }
    }

    dirty
}

#[allow(dead_code)]
fn changed_data_deps(old: &UiState, new: &UiState) -> Vec<DataDep> {
    let mut changed = Vec::new();

    if old.session.status != new.session.status || old.session.position != new.session.position {
        changed.push(DataDep::SessionStatus);
    }

    if old.cursor.layer != new.cursor.layer
        || old.cursor.token_position != new.cursor.token_position
        || old.cursor.component != new.cursor.component
        || old.cursor.focused_view != new.cursor.focused_view
    {
        changed.push(DataDep::CursorPosition);
    }

    if old.mode != new.mode {
        changed.push(DataDep::Mode);
    }

    if old.session.active_interventions != new.session.active_interventions {
        changed.push(DataDep::Interventions);
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::mode::Mode;
    use crate::state::{ViewKind, ViewSlot, initial_ui_state};
    use rocket_surgeon_protocol::types::Status;

    fn two_view_state() -> UiState {
        let mut state = initial_ui_state();
        state.views = vec![
            ViewSlot {
                id: ViewId(0),
                kind: ViewKind::LayerStack,
                data_deps: vec![DataDep::CursorPosition],
            },
            ViewSlot {
                id: ViewId(1),
                kind: ViewKind::StatusBar,
                data_deps: vec![DataDep::SessionStatus, DataDep::Mode],
            },
        ];
        state
    }

    #[test]
    fn no_change_no_dirty() {
        let state = two_view_state();
        let dirty = compute_dirty(&state, &state);
        assert!(dirty.is_empty());
    }

    #[test]
    fn cursor_change_dirties_cursor_views() {
        let old = two_view_state();
        let mut new = old.clone();
        new.cursor.layer = 5;
        let dirty = compute_dirty(&old, &new);
        assert!(dirty.contains(&ViewId(0)));
        assert!(!dirty.contains(&ViewId(1)));
    }

    #[test]
    fn status_change_dirties_status_views() {
        let old = two_view_state();
        let mut new = old.clone();
        new.session.status = Status::Stopped;
        let dirty = compute_dirty(&old, &new);
        assert!(!dirty.contains(&ViewId(0)));
        assert!(dirty.contains(&ViewId(1)));
    }

    #[test]
    fn mode_change_dirties_mode_views() {
        let old = two_view_state();
        let mut new = old.clone();
        new.mode = Mode::Command;
        let dirty = compute_dirty(&old, &new);
        assert!(!dirty.contains(&ViewId(0)));
        assert!(dirty.contains(&ViewId(1)));
    }

    #[test]
    fn multiple_changes_dirty_union() {
        let old = two_view_state();
        let mut new = old.clone();
        new.cursor.layer = 5;
        new.session.status = Status::Stopped;
        let dirty = compute_dirty(&old, &new);
        assert!(dirty.contains(&ViewId(0)));
        assert!(dirty.contains(&ViewId(1)));
    }
}
