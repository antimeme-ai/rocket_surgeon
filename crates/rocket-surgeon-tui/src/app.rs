//! The application: owns the UI state and the panel layout, applies actions,
//! and renders. The single owner of [`UiState`] and [`Layout`].

use ratatui::Frame;

use crate::input::events::InputEvent;
use crate::input::terminal::decode;
use crate::render::compositor;
use crate::state::reducer::apply_input;
use crate::state::{UiState, ViewId, ViewKind, ViewSlot, initial_ui_state};
use crate::tiling::Layout;

/// Whether the loop should continue or exit after handling an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Quit,
}

/// The running TUI application.
pub struct App {
    state: UiState,
    layout: Layout,
}

impl App {
    #[must_use]
    pub fn new() -> Self {
        let mut state = initial_ui_state();
        state.views = default_views();
        Self {
            state,
            layout: default_layout(),
        }
    }

    /// Decode a raw terminal event against the live input mode and apply it.
    /// Returns [`Flow::Quit`] when the user asked to exit.
    pub fn handle_terminal(&mut self, event: &crossterm::event::Event) -> Flow {
        match decode(event, self.state.mode) {
            Some(InputEvent::Quit) => Flow::Quit,
            Some(input) => {
                apply_input(&mut self.state, &input);
                Flow::Continue
            }
            None => Flow::Continue,
        }
    }

    /// Render the current state into `frame` (immediate mode — every frame).
    pub fn draw(&self, frame: &mut Frame<'_>) {
        compositor::render_frame(frame, &self.layout, &self.state);
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

fn default_views() -> Vec<ViewSlot> {
    vec![
        ViewSlot {
            id: ViewId(0),
            kind: ViewKind::LayerStack,
        },
        ViewSlot {
            id: ViewId(1),
            kind: ViewKind::StatusBar,
        },
    ]
}

fn default_layout() -> Layout {
    Layout::vsplit(Layout::single(ViewId(0)), Layout::single(ViewId(1)), 0.95)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::mode::Mode;
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    fn key(code: KeyCode) -> Event {
        Event::Key(KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
    }

    #[test]
    fn new_app_has_default_views() {
        let app = App::new();
        assert_eq!(app.state.views.len(), 2);
    }

    #[test]
    fn handle_terminal_navigation_moves_cursor() {
        let mut app = App::new();
        let flow = app.handle_terminal(&key(KeyCode::Char('j')));
        assert_eq!(flow, Flow::Continue);
        assert_eq!(app.state.cursor.layer, 1);
    }

    #[test]
    fn handle_terminal_quit_returns_quit() {
        let mut app = App::new();
        assert_eq!(app.handle_terminal(&key(KeyCode::Char('q'))), Flow::Quit);
    }

    #[test]
    fn handle_terminal_colon_enters_command_mode() {
        let mut app = App::new();
        app.handle_terminal(&key(KeyCode::Char(':')));
        assert_eq!(app.state.mode, Mode::Command);
    }

    #[test]
    fn handle_terminal_unmapped_key_is_continue() {
        let mut app = App::new();
        assert_eq!(app.handle_terminal(&key(KeyCode::F(9))), Flow::Continue);
    }
}
