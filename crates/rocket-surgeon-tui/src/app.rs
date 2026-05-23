//! The application: owns the UI state and the panel layout, applies actions,
//! and renders. The single owner of [`UiState`] and [`Layout`].

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders, Paragraph};
use rocket_surgeon_protocol::types::Status;

use crate::action::{Action, DaemonEvent, Effect};
use crate::components::Component;
use crate::components::command_line::CommandLine;
use crate::components::layer_stack::LayerStack;
use crate::components::status_bar::StatusBar;
use crate::input::events::InputEvent;
use crate::input::terminal::decode;
use crate::state::reducer::apply_input;
use crate::state::{UiState, ViewId, ViewKind, ViewSlot, initial_ui_state};
use crate::tiling::Layout;

/// Whether the loop should continue or exit after handling an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Quit,
}

/// The result of [`App::update`]: whether the loop continues, and any
/// [`Effect`] the loop must route to the daemon task.
#[derive(Debug, PartialEq, Eq)]
pub struct Outcome {
    pub flow: Flow,
    pub effect: Option<Effect>,
}

impl Outcome {
    /// Continue the loop, dispatching no effect — the common case.
    const CONTINUE: Self = Self {
        flow: Flow::Continue,
        effect: None,
    };

    /// Quit the loop.
    const QUIT: Self = Self {
        flow: Flow::Quit,
        effect: None,
    };
}

/// The running TUI application.
pub struct App {
    state: UiState,
    layout: Layout,
    layer_stack: LayerStack,
    status_bar: StatusBar,
    command_line: CommandLine,
}

impl App {
    #[must_use]
    pub fn new() -> Self {
        let mut state = initial_ui_state();
        state.views = default_views();
        Self {
            state,
            layout: default_layout(),
            layer_stack: LayerStack,
            status_bar: StatusBar,
            command_line: CommandLine,
        }
    }

    /// Apply one [`Action`] to the application — the loop's single entry point.
    ///
    /// Folds terminal input, daemon events, and ticks into the model and
    /// returns an [`Outcome`]: whether to keep running, and any [`Effect`] the
    /// loop must route to the daemon task.
    pub fn update(&mut self, action: &Action) -> Outcome {
        match action {
            Action::Terminal(event) => self.apply_terminal(event),
            Action::Daemon(event) => {
                self.apply_daemon(event);
                Outcome::CONTINUE
            }
            Action::Tick => Outcome::CONTINUE,
        }
    }

    /// Decode a raw terminal event against the live input mode and apply it,
    /// carrying up any [`Effect`] an executed command produced.
    fn apply_terminal(&mut self, event: &crossterm::event::Event) -> Outcome {
        match decode(event, self.state.mode) {
            Some(InputEvent::Quit) => Outcome::QUIT,
            Some(input) => Outcome {
                flow: Flow::Continue,
                effect: apply_input(&mut self.state, &input),
            },
            None => Outcome::CONTINUE,
        }
    }

    /// Apply a daemon-link event to the session snapshot.
    fn apply_daemon(&mut self, event: &DaemonEvent) {
        match event {
            DaemonEvent::Connected { protocol_version } => {
                self.state.session.status = Status::Initialized;
                protocol_version.clone_into(&mut self.state.session.protocol_version);
            }
            DaemonEvent::Disconnected => {
                self.state.session.status = Status::Uninitialized;
            }
            DaemonEvent::TickStopped(position) => {
                self.state.session.status = Status::Stopped;
                self.state.session.position = Some(position.clone());
            }
        }
    }

    /// Render the current state into `frame` (immediate mode — every frame).
    ///
    /// Walks the [`Layout`] to allocate a rect per [`ViewId`], finds each
    /// view's [`ViewSlot`], and dispatches to the owning [`Component`].
    /// `ViewKind`s without a component yet fall through to the placeholder.
    pub fn draw(&self, frame: &mut Frame<'_>) {
        let allocations = self.layout.resolve(frame.area());
        for (view_id, rect) in &allocations {
            let kind = self
                .state
                .views
                .iter()
                .find(|v| &v.id == view_id)
                .map(|v| &v.kind);
            match kind {
                Some(ViewKind::LayerStack) => self.layer_stack.draw(frame, *rect, &self.state),
                Some(ViewKind::StatusBar) => self.status_bar.draw(frame, *rect, &self.state),
                Some(ViewKind::CommandLine) => self.command_line.draw(frame, *rect, &self.state),
                _ => Self::draw_placeholder(frame, *rect, view_id),
            }
        }
    }

    /// Render a bordered placeholder for a [`ViewKind`] that has no component
    /// yet (`LayerStack`, `TensorDetail`, …, built per-panel in slice 5).
    fn draw_placeholder(frame: &mut Frame<'_>, rect: Rect, view_id: &ViewId) {
        let block = Block::default()
            .title(format!("View {}", view_id.0))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));
        let inner_text = Paragraph::new("(awaiting data)")
            .style(Style::default().fg(Color::DarkGray))
            .block(block);
        frame.render_widget(inner_text, rect);
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
        ViewSlot {
            id: ViewId(2),
            kind: ViewKind::CommandLine,
        },
    ]
}

/// The main panel fills the screen; the status bar and command line share the
/// bottom row, one above the other.
fn default_layout() -> Layout {
    Layout::vsplit(
        Layout::single(ViewId(0)),
        Layout::vsplit(Layout::single(ViewId(1)), Layout::single(ViewId(2)), 0.5),
        0.92,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::mode::Mode;
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

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
        assert_eq!(app.state.views.len(), 3);
        assert!(
            app.state
                .views
                .iter()
                .any(|v| v.kind == ViewKind::CommandLine)
        );
    }

    /// Drive one terminal key through `update`.
    fn press(app: &mut App, code: KeyCode) -> Outcome {
        app.update(&Action::Terminal(key(code)))
    }

    fn sample_position() -> rocket_surgeon_protocol::types::TickPosition {
        serde_json::from_value(serde_json::json!({
            "tick_id": 3, "direction": "forward", "rank": 0, "layer": 1,
            "component": "mlp", "event": "output", "replay_of": null,
            "phase": {"type": "decode"}, "token_position": null, "clock": null
        }))
        .expect("position deserializes")
    }

    #[test]
    fn update_navigation_moves_cursor() {
        let mut app = App::new();
        let outcome = press(&mut app, KeyCode::Char('j'));
        assert_eq!(outcome.flow, Flow::Continue);
        assert_eq!(outcome.effect, None);
        assert_eq!(app.state.cursor.layer, 1);
    }

    #[test]
    fn update_quit_key_returns_quit() {
        let mut app = App::new();
        assert_eq!(press(&mut app, KeyCode::Char('q')).flow, Flow::Quit);
    }

    #[test]
    fn update_colon_enters_command_mode() {
        let mut app = App::new();
        press(&mut app, KeyCode::Char(':'));
        assert_eq!(app.state.mode, Mode::Command);
    }

    #[test]
    fn update_unmapped_key_is_continue_without_effect() {
        let mut app = App::new();
        assert_eq!(press(&mut app, KeyCode::F(9)), Outcome::CONTINUE);
    }

    #[test]
    fn update_executed_step_command_emits_effect() {
        let mut app = App::new();
        // ":" enters command mode; typing "step" fills the buffer; Enter runs it.
        for code in [
            KeyCode::Char(':'),
            KeyCode::Char('s'),
            KeyCode::Char('t'),
            KeyCode::Char('e'),
            KeyCode::Char('p'),
        ] {
            assert_eq!(press(&mut app, code).effect, None);
        }
        let outcome = press(&mut app, KeyCode::Enter);
        assert_eq!(outcome.effect, Some(Effect::RequestStep { count: 1 }));
        assert_eq!(outcome.flow, Flow::Continue);
    }

    #[test]
    fn update_daemon_connected_sets_initialized() {
        let mut app = App::new();
        let outcome = app.update(&Action::Daemon(DaemonEvent::Connected {
            protocol_version: "0.3.0".into(),
        }));
        assert_eq!(outcome, Outcome::CONTINUE);
        assert_eq!(app.state.session.status, Status::Initialized);
        assert_eq!(app.state.session.protocol_version, "0.3.0");
    }

    #[test]
    fn update_daemon_disconnected_resets_status() {
        let mut app = App::new();
        app.update(&Action::Daemon(DaemonEvent::Connected {
            protocol_version: "0.3.0".into(),
        }));
        app.update(&Action::Daemon(DaemonEvent::Disconnected));
        assert_eq!(app.state.session.status, Status::Uninitialized);
    }

    #[test]
    fn update_daemon_tick_stopped_updates_position() {
        let mut app = App::new();
        app.update(&Action::Daemon(DaemonEvent::TickStopped(sample_position())));
        assert_eq!(app.state.session.status, Status::Stopped);
        assert_eq!(app.state.session.position.as_ref().unwrap().tick_id, 3);
    }

    #[test]
    fn update_tick_is_continue_without_effect() {
        let mut app = App::new();
        assert_eq!(app.update(&Action::Tick), Outcome::CONTINUE);
    }

    #[test]
    fn draw_renders_without_panic() {
        let app = App::new();
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
    }

    #[test]
    fn draw_dispatches_all_default_views() {
        // The default layout resolves to three rects, one per default view;
        // walking it must not panic and must cover the full screen.
        let app = App::new();
        let allocations = app.layout.resolve(Rect::new(0, 0, 80, 24));
        assert_eq!(allocations.len(), 3);
        for slot in &app.state.views {
            assert!(
                allocations.iter().any(|(id, _)| id == &slot.id),
                "view {:?} has no rect",
                slot.id
            );
        }
    }

    #[test]
    fn draw_renders_placeholder_for_unmapped_kind() {
        // LayerStack now has a component (slice 5a); swap the main panel to
        // an unmapped kind so the placeholder path is still exercised.
        let mut app = App::new();
        app.state.views[0].kind = ViewKind::TensorDetail;
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();

        let buf = terminal.backend().buffer().clone();
        let content: String = (0..buf.area.width)
            .map(|x| buf[(x, 0)].symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("View 0"));
    }
}
