//! The status bar: a single-line panel showing the input mode, session
//! status, cursor position, and any pending-request count.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use super::Component;
use crate::state::UiState;

/// The status-bar panel. Stateless — it renders purely from [`UiState`].
pub struct StatusBar;

impl Component for StatusBar {
    fn draw(&self, frame: &mut Frame<'_>, area: Rect, state: &UiState) {
        let mode_str = format!("{:?}", state.mode);
        let status_str = format!("{:?}", state.session.status);
        let pos_str = format!(
            "L{} T{} {}",
            state.cursor.layer, state.cursor.token_position, state.cursor.component
        );
        let pending = if state.pending_requests > 0 {
            format!(" [{}]", state.pending_requests)
        } else {
            String::new()
        };

        let text = format!(" {mode_str} | {status_str} | {pos_str}{pending}");
        let bar = Paragraph::new(Line::from(text))
            .style(Style::default().bg(Color::DarkGray).fg(Color::White));
        frame.render_widget(bar, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    use crate::state::initial_ui_state;

    #[test]
    fn status_bar_shows_mode() {
        let backend = TestBackend::new(80, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let state = initial_ui_state();

        terminal
            .draw(|frame| {
                StatusBar.draw(frame, Rect::new(0, 0, 80, 1), &state);
            })
            .unwrap();

        let buf = terminal.backend().buffer().clone();
        let content: String = (0..buf.area.width)
            .map(|x| buf[(x, 0)].symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("Normal"));
    }

    #[test]
    fn status_bar_renders_without_panic() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let state = initial_ui_state();

        terminal
            .draw(|frame| {
                StatusBar.draw(frame, Rect::new(0, 23, 80, 1), &state);
            })
            .unwrap();
    }
}
