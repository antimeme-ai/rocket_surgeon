//! The command line: a single-line panel showing either the in-progress
//! `:`-command being typed (in [`Mode::Command`](crate::input::mode::Mode))
//! or the current status line.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use super::Component;
use crate::input::mode::Mode;
use crate::state::UiState;

/// The command-line panel. Stateless — it renders purely from [`UiState`].
pub struct CommandLine;

impl Component for CommandLine {
    fn draw(&self, frame: &mut Frame<'_>, area: Rect, state: &UiState) {
        let text = if state.mode == Mode::Command {
            format!(":{}", state.command_buffer)
        } else {
            state.status_line.clone()
        };
        let para = Paragraph::new(Line::from(text));
        frame.render_widget(para, area);
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
    fn command_line_renders_without_panic() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let state = initial_ui_state();

        terminal
            .draw(|frame| {
                CommandLine.draw(frame, Rect::new(0, 23, 80, 1), &state);
            })
            .unwrap();
    }

    #[test]
    fn command_line_shows_command_buffer_in_command_mode() {
        let backend = TestBackend::new(80, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = initial_ui_state();
        state.mode = Mode::Command;
        "step 4".clone_into(&mut state.command_buffer);

        terminal
            .draw(|frame| {
                CommandLine.draw(frame, Rect::new(0, 0, 80, 1), &state);
            })
            .unwrap();

        let buf = terminal.backend().buffer().clone();
        let content: String = (0..buf.area.width)
            .map(|x| buf[(x, 0)].symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains(":step 4"));
    }

    #[test]
    fn command_line_shows_status_line_in_normal_mode() {
        let backend = TestBackend::new(80, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = initial_ui_state();
        "ready".clone_into(&mut state.status_line);

        terminal
            .draw(|frame| {
                CommandLine.draw(frame, Rect::new(0, 0, 80, 1), &state);
            })
            .unwrap();

        let buf = terminal.backend().buffer().clone();
        let content: String = (0..buf.area.width)
            .map(|x| buf[(x, 0)].symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("ready"));
    }
}
