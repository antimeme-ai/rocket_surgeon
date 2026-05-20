use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::state::{UiState, ViewId, ViewKind};
use crate::tiling::Layout;

pub fn render_frame(frame: &mut Frame<'_>, layout: &Layout, state: &UiState) {
    let allocations = layout.resolve(frame.area());

    for (view_id, rect) in &allocations {
        let view = state.views.iter().find(|v| &v.id == view_id);
        match view.map(|v| &v.kind) {
            Some(ViewKind::StatusBar) => render_status_bar(frame, *rect, state),
            Some(ViewKind::CommandLine) => render_command_line(frame, *rect, state),
            _ => render_placeholder(frame, *rect, view_id),
        }
    }
}

fn render_status_bar(frame: &mut Frame<'_>, rect: Rect, state: &UiState) {
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

    let text = format!(" {} | {} | {}{}", mode_str, status_str, pos_str, pending);
    let bar = Paragraph::new(Line::from(text))
        .style(Style::default().bg(Color::DarkGray).fg(Color::White));
    frame.render_widget(bar, rect);
}

fn render_command_line(frame: &mut Frame<'_>, rect: Rect, state: &UiState) {
    let text = if state.mode == crate::input::mode::Mode::Command {
        format!(":{}", state.command_buffer)
    } else {
        state.status_line.clone()
    };
    let para = Paragraph::new(Line::from(text));
    frame.render_widget(para, rect);
}

fn render_placeholder(frame: &mut Frame<'_>, rect: Rect, view_id: &ViewId) {
    let block = Block::default()
        .title(format!("View {}", view_id.0))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner_text = Paragraph::new("(awaiting data)")
        .style(Style::default().fg(Color::DarkGray))
        .block(block);
    frame.render_widget(inner_text, rect);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use crate::state::{DataDep, ViewSlot, initial_ui_state};

    fn test_state() -> UiState {
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
    fn compositor_renders_without_panic() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let state = test_state();
        let layout = Layout::vsplit(Layout::single(ViewId(0)), Layout::single(ViewId(1)), 0.9);

        terminal
            .draw(|frame| {
                render_frame(frame, &layout, &state);
            })
            .unwrap();
    }

    #[test]
    fn status_bar_shows_mode() {
        let backend = TestBackend::new(80, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let state = test_state();
        let layout = Layout::single(ViewId(1));

        terminal
            .draw(|frame| {
                render_frame(frame, &layout, &state);
            })
            .unwrap();

        let buf = terminal.backend().buffer().clone();
        let content: String = (0..buf.area.width)
            .map(|x| buf[(x, 0)].symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("Normal"));
    }
}
