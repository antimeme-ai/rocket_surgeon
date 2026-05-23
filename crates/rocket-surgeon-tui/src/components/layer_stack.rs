//! The layer-stack panel — a vertical list of the model's transformer layers
//! with the current `cursor.layer` highlighted.
//!
//! Stateless: ratatui's `ListState` is constructed per-frame from
//! `state.cursor.layer`, and the widget itself handles scrolling so the
//! selected row stays visible. When `state.session.capabilities.num_layers` is
//! absent — no attach yet, or capabilities not captured by the link — a hint
//! sits in the same bordered block, so the panel does not flicker on attach.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph};

use super::Component;
use crate::state::UiState;

/// The layer-stack panel. Stateless — see the module docs.
pub struct LayerStack;

impl Component for LayerStack {
    fn draw(&self, frame: &mut Frame<'_>, area: Rect, state: &UiState) {
        let block = Block::default().title("Layers").borders(Borders::ALL);

        let num_layers = state
            .session
            .capabilities
            .as_ref()
            .and_then(|c| c.num_layers);

        match num_layers {
            Some(n) if n > 0 => {
                let items: Vec<ListItem<'_>> = (0..n)
                    .map(|i| ListItem::new(format!("Layer {i:>3}")))
                    .collect();
                let list = List::new(items)
                    .block(block)
                    .highlight_style(Style::default().bg(Color::Cyan).fg(Color::Black));
                let selected = state.cursor.layer.min(n - 1) as usize;
                let mut list_state = ListState::default();
                list_state.select(Some(selected));
                frame.render_stateful_widget(list, area, &mut list_state);
            }
            _ => {
                let hint = Paragraph::new("(awaiting capabilities — attach a model to see layers)")
                    .style(Style::default().fg(Color::DarkGray))
                    .block(block);
                frame.render_widget(hint, area);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use crate::state::{UiState, initial_ui_state};

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

    fn render(state: &UiState) -> Terminal<TestBackend> {
        let backend = TestBackend::new(30, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let area = Rect::new(0, 0, 30, 24);
        terminal
            .draw(|frame| LayerStack.draw(frame, area, state))
            .unwrap();
        terminal
    }

    fn pane_text(terminal: &Terminal<TestBackend>) -> String {
        let buf = terminal.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push(buf[(x, y)].symbol().chars().next().unwrap_or(' '));
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn layer_stack_no_caps_shows_hint() {
        let state = initial_ui_state();
        let terminal = render(&state);
        let text = pane_text(&terminal);
        assert!(text.contains("awaiting capabilities"), "got:\n{text}");
    }

    #[test]
    fn layer_stack_with_caps_lists_layers() {
        let mut state = initial_ui_state();
        state.session.capabilities = Some(capabilities(8));
        let terminal = render(&state);
        let text = pane_text(&terminal);
        assert!(text.contains("Layer   0"), "missing first row:\n{text}");
        assert!(text.contains("Layer   7"), "missing last row:\n{text}");
    }

    #[test]
    fn layer_stack_highlights_cursor_layer() {
        let mut state = initial_ui_state();
        state.session.capabilities = Some(capabilities(8));
        state.cursor.layer = 3;
        let terminal = render(&state);
        let buf = terminal.backend().buffer().clone();
        // Find the row containing "Layer   3"; its content cells (just inside
        // the left border) should carry the highlight bg.
        let mut found: Option<u16> = None;
        for y in 1..(buf.area.height - 1) {
            let row: String = (1..(buf.area.width - 1))
                .map(|x| buf[(x, y)].symbol().chars().next().unwrap_or(' '))
                .collect();
            if row.contains("Layer   3") {
                found = Some(y);
                break;
            }
        }
        let y = found.expect("Layer 3 row rendered");
        let bg = buf[(1, y)].style().bg;
        assert_eq!(
            bg,
            Some(Color::Cyan),
            "selected row should carry the highlight bg",
        );
    }
}
