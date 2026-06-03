//! The probe-watch panel — live per-probe fire counts and most-recent point.
//!
//! Reads `UiState::probe_stats` (accumulated from `DaemonEvent::ProbeFired`)
//! and renders a table of probe id × fire count × last probe point. When
//! `state.session.defined_probes` has entries, the panel falls back to those
//! to surface probes that are registered but have not yet fired.
//!
//! Stateless. Ordering: by fire count descending, then alphabetic by id —
//! stable across renders so the user's eye doesn't have to chase rows.

use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};

use super::Component;
use crate::state::UiState;

/// The probe-watch panel. Stateless — see the module docs.
pub struct ProbeWatch;

impl Component for ProbeWatch {
    fn draw(&self, frame: &mut Frame<'_>, area: Rect, state: &UiState) {
        let block = Block::default().title("Probes").borders(Borders::ALL);

        if state.probe_stats.is_empty() && state.session.defined_probes.is_empty() {
            let hint = Paragraph::new("(no probes defined — register one with rocket/probe)")
                .style(Style::default().fg(Color::DarkGray))
                .block(block);
            frame.render_widget(hint, area);
            return;
        }

        let rows = build_rows(state);
        let header = Row::new(vec![
            Cell::from("probe").style(Style::default().fg(Color::DarkGray)),
            Cell::from("fires").style(Style::default().fg(Color::DarkGray)),
            Cell::from("last").style(Style::default().fg(Color::DarkGray)),
            Cell::from("point").style(Style::default().fg(Color::DarkGray)),
        ]);

        let table = Table::new(
            rows,
            [
                Constraint::Length(16),
                Constraint::Length(7),
                Constraint::Length(10),
                Constraint::Min(20),
            ],
        )
        .header(header)
        .block(block);

        frame.render_widget(table, area);
    }
}

fn build_rows(state: &UiState) -> Vec<Row<'static>> {
    // Merge: every probe that has ever fired, plus every defined probe that
    // hasn't fired yet. Sort: fired probes first (by count desc, then id),
    // then never-fired probes (by id).
    let mut fired: Vec<(String, u32, u64, String)> = state
        .probe_stats
        .iter()
        .map(|(id, stats)| {
            (
                id.clone(),
                stats.fire_count,
                stats.last_tick_id,
                stats.last_point.clone(),
            )
        })
        .collect();
    fired.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut rows: Vec<Row<'static>> = fired
        .into_iter()
        .map(|(id, count, tick, point)| {
            Row::new(vec![
                Cell::from(id).style(Style::default().fg(Color::Cyan)),
                Cell::from(count.to_string()).style(Style::default().fg(Color::White)),
                Cell::from(format!("t{tick}")).style(Style::default().fg(Color::DarkGray)),
                Cell::from(point).style(Style::default().fg(Color::White)),
            ])
        })
        .collect();

    let mut never_fired: Vec<&str> = state
        .session
        .defined_probes
        .iter()
        .filter(|p| !state.probe_stats.contains_key(&p.id))
        .map(|p| p.id.as_str())
        .collect();
    never_fired.sort_unstable();
    for id in never_fired {
        rows.push(Row::new(vec![
            Cell::from(id.to_owned()).style(Style::default().fg(Color::DarkGray)),
            Cell::from("0").style(Style::default().fg(Color::DarkGray)),
            Cell::from("—").style(Style::default().fg(Color::DarkGray)),
            Cell::from(
                state
                    .session
                    .defined_probes
                    .iter()
                    .find(|p| p.id == id)
                    .map(|p| p.point.clone())
                    .unwrap_or_default(),
            )
            .style(Style::default().fg(Color::DarkGray)),
        ]));
    }

    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use rocket_surgeon_protocol::types::{ProbeAction, ProbeDefinition};

    use crate::state::{ProbeStats, UiState, initial_ui_state};

    fn render(state: &UiState) -> Terminal<TestBackend> {
        let backend = TestBackend::new(60, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let area = Rect::new(0, 0, 60, 12);
        terminal
            .draw(|frame| ProbeWatch.draw(frame, area, state))
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

    fn defined(id: &str, point: &str) -> ProbeDefinition {
        ProbeDefinition {
            id: id.to_owned(),
            point: point.to_owned(),
            action: ProbeAction::Capture,
            config: None,
            enabled: true,
            priority: 0,
        }
    }

    fn stats(count: u32, tick: u64, point: &str) -> ProbeStats {
        ProbeStats {
            fire_count: count,
            last_tick_id: tick,
            last_point: point.to_owned(),
        }
    }

    #[test]
    fn empty_state_shows_hint() {
        let state = initial_ui_state();
        let text = pane_text(&render(&state));
        assert!(
            text.contains("no probes defined"),
            "expected empty-state hint, got:\n{text}",
        );
    }

    #[test]
    fn fired_probe_appears_with_count_and_point() {
        let mut state = initial_ui_state();
        state.probe_stats.insert(
            "attn-watch".to_owned(),
            stats(3, 42, "L7::attn.o_proj:output"),
        );
        let text = pane_text(&render(&state));
        assert!(text.contains("attn-watch"), "missing probe id:\n{text}");
        assert!(text.contains(" 3 "), "missing fire count:\n{text}");
        assert!(text.contains("t42"), "missing last tick:\n{text}");
        assert!(
            text.contains("L7::attn.o_proj"),
            "missing last point:\n{text}",
        );
    }

    #[test]
    fn fired_probes_sort_by_count_descending() {
        let mut state = initial_ui_state();
        state
            .probe_stats
            .insert("low".to_owned(), stats(1, 1, "L0::a"));
        state
            .probe_stats
            .insert("high".to_owned(), stats(10, 5, "L0::b"));
        state
            .probe_stats
            .insert("mid".to_owned(), stats(5, 3, "L0::c"));

        let text = pane_text(&render(&state));
        let high_pos = text.find("high").expect("high row present");
        let mid_pos = text.find("mid").expect("mid row present");
        let low_pos = text.find("low").expect("low row present");
        assert!(
            high_pos < mid_pos && mid_pos < low_pos,
            "expected sort order high→mid→low, got positions: high={high_pos} mid={mid_pos} low={low_pos}\n{text}",
        );
    }

    #[test]
    fn defined_but_never_fired_appears_below_fired() {
        let mut state = initial_ui_state();
        state
            .probe_stats
            .insert("active".to_owned(), stats(2, 7, "L0::x"));
        state
            .session
            .defined_probes
            .push(defined("dormant", "L*::*:output"));

        let text = pane_text(&render(&state));
        let active_pos = text.find("active").expect("active row");
        let dormant_pos = text.find("dormant").expect("dormant row");
        assert!(
            active_pos < dormant_pos,
            "fired probes must precede never-fired ones; got active={active_pos} dormant={dormant_pos}\n{text}",
        );
        // Defined-but-never-fired surfaces the registered point pattern.
        assert!(
            text.contains("L*::*:output"),
            "missing defined-point pattern:\n{text}",
        );
    }

    #[test]
    fn header_row_labels_the_columns() {
        let mut state = initial_ui_state();
        state
            .probe_stats
            .insert("x".to_owned(), stats(1, 1, "L0::x"));
        let text = pane_text(&render(&state));
        assert!(text.contains("probe"), "missing 'probe' header:\n{text}");
        assert!(text.contains("fires"), "missing 'fires' header:\n{text}");
        assert!(text.contains("last"), "missing 'last' header:\n{text}");
        assert!(text.contains("point"), "missing 'point' header:\n{text}");
    }
}
