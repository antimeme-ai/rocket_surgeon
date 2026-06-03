//! The tensor-detail panel — research-grade summary stats for the tensor
//! captured at the cursor's current probe point.
//!
//! Stateless: reads `(cursor.layer, cursor.component, session.position.tick_id)`,
//! constructs the cache key, and peeks into `state.tensor_cache`. The peek is
//! deliberately read-only (no LRU promotion) — immediate-mode render loops
//! must not mutate cache ordering each frame.
//!
//! Until the inspect-response → cache wiring lands, the panel shows a hint
//! when its key is absent: "(no tensor captured at cursor)". Tests insert
//! manually to exercise the rendered output.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use rocket_surgeon_protocol::types::TensorSummary;

use super::Component;
use crate::state::UiState;
use crate::state::cache::CacheKey;

/// The tensor-detail panel. Stateless — see the module docs.
pub struct TensorDetail;

impl Component for TensorDetail {
    fn draw(&self, frame: &mut Frame<'_>, area: Rect, state: &UiState) {
        let block = Block::default().title("Tensor").borders(Borders::ALL);

        let Some(key) = cache_key_for_cursor(state) else {
            render_hint(
                frame,
                area,
                block,
                "(awaiting tick position — step the model)",
            );
            return;
        };

        match state.tensor_cache.peek(&key) {
            Some(summary) => {
                let para = Paragraph::new(summary_lines(&key, summary)).block(block);
                frame.render_widget(para, area);
            }
            None => {
                render_hint(
                    frame,
                    area,
                    block,
                    "(no tensor captured at cursor — fire a probe and step)",
                );
            }
        }
    }
}

/// Build the cache lookup key for the cursor's current focus. Returns `None`
/// when no tick has been observed yet — there's nothing to address.
fn cache_key_for_cursor(state: &UiState) -> Option<CacheKey> {
    let tick_id = state.session.position.as_ref().map(|p| p.tick_id)?;
    let component = if state.cursor.component.is_empty() {
        return None;
    } else {
        &state.cursor.component
    };
    // Probe-point format mirrors the perfetto sink's component display name:
    // `L{layer}::{component}`. Daemons routing inspect results into the
    // cache should use the same shape.
    Some(CacheKey {
        tick_id,
        probe_point: format!("L{}::{}", state.cursor.layer, component),
    })
}

fn render_hint(frame: &mut Frame<'_>, area: Rect, block: Block<'_>, msg: &str) {
    let para = Paragraph::new(msg)
        .style(Style::default().fg(Color::DarkGray))
        .block(block);
    frame.render_widget(para, area);
}

fn summary_lines(key: &CacheKey, summary: &TensorSummary) -> Vec<Line<'static>> {
    let label_style = Style::default().fg(Color::DarkGray);
    let value_style = Style::default().fg(Color::White);
    let stats = &summary.stats;

    vec![
        Line::from(vec![
            Span::styled(key.probe_point.clone(), Style::default().fg(Color::Cyan)),
            Span::styled(format!("  @ tick {}", key.tick_id), label_style),
        ]),
        Line::from(""),
        stat_line(
            "shape",
            format!("{:?}", summary.shape),
            value_style,
            label_style,
        ),
        stat_line(
            "dtype",
            format!("{:?}", summary.dtype),
            value_style,
            label_style,
        ),
        stat_line("device", summary.device.clone(), value_style, label_style),
        Line::from(""),
        stat_line(
            "mean",
            format!("{:.4}", stats.mean),
            value_style,
            label_style,
        ),
        stat_line("std", format!("{:.4}", stats.std), value_style, label_style),
        stat_line(
            "min..max",
            format!("{:.4} .. {:.4}", stats.min, stats.max),
            value_style,
            label_style,
        ),
        stat_line(
            "abs_max",
            format!("{:.4}", stats.abs_max),
            value_style,
            label_style,
        ),
        stat_line(
            "l2_norm",
            format!("{:.4}", stats.l2_norm),
            value_style,
            label_style,
        ),
        stat_line(
            "sparsity",
            format!("{:.2}%", stats.sparsity * 100.0),
            value_style,
            label_style,
        ),
    ]
}

fn stat_line(label: &str, value: String, value_style: Style, label_style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:>10}  "), label_style),
        Span::styled(value, value_style),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use rocket_surgeon_protocol::types::{
        DType, Histogram, Phase, StepDirection, TensorStats, TickEvent, TickPosition,
    };

    use crate::state::cache::CacheKey;
    use crate::state::{UiState, initial_ui_state};

    fn position(tick_id: u64) -> TickPosition {
        TickPosition {
            tick_id,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer: 0,
            component: String::new(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        }
    }

    fn summary() -> TensorSummary {
        TensorSummary {
            tensor_id: "abc".to_owned(),
            shape: vec![32, 16, 128],
            dtype: DType::Float32,
            device: "cuda:0".to_owned(),
            sharding: None,
            stats: TensorStats {
                mean: 0.5,
                std: 0.1,
                min: -1.0,
                max: 2.0,
                abs_max: 2.0,
                sparsity: 0.0234,
                l2_norm: 1.234,
                histogram: Histogram {
                    bins: 0,
                    edges: vec![],
                    counts: vec![],
                },
            },
            top_k: vec![],
        }
    }

    fn render(state: &UiState) -> Terminal<TestBackend> {
        let backend = TestBackend::new(40, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        let area = Rect::new(0, 0, 40, 16);
        terminal
            .draw(|frame| TensorDetail.draw(frame, area, state))
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
    fn no_position_yet_shows_awaiting_hint() {
        let state = initial_ui_state();
        let text = pane_text(&render(&state));
        assert!(
            text.contains("awaiting tick position"),
            "expected awaiting-tick hint, got:\n{text}",
        );
    }

    #[test]
    fn position_but_empty_component_shows_awaiting_hint() {
        let mut state = initial_ui_state();
        state.session.position = Some(position(1));
        // cursor.component is "" by default
        let text = pane_text(&render(&state));
        assert!(
            text.contains("awaiting tick position"),
            "an empty cursor component should also fall through to the awaiting hint, got:\n{text}",
        );
    }

    #[test]
    fn key_present_but_no_cache_entry_shows_miss_hint() {
        let mut state = initial_ui_state();
        state.session.position = Some(position(42));
        state.cursor.layer = 7;
        state.cursor.component = "attn.o_proj".to_owned();
        let text = pane_text(&render(&state));
        assert!(
            text.contains("no tensor captured at cursor"),
            "expected miss hint, got:\n{text}",
        );
    }

    #[test]
    fn cache_hit_renders_summary_stats() {
        let mut state = initial_ui_state();
        state.session.position = Some(position(42));
        state.cursor.layer = 7;
        state.cursor.component = "attn.o_proj".to_owned();
        let key = CacheKey {
            tick_id: 42,
            probe_point: "L7::attn.o_proj".to_owned(),
        };
        state.tensor_cache.insert(key, summary());

        let text = pane_text(&render(&state));
        assert!(
            text.contains("L7::attn.o_proj"),
            "missing probe point:\n{text}"
        );
        assert!(text.contains("@ tick 42"), "missing tick label:\n{text}");
        assert!(text.contains("0.5000"), "missing mean:\n{text}");
        assert!(text.contains("1.2340"), "missing l2_norm:\n{text}");
        assert!(text.contains("[32, 16, 128]"), "missing shape:\n{text}");
        assert!(text.contains("cuda:0"), "missing device:\n{text}");
    }

    #[test]
    fn cache_hit_does_not_promote_on_render() {
        // Peek must not mutate LRU order — two renders in a row must not
        // change which entry would be evicted next. Construct a tiny cache
        // that will evict on the next insert; render twice; verify the
        // expected entry survives.
        let mut state = initial_ui_state();
        state.session.position = Some(position(1));
        state.cursor.layer = 0;
        state.cursor.component = "attn".to_owned();

        // Cache holds 2. Insert A and B; render A twice; insert C; B must
        // survive (insertion order: A, B → A evicted by C if peek did not
        // promote A).
        state.tensor_cache = crate::state::cache::TensorCache::new(2);
        let key_a = CacheKey {
            tick_id: 1,
            probe_point: "L0::attn".to_owned(),
        };
        let key_b = CacheKey {
            tick_id: 1,
            probe_point: "L0::b".to_owned(),
        };
        let key_c = CacheKey {
            tick_id: 1,
            probe_point: "L0::c".to_owned(),
        };
        state.tensor_cache.insert(key_a.clone(), summary());
        state.tensor_cache.insert(key_b.clone(), summary());

        // Render A twice. If peek promoted A, B becomes least-recent.
        let _ = render(&state);
        let _ = render(&state);

        // Insert C — should evict whichever entry is at the front of the LRU
        // queue (A, per insertion order, since peek does not promote).
        state.tensor_cache.insert(key_c, summary());

        assert!(
            !state.tensor_cache.contains(&key_a),
            "A should have been evicted (peek must not promote)",
        );
        assert!(
            state.tensor_cache.contains(&key_b),
            "B should have survived (peek must not promote A)",
        );
    }
}
