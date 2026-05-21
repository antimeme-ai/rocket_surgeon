use ratatui::layout::Rect;

use crate::state::{UiState, ViewId};

#[derive(Debug, Clone, PartialEq)]
pub enum Layout {
    Single(ViewId),
    // In-flight scaffolding: horizontal splits are produced by the
    // context-reactive `propose_layout`, which is unit-tested but not yet
    // wired into the `main.rs` render loop. `dead_code` is a false positive.
    #[allow(dead_code)]
    HSplit {
        left: Box<Self>,
        right: Box<Self>,
        ratio: f32,
    },
    VSplit {
        top: Box<Self>,
        bottom: Box<Self>,
        ratio: f32,
    },
}

impl Layout {
    pub fn single(view: ViewId) -> Self {
        Self::Single(view)
    }

    /// In-flight scaffolding: used by `propose_layout`, which is unit-tested
    /// but not yet wired into the render loop.
    #[allow(dead_code)]
    pub fn hsplit(left: Self, right: Self, ratio: f32) -> Self {
        Self::HSplit {
            left: Box::new(left),
            right: Box::new(right),
            ratio: ratio.clamp(0.1, 0.9),
        }
    }

    pub fn vsplit(top: Self, bottom: Self, ratio: f32) -> Self {
        Self::VSplit {
            top: Box::new(top),
            bottom: Box::new(bottom),
            ratio: ratio.clamp(0.1, 0.9),
        }
    }

    pub fn resolve(&self, area: Rect) -> Vec<(ViewId, Rect)> {
        let mut result = Vec::new();
        self.resolve_into(area, &mut result);
        result
    }

    fn resolve_into(&self, area: Rect, out: &mut Vec<(ViewId, Rect)>) {
        match self {
            Self::Single(id) => {
                out.push((id.clone(), area));
            }
            Self::HSplit { left, right, ratio } => {
                let left_width = (f32::from(area.width) * ratio) as u16;
                let right_width = area.width.saturating_sub(left_width);
                left.resolve_into(Rect::new(area.x, area.y, left_width, area.height), out);
                right.resolve_into(
                    Rect::new(area.x + left_width, area.y, right_width, area.height),
                    out,
                );
            }
            Self::VSplit { top, bottom, ratio } => {
                let top_height = (f32::from(area.height) * ratio) as u16;
                let bottom_height = area.height.saturating_sub(top_height);
                top.resolve_into(Rect::new(area.x, area.y, area.width, top_height), out);
                bottom.resolve_into(
                    Rect::new(area.x, area.y + top_height, area.width, bottom_height),
                    out,
                );
            }
        }
    }

    /// In-flight scaffolding: split-ratio adjustment for the (not yet wired)
    /// interactive resize controls; exercised by unit tests.
    #[allow(dead_code)]
    pub fn adjust_ratio(&mut self, delta: f32) {
        match self {
            Self::HSplit { ratio, .. } | Self::VSplit { ratio, .. } => {
                *ratio = (*ratio + delta).clamp(0.1, 0.9);
            }
            Self::Single(_) => {}
        }
    }

    /// In-flight scaffolding: enumerates the views a layout references, for
    /// the (not yet wired) layout-proposal diffing; exercised by unit tests.
    #[allow(dead_code)]
    pub fn view_ids(&self) -> Vec<ViewId> {
        let mut ids = Vec::new();
        self.collect_ids(&mut ids);
        ids
    }

    #[allow(dead_code)]
    fn collect_ids(&self, ids: &mut Vec<ViewId>) {
        match self {
            Self::Single(id) => ids.push(id.clone()),
            Self::HSplit { left, right, .. } => {
                left.collect_ids(ids);
                right.collect_ids(ids);
            }
            Self::VSplit { top, bottom, .. } => {
                top.collect_ids(ids);
                bottom.collect_ids(ids);
            }
        }
    }
}

/// In-flight scaffolding: context-reactive layout proposal. Unit-tested but
/// not yet wired into the `main.rs` render loop, which uses a fixed layout.
#[allow(dead_code)]
pub fn propose_layout(old: &UiState, new: &UiState) -> Option<Layout> {
    if old.cursor.component != new.cursor.component && !new.cursor.component.is_empty() {
        return Some(Layout::hsplit(
            Layout::single(ViewId(0)),
            Layout::single(ViewId(2)),
            0.6,
        ));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::initial_ui_state;

    fn full_screen() -> Rect {
        Rect::new(0, 0, 200, 60)
    }

    #[test]
    fn single_layout_gives_full_area() {
        let layout = Layout::single(ViewId(0));
        let rects = layout.resolve(full_screen());
        assert_eq!(rects.len(), 1);
        assert_eq!(rects[0].0, ViewId(0));
        assert_eq!(rects[0].1.width, 200);
        assert_eq!(rects[0].1.height, 60);
    }

    #[test]
    fn hsplit_divides_width() {
        let layout = Layout::hsplit(Layout::single(ViewId(0)), Layout::single(ViewId(1)), 0.5);
        let rects = layout.resolve(full_screen());
        assert_eq!(rects.len(), 2);
        assert_eq!(rects[0].1.width, 100);
        assert_eq!(rects[1].1.width, 100);
        assert_eq!(rects[0].1.x, 0);
        assert_eq!(rects[1].1.x, 100);
    }

    #[test]
    fn vsplit_divides_height() {
        let layout = Layout::vsplit(Layout::single(ViewId(0)), Layout::single(ViewId(1)), 0.5);
        let rects = layout.resolve(full_screen());
        assert_eq!(rects.len(), 2);
        assert_eq!(rects[0].1.height, 30);
        assert_eq!(rects[1].1.height, 30);
        assert_eq!(rects[0].1.y, 0);
        assert_eq!(rects[1].1.y, 30);
    }

    #[test]
    fn nested_layout() {
        let layout = Layout::hsplit(
            Layout::single(ViewId(0)),
            Layout::vsplit(Layout::single(ViewId(1)), Layout::single(ViewId(2)), 0.5),
            0.5,
        );
        let rects = layout.resolve(full_screen());
        assert_eq!(rects.len(), 3);
        assert_eq!(rects[0].1.width, 100);
        assert_eq!(rects[1].1.width, 100);
        assert_eq!(rects[2].1.width, 100);
        assert_eq!(rects[1].1.height, 30);
        assert_eq!(rects[2].1.height, 30);
    }

    #[test]
    fn ratio_clamped() {
        let layout = Layout::hsplit(Layout::single(ViewId(0)), Layout::single(ViewId(1)), 0.0);
        match layout {
            Layout::HSplit { ratio, .. } => assert!((ratio - 0.1).abs() < f32::EPSILON),
            _ => panic!("expected HSplit"),
        }
    }

    #[test]
    fn adjust_ratio() {
        let mut layout = Layout::hsplit(Layout::single(ViewId(0)), Layout::single(ViewId(1)), 0.5);
        layout.adjust_ratio(0.1);
        match &layout {
            Layout::HSplit { ratio, .. } => assert!((ratio - 0.6).abs() < f32::EPSILON),
            _ => panic!("expected HSplit"),
        }
    }

    #[test]
    fn adjust_ratio_clamps() {
        let mut layout = Layout::hsplit(Layout::single(ViewId(0)), Layout::single(ViewId(1)), 0.85);
        layout.adjust_ratio(0.2);
        match &layout {
            Layout::HSplit { ratio, .. } => assert!((ratio - 0.9).abs() < f32::EPSILON),
            _ => panic!("expected HSplit"),
        }
    }

    #[test]
    fn view_ids_collects_all() {
        let layout = Layout::hsplit(
            Layout::single(ViewId(0)),
            Layout::vsplit(Layout::single(ViewId(1)), Layout::single(ViewId(2)), 0.5),
            0.5,
        );
        let ids = layout.view_ids();
        assert_eq!(ids, vec![ViewId(0), ViewId(1), ViewId(2)]);
    }

    #[test]
    fn propose_layout_on_component_change() {
        let mut old = initial_ui_state();
        old.cursor.component = "mlp".into();

        let mut new = initial_ui_state();
        new.cursor.component = "attn.o_proj".into();

        let proposal = propose_layout(&old, &new);
        assert!(proposal.is_some());
        let ids = proposal.unwrap().view_ids();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn propose_layout_no_change() {
        let state = initial_ui_state();
        let proposal = propose_layout(&state, &state);
        assert!(proposal.is_none());
    }
}
