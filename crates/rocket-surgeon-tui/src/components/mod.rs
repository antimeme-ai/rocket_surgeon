//! View components.
//!
//! Each panel of the TUI is a [`Component`]: a self-contained unit that knows
//! how to render itself into an allocated rect, given the shared [`UiState`].
//! `App` walks the [`Layout`](crate::tiling::Layout), matches each
//! [`ViewKind`](crate::state::ViewKind) to its owning component, and calls
//! [`Component::draw`].
//!
//! Components hold only view-local state (none yet — `StatusBar` and
//! `CommandLine` are stateless unit structs). App-wide state stays centralized
//! in `UiState` and is passed in by reference.

pub mod command_line;
pub mod layer_stack;
pub mod status_bar;

use ratatui::Frame;
use ratatui::layout::Rect;

use crate::state::UiState;

/// A renderable view panel.
///
/// Immediate-mode: `draw` is called every frame with the panel's allocated
/// `area`. A panel-level `update` method will land once a component has
/// view-local behaviour to co-locate — the per-panel components of BEAD-0015
/// slice 5. It is deliberately absent here: slice 4's effect channel routes
/// through `App::update`, so a `Component::update` now would be an unused
/// trait method — dead scaffolding.
pub trait Component {
    /// Draw the component into `area` within `frame`, reading shared state.
    fn draw(&self, frame: &mut Frame<'_>, area: Rect, state: &UiState);
}
