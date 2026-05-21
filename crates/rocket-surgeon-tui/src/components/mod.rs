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
pub mod status_bar;

use ratatui::Frame;
use ratatui::layout::Rect;

use crate::state::UiState;

/// A renderable view panel.
///
/// Immediate-mode: `draw` is called every frame with the panel's allocated
/// `area`. An `update` method lands in BEAD-0015 slice 4 alongside the effect
/// channel; it is deliberately absent here — an unused trait method now would
/// be dead scaffolding.
pub trait Component {
    /// Draw the component into `area` within `frame`, reading shared state.
    fn draw(&self, frame: &mut Frame<'_>, area: Rect, state: &UiState);
}
