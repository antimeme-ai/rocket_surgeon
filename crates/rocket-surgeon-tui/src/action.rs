//! The unified event type driving the TUI loop.
//!
//! Every event source — the terminal, and (BEAD-0015 slice 2) the daemon
//! link — produces an [`Action`]; the loop drains a single channel of them.
//! Slice 1 has only terminal input and a redraw tick.

/// An event delivered to the application loop.
#[derive(Debug)]
pub enum Action {
    /// A raw terminal event, decoded against the live input mode by `App`.
    Terminal(crossterm::event::Event),
    /// A periodic redraw tick.
    Tick,
}
