//! The unified event type driving the TUI loop.
//!
//! Every event source — the terminal, and the daemon link — produces an
//! [`Action`]; the loop drains a single channel of them.

use rocket_surgeon_protocol::types::TickPosition;

/// An event delivered to the application loop.
#[derive(Debug)]
pub enum Action {
    /// A raw terminal event, decoded against the live input mode by `App`.
    Terminal(crossterm::event::Event),
    /// A periodic redraw tick.
    Tick,
    /// A state change observed on the daemon link.
    Daemon(DaemonEvent),
}

/// A change in the daemon connection or session, mapped from the daemon's
/// JSON-RPC handshake and notification stream (BEAD-0015 slice 2).
#[derive(Debug)]
pub enum DaemonEvent {
    /// The link is up; the handshake reported this protocol version.
    Connected { protocol_version: String },
    /// The link dropped.
    Disconnected,
    /// The daemon stopped at a tick (`tick.stopped` notification).
    TickStopped(TickPosition),
}

/// A command issued by the application toward the daemon link.
///
/// The mirror of [`DaemonEvent`]: where `DaemonEvent` carries daemon→app state
/// changes, `Effect` carries app→daemon requests. `App::update` returns an
/// `Effect`; the loop forwards it to the daemon task, which turns it into a
/// `rocket/*` JSON-RPC request (BEAD-0015 slice 4).
#[derive(Debug, PartialEq, Eq)]
pub enum Effect {
    /// Advance the forward pass by `count` ticks — a `rocket/step` request.
    RequestStep { count: u32 },
}
