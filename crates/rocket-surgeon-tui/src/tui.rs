//! Terminal lifecycle and the event source feeding the application loop.
//!
//! `Tui` owns the ratatui terminal, toggles raw mode + the alternate screen on
//! construction/teardown, and spawns two tasks — a blocking crossterm reader
//! and a redraw ticker — that merge into one [`Action`] channel. A daemon
//! task feeds the same channel once the link is wired (BEAD-0015 slice 2).

use std::io::{self, Stdout};
use std::time::Duration;

use anyhow::Result;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use crossterm::{event, execute};
use ratatui::backend::CrosstermBackend;
use ratatui::{Frame, Terminal};
use tokio::sync::mpsc;

use crate::action::Action;

type Backend = CrosstermBackend<Stdout>;

/// Owns the terminal and the merged [`Action`] event stream.
///
/// Panic-safety: terminal restoration relies on stack unwinding running the
/// [`Drop`] impl. A spawned task panicking under `panic = "abort"` would skip
/// it; the workspace builds unwind, so this holds.
pub struct Tui {
    terminal: Terminal<Backend>,
    actions: mpsc::Receiver<Action>,
    restored: bool,
}

impl Tui {
    /// Enter raw mode + the alternate screen and start the event tasks.
    /// `fps` bounds the redraw tick.
    pub fn new(fps: u32) -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;

        let (tx, actions) = mpsc::channel(256);
        spawn_input_reader(tx.clone());
        spawn_ticker(tx, fps);

        Ok(Self {
            terminal,
            actions,
            restored: false,
        })
    }

    /// Await the next action; `None` once every event task has stopped.
    pub async fn next_action(&mut self) -> Option<Action> {
        self.actions.recv().await
    }

    /// Draw a frame. Wraps the owned terminal so callers never reach the
    /// `Terminal` directly and cannot desync its alternate-screen state.
    pub fn draw(&mut self, render: impl FnOnce(&mut Frame<'_>)) -> Result<()> {
        self.terminal.draw(render)?;
        Ok(())
    }

    /// Leave the alternate screen and restore the terminal. Idempotent — also
    /// runs from `Drop` as a panic-safety backstop. `restored` is set only
    /// after every step succeeds, so a partial failure lets `Drop` retry.
    pub fn restore(&mut self) -> Result<()> {
        if self.restored {
            return Ok(());
        }
        disable_raw_mode()?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
        self.terminal.show_cursor()?;
        self.restored = true;
        Ok(())
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// Read crossterm events on the blocking pool and forward them as `Action`s.
///
/// The task is detached, not joined: it observes the receiver closing only on
/// its next `blocking_send` (within one ~100 ms poll cycle) and otherwise ends
/// when the process exits. That latency is harmless at slice-1 shutdown; a
/// deterministic cancel path can come with the daemon task in slice 2.
fn spawn_input_reader(tx: mpsc::Sender<Action>) {
    tokio::task::spawn_blocking(move || {
        loop {
            match event::poll(Duration::from_millis(100)) {
                Ok(true) => match event::read() {
                    Ok(ev) => {
                        if tx.blocking_send(Action::Terminal(ev)).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                },
                Ok(false) => {}
                Err(_) => break,
            }
        }
    });
}

/// Emit a redraw `Tick` at `fps`.
fn spawn_ticker(tx: mpsc::Sender<Action>, fps: u32) {
    let period = Duration::from_secs_f64(1.0 / f64::from(fps.max(1)));
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(period);
        loop {
            interval.tick().await;
            if tx.send(Action::Tick).await.is_err() {
                break;
            }
        }
    });
}
