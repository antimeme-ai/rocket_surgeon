#![forbid(unsafe_code)]

mod action;
mod app;
mod client;
mod components;
mod daemon;
mod input;
mod render;
mod state;
mod tiling;
mod tui;

use std::io;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

use app::{App, Flow};
use render::capability;
use tui::Tui;

#[derive(Parser)]
#[command(name = "rocket-surgeon-tui", about = "Terminal UI for rocket-surgeon")]
struct Cli {
    /// Path to the `rocket-surgeon` daemon binary. Defaults to a sibling of
    /// this executable — the same pattern the daemon uses for its own
    /// `--orchestrator-bin` / `--worker-bin` defaults (BEAD-0020).
    #[arg(long)]
    daemon_bin: Option<PathBuf>,

    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u32).range(1..=240))]
    fps: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let daemon_bin = match cli.daemon_bin {
        Some(path) => path,
        None => default_daemon_bin()?,
    };
    let caps = capability::detect();

    tracing_subscriber::fmt()
        .with_env_filter("rocket_surgeon_tui=debug")
        .with_writer(io::stderr)
        .init();

    tracing::info!(
        graphics = ?caps.graphics,
        color = ?caps.color,
        size = format!("{}x{}", caps.width, caps.height),
        daemon = %daemon_bin.display(),
        "terminal capabilities detected"
    );

    let mut tui = Tui::new(cli.fps)?;
    let result = run(&mut tui, daemon_bin).await;
    tui.restore()?;
    result
}

/// Sibling-of-current-exe lookup, mirroring the daemon's own binary defaults.
fn default_daemon_bin() -> anyhow::Result<PathBuf> {
    let exe = std::env::current_exe().context("locating TUI binary")?;
    let dir = exe.parent().context("TUI binary has no parent directory")?;
    Ok(dir.join("rocket-surgeon"))
}

/// The application loop: redraw, take the next action, apply it, and route any
/// resulting effect to the daemon task. Immediate mode — every iteration
/// redraws, so a `Tick` is enough to refresh.
async fn run(tui: &mut Tui, daemon_bin: PathBuf) -> anyhow::Result<()> {
    let mut app = App::new();
    let effects = daemon::spawn(daemon_bin, tui.action_sender());
    loop {
        tui.draw(|frame| app.draw(frame))?;
        let Some(action) = tui.next_action().await else {
            // Every event task has stopped — nothing left to drive the loop.
            return Ok(());
        };
        let outcome = app.update(&action);
        if let Some(effect) = outcome.effect {
            // A closed channel means the daemon task has already exited; its
            // `Disconnected` action will arrive on a later iteration. Drop the
            // effect rather than failing the loop.
            let _ = effects.send(effect).await;
        }
        if outcome.flow == Flow::Quit {
            return Ok(());
        }
    }
}
