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

use clap::Parser;

use action::Action;
use app::{App, Flow};
use render::capability;
use tui::Tui;

#[derive(Parser)]
#[command(name = "rocket-surgeon-tui", about = "Terminal UI for rocket-surgeon")]
struct Cli {
    /// Daemon Unix socket to attach to (wired in BEAD-0015 slice 2).
    #[arg(long, default_value = "/tmp/rocket-surgeon.sock")]
    socket: String,

    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u32).range(1..=240))]
    fps: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let caps = capability::detect();

    tracing_subscriber::fmt()
        .with_env_filter("rocket_surgeon_tui=debug")
        .with_writer(io::stderr)
        .init();

    tracing::info!(
        graphics = ?caps.graphics,
        color = ?caps.color,
        size = format!("{}x{}", caps.width, caps.height),
        socket = %cli.socket,
        "terminal capabilities detected"
    );

    let mut tui = Tui::new(cli.fps)?;
    let result = run(&mut tui, cli.socket).await;
    tui.restore()?;
    result
}

/// The application loop: redraw, then wait for the next action. Immediate
/// mode — every iteration redraws, so a `Tick` is enough to refresh.
async fn run(tui: &mut Tui, socket: String) -> anyhow::Result<()> {
    let mut app = App::new();
    daemon::spawn(socket, tui.action_sender());
    loop {
        tui.draw(|frame| app.draw(frame))?;
        match tui.next_action().await {
            Some(Action::Terminal(ev)) => {
                if app.handle_terminal(&ev) == Flow::Quit {
                    return Ok(());
                }
            }
            Some(Action::Daemon(ev)) => app.handle_daemon(&ev),
            Some(Action::Tick) => {}
            // Every event task has stopped — nothing left to drive the loop.
            None => return Ok(()),
        }
    }
}
