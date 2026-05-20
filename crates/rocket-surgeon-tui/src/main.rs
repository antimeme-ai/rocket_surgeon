mod client;
mod input;
mod render;
mod state;
mod tiling;

use std::io;
use std::time::Duration;

use clap::Parser;
use crossterm::event;
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use render::capability;
use render::compositor;
use state::reducer::{UiEvent, reduce};
use state::{DataDep, ViewId, ViewKind, ViewSlot, initial_ui_state};
use tiling::Layout;

#[derive(Parser)]
#[command(name = "rocket-surgeon-tui", about = "Terminal UI for rocket-surgeon")]
struct Cli {
    #[arg(long, default_value = "/tmp/rocket-surgeon.sock")]
    socket: String,

    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u32).range(1..=240))]
    fps: u32,
}

fn main() -> anyhow::Result<()> {
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
        "terminal capabilities detected"
    );

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_loop(&mut terminal, &cli);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    cli: &Cli,
) -> anyhow::Result<()> {
    let frame_budget = Duration::from_millis(1000 / cli.fps as u64);

    let mut state = initial_ui_state();
    state.views = default_views();
    let layout = default_layout();

    for view in &state.views {
        state.dirty.insert(view.id.clone());
    }

    loop {
        if !state.dirty.is_empty() {
            terminal.draw(|frame| {
                compositor::render_frame(frame, &layout, &state);
            })?;
            state.dirty.clear();
        }

        if event::poll(frame_budget)? {
            if let Ok(ev) = event::read() {
                if let Some(input_event) = input::terminal::decode(ev, state.mode) {
                    if matches!(input_event, input::events::InputEvent::Quit) {
                        return Ok(());
                    }
                    state = reduce(state, UiEvent::Input(input_event));
                }
            }
        }
    }
}

fn default_views() -> Vec<ViewSlot> {
    vec![
        ViewSlot {
            id: ViewId(0),
            kind: ViewKind::LayerStack,
            data_deps: vec![DataDep::CursorPosition, DataDep::SessionStatus],
        },
        ViewSlot {
            id: ViewId(1),
            kind: ViewKind::StatusBar,
            data_deps: vec![DataDep::SessionStatus, DataDep::Mode],
        },
    ]
}

fn default_layout() -> Layout {
    Layout::vsplit(Layout::single(ViewId(0)), Layout::single(ViewId(1)), 0.95)
}
