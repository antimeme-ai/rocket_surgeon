mod dispatch;
mod server;
mod session;
mod tensor_stats;
mod tensor_store;
mod trace_log;

use std::io::{self, BufReader};

use clap::Parser;
use tracing::{error, info};

use crate::dispatch::dispatch;
use crate::server::{read_message, write_message};
use crate::session::Session;
use crate::trace_log::{Direction, TraceLog};

#[derive(Parser)]
#[command(
    name = "rocket-surgeon",
    about = "Multi-GPU transformer forward pass debugger"
)]
struct Cli {
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[allow(clippy::significant_drop_tightening)]
fn main() {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .with_writer(io::stderr)
        .init();

    info!("rocket-surgeon starting");

    let mut session = Session::new();
    let mut trace_log = TraceLog::new();
    let mut reader = BufReader::new(io::stdin().lock());
    let mut writer = io::stdout().lock();

    loop {
        let raw = match read_message(&mut reader) {
            Ok(msg) => msg,
            Err(e) => {
                info!("connection closed: {e}");
                break;
            }
        };

        trace_log.record(Direction::Inbound, &raw);

        let request: rocket_surgeon_protocol::jsonrpc::Request = match serde_json::from_str(&raw) {
            Ok(r) => r,
            Err(e) => {
                let err_resp = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": rocket_surgeon_protocol::jsonrpc::PARSE_ERROR,
                        "message": format!("Parse error: {e}"),
                    }
                });
                let resp_json = err_resp.to_string();
                trace_log.record(Direction::Outbound, &resp_json);
                if let Err(e) = write_message(&mut writer, &resp_json) {
                    error!("failed to write response: {e}");
                    break;
                }
                continue;
            }
        };

        let response = dispatch(&mut session, &request);
        let resp_json = serde_json::to_string(&response).expect("serialize response");

        trace_log.record(Direction::Outbound, &resp_json);

        if let Err(e) = write_message(&mut writer, &resp_json) {
            error!("failed to write response: {e}");
            break;
        }
    }

    info!(
        "rocket-surgeon shutting down ({} messages traced)",
        trace_log.len()
    );
}
