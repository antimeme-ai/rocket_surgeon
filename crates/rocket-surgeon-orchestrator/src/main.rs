#![forbid(unsafe_code)]

mod dispatch;
mod worker_handle;

use std::io::{self, BufReader};

use clap::Parser;
use rocket_surgeon_transport::framing::{read_message, write_message};
use tracing::{error, info};

use crate::dispatch::{OrchestratorState, dispatch};

#[derive(Parser)]
#[command(
    name = "rs-orchestrator",
    about = "Worker lifecycle orchestrator for rocket-surgeon"
)]
struct Cli {
    /// Log level filter (e.g. "info", "debug", "trace")
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Path to the rs-worker binary
    #[arg(long)]
    worker_bin: String,
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

    info!("rs-orchestrator starting");

    let mut state = OrchestratorState {
        worker: None,
        worker_bin: cli.worker_bin,
        log_level: cli.log_level,
    };

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
                if let Err(e) = write_message(&mut writer, &resp_json) {
                    error!("failed to write response: {e}");
                    break;
                }
                continue;
            }
        };

        let response = dispatch(&mut state, &request);
        let resp_json = serde_json::to_string(&response).expect("serialize response");

        if let Err(e) = write_message(&mut writer, &resp_json) {
            error!("failed to write response: {e}");
            break;
        }
    }

    info!("rs-orchestrator shutting down");
}
