mod adapter;
mod bridge;
mod capture;
mod dispatch;
mod step_driver;
mod tick;

use std::io::{self, BufReader};

use clap::Parser;
use pyo3::prelude::*;
use rocket_surgeon_transport::framing::{read_message, write_message};
use tracing::{error, info, warn};

use crate::dispatch::{WorkerState, dispatch};

#[derive(Parser)]
#[command(name = "rs-worker", about = "Per-rank model worker for rocket-surgeon")]
struct Cli {
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[allow(clippy::significant_drop_tightening)]
fn main() {
    pyo3::prepare_freethreaded_python();

    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .with_writer(io::stderr)
        .init();

    info!("rs-worker starting");

    align_subprocess_interpreter();

    let mut state = WorkerState::new();
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
        let resp_json = match serde_json::to_string(&response) {
            Ok(j) => j,
            Err(e) => {
                error!("failed to serialize response: {e}");
                let fallback = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                        "message": format!("serialization failed: {e}"),
                    }
                });
                fallback.to_string()
            }
        };

        if let Err(e) = write_message(&mut writer, &resp_json) {
            error!("failed to write response: {e}");
            break;
        }
    }

    info!("rs-worker shutting down");
}

/// Repoint the embedded interpreter's subprocess-launch paths at a real
/// Python interpreter.
///
/// The worker embeds `CPython`, which reports the `rs-worker` binary as
/// `sys.executable`. Without this, `multiprocessing` (the `spawn` start
/// method) and torch compile / distributed workers re-exec `rs-worker -c …`,
/// which the worker CLI rejects. Best-effort: a failure here only costs
/// subprocess-spawning features, so it is logged rather than fatal.
fn align_subprocess_interpreter() {
    let chosen = Python::with_gil(|py| {
        py.import("rocket_surgeon.runtime")?
            .call_method0("align_subprocess_interpreter")?
            .extract::<Option<String>>()
    });
    match chosen {
        Ok(Some(path)) => info!(interpreter = %path, "aligned subprocess interpreter"),
        Ok(None) => {}
        Err(e) => warn!("could not align subprocess interpreter: {e}"),
    }
}
