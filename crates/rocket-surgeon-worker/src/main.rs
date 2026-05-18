mod bridge;
mod dispatch;

use std::io::{self, BufReader};

use clap::Parser;
use rocket_surgeon_transport::framing::{read_message, write_message};
use tracing::{error, info};

use crate::dispatch::dispatch;

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

        let response = dispatch(&request);
        let resp_json = serde_json::to_string(&response).expect("serialize response");

        if let Err(e) = write_message(&mut writer, &resp_json) {
            error!("failed to write response: {e}");
            break;
        }
    }

    info!("rs-worker shutting down");
}
