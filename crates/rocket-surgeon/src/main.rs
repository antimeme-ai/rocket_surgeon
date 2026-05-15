mod dispatch;
mod orchestrator_handle;
mod server;
mod session;
mod tensor_stats;
mod tensor_store;
mod trace_log;

use std::io::{self, BufReader};

use clap::Parser;
use tracing::{error, info, warn};

use crate::dispatch::dispatch;
use crate::orchestrator_handle::OrchestratorHandle;
use crate::server::{read_message, write_message};
use crate::session::Session;
use crate::trace_log::{Direction, TraceLog};

use rocket_surgeon_protocol::messages::{AttachRequest, HostAttachRequest, method};

#[derive(Parser)]
#[command(
    name = "rocket-surgeon",
    about = "Multi-GPU transformer forward pass debugger"
)]
struct Cli {
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Path to the rs-orchestrator binary (default: sibling of this binary)
    #[arg(long, env = "RS_ORCHESTRATOR_BIN")]
    orchestrator_bin: Option<String>,

    /// Path to the rs-worker binary (default: sibling of this binary)
    #[arg(long, env = "RS_WORKER_BIN")]
    worker_bin: Option<String>,
}

/// Locate a sibling binary next to the current executable.
fn find_sibling_binary(name: &str) -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let sibling = dir.join(name);
    if sibling.is_file() {
        Some(sibling.to_string_lossy().into_owned())
    } else {
        None
    }
}

/// Spawn an orchestrator and send `_host/attach`. Returns the handle and
/// model handle on success, or logs a warning and returns `None`.
fn spawn_and_attach(
    request: &rocket_surgeon_protocol::jsonrpc::Request,
    orchestrator_bin: Option<&str>,
    worker_bin: Option<&str>,
    log_level: &str,
) -> Option<(OrchestratorHandle, u64)> {
    let params = request.params.as_ref()?;
    let attach_req: AttachRequest = match serde_json::from_value(params.clone()) {
        Ok(r) => r,
        Err(e) => {
            warn!("failed to parse attach params for orchestrator: {e}");
            return None;
        }
    };

    let host_req = HostAttachRequest {
        model_source: attach_req.model_path,
        model_family: attach_req.model_family,
        device: attach_req.device,
        dtype: attach_req.dtype,
        rank: 0,
        config: attach_req.config,
    };

    let (Some(orch_bin), Some(wrk_bin)) = (orchestrator_bin, worker_bin) else {
        warn!("orchestrator or worker binary not found; running without backend");
        return None;
    };

    let mut orch = match OrchestratorHandle::spawn(orch_bin, wrk_bin, log_level) {
        Ok(o) => o,
        Err(e) => {
            warn!("failed to spawn orchestrator: {e}");
            return None;
        }
    };

    match orch.attach(&host_req) {
        Ok(host_resp) => {
            info!(
                model_handle = host_resp.model_handle,
                num_layers = host_resp.num_layers,
                num_heads = host_resp.num_heads,
                hidden_dim = host_resp.hidden_dim,
                module_count = host_resp.module_tree.len(),
                "orchestrator attached model"
            );
            Some((orch, host_resp.model_handle))
        }
        Err(e) => {
            warn!("orchestrator attach failed: {e}");
            None
        }
    }
}

/// Send `_host/detach` to the orchestrator and drop it.
fn detach_orchestrator(
    orchestrator: &mut Option<OrchestratorHandle>,
    model_handle: &mut Option<u64>,
) {
    if let Some(handle) = model_handle.take() {
        if let Some(orch) = orchestrator {
            if let Err(e) = orch.detach(handle) {
                warn!("orchestrator detach failed: {e}");
            }
        }
    }
    // Drop orchestrator — kills child processes
    *orchestrator = None;
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

    let orchestrator_bin = cli
        .orchestrator_bin
        .or_else(|| find_sibling_binary("rs-orchestrator"));
    let worker_bin = cli.worker_bin.or_else(|| find_sibling_binary("rs-worker"));

    let mut session = Session::new();
    let mut trace_log = TraceLog::new();
    let mut reader = BufReader::new(io::stdin().lock());
    let mut writer = io::stdout().lock();
    let mut orchestrator: Option<OrchestratorHandle> = None;
    let mut model_handle: Option<u64> = None;

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

        if response.error.is_none() && request.method == method::ATTACH {
            if let Some((orch, handle)) = spawn_and_attach(
                &request,
                orchestrator_bin.as_deref(),
                worker_bin.as_deref(),
                &cli.log_level,
            ) {
                orchestrator = Some(orch);
                model_handle = Some(handle);
            }
        }

        if response.error.is_none() && request.method == method::DETACH {
            detach_orchestrator(&mut orchestrator, &mut model_handle);
        }

        let resp_json = serde_json::to_string(&response).expect("serialize response");

        trace_log.record(Direction::Outbound, &resp_json);

        if let Err(e) = write_message(&mut writer, &resp_json) {
            error!("failed to write response: {e}");
            break;
        }
    }

    // Cleanup: drop orchestrator on exit
    drop(orchestrator);

    info!(
        "rocket-surgeon shutting down ({} messages traced)",
        trace_log.len()
    );
}
