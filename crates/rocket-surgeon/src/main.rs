mod dispatch;
mod notifications;
mod orchestrator_handle;
mod server;
mod session;
mod tensor_stats;
mod tensor_store;
mod trace_log;

use std::io::{self, BufReader};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use tracing::{error, info, warn};

use crate::dispatch::{
    dispatch, handle_attach, handle_inspect, handle_probe, handle_step, handle_subscribe,
    handle_unsubscribe, handle_view,
};
use crate::notifications::send_notification;
use crate::orchestrator_handle::OrchestratorHandle;
use crate::server::{read_message, write_message};
use crate::session::Session;
use crate::tensor_store::TensorStore;
use crate::trace_log::{Direction, TraceLog};

use rocket_surgeon_probes::registry::ProbeRegistry;
use rocket_surgeon_protocol::jsonrpc::{Response, RpcError};
use rocket_surgeon_protocol::messages::{
    AttachRequest, HostAttachRequest, HostUpdateProbesRequest, ProbeRequest, TickHeartbeatEvent,
    TickStoppedEvent, event, method,
};
use rocket_surgeon_protocol::types::GranularityScope;

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

/// Spawn the orchestrator and send `_host/attach`. Returns the handle and
/// the full `HostAttachResponse` on success, or a human-readable error
/// message on failure (parse failure, missing binaries, spawn failure,
/// backend rejection). BEAD-0008: the error string becomes the body of a
/// `BACKEND_ATTACH_FAILED` response to the client.
fn spawn_and_attach(
    request: &rocket_surgeon_protocol::jsonrpc::Request,
    orchestrator_bin: Option<&str>,
    worker_bin: Option<&str>,
    log_level: &str,
) -> Result<
    (
        OrchestratorHandle,
        rocket_surgeon_protocol::messages::HostAttachResponse,
    ),
    String,
> {
    let params = request
        .params
        .as_ref()
        .ok_or_else(|| "attach request missing params".to_owned())?;
    let attach_req: AttachRequest = serde_json::from_value(params.clone())
        .map_err(|e| format!("failed to parse attach params: {e}"))?;

    let host_req = HostAttachRequest {
        model_source: attach_req.model_path,
        model_family: attach_req.model_family,
        device: attach_req.device,
        dtype: attach_req.dtype,
        rank: 0,
        config: attach_req.config,
    };

    let (Some(orch_bin), Some(wrk_bin)) = (orchestrator_bin, worker_bin) else {
        return Err("orchestrator or worker binary not found".to_owned());
    };

    let mut orch = OrchestratorHandle::spawn(orch_bin, wrk_bin, log_level)
        .map_err(|e| format!("failed to spawn orchestrator: {e}"))?;

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
            Ok((orch, host_resp))
        }
        Err(e) => Err(format!("orchestrator attach failed: {e}")),
    }
}

fn resolve_granularity(
    explicit: Option<rocket_surgeon_protocol::types::TickGranularity>,
    scopes: &[GranularityScope],
) -> Option<rocket_surgeon_protocol::types::TickGranularity> {
    if explicit.is_some() {
        return explicit;
    }
    scopes.first().map(|s| s.granularity)
}

/// Try to step via the orchestrator. Returns `Some(HostStepResponse)` on success.
fn try_orchestrator_step(
    orchestrator: &mut Option<OrchestratorHandle>,
    model_handle: Option<u64>,
    request: &rocket_surgeon_protocol::jsonrpc::Request,
    granularity_scopes: &[GranularityScope],
) -> Option<rocket_surgeon_protocol::messages::HostStepResponse> {
    let (orch, mh) = (orchestrator.as_mut()?, model_handle?);
    let step_req: rocket_surgeon_protocol::messages::StepRequest = request
        .params
        .as_ref()
        .map_or(
            Ok(rocket_surgeon_protocol::messages::StepRequest {
                direction: rocket_surgeon_protocol::types::StepDirection::Forward,
                count: 1,
                granularity: None,
            }),
            |p| serde_json::from_value(p.clone()),
        )
        .ok()?;

    let granularity = resolve_granularity(step_req.granularity, granularity_scopes);

    let host_req = rocket_surgeon_protocol::messages::HostStepRequest {
        model_handle: mh,
        count: step_req.count,
        direction: step_req.direction,
        granularity,
        max_events: None,
    };
    match orch.step(&host_req) {
        Ok(hr) => Some(hr),
        Err(e) => {
            warn!("orchestrator step failed: {e}");
            None
        }
    }
}

/// Try to inspect via the orchestrator.
/// Returns the `HostInspectResponse` on success, a forwarded error `Response` if
/// the orchestrator returned an RPC error, or `None` if no orchestrator is available.
fn try_orchestrator_inspect(
    orchestrator: &mut Option<OrchestratorHandle>,
    model_handle: Option<u64>,
    request: &rocket_surgeon_protocol::jsonrpc::Request,
) -> Result<Option<rocket_surgeon_protocol::messages::HostInspectResponse>, Box<Response>> {
    let Some(orch) = orchestrator.as_mut() else {
        return Ok(None);
    };
    let Some(mh) = model_handle else {
        return Ok(None);
    };
    let inspect_req: rocket_surgeon_protocol::messages::InspectRequest = match request
        .params
        .as_ref()
        .map(|p| serde_json::from_value(p.clone()))
    {
        Some(Ok(r)) => r,
        _ => return Ok(None),
    };

    let host_req = rocket_surgeon_protocol::messages::HostInspectRequest {
        model_handle: mh,
        target: inspect_req.target,
        detail: inspect_req.detail,
        slices: inspect_req.slices,
    };

    let raw_response = match orch.inspect_raw(&host_req) {
        Ok(r) => r,
        Err(e) => {
            warn!("orchestrator inspect transport error: {e}");
            return Err(Box::new(Response::error(
                request.id.clone(),
                RpcError {
                    code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                    message: format!("orchestrator transport error: {e}"),
                    data: None,
                },
            )));
        }
    };

    if let Some(err) = raw_response.error {
        warn!("orchestrator inspect failed: {}", err.message);
        return Err(Box::new(Response::error(request.id.clone(), err)));
    }

    match raw_response.result {
        Some(value) => {
            let hr: rocket_surgeon_protocol::messages::HostInspectResponse =
                serde_json::from_value(value).map_err(|e| {
                    Box::new(Response::error(
                        request.id.clone(),
                        RpcError {
                            code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                            message: format!("failed to parse orchestrator response: {e}"),
                            data: None,
                        },
                    ))
                })?;
            Ok(Some(hr))
        }
        None => Ok(None),
    }
}

fn try_orchestrator_view(
    orchestrator: &mut Option<OrchestratorHandle>,
    model_handle: Option<u64>,
    request: &rocket_surgeon_protocol::jsonrpc::Request,
) -> Result<Option<rocket_surgeon_protocol::messages::HostViewResponse>, Box<Response>> {
    let Some(orch) = orchestrator.as_mut() else {
        return Ok(None);
    };
    let Some(mh) = model_handle else {
        return Ok(None);
    };
    let view_req: rocket_surgeon_protocol::messages::ViewRequest = match request
        .params
        .as_ref()
        .map(|p| serde_json::from_value(p.clone()))
    {
        Some(Ok(r)) => r,
        _ => return Ok(None),
    };

    let host_req = rocket_surgeon_protocol::messages::HostViewRequest {
        model_handle: mh,
        view: view_req.view,
        params: view_req.params,
    };

    let raw_response = match orch.view_raw(&host_req) {
        Ok(r) => r,
        Err(e) => {
            warn!("orchestrator view transport error: {e}");
            return Err(Box::new(Response::error(
                request.id.clone(),
                RpcError {
                    code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                    message: format!("orchestrator transport error: {e}"),
                    data: None,
                },
            )));
        }
    };

    if let Some(err) = raw_response.error {
        warn!("orchestrator view failed: {}", err.message);
        return Err(Box::new(Response::error(request.id.clone(), err)));
    }

    match raw_response.result {
        Some(value) => {
            let hr: rocket_surgeon_protocol::messages::HostViewResponse =
                serde_json::from_value(value).map_err(|e| {
                    Box::new(Response::error(
                        request.id.clone(),
                        RpcError {
                            code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                            message: format!("failed to parse orchestrator response: {e}"),
                            data: None,
                        },
                    ))
                })?;
            Ok(Some(hr))
        }
        None => Ok(None),
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

fn propagate_probes(
    orchestrator: &mut Option<OrchestratorHandle>,
    model_handle: Option<u64>,
    registry: &ProbeRegistry,
) {
    let (Some(orch), Some(mh)) = (orchestrator.as_mut(), model_handle) else {
        return;
    };
    let enabled = registry.list().into_iter().filter(|p| p.enabled).collect();
    let req = HostUpdateProbesRequest {
        model_handle: mh,
        active_probes: enabled,
    };
    if let Err(e) = orch.update_probes(&req) {
        warn!("failed to propagate probes to worker: {e}");
    }
}

fn default_position() -> rocket_surgeon_protocol::types::TickPosition {
    rocket_surgeon_protocol::types::TickPosition {
        tick_id: 0,
        direction: rocket_surgeon_protocol::types::StepDirection::Forward,
        rank: Some(0),
        layer: 0,
        component: String::new(),
        event: rocket_surgeon_protocol::types::TickEvent::Output,
        replay_of: None,
    }
}

#[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
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
    let mut tensor_store = TensorStore::new();
    let mut trace_log = TraceLog::new();
    let (stdin_tx, stdin_rx) = mpsc::channel::<Result<String, String>>();
    thread::spawn(move || {
        let stdin = io::stdin();
        let mut reader = BufReader::new(stdin.lock());
        loop {
            match read_message(&mut reader) {
                Ok(msg) => {
                    if stdin_tx.send(Ok(msg)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = stdin_tx.send(Err(e.to_string()));
                    break;
                }
            }
        }
    });

    let started_at = Instant::now();
    let mut last_heartbeat = Instant::now();
    let mut stopped_since = Instant::now();
    let mut writer = io::stdout().lock();
    let mut orchestrator: Option<OrchestratorHandle> = None;
    let mut model_handle: Option<u64> = None;
    let mut shm_consumer: Option<rocket_surgeon_shm::ring::DoomRingConsumer> = None;
    let mut probe_registry = ProbeRegistry::new();
    let mut granularity_scopes: Vec<GranularityScope> = Vec::new();
    let mut events_enabled = false;
    let mut notification_seq: u64 = 0;
    let mut last_stale_sweep = Instant::now();

    let stale_names = rocket_surgeon_shm::cleanup::discover_stale_region_names();
    if !stale_names.is_empty() {
        let count = rocket_surgeon_shm::cleanup::sweep_stale_regions(&stale_names);
        if count > 0 {
            info!(count, "cleaned up stale shm regions from previous sessions");
        }
    }

    loop {
        let raw = if events_enabled {
            if last_heartbeat.elapsed() >= Duration::from_secs(1) {
                let position = session
                    .state()
                    .position
                    .clone()
                    .unwrap_or_else(default_position);
                let hb = TickHeartbeatEvent {
                    position,
                    uptime_seconds: started_at.elapsed().as_secs_f64(),
                    elapsed_stopped_sec: stopped_since.elapsed().as_secs_f64(),
                    per_rank_status: vec![],
                };
                let params = serde_json::to_value(&hb).expect("serialize heartbeat");
                if let Err(e) = send_notification(
                    &mut writer,
                    &mut notification_seq,
                    event::TICK_HEARTBEAT,
                    params,
                ) {
                    error!("failed to send heartbeat: {e}");
                    break;
                }
                last_heartbeat = Instant::now();
            }

            match stdin_rx.recv_timeout(Duration::from_secs(1)) {
                Ok(Ok(msg)) => msg,
                Ok(Err(e)) => {
                    info!("connection closed: {e}");
                    break;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    info!("reader thread disconnected");
                    break;
                }
            }
        } else {
            match stdin_rx.recv() {
                Ok(Ok(msg)) => msg,
                Ok(Err(e)) => {
                    info!("connection closed: {e}");
                    break;
                }
                Err(_) => {
                    info!("reader thread disconnected");
                    break;
                }
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

        let mut step_host_response = None;
        // ATTACH: capture the host response so we can wire shm + orchestrator
        // state below if the response ends up successful. BEAD-0008 — the
        // client-facing response now reflects real backend metadata; failed
        // backend attaches surface as BACKEND_ATTACH_FAILED.
        let mut attach_committed: Option<(
            OrchestratorHandle,
            rocket_surgeon_protocol::messages::HostAttachResponse,
        )> = None;
        let response = if request.method == method::ATTACH {
            // BEAD-0008 review (H-1): cheap session-state validation BEFORE
            // we pay to spawn the orchestrator and load the model. Without
            // this precheck, a duplicate attach for a 70B model would spawn
            // a worker, load weights, allocate shm, then get rejected.
            let parsed = request
                .params
                .as_ref()
                .map(|p| serde_json::from_value::<AttachRequest>(p.clone()));
            match parsed {
                Some(Ok(attach_req)) => {
                    if let Err(ref e) = session.validate_attach(&attach_req) {
                        // Build a session-error response directly — no backend spawn.
                        let rpc_err = rocket_surgeon_protocol::jsonrpc::RpcError::from_error_data(
                            e.error_data().clone(),
                        );
                        Response::error(request.id.clone(), rpc_err)
                    } else {
                        match spawn_and_attach(
                            &request,
                            orchestrator_bin.as_deref(),
                            worker_bin.as_deref(),
                            &cli.log_level,
                        ) {
                            Ok((orch, host_resp)) => {
                                let resp = handle_attach(&mut session, &request, Ok(&host_resp));
                                if resp.error.is_none() {
                                    attach_committed = Some((orch, host_resp));
                                } else {
                                    // Shouldn't happen — we just validated.
                                    // Drop orch to kill the worker subprocess.
                                    drop(orch);
                                }
                                resp
                            }
                            Err(msg) => {
                                warn!("backend attach failed: {msg}");
                                handle_attach(&mut session, &request, Err(msg.as_str()))
                            }
                        }
                    }
                }
                // Bad params (missing or malformed) — let handle_attach build
                // the INVALID_PARAMS response. No backend spawn either way.
                _ => handle_attach(
                    &mut session,
                    &request,
                    Err("attach params missing or malformed"),
                ),
            }
        } else if request.method == method::STEP {
            step_host_response = try_orchestrator_step(
                &mut orchestrator,
                model_handle,
                &request,
                &granularity_scopes,
            );
            handle_step(&mut session, &request, step_host_response.as_ref())
        } else if request.method == method::INSPECT {
            match try_orchestrator_inspect(&mut orchestrator, model_handle, &request) {
                Ok(host_response) => handle_inspect(
                    &session,
                    &request,
                    host_response.as_ref(),
                    &mut tensor_store,
                    shm_consumer.as_mut(),
                ),
                Err(err_response) => *err_response,
            }
        } else if request.method == method::VIEW {
            match try_orchestrator_view(&mut orchestrator, model_handle, &request) {
                Ok(host_response) => handle_view(&session, &request, host_response.as_ref()),
                Err(err_response) => *err_response,
            }
        } else if request.method == method::PROBE {
            if let Some(params) = &request.params {
                if let Ok(ProbeRequest::SetGranularity { scopes }) =
                    serde_json::from_value::<ProbeRequest>(params.clone())
                {
                    info!(num_scopes = scopes.len(), "granularity scopes updated");
                    granularity_scopes = scopes;
                }
            }
            let resp = handle_probe(&session, &request, &mut probe_registry);
            if resp.error.is_none() {
                propagate_probes(&mut orchestrator, model_handle, &probe_registry);
            }
            resp
        } else if request.method == method::SUBSCRIBE {
            let resp = handle_subscribe(&session, &request);
            if resp.error.is_none() {
                events_enabled = true;
            }
            resp
        } else if request.method == method::UNSUBSCRIBE {
            events_enabled = false;
            handle_unsubscribe(&session, &request)
        } else {
            dispatch(&mut session, &request)
        };

        if let Some((orch, host_resp)) = attach_committed {
            // Tripwire: validate_attach should have rejected a duplicate
            // attach before we ever got here. If somehow we have both an
            // existing orchestrator and a freshly-attached one, the old one
            // would leak. Kill the old one explicitly.
            if let Some(old) = orchestrator.take() {
                warn!("replacing existing orchestrator on re-attach (unexpected)");
                drop(old);
            }
            orchestrator = Some(orch);
            model_handle = Some(host_resp.model_handle);
            shm_consumer = host_resp.shm_name.and_then(|name| {
                match rocket_surgeon_shm::ring::DoomRingConsumer::open(&name) {
                    Ok(c) => {
                        info!(shm_name = %name, "opened shared memory ring buffer");
                        Some(c)
                    }
                    Err(e) => {
                        warn!("failed to open shm ring '{name}', using base64: {e}");
                        None
                    }
                }
            });
        }

        if response.error.is_none() && request.method == method::DETACH {
            detach_orchestrator(&mut orchestrator, &mut model_handle);
            shm_consumer = None;
        }

        if last_stale_sweep.elapsed() >= Duration::from_secs(60) {
            let stale = rocket_surgeon_shm::cleanup::discover_stale_region_names();
            if !stale.is_empty() {
                let swept = rocket_surgeon_shm::cleanup::sweep_stale_regions(&stale);
                if swept > 0 {
                    info!(count = swept, "periodic stale shm sweep");
                }
            }
            last_stale_sweep = Instant::now();
        }

        let resp_json = serde_json::to_string(&response).expect("serialize response");

        trace_log.record(Direction::Outbound, &resp_json);

        if let Err(e) = write_message(&mut writer, &resp_json) {
            error!("failed to write response: {e}");
            break;
        }

        if events_enabled && response.error.is_none() && request.method == method::STEP {
            stopped_since = Instant::now();

            let position = session
                .state()
                .position
                .clone()
                .unwrap_or_else(default_position);
            let stopped = TickStoppedEvent {
                position,
                state: session.state().status,
            };
            let params = serde_json::to_value(&stopped).expect("serialize tick.stopped");
            if let Err(e) = send_notification(
                &mut writer,
                &mut notification_seq,
                event::TICK_STOPPED,
                params,
            ) {
                error!("failed to send tick.stopped: {e}");
                break;
            }

            if let Some(ref hr) = step_host_response {
                for pe in &hr.events {
                    let params = serde_json::to_value(pe).expect("serialize probe.fired");
                    if let Err(e) = send_notification(
                        &mut writer,
                        &mut notification_seq,
                        event::PROBE_FIRED,
                        params,
                    ) {
                        error!("failed to send probe.fired: {e}");
                        break;
                    }
                }
            }

            last_heartbeat = Instant::now();
        }
    }

    // Cleanup: drop orchestrator on exit
    drop(orchestrator);

    info!(
        "rocket-surgeon shutting down ({} messages traced)",
        trace_log.len()
    );
}
