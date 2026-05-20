use base64::Engine;
use rocket_surgeon_probes::registry::{ProbeRegistry, RegistryError};
use rocket_surgeon_protocol::errors::{ErrorCode, ErrorData};
use rocket_surgeon_protocol::jsonrpc::{METHOD_NOT_FOUND, Request, RequestId, Response, RpcError};
use rocket_surgeon_protocol::messages::{
    AttachRequest, EventType, HostAttachResponse, HostViewResponse, InitializeRequest,
    InspectRequest, ProbeRequest, ProbeResponse, StepRequest, SubscribeRequest, SubscribeResponse,
    UnsubscribeRequest, UnsubscribeResponse, ViewRequest, ViewResponse, method,
};
use rocket_surgeon_protocol::types::{DType, Phase, StepDirection, TickEvent, TickPosition};

use crate::session::{Session, SessionError};
use crate::tensor_store::TensorStore;

/// Canonical, actionable recovery hint for an error code.
///
/// Every error response the daemon emits MUST carry a `recovery_hint` — a
/// short, imperative string telling the caller how to recover. This is the
/// single source of truth (TCK `error-expressiveness.feature`, scenario
/// `ErrorData includes recovery_hint`). Hints are kept terse and free of
/// tenant-identifying detail so they are safe to surface verbatim.
pub fn recovery_hint_for(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::InvalidState => {
            "Check `state.status` and `valid_states`, then issue an action allowed from the current state."
        }
        ErrorCode::InvalidTarget => {
            "Pick a target from `context.nearest_matches` or `context.valid_components_at_layer`."
        }
        ErrorCode::InvalidRecipe => "Fix the recipe params and resubmit the intervention.",
        ErrorCode::ModelNotAttached | ErrorCode::BackendAttachFailed => {
            "Call `rocket/attach` with a valid model before retrying."
        }
        ErrorCode::ModelAlreadyAttached => "Call `rocket/detach` first, then attach the new model.",
        ErrorCode::TensorNotFound => {
            "Run `rocket/step` to advance the forward pass so the tensor is captured, then retry."
        }
        ErrorCode::CheckpointNotFound => {
            "List checkpoints via `rocket/status` and restore an id that exists."
        }
        ErrorCode::ProbeNotFound => {
            "List probes with `rocket/probe action=list` and use a known id."
        }
        ErrorCode::CapabilityNotSupported => {
            "Check `initialize` capabilities; this build does not support the requested feature."
        }
        ErrorCode::SliceOutOfBounds => {
            "Inspect with `detail=summary` to read the tensor shape, then request an in-bounds slice."
        }
        ErrorCode::ResponseTooLarge => {
            "Narrow the target or request a slice instead of the full tensor."
        }
        ErrorCode::HostError => {
            "Inspect the daemon logs; the host worker hit an unrecoverable fault."
        }
        ErrorCode::GpuOom => "Free GPU memory or attach a smaller model, then retry.",
        ErrorCode::NcclTimeout => {
            "Check inter-rank connectivity; restart the session if ranks are unreachable."
        }
        ErrorCode::ReplayDivergence => {
            "Re-capture from a fresh checkpoint; the replay diverged from the recorded run."
        }
        ErrorCode::UnsupportedModel => "Attach a model from a supported family (see `suggestion`).",
        ErrorCode::CompiledModel => {
            "Remove the torch.compile() wrapper and re-export the model in eager mode."
        }
        ErrorCode::InvalidParams => "Fix the request params to match the method schema and retry.",
        ErrorCode::DuplicateProbeId => {
            "Choose a unique probe id, or remove the existing probe first."
        }
        ErrorCode::InvalidPoint => {
            "Fix the probe point grammar (family:rank:layer:component:slot:event)."
        }
        ErrorCode::ViewDataUnavailable => {
            "Step the forward pass so view data is computed, then retry the view."
        }
        ErrorCode::BranchNotFound => "List branches and target one that exists.",
        ErrorCode::BranchMergeRefused => {
            "Resolve the divergence before merging, or merge into a compatible branch."
        }
        ErrorCode::VramExhausted => {
            "Drop branches or checkpoints to reclaim VRAM (see `context.recommendation`)."
        }
        ErrorCode::CrossRequestKv => {
            "Scope KV access to the originating request; cross-request KV reads are not allowed."
        }
        ErrorCode::KvEvicted => "Re-run the prefill; the KV-cache entry was evicted.",
    }
}

/// Build `ErrorData` with the canonical `recovery_hint` already populated.
fn error_with_hint(code: ErrorCode, suggestion: impl Into<String>) -> ErrorData {
    let mut data = ErrorData::new(code, suggestion);
    data.recovery_hint = Some(recovery_hint_for(code).to_owned());
    data
}

/// Build `ErrorData` with both a `recovery_hint` and structured `context`.
fn error_with_context(
    code: ErrorCode,
    suggestion: impl Into<String>,
    context: serde_json::Value,
) -> ErrorData {
    let mut data = error_with_hint(code, suggestion);
    data.context = Some(context);
    data
}

/// Canonical leaf-component names a target's `component` field may name.
///
/// Targets follow the grammar `family:rank:layer:component:event`, where
/// `component` is a (possibly dotted) path such as `attn.o_proj` or a bare
/// leaf such as `o_proj`. Validation matches the *final* dotted segment
/// against this vocabulary. When a caller names a component outside this
/// set we cannot resolve it to a tensor, so the daemon answers
/// `INVALID_TARGET` with nearest matches drawn from this vocabulary (TCK
/// `error-expressiveness.feature`, scenario `INVALID_TARGET includes
/// nearest matches`).
const CANONICAL_COMPONENTS: &[&str] = &[
    "embed",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "scores",
    "mlp",
    "gate_proj",
    "up_proj",
    "down_proj",
    "router",
    "residual_pre",
    "residual_post",
    "ln_1",
    "ln_2",
    "lm_head",
];

/// The final dotted segment of a component path (`attn.o_proj` -> `o_proj`).
fn component_leaf(component: &str) -> &str {
    component.rsplit('.').next().unwrap_or(component)
}

/// Levenshtein edit distance between two byte strings.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Extract the `component` field (index 3) from an inspect target string of
/// the form `family:rank:layer:component:event`. Returns `None` when the
/// target does not have enough colon-separated fields.
fn target_component(target: &str) -> Option<&str> {
    target.split(':').nth(3).filter(|c| !c.is_empty())
}

/// Closest canonical components to `attempted`, ranked by edit distance.
///
/// Always returns at least one entry (the global nearest) so the TCK's
/// "non-empty array" expectation holds even for wildly malformed input.
fn nearest_components(attempted: &str) -> Vec<&'static str> {
    let mut scored: Vec<(usize, &'static str)> = CANONICAL_COMPONENTS
        .iter()
        .map(|&c| (edit_distance(attempted, c), c))
        .collect();
    scored.sort_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(y.1)));
    let best = scored.first().map_or(usize::MAX, |s| s.0);
    // Keep the global nearest plus anything within a small radius of it.
    scored
        .into_iter()
        .filter(|&(dist, _)| dist <= best.saturating_add(2))
        .take(5)
        .map(|(_, c)| c)
        .collect()
}

/// Build the `INVALID_TARGET` error for an inspect target whose `component`
/// field names something the daemon cannot resolve. The structured context
/// carries the attempted component, nearest spelling matches, and the full
/// set of valid components so a caller (human or LLM) can self-correct.
fn invalid_target_error(target: &str) -> ErrorData {
    let attempted = target_component(target).unwrap_or(target);
    let nearest = nearest_components(component_leaf(attempted));
    let suggestion = match nearest.first() {
        Some(best) => format!("Unknown component '{attempted}' in target — did you mean '{best}'?"),
        None => format!("Unknown component '{attempted}' in target"),
    };
    error_with_context(
        ErrorCode::InvalidTarget,
        suggestion,
        serde_json::json!({
            "attempted": attempted,
            "nearest_matches": nearest,
            "valid_components_at_layer": CANONICAL_COMPONENTS,
        }),
    )
}

/// Build an `INVALID_PARAMS` response for a params-deserialization failure.
///
/// Routed through `ErrorData` (rather than a bare `RpcError`) so the
/// response carries a `recovery_hint` and the raw serde diagnostic in
/// `context.parse_error`, per TCK `error-expressiveness.feature`.
fn invalid_params_response(id: RequestId, err: &serde_json::Error) -> Response {
    let data = error_with_context(
        ErrorCode::InvalidParams,
        format!("Invalid params: {err}"),
        serde_json::json!({ "parse_error": err.to_string() }),
    );
    Response::error(id, RpcError::from_error_data(data))
}

fn session_error_to_response(id: RequestId, err: &SessionError) -> Response {
    let rpc_error = RpcError::from_error_data(err.error_data().clone());
    Response::error(id, rpc_error)
}

fn serialize_envelope<T: serde::Serialize>(id: RequestId, envelope: T) -> Response {
    match serde_json::to_value(envelope) {
        Ok(value) => Response::success(id, value),
        Err(e) => Response::error(
            id,
            RpcError {
                code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                message: format!("Internal serialization error: {e}"),
                data: None,
            },
        ),
    }
}

pub fn dispatch(session: &mut Session, request: &Request) -> Response {
    match request.method.as_str() {
        method::INITIALIZE => handle_initialize(session, request),
        // ATTACH is routed by main.rs, which supplies the backend result.
        // dispatch() never builds the attach response itself — see
        // `handle_attach` for the full flow.
        method::ATTACH => Response::error(
            request.id.clone(),
            RpcError {
                code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                message: "attach must be dispatched via main loop".to_owned(),
                data: None,
            },
        ),
        method::DETACH => handle_detach(session, request),
        method::STATUS => handle_status(session, request),
        method::STEP => handle_step(session, request, None),
        method::INSPECT => handle_inspect_no_store(session, request),
        method::SUBSCRIBE => handle_subscribe(session, request),
        method::UNSUBSCRIBE => handle_unsubscribe(session, request),
        method::VIEW => handle_view(session, request, None),
        method::INTERVENE | method::CHECKPOINT | method::REPLAY => {
            handle_stub_requires_stopped(session, request)
        }
        _ => Response::error(
            request.id.clone(),
            RpcError {
                code: METHOD_NOT_FOUND,
                message: format!("Method not found: {}", request.method),
                data: None,
            },
        ),
    }
}

fn parse_params<T: serde::de::DeserializeOwned>(request: &Request) -> Result<T, serde_json::Error> {
    let params = request
        .params
        .clone()
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
    serde_json::from_value(params)
}

fn handle_initialize(session: &mut Session, request: &Request) -> Response {
    let req: InitializeRequest = match parse_params(request) {
        Ok(r) => r,
        Err(e) => return invalid_params_response(request.id.clone(), &e),
    };

    match session.initialize(&req) {
        Ok(envelope) => serialize_envelope(request.id.clone(), envelope),
        Err(ref e) => session_error_to_response(request.id.clone(), e),
    }
}

/// Build the client-facing `attach` response.
///
/// `backend_result` is the outcome of the worker round-trip:
/// - `Ok(host)` → validate (cheap), then commit with real metadata.
/// - `Err(msg)` → return `BACKEND_ATTACH_FAILED` carrying the backend's
///   error message in `data.context.backend_error`. No session mutation.
///
/// Per BEAD-0008 review (H-2), the response's `model_family` is taken from
/// `host.model_type` (what the worker loaded), not the client's claim.
/// Per BEAD-0008 review (M-4), worker metadata is sanity-checked; zero
/// values trigger `BACKEND_ATTACH_FAILED`.
pub fn handle_attach(
    session: &mut Session,
    request: &Request,
    backend_result: Result<&HostAttachResponse, &str>,
) -> Response {
    let req: AttachRequest = match parse_params(request) {
        Ok(r) => r,
        Err(e) => return invalid_params_response(request.id.clone(), &e),
    };

    let host = match backend_result {
        Ok(h) => h,
        Err(message) => {
            let data = error_with_context(
                ErrorCode::BackendAttachFailed,
                message.to_owned(),
                serde_json::json!({ "backend_error": message }),
            );
            return Response::error(request.id.clone(), RpcError::from_error_data(data));
        }
    };

    if host.num_layers == 0 || host.num_heads == 0 || host.hidden_dim == 0 {
        let message = format!(
            "worker returned invalid metadata: num_layers={}, num_heads={}, hidden_dim={}",
            host.num_layers, host.num_heads, host.hidden_dim
        );
        let data = error_with_context(
            ErrorCode::BackendAttachFailed,
            message.clone(),
            serde_json::json!({
                "backend_error": message,
                "num_layers": host.num_layers,
                "num_heads": host.num_heads,
                "hidden_dim": host.hidden_dim,
            }),
        );
        return Response::error(request.id.clone(), RpcError::from_error_data(data));
    }

    if let Err(ref e) = session.validate_attach(&req) {
        return session_error_to_response(request.id.clone(), e);
    }

    let envelope = session.commit_attach(
        &req,
        &host.model_type,
        host.num_layers,
        host.num_heads,
        host.hidden_dim,
    );
    serialize_envelope(request.id.clone(), envelope)
}

fn handle_detach(session: &mut Session, request: &Request) -> Response {
    match session.detach() {
        Ok(envelope) => serialize_envelope(request.id.clone(), envelope),
        Err(ref e) => session_error_to_response(request.id.clone(), e),
    }
}

fn handle_status(session: &Session, request: &Request) -> Response {
    match session.status() {
        Ok(envelope) => serialize_envelope(request.id.clone(), envelope),
        Err(ref e) => session_error_to_response(request.id.clone(), e),
    }
}

pub fn handle_step(
    session: &mut Session,
    request: &Request,
    host_response: Option<&rocket_surgeon_protocol::messages::HostStepResponse>,
) -> Response {
    let req: StepRequest = match parse_params(request) {
        Ok(r) => r,
        Err(e) => return invalid_params_response(request.id.clone(), &e),
    };

    let (position, forward_complete) = if let Some(hr) = host_response {
        (hr.position.clone(), hr.forward_complete)
    } else {
        let tick_id = session.state().tick_id.unwrap_or(0) + u64::from(req.count);
        let pos = TickPosition {
            tick_id,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer: 0,
            component: String::new(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        };
        (pos, false)
    };

    match session.step(&req, &position, forward_complete) {
        Ok(envelope) => serialize_envelope(request.id.clone(), envelope),
        Err(ref e) => session_error_to_response(request.id.clone(), e),
    }
}

fn handle_stub_requires_stopped(session: &Session, request: &Request) -> Response {
    if let Err(e) = session.require_stopped(&request.method) {
        return session_error_to_response(request.id.clone(), &e);
    }

    serialize_envelope(
        request.id.clone(),
        session.envelope(serde_json::json!({
            "stub": true,
            "method": request.method,
            "message": format!("{} is not yet implemented", request.method),
        })),
    )
}

fn handle_inspect_no_store(session: &Session, request: &Request) -> Response {
    if let Err(e) = session.require_stopped(&request.method) {
        return session_error_to_response(request.id.clone(), &e);
    }
    Response::error(
        request.id.clone(),
        RpcError::from_error_data(error_with_hint(
            ErrorCode::TensorNotFound,
            "No orchestrator available to capture tensors",
        )),
    )
}

fn read_from_shm(
    consumer: Option<&rocket_surgeon_shm::ring::DoomRingConsumer>,
    shm_offset: u64,
    byte_length: u64,
) -> Result<Vec<u8>, String> {
    let consumer = consumer.ok_or("shm consumer not available")?;
    let data_start = shm_offset as usize + rocket_surgeon_shm::PROBE_FRAME_HEADER_SIZE;
    consumer
        .read_absolute(data_start, byte_length as usize)
        .map_err(|e| format!("shm read at offset {data_start}: {e}"))
}

fn parse_dtype(s: &str) -> Option<DType> {
    match s {
        "float16" => Some(DType::Float16),
        "bfloat16" => Some(DType::Bfloat16),
        "float32" => Some(DType::Float32),
        "float64" => Some(DType::Float64),
        "int8" => Some(DType::Int8),
        "int16" => Some(DType::Int16),
        "int32" => Some(DType::Int32),
        "int64" => Some(DType::Int64),
        "uint8" => Some(DType::Uint8),
        "bool" => Some(DType::Bool),
        _ => None,
    }
}

pub fn handle_inspect(
    session: &Session,
    request: &Request,
    host_response: Option<&rocket_surgeon_protocol::messages::HostInspectResponse>,
    store: &mut TensorStore,
    shm_consumer: Option<&mut rocket_surgeon_shm::ring::DoomRingConsumer>,
) -> Response {
    let req: InspectRequest = match parse_params(request) {
        Ok(r) => r,
        Err(e) => return invalid_params_response(request.id.clone(), &e),
    };

    if let Err(ref e) = session.require_stopped("rocket/inspect") {
        return session_error_to_response(request.id.clone(), e);
    }

    // Validate the target's `component` field before consulting the host:
    // an unresolvable component is a caller mistake (INVALID_TARGET), not a
    // "tensor not captured yet" condition. The structured context lets the
    // caller self-correct without a round-trip.
    if let Some(component) = target_component(&req.target) {
        // A `*` wildcard component matches everything — leave resolution to
        // the host. Otherwise the leaf must be a known canonical component.
        if component != "*" && !CANONICAL_COMPONENTS.contains(&component_leaf(component)) {
            return Response::error(
                request.id.clone(),
                RpcError::from_error_data(invalid_target_error(&req.target)),
            );
        }
    }

    let captured = match host_response {
        Some(hr) => &hr.tensors,
        None => {
            return Response::error(
                request.id.clone(),
                RpcError::from_error_data(error_with_hint(
                    ErrorCode::TensorNotFound,
                    "No orchestrator available to capture tensors",
                )),
            );
        }
    };

    if captured.is_empty() {
        return Response::error(
            request.id.clone(),
            RpcError::from_error_data(error_with_hint(
                ErrorCode::TensorNotFound,
                "Target matched components but no tensors have been captured yet (step first)",
            )),
        );
    }

    ingest_and_respond(session, request, &req, captured, store, shm_consumer)
}

#[allow(clippy::too_many_lines)]
fn ingest_and_respond(
    session: &Session,
    request: &Request,
    req: &InspectRequest,
    captured: &[rocket_surgeon_protocol::messages::CapturedTensor],
    store: &mut TensorStore,
    shm_consumer: Option<&mut rocket_surgeon_shm::ring::DoomRingConsumer>,
) -> Response {
    let engine = base64::engine::general_purpose::STANDARD;
    let mut summaries = Vec::new();
    let mut first_tensor_id = None;
    let mut shm_slots_consumed: u64 = 0;

    for ct in captured {
        let Some(dtype) = parse_dtype(&ct.dtype) else {
            return Response::error(
                request.id.clone(),
                RpcError {
                    code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                    message: format!("unknown dtype from worker: {}", ct.dtype),
                    data: None,
                },
            );
        };

        let has_shm = ct.shm_offset.is_some() && ct.byte_length.is_some();

        if store.contains(&ct.tensor_id) {
            if first_tensor_id.is_none() {
                first_tensor_id = Some(ct.tensor_id.clone());
            }
            let summary = store.summarize(&ct.tensor_id).expect("tensor exists");
            summaries.push(summary);
            if has_shm {
                shm_slots_consumed += 1;
            }
            continue;
        }

        let handle = if let (Some(shm_offset), Some(byte_length)) = (ct.shm_offset, ct.byte_length)
        {
            match read_from_shm(shm_consumer.as_deref(), shm_offset, byte_length) {
                Ok(bytes) => {
                    shm_slots_consumed += 1;
                    store.insert_with_id(
                        ct.tensor_id.clone(),
                        bytes,
                        ct.shape.clone(),
                        dtype,
                        ct.device.clone(),
                    )
                }
                Err(e) => {
                    return Response::error(
                        request.id.clone(),
                        RpcError {
                            code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                            message: format!("shm read failed: {e}"),
                            data: None,
                        },
                    );
                }
            }
        } else if let Some(ref b64) = ct.data_base64 {
            match engine.decode(b64) {
                Ok(bytes) => store.insert(bytes, ct.shape.clone(), dtype, ct.device.clone()),
                Err(e) => {
                    return Response::error(
                        request.id.clone(),
                        RpcError {
                            code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                            message: format!("base64 decode failed: {e}"),
                            data: None,
                        },
                    );
                }
            }
        } else {
            return Response::error(
                request.id.clone(),
                RpcError {
                    code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                    message: "CapturedTensor has neither shm_offset nor data_base64".into(),
                    data: None,
                },
            );
        };

        if first_tensor_id.is_none() {
            first_tensor_id = Some(handle.tensor_id.clone());
        }

        let summary = store.summarize(&handle.tensor_id).expect("just inserted");
        summaries.push(summary);
    }

    if shm_slots_consumed > 0 {
        if let Some(consumer) = shm_consumer {
            if let Err(e) = consumer.advance_by(shm_slots_consumed) {
                tracing::warn!(
                    count = shm_slots_consumed,
                    "failed to advance shm consumer: {e}"
                );
            }
        }
    }

    let slice_data = if req.detail == rocket_surgeon_protocol::messages::InspectDetail::Slice {
        if let (Some(slices), Some(tid)) = (&req.slices, &first_tensor_id) {
            if let Some(&[offset, len]) = slices.first() {
                match store.slice(tid, offset, len) {
                    Ok(bytes) => Some(engine.encode(&bytes)),
                    Err(crate::tensor_store::StoreError::SliceOutOfBounds { .. }) => {
                        return Response::error(
                            request.id.clone(),
                            RpcError::from_error_data(error_with_hint(
                                ErrorCode::SliceOutOfBounds,
                                "Slice indices exceed tensor data size",
                            )),
                        );
                    }
                    Err(e) => {
                        return Response::error(
                            request.id.clone(),
                            RpcError {
                                code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                                message: format!("slice failed: {e}"),
                                data: None,
                            },
                        );
                    }
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    match session.inspect(&summaries, slice_data) {
        Ok(envelope) => serialize_envelope(request.id.clone(), envelope),
        Err(ref e) => session_error_to_response(request.id.clone(), e),
    }
}

fn probe_success(
    session: &Session,
    id: RequestId,
    registry: &ProbeRegistry,
    probe_id: Option<String>,
) -> Response {
    let resp = ProbeResponse {
        probes: registry.list(),
        probe_id,
    };
    serialize_envelope(id, session.envelope(resp))
}

fn registry_error_to_response(id: RequestId, err: RegistryError) -> Response {
    match err {
        RegistryError::DuplicateId { id: pid } => Response::error(
            id,
            RpcError::from_error_data(error_with_context(
                ErrorCode::DuplicateProbeId,
                format!("Probe with id '{pid}' already exists"),
                serde_json::json!({ "probe_id": pid }),
            )),
        ),
        RegistryError::InvalidPoint(e) => Response::error(
            id,
            RpcError::from_error_data(error_with_context(
                ErrorCode::InvalidPoint,
                format!("Invalid probe point: {e}"),
                serde_json::json!({ "parse_error": e.to_string() }),
            )),
        ),
        RegistryError::NotFound { id: pid } => Response::error(
            id,
            RpcError::from_error_data(error_with_context(
                ErrorCode::ProbeNotFound,
                format!("Probe '{pid}' not found"),
                serde_json::json!({ "probe_id": pid }),
            )),
        ),
    }
}

pub fn handle_probe(
    session: &Session,
    request: &Request,
    registry: &mut ProbeRegistry,
) -> Response {
    if let Err(ref e) = session.require_stopped("rocket/probe") {
        return session_error_to_response(request.id.clone(), e);
    }

    let req: ProbeRequest = match parse_params(request) {
        Ok(r) => r,
        Err(e) => return invalid_params_response(request.id.clone(), &e),
    };

    match req {
        ProbeRequest::Define { probe } => match registry.define(*probe) {
            Ok(id) => probe_success(session, request.id.clone(), registry, Some(id)),
            Err(e) => registry_error_to_response(request.id.clone(), e),
        },
        ProbeRequest::List {} | ProbeRequest::SetGranularity { .. } => {
            probe_success(session, request.id.clone(), registry, None)
        }
        ProbeRequest::Enable { probe_id } => match registry.enable(&probe_id) {
            Ok(_) => probe_success(session, request.id.clone(), registry, Some(probe_id)),
            Err(e) => registry_error_to_response(request.id.clone(), e),
        },
        ProbeRequest::Disable { probe_id } => match registry.disable(&probe_id) {
            Ok(_) => probe_success(session, request.id.clone(), registry, Some(probe_id)),
            Err(e) => registry_error_to_response(request.id.clone(), e),
        },
        ProbeRequest::Remove { probe_id } => match registry.remove(&probe_id) {
            Ok(_) => probe_success(session, request.id.clone(), registry, Some(probe_id)),
            Err(e) => registry_error_to_response(request.id.clone(), e),
        },
    }
}

pub fn handle_subscribe(session: &Session, request: &Request) -> Response {
    let _req: SubscribeRequest = match parse_params(request) {
        Ok(r) => r,
        Err(e) => {
            return Response::error(
                request.id.clone(),
                RpcError {
                    code: rocket_surgeon_protocol::jsonrpc::INVALID_PARAMS,
                    message: format!("Invalid params: {e}"),
                    data: None,
                },
            );
        }
    };

    if let Err(ref e) = session.require_stopped("rocket/subscribe") {
        return session_error_to_response(request.id.clone(), e);
    }

    let resp = SubscribeResponse {
        available_events: vec![
            EventType::TickStopped,
            EventType::TickHeartbeat,
            EventType::ProbeFired,
        ],
        status: session.state().status,
    };
    serialize_envelope(request.id.clone(), session.envelope(resp))
}

pub fn handle_unsubscribe(session: &Session, request: &Request) -> Response {
    let _req: UnsubscribeRequest = match parse_params(request) {
        Ok(r) => r,
        Err(e) => return invalid_params_response(request.id.clone(), &e),
    };

    let resp = UnsubscribeResponse {
        status: session.state().status,
    };
    serialize_envelope(request.id.clone(), session.envelope(resp))
}

pub fn handle_view(
    session: &Session,
    request: &Request,
    host_response: Option<&HostViewResponse>,
) -> Response {
    let _req: ViewRequest = match parse_params(request) {
        Ok(r) => r,
        Err(e) => return invalid_params_response(request.id.clone(), &e),
    };

    if let Err(ref e) = session.require_stopped("rocket/view") {
        return session_error_to_response(request.id.clone(), e);
    }

    let Some(hr) = host_response else {
        return Response::error(
            request.id.clone(),
            RpcError::from_error_data(error_with_hint(
                ErrorCode::ViewDataUnavailable,
                "No orchestrator available to compute view",
            )),
        );
    };

    let resp = ViewResponse {
        view: hr.view,
        data: hr.data.clone(),
    };
    serialize_envelope(request.id.clone(), session.envelope(resp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocket_surgeon_protocol::jsonrpc::{JSONRPC_VERSION, RequestId};
    use rocket_surgeon_protocol::types::Status;

    fn make_request(method: &str, params: serde_json::Value) -> Request {
        Request {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id: RequestId::Number(1),
            method: method.to_owned(),
            params: if params.is_null() { None } else { Some(params) },
        }
    }

    fn init_params() -> serde_json::Value {
        serde_json::json!({
            "client_name": "test",
            "protocol_version": "0.3.0"
        })
    }

    fn attach_params() -> serde_json::Value {
        serde_json::json!({
            "model_path": "/models/test",
            "model_family": "llama",
            "device": "cuda:0",
            "num_ranks": 1
        })
    }

    // BEAD-0008: attach now goes through `handle_attach` directly with a
    // backend response, not through `dispatch`. Distinctive numerics (not
    // the deleted llama stub 32/32/4096) so reviewers can tell at a glance
    // that the tests care about the *flow*, not specific magic numbers.
    fn fake_host_attach_response() -> HostAttachResponse {
        HostAttachResponse {
            model_handle: 1,
            num_layers: 7,
            num_heads: 3,
            hidden_dim: 256,
            module_tree: vec![],
            model_type: "llama".to_owned(),
            component_vocabulary: vec![],
            shm_name: None,
        }
    }

    fn test_attach_dispatch(session: &mut Session) -> Response {
        let req = make_request("attach", attach_params());
        let host = fake_host_attach_response();
        handle_attach(session, &req, Ok(&host))
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        let mut session = Session::new();
        let req = make_request("nonexistent/method", serde_json::Value::Null);
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, METHOD_NOT_FOUND);
        assert!(resp.result.is_none());
    }

    #[test]
    fn dispatch_initialize_succeeds() {
        let mut session = Session::new();
        let req = make_request("initialize", init_params());
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_none());
        assert!(resp.result.is_some());
        assert_eq!(session.state().status, Status::Initialized);
    }

    #[test]
    fn dispatch_wraps_response_in_envelope() {
        let mut session = Session::new();
        let req = make_request("initialize", init_params());
        let resp = dispatch(&mut session, &req);
        let result = resp.result.unwrap();
        assert!(result.get("state").is_some());
        assert!(result.get("data").is_some());

        let state = &result["state"];
        assert!(state.get("session_id").is_some());
        assert!(state.get("status").is_some());
        assert!(state.get("available_actions").is_some());
    }

    #[test]
    fn dispatch_attach_returns_internal_error_because_routing_lives_in_main() {
        // BEAD-0008: dispatch() refuses ATTACH because the daemon's main loop
        // routes it directly to handle_attach with the backend response.
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let resp = dispatch(&mut session, &make_request("attach", attach_params()));
        assert!(resp.error.is_some());
        assert_eq!(
            resp.error.as_ref().unwrap().code,
            rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR
        );
    }

    #[test]
    fn handle_attach_with_host_response_succeeds() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let resp = test_attach_dispatch(&mut session);
        assert!(resp.error.is_none());
        assert_eq!(session.state().status, Status::Stopped);

        // Real metadata flowed through from the fake HostAttachResponse —
        // not the stub-fabricated values that the old code path produced.
        let result = resp.result.as_ref().unwrap();
        assert_eq!(result["data"]["num_layers"], 7);
        assert_eq!(result["data"]["num_heads"], 3);
        assert_eq!(result["data"]["hidden_dim"], 256);
        // H-2: model_family in the response comes from the worker's
        // model_type, not the client's claimed family.
        assert_eq!(result["data"]["model_family"], "llama");
    }

    #[test]
    fn handle_attach_rejects_zero_metadata_as_backend_attach_failed() {
        // M-4: worker that returns garbage metadata (e.g. 0 layers) should
        // be treated the same as a failed attach, not let through to clients.
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let bad_host = HostAttachResponse {
            num_layers: 0,
            ..fake_host_attach_response()
        };
        let req = make_request("attach", attach_params());
        let resp = handle_attach(&mut session, &req, Ok(&bad_host));

        let err = resp.error.as_ref().expect("expected error response");
        assert_eq!(err.code, ErrorCode::BackendAttachFailed.numeric_code());
        assert_eq!(session.state().status, Status::Initialized);
    }

    #[test]
    fn handle_attach_uses_worker_model_type_over_client_claim() {
        // H-2: when the client claims "llama" but the worker reports
        // model_type "mixtral", the response reflects the worker.
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let host_mixtral = HostAttachResponse {
            model_type: "mixtral".to_owned(),
            ..fake_host_attach_response()
        };
        let req = make_request("attach", attach_params()); // claims "llama"
        let resp = handle_attach(&mut session, &req, Ok(&host_mixtral));

        assert!(resp.error.is_none());
        let result = resp.result.as_ref().unwrap();
        assert_eq!(result["data"]["model_family"], "mixtral");
    }

    #[test]
    fn handle_attach_returns_backend_attach_failed_when_backend_missing() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let req = make_request("attach", attach_params());
        let resp = handle_attach(&mut session, &req, Err("worker died loading torch"));

        let err = resp.error.as_ref().expect("expected error response");
        assert_eq!(err.code, ErrorCode::BackendAttachFailed.numeric_code());
        let data = err.data.as_ref().expect("error data present");
        assert_eq!(data.error_code, ErrorCode::BackendAttachFailed);
        assert_eq!(
            data.severity,
            rocket_surgeon_protocol::errors::Severity::Recoverable
        );
        let context = data.context.as_ref().expect("context present");
        assert!(context["backend_error"].as_str().unwrap().contains("torch"));
        // Session state was not mutated by the failed attach.
        assert_eq!(session.state().status, Status::Initialized);
        assert!(session.state().model_id.is_none());
    }

    #[test]
    fn dispatch_detach_succeeds() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let req = make_request("detach", serde_json::Value::Null);
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_none());
        assert_eq!(session.state().status, Status::Initialized);
    }

    #[test]
    fn dispatch_status_from_stopped() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let req = make_request("rocket/status", serde_json::Value::Null);
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_none());
    }

    #[test]
    fn dispatch_stub_method_from_stopped() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let req = make_request("rocket/intervene", serde_json::json!({}));
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert!(result.get("state").is_some());
        let data = &result["data"];
        assert_eq!(data["stub"], true);
        assert_eq!(data["method"], "rocket/intervene");
    }

    #[test]
    fn dispatch_stub_method_when_not_stopped_returns_error() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let req = make_request("rocket/intervene", serde_json::json!({}));
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_some());
    }

    #[test]
    fn dispatch_step_from_stopped_returns_step_response() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let req = make_request(
            "rocket/step",
            serde_json::json!({
                "direction": "forward",
                "count": 1,
                "granularity": "component"
            }),
        );
        let resp = dispatch(&mut session, &req);
        assert!(
            resp.error.is_none(),
            "Expected success, got: {:?}",
            resp.error
        );
        let result = resp.result.unwrap();
        let data = &result["data"];
        assert_eq!(data["ticks_executed"], 1);
        assert!(data["stopped_at"]["tick_id"].is_number());
        assert!(data["stopped_at"]["component"].is_string());
        assert!(data["stopped_at"]["layer"].is_number());
        let state = &result["state"];
        assert_eq!(state["status"], "stopped");
    }

    #[test]
    fn dispatch_step_backward_returns_capability_error() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let req = make_request(
            "rocket/step",
            serde_json::json!({
                "direction": "backward",
                "count": 1,
                "granularity": "component"
            }),
        );
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_some());
    }

    #[test]
    fn dispatch_step_invalid_params_returns_error() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let req = make_request("rocket/step", serde_json::json!({"wrong": true}));
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_some());
        assert_eq!(
            resp.error.as_ref().unwrap().code,
            rocket_surgeon_protocol::jsonrpc::INVALID_PARAMS,
        );
    }

    #[test]
    fn dispatch_step_when_not_stopped_returns_error() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let req = make_request(
            "rocket/step",
            serde_json::json!({
                "direction": "forward",
                "count": 1
            }),
        );
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_some());
    }

    #[test]
    fn dispatch_step_tick_id_increments() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let step_params = serde_json::json!({
            "direction": "forward",
            "count": 1,
            "granularity": "component"
        });

        let resp1 = dispatch(
            &mut session,
            &make_request("rocket/step", step_params.clone()),
        );
        let tick1 = resp1.result.as_ref().unwrap()["state"]["tick_id"]
            .as_u64()
            .unwrap();

        let resp2 = dispatch(&mut session, &make_request("rocket/step", step_params));
        let tick2 = resp2.result.as_ref().unwrap()["state"]["tick_id"]
            .as_u64()
            .unwrap();

        assert!(
            tick2 > tick1,
            "tick_id should monotonically increase: {tick1} -> {tick2}"
        );
    }

    #[test]
    fn dispatch_invalid_params_returns_error() {
        let mut session = Session::new();
        let req = make_request("initialize", serde_json::json!({"wrong_field": 42}));
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_some());
        assert_eq!(
            resp.error.as_ref().unwrap().code,
            rocket_surgeon_protocol::jsonrpc::INVALID_PARAMS,
        );
    }

    #[test]
    fn dispatch_preserves_request_id() {
        let mut session = Session::new();
        let mut req = make_request("initialize", init_params());
        req.id = RequestId::String("abc-123".to_owned());
        let resp = dispatch(&mut session, &req);
        assert_eq!(resp.id, RequestId::String("abc-123".to_owned()));
    }

    #[test]
    fn dispatch_error_preserves_request_id() {
        let mut session = Session::new();
        let mut req = make_request("nonexistent", serde_json::Value::Null);
        req.id = RequestId::Number(42);
        let resp = dispatch(&mut session, &req);
        assert_eq!(resp.id, RequestId::Number(42));
    }

    #[test]
    fn dispatch_jsonrpc_version() {
        let mut session = Session::new();
        let req = make_request("initialize", init_params());
        let resp = dispatch(&mut session, &req);
        assert_eq!(resp.jsonrpc, JSONRPC_VERSION);
    }

    use crate::tensor_store::TensorStore;
    use base64::Engine;
    use rocket_surgeon_protocol::errors::ErrorCode;
    use rocket_surgeon_protocol::messages::{CapturedTensor, HostInspectResponse};

    #[test]
    fn dispatch_inspect_from_stopped_with_host_response() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let mut store = TensorStore::new();
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let host_resp = HostInspectResponse {
            tensors: vec![CapturedTensor {
                module_path: "model.layers.0.self_attn.q_proj".to_owned(),
                canonical: "q_proj".to_owned(),
                layer: 0,
                shape: vec![4],
                dtype: "float32".to_owned(),
                device: "cpu".to_owned(),
                tensor_id: "a".repeat(64),
                shm_name: None,
                shm_offset: None,
                byte_length: None,
                data_base64: Some(base64::engine::general_purpose::STANDARD.encode(&data)),
            }],
        };

        let req = make_request(
            "rocket/inspect",
            serde_json::json!({
                "target": "model:0:0:q_proj:output",
                "detail": "summary"
            }),
        );
        let resp = handle_inspect(&session, &req, Some(&host_resp), &mut store, None);
        assert!(
            resp.error.is_none(),
            "Expected success, got: {:?}",
            resp.error
        );
        let result = resp.result.unwrap();
        let data = &result["data"];
        assert!(data["tensors"].is_array());
        assert_eq!(data["tensors"].as_array().unwrap().len(), 1);
        let tensor = &data["tensors"][0];
        assert_eq!(tensor["tensor_id"].as_str().unwrap().len(), 64);
        assert!(tensor["stats"]["mean"].is_number());
    }

    #[test]
    fn dispatch_inspect_when_not_stopped_returns_error() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let mut store = TensorStore::new();
        let req = make_request(
            "rocket/inspect",
            serde_json::json!({
                "target": "model:0:0:q_proj:output",
                "detail": "summary"
            }),
        );
        let resp = handle_inspect(&session, &req, None, &mut store, None);
        assert!(resp.error.is_some());
    }

    #[test]
    fn dispatch_inspect_with_slice() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let mut store = TensorStore::new();
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let host_resp = HostInspectResponse {
            tensors: vec![CapturedTensor {
                module_path: "model.layers.0.self_attn.q_proj".to_owned(),
                canonical: "q_proj".to_owned(),
                layer: 0,
                shape: vec![4],
                dtype: "float32".to_owned(),
                device: "cpu".to_owned(),
                tensor_id: "a".repeat(64),
                shm_name: None,
                shm_offset: None,
                byte_length: None,
                data_base64: Some(base64::engine::general_purpose::STANDARD.encode(&data)),
            }],
        };

        let req = make_request(
            "rocket/inspect",
            serde_json::json!({
                "target": "model:0:0:q_proj:output",
                "detail": "slice",
                "slices": [[0, 8]]
            }),
        );
        let resp = handle_inspect(&session, &req, Some(&host_resp), &mut store, None);
        assert!(
            resp.error.is_none(),
            "Expected success, got: {:?}",
            resp.error
        );
        let result = resp.result.unwrap();
        assert!(result["data"]["slice_data"].is_string());
    }

    #[test]
    fn dispatch_inspect_empty_host_response_returns_tensor_not_found() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let mut store = TensorStore::new();
        let host_resp = HostInspectResponse { tensors: vec![] };

        let req = make_request(
            "rocket/inspect",
            serde_json::json!({
                "target": "model:0:0:q_proj:output",
                "detail": "summary"
            }),
        );
        let resp = handle_inspect(&session, &req, Some(&host_resp), &mut store, None);
        assert!(resp.error.is_some());
        let err_data = resp.error.as_ref().unwrap().data.as_ref().unwrap();
        assert_eq!(err_data.error_code, ErrorCode::TensorNotFound);
    }

    #[test]
    fn dispatch_inspect_slice_out_of_bounds() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let mut store = TensorStore::new();
        let data = vec![0u8; 16]; // 4 x f32

        let host_resp = HostInspectResponse {
            tensors: vec![CapturedTensor {
                module_path: "model.layers.0.self_attn.q_proj".to_owned(),
                canonical: "q_proj".to_owned(),
                layer: 0,
                shape: vec![4],
                dtype: "float32".to_owned(),
                device: "cpu".to_owned(),
                tensor_id: "a".repeat(64),
                shm_name: None,
                shm_offset: None,
                byte_length: None,
                data_base64: Some(base64::engine::general_purpose::STANDARD.encode(&data)),
            }],
        };

        let req = make_request(
            "rocket/inspect",
            serde_json::json!({
                "target": "model:0:0:q_proj:output",
                "detail": "slice",
                "slices": [[0, 999_999_999]]
            }),
        );
        let resp = handle_inspect(&session, &req, Some(&host_resp), &mut store, None);
        assert!(resp.error.is_some());
        let err_data = resp.error.as_ref().unwrap().data.as_ref().unwrap();
        assert_eq!(err_data.error_code, ErrorCode::SliceOutOfBounds);
    }

    #[test]
    fn dispatch_inspect_no_host_response_without_orchestrator() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let mut store = TensorStore::new();
        let req = make_request(
            "rocket/inspect",
            serde_json::json!({
                "target": "model:0:0:q_proj:output",
                "detail": "summary"
            }),
        );
        let resp = handle_inspect(&session, &req, None, &mut store, None);
        assert!(resp.error.is_some());
        let err_data = resp.error.as_ref().unwrap().data.as_ref().unwrap();
        assert_eq!(err_data.error_code, ErrorCode::TensorNotFound);
    }

    // --- Probe tests ---

    #[test]
    fn dispatch_probe_define_returns_probe_id() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let mut registry = rocket_surgeon_probes::registry::ProbeRegistry::new();
        let req = make_request(
            "rocket/probe",
            serde_json::json!({
                "action": "define",
                "probe": {
                    "id": "p-test-1",
                    "point": "llama:0:12:attn.o_proj:0:output",
                    "action": "capture",
                    "config": {"summary": true},
                    "enabled": true,
                    "priority": 0
                }
            }),
        );
        let resp = handle_probe(&session, &req, &mut registry);
        assert!(
            resp.error.is_none(),
            "Expected success, got: {:?}",
            resp.error
        );
        let result = resp.result.unwrap();
        let data = &result["data"];
        assert_eq!(data["probe_id"], "p-test-1");
        assert!(data["probes"].is_array());
        assert_eq!(data["probes"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn dispatch_probe_list_returns_all() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let mut registry = rocket_surgeon_probes::registry::ProbeRegistry::new();
        handle_probe(
            &session,
            &make_request(
                "rocket/probe",
                serde_json::json!({
                    "action": "define",
                    "probe": {
                        "id": "p1",
                        "point": "llama:0:12:attn.o_proj:0:output",
                        "action": "capture"
                    }
                }),
            ),
            &mut registry,
        );
        handle_probe(
            &session,
            &make_request(
                "rocket/probe",
                serde_json::json!({
                    "action": "define",
                    "probe": {
                        "id": "p2",
                        "point": "llama:0:8:mlp:0:output",
                        "action": "trace"
                    }
                }),
            ),
            &mut registry,
        );

        let resp = handle_probe(
            &session,
            &make_request("rocket/probe", serde_json::json!({"action": "list"})),
            &mut registry,
        );
        assert!(resp.error.is_none());
        let data = &resp.result.unwrap()["data"];
        assert_eq!(data["probes"].as_array().unwrap().len(), 2);
        assert!(data["probe_id"].is_null());
    }

    #[test]
    fn handle_subscribe_from_stopped_returns_available_events() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let req = make_request("rocket/subscribe", serde_json::json!({}));
        let resp = handle_subscribe(&session, &req);
        assert!(
            resp.error.is_none(),
            "Expected success, got: {:?}",
            resp.error
        );
        let result = resp.result.unwrap();
        let data = &result["data"];
        let events = data["available_events"].as_array().unwrap();
        assert_eq!(events.len(), 3);
        assert!(events.iter().any(|e| e == "tick.stopped"));
        assert!(events.iter().any(|e| e == "tick.heartbeat"));
        assert!(events.iter().any(|e| e == "probe.fired"));
        assert_eq!(data["status"], "stopped");
    }

    #[test]
    fn handle_subscribe_when_not_stopped_returns_error() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let req = make_request("rocket/subscribe", serde_json::json!({}));
        let resp = handle_subscribe(&session, &req);
        assert!(resp.error.is_some());
        let err_data = resp.error.as_ref().unwrap().data.as_ref().unwrap();
        assert_eq!(err_data.error_code, ErrorCode::ModelNotAttached);
    }

    #[test]
    fn handle_unsubscribe_from_stopped_returns_status() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let req = make_request("rocket/unsubscribe", serde_json::json!({}));
        let resp = handle_unsubscribe(&session, &req);
        assert!(
            resp.error.is_none(),
            "Expected success, got: {:?}",
            resp.error
        );
        let result = resp.result.unwrap();
        assert_eq!(result["data"]["status"], "stopped");
    }

    #[test]
    fn handle_unsubscribe_from_initialized_returns_status() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let req = make_request("rocket/unsubscribe", serde_json::json!({}));
        let resp = handle_unsubscribe(&session, &req);
        assert!(resp.error.is_none());
        assert_eq!(resp.result.unwrap()["data"]["status"], "initialized");
    }

    #[test]
    fn dispatch_probe_enable_nonexistent_returns_error() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let mut registry = rocket_surgeon_probes::registry::ProbeRegistry::new();
        let resp = handle_probe(
            &session,
            &make_request(
                "rocket/probe",
                serde_json::json!({
                    "action": "enable",
                    "probe_id": "nonexistent"
                }),
            ),
            &mut registry,
        );
        assert!(resp.error.is_some());
        let err = resp.error.as_ref().unwrap();
        let data = err.data.as_ref().unwrap();
        assert_eq!(data.error_code, ErrorCode::ProbeNotFound);
    }

    // --- View tests ---

    use rocket_surgeon_protocol::messages::HostViewResponse;
    use rocket_surgeon_protocol::types::BuiltInView;

    #[test]
    fn handle_view_from_stopped_returns_view_response() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let req = make_request(
            "rocket/view",
            serde_json::json!({"view": "residual_stream_norm"}),
        );
        let resp = handle_view(&session, &req, None);
        assert!(resp.error.is_some());
    }

    #[test]
    fn handle_view_when_not_stopped_returns_error() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let req = make_request(
            "rocket/view",
            serde_json::json!({"view": "residual_stream_norm"}),
        );
        let resp = handle_view(&session, &req, None);
        assert!(resp.error.is_some());
        let err_data = resp.error.as_ref().unwrap().data.as_ref().unwrap();
        assert_eq!(err_data.error_code, ErrorCode::ModelNotAttached);
    }

    #[test]
    fn handle_view_with_invalid_params_returns_error() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let req = make_request("rocket/view", serde_json::json!("not an object"));
        let resp = handle_view(&session, &req, None);
        assert!(resp.error.is_some());
    }

    #[test]
    fn handle_view_with_host_response_returns_success() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let host_resp = HostViewResponse {
            view: BuiltInView::ResidualStreamNorm,
            data: serde_json::json!({"norms": [0.5, 0.6], "num_layers": 2, "norm_type": "l2"}),
        };

        let req = make_request(
            "rocket/view",
            serde_json::json!({"view": "residual_stream_norm"}),
        );
        let resp = handle_view(&session, &req, Some(&host_resp));
        assert!(
            resp.error.is_none(),
            "Expected success, got: {:?}",
            resp.error
        );
        let data = &resp.result.unwrap()["data"];
        assert_eq!(data["view"], "residual_stream_norm");
        assert!(data["data"]["norms"].is_array());
    }

    // --- Error expressiveness (TCK protocol/error-expressiveness.feature) ---

    /// Every `ErrorCode` maps to a non-empty, actionable recovery hint.
    #[test]
    fn every_error_code_has_a_recovery_hint() {
        use rocket_surgeon_protocol::errors::ErrorCode::{
            BackendAttachFailed, BranchMergeRefused, BranchNotFound, CapabilityNotSupported,
            CheckpointNotFound, CompiledModel, CrossRequestKv, DuplicateProbeId, GpuOom, HostError,
            InvalidParams, InvalidPoint, InvalidRecipe, InvalidState, InvalidTarget, KvEvicted,
            ModelAlreadyAttached, ModelNotAttached, NcclTimeout, ProbeNotFound, ReplayDivergence,
            ResponseTooLarge, SliceOutOfBounds, TensorNotFound, UnsupportedModel,
            ViewDataUnavailable, VramExhausted,
        };
        for code in [
            InvalidState,
            InvalidTarget,
            InvalidRecipe,
            ModelNotAttached,
            TensorNotFound,
            CheckpointNotFound,
            ProbeNotFound,
            CapabilityNotSupported,
            SliceOutOfBounds,
            ResponseTooLarge,
            HostError,
            GpuOom,
            NcclTimeout,
            ReplayDivergence,
            UnsupportedModel,
            CompiledModel,
            ModelAlreadyAttached,
            InvalidParams,
            DuplicateProbeId,
            InvalidPoint,
            ViewDataUnavailable,
            BackendAttachFailed,
            BranchNotFound,
            BranchMergeRefused,
            VramExhausted,
            CrossRequestKv,
            KvEvicted,
        ] {
            let hint = recovery_hint_for(code);
            assert!(!hint.is_empty(), "{code:?} has an empty recovery hint");
        }
    }

    /// Scenario: `ErrorData` includes `recovery_hint` — a representative
    /// daemon error response carries a non-null `recovery_hint` string.
    #[test]
    fn error_response_carries_recovery_hint() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        // rocket/view before attach -> MODEL_NOT_ATTACHED with a hint.
        let resp = handle_view(
            &session,
            &make_request(
                "rocket/view",
                serde_json::json!({"view": "residual_stream_norm"}),
            ),
            None,
        );
        let data = resp.error.as_ref().unwrap().data.as_ref().unwrap();
        assert!(
            data.recovery_hint.as_deref().is_some_and(|h| !h.is_empty()),
            "recovery_hint should be a non-empty string"
        );
    }

    /// `INVALID_PARAMS` responses are routed through `ErrorData` so they
    /// too carry a `recovery_hint` and the raw serde diagnostic in context.
    #[test]
    fn invalid_params_response_carries_recovery_hint_and_context() {
        let mut session = Session::new();
        let resp = dispatch(
            &mut session,
            &make_request("initialize", serde_json::json!({"wrong_field": 42})),
        );
        let err = resp.error.as_ref().unwrap();
        assert_eq!(err.code, rocket_surgeon_protocol::jsonrpc::INVALID_PARAMS);
        let data = err.data.as_ref().expect("INVALID_PARAMS carries ErrorData");
        assert!(data.recovery_hint.as_deref().is_some_and(|h| !h.is_empty()));
        assert!(data.context.as_ref().unwrap()["parse_error"].is_string());
    }

    /// Scenario: `INVALID_TARGET` includes nearest matches — inspecting a
    /// target whose component is misspelled yields `INVALID_TARGET` with the
    /// attempted name, non-empty `nearest_matches`, and valid components.
    #[test]
    fn invalid_target_includes_nearest_matches() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let mut store = TensorStore::new();
        let req = make_request(
            "rocket/inspect",
            serde_json::json!({
                "target": "llama:*:12:attn.out_proj:output",
                "detail": "summary"
            }),
        );
        let resp = handle_inspect(&session, &req, None, &mut store, None);
        let data = resp.error.as_ref().unwrap().data.as_ref().unwrap();
        assert_eq!(data.error_code, ErrorCode::InvalidTarget);
        assert!(data.recovery_hint.as_deref().is_some_and(|h| !h.is_empty()));

        let ctx = data
            .context
            .as_ref()
            .expect("INVALID_TARGET carries context");
        assert_eq!(ctx["attempted"], "attn.out_proj");
        let nearest = ctx["nearest_matches"]
            .as_array()
            .expect("nearest_matches array");
        assert!(!nearest.is_empty(), "nearest_matches must be non-empty");
        // `o_proj` is one edit away from `out_proj`.
        assert!(nearest.iter().any(|m| m == "o_proj"));
        assert!(ctx["valid_components_at_layer"].is_array());
    }

    /// A well-formed target with a known component is not rejected as
    /// `INVALID_TARGET` — it falls through to the normal capture path.
    #[test]
    fn valid_target_component_is_not_rejected() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        test_attach_dispatch(&mut session);

        let mut store = TensorStore::new();
        let req = make_request(
            "rocket/inspect",
            serde_json::json!({
                "target": "model:0:0:q_proj:output",
                "detail": "summary"
            }),
        );
        let resp = handle_inspect(&session, &req, None, &mut store, None);
        let data = resp.error.as_ref().unwrap().data.as_ref().unwrap();
        // Falls through to the host path -> TENSOR_NOT_FOUND, not INVALID_TARGET.
        assert_eq!(data.error_code, ErrorCode::TensorNotFound);
    }

    #[test]
    fn nearest_components_is_always_non_empty() {
        assert!(!nearest_components("totally_bogus_xyz").is_empty());
        assert!(!nearest_components("").is_empty());
    }
}
