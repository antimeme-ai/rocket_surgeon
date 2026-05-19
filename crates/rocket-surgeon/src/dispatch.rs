use base64::Engine;
use rocket_surgeon_probes::registry::{ProbeRegistry, RegistryError};
use rocket_surgeon_protocol::errors::{ErrorCode, ErrorData};
use rocket_surgeon_protocol::jsonrpc::{METHOD_NOT_FOUND, Request, RequestId, Response, RpcError};
use rocket_surgeon_protocol::messages::{
    AttachRequest, EventType, HostViewResponse, InitializeRequest, InspectRequest, ProbeRequest,
    ProbeResponse, StepRequest, SubscribeRequest, SubscribeResponse, UnsubscribeRequest,
    UnsubscribeResponse, ViewRequest, ViewResponse, method,
};
use rocket_surgeon_protocol::types::{DType, StepDirection, TickEvent, TickPosition};

use crate::session::{Session, SessionError};
use crate::tensor_store::TensorStore;

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
        method::ATTACH => handle_attach(session, request),
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

    match session.initialize(&req) {
        Ok(envelope) => serialize_envelope(request.id.clone(), envelope),
        Err(ref e) => session_error_to_response(request.id.clone(), e),
    }
}

fn handle_attach(session: &mut Session, request: &Request) -> Response {
    let req: AttachRequest = match parse_params(request) {
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

    match session.attach(&req) {
        Ok(envelope) => serialize_envelope(request.id.clone(), envelope),
        Err(ref e) => session_error_to_response(request.id.clone(), e),
    }
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
        RpcError::from_error_data(ErrorData::new(
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

    if let Err(ref e) = session.require_stopped("rocket/inspect") {
        return session_error_to_response(request.id.clone(), e);
    }

    let captured = match host_response {
        Some(hr) => &hr.tensors,
        None => {
            return Response::error(
                request.id.clone(),
                RpcError::from_error_data(ErrorData::new(
                    ErrorCode::TensorNotFound,
                    "No orchestrator available to capture tensors",
                )),
            );
        }
    };

    if captured.is_empty() {
        return Response::error(
            request.id.clone(),
            RpcError::from_error_data(ErrorData::new(
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
                            RpcError::from_error_data(ErrorData::new(
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
            RpcError::from_error_data(ErrorData::new(
                ErrorCode::DuplicateProbeId,
                format!("Probe with id '{pid}' already exists"),
            )),
        ),
        RegistryError::InvalidPoint(e) => Response::error(
            id,
            RpcError::from_error_data(ErrorData::new(
                ErrorCode::InvalidPoint,
                format!("Invalid probe point: {e}"),
            )),
        ),
        RegistryError::NotFound { id: pid } => Response::error(
            id,
            RpcError::from_error_data(ErrorData::new(
                ErrorCode::ProbeNotFound,
                format!("Probe '{pid}' not found"),
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

    if let Err(ref e) = session.require_stopped("rocket/view") {
        return session_error_to_response(request.id.clone(), e);
    }

    let Some(hr) = host_response else {
        return Response::error(
            request.id.clone(),
            RpcError::from_error_data(ErrorData::new(
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
            "protocol_version": "0.1.0"
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
    fn dispatch_attach_succeeds() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let req = make_request("attach", attach_params());
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_none());
        assert_eq!(session.state().status, Status::Stopped);
    }

    #[test]
    fn dispatch_detach_succeeds() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        dispatch(&mut session, &make_request("attach", attach_params()));

        let req = make_request("detach", serde_json::Value::Null);
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_none());
        assert_eq!(session.state().status, Status::Initialized);
    }

    #[test]
    fn dispatch_status_from_stopped() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        dispatch(&mut session, &make_request("attach", attach_params()));

        let req = make_request("rocket/status", serde_json::Value::Null);
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_none());
    }

    #[test]
    fn dispatch_stub_method_from_stopped() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

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
        dispatch(&mut session, &make_request("attach", attach_params()));

        let req = make_request("rocket/view", serde_json::json!("not an object"));
        let resp = handle_view(&session, &req, None);
        assert!(resp.error.is_some());
    }

    #[test]
    fn handle_view_with_host_response_returns_success() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));
        dispatch(&mut session, &make_request("attach", attach_params()));

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
}
