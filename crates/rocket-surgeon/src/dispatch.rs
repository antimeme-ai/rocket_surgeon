use rocket_surgeon_protocol::jsonrpc::{METHOD_NOT_FOUND, Request, RequestId, Response, RpcError};
use rocket_surgeon_protocol::messages::{AttachRequest, InitializeRequest, StepRequest, method};
use rocket_surgeon_protocol::types::{StepDirection, TickEvent, TickPosition};

use crate::session::{Session, SessionError};

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
        method::STEP => handle_step(session, request),
        method::INSPECT
        | method::INTERVENE
        | method::PROBE
        | method::CHECKPOINT
        | method::REPLAY
        | method::SUBSCRIBE => handle_stub_requires_stopped(session, request),
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

fn handle_step(session: &mut Session, request: &Request) -> Response {
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

    let tick_id = session.state().tick_id.unwrap_or(0) + u64::from(req.count);
    let synthetic_position = TickPosition {
        tick_id,
        direction: StepDirection::Forward,
        rank: Some(0),
        layer: 0,
        component: String::new(),
        event: TickEvent::Output,
        replay_of: None,
    };

    match session.step(&req, &synthetic_position, false) {
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

        let req = make_request("rocket/inspect", serde_json::json!({}));
        let resp = dispatch(&mut session, &req);
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert!(result.get("state").is_some());
        let data = &result["data"];
        assert_eq!(data["stub"], true);
        assert_eq!(data["method"], "rocket/inspect");
    }

    #[test]
    fn dispatch_stub_method_when_not_stopped_returns_error() {
        let mut session = Session::new();
        dispatch(&mut session, &make_request("initialize", init_params()));

        let req = make_request("rocket/inspect", serde_json::json!({}));
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
}
