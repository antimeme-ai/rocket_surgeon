use rocket_surgeon_protocol::jsonrpc::{
    INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND, Request, RequestId, Response, RpcError,
};
use rocket_surgeon_protocol::messages::internal;
use rocket_surgeon_protocol::messages::{HostAttachRequest, HostAttachResponse};
use rocket_surgeon_protocol::messages::{
    HostConfigureHooksRequest, HostConfigureHooksResponse, HostDetachRequest, HostDetachResponse,
    HostStepRequest, HostStepResponse, HostUpdateProbesRequest, HostUpdateProbesResponse,
};
use tracing::error;

use crate::bridge;

pub fn dispatch(request: &Request) -> Response {
    match request.method.as_str() {
        internal::HOST_ATTACH => handle_host_attach(request),
        internal::HOST_DETACH => handle_host_detach(request),
        internal::HOST_CONFIGURE_HOOKS => handle_host_configure_hooks(request),
        internal::HOST_STEP => handle_host_step(request),
        internal::HOST_UPDATE_PROBES => handle_host_update_probes(request),
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

fn parse_params<T: serde::de::DeserializeOwned>(request: &Request) -> Result<T, Box<Response>> {
    let params = request
        .params
        .clone()
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
    serde_json::from_value(params).map_err(|e| {
        Box::new(Response::error(
            request.id.clone(),
            RpcError {
                code: INVALID_PARAMS,
                message: format!("Invalid params: {e}"),
                data: None,
            },
        ))
    })
}

fn internal_error(id: RequestId, message: String) -> Response {
    error!("{message}");
    Response::error(
        id,
        RpcError {
            code: INTERNAL_ERROR,
            message,
            data: None,
        },
    )
}

fn handle_host_attach(request: &Request) -> Response {
    let req: HostAttachRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let dtype_str = match req.dtype {
        Some(rocket_surgeon_protocol::types::DType::Float16) => "float16",
        Some(rocket_surgeon_protocol::types::DType::Bfloat16) => "bfloat16",
        Some(rocket_surgeon_protocol::types::DType::Float32) | None => "float32",
        Some(other) => {
            return Response::error(
                request.id.clone(),
                RpcError {
                    code: INVALID_PARAMS,
                    message: format!("Unsupported dtype: {other:?}"),
                    data: None,
                },
            );
        }
    };

    let handle = match bridge::load_model(&req.model_source, &req.device, dtype_str) {
        Ok(h) => h,
        Err(e) => return internal_error(request.id.clone(), format!("load_model failed: {e}")),
    };

    let info = match bridge::model_metadata(handle) {
        Ok(i) => i,
        Err(e) => {
            return internal_error(request.id.clone(), format!("model_metadata failed: {e}"));
        }
    };

    let config = match bridge::get_model_config(handle) {
        Ok(c) => c,
        Err(e) => {
            return internal_error(request.id.clone(), format!("model_config failed: {e}"));
        }
    };

    let modules = match bridge::discover_modules(handle) {
        Ok(m) => m,
        Err(e) => {
            return internal_error(request.id.clone(), format!("discover_modules failed: {e}"));
        }
    };

    let mut component_map = match crate::adapter::resolve(&modules, &config, req.rank) {
        Ok(m) => m,
        Err(e) => {
            return internal_error(
                request.id.clone(),
                format!("adapter resolution failed: {e}"),
            );
        }
    };

    let execution_order = match bridge::discover_execution_order(handle) {
        Ok(o) => o,
        Err(e) => {
            return internal_error(
                request.id.clone(),
                format!("discover_execution_order failed: {e}"),
            );
        }
    };
    crate::adapter::apply_execution_order(&mut component_map, &execution_order);

    let resp = HostAttachResponse {
        model_handle: info.handle,
        num_layers: info.num_layers,
        num_heads: info.num_heads,
        hidden_dim: info.hidden_dim,
        module_tree: info.module_tree,
        model_type: config.model_type,
        component_vocabulary: component_map.vocabulary,
    };

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}

fn handle_host_detach(request: &Request) -> Response {
    let req: HostDetachRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    match bridge::unload_model(req.model_handle) {
        Ok(()) => {}
        Err(e) => {
            return internal_error(request.id.clone(), format!("unload_model failed: {e}"));
        }
    }

    let resp = HostDetachResponse { released: true };

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}

fn handle_host_configure_hooks(request: &Request) -> Response {
    let _req: HostConfigureHooksRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let resp = HostConfigureHooksResponse {
        sentinel_count: 0,
        capture_count: 0,
    };

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}

fn handle_host_step(request: &Request) -> Response {
    let _req: HostStepRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let resp = HostStepResponse {
        position: rocket_surgeon_protocol::types::TickPosition {
            tick_id: 0,
            direction: rocket_surgeon_protocol::types::StepDirection::Forward,
            rank: Some(0),
            layer: 0,
            component: String::new(),
            event: rocket_surgeon_protocol::types::TickEvent::Output,
            replay_of: None,
        },
        capture: None,
        forward_complete: false,
    };

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}

fn handle_host_update_probes(request: &Request) -> Response {
    let req: HostUpdateProbesRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let resp = HostUpdateProbesResponse {
        probes_active: req.active_probes.len() as u32,
    };

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocket_surgeon_protocol::jsonrpc::{JSONRPC_VERSION, RequestId};

    fn make_request(method: &str, params: serde_json::Value) -> Request {
        Request {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id: RequestId::Number(1),
            method: method.to_owned(),
            params: if params.is_null() { None } else { Some(params) },
        }
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        let req = make_request("nonexistent/method", serde_json::Value::Null);
        let resp = dispatch(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, METHOD_NOT_FOUND);
    }

    #[test]
    fn host_attach_invalid_params_returns_error() {
        let req = make_request(
            internal::HOST_ATTACH,
            serde_json::json!({"wrong_field": 42}),
        );
        let resp = dispatch(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }

    #[test]
    fn host_detach_invalid_params_returns_error() {
        let req = make_request(
            internal::HOST_DETACH,
            serde_json::json!({"wrong_field": 42}),
        );
        let resp = dispatch(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }

    #[test]
    fn host_attach_unsupported_dtype_returns_error() {
        let req = make_request(
            internal::HOST_ATTACH,
            serde_json::json!({
                "model_source": "test",
                "model_family": "llama",
                "device": "cpu",
                "dtype": "int8",
                "rank": 0
            }),
        );
        let resp = dispatch(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
        assert!(
            resp.error
                .as_ref()
                .unwrap()
                .message
                .contains("Unsupported dtype")
        );
    }

    #[test]
    fn dispatch_preserves_request_id() {
        let mut req = make_request("nonexistent", serde_json::Value::Null);
        req.id = RequestId::String("test-id-42".to_owned());
        let resp = dispatch(&req);
        assert_eq!(resp.id, RequestId::String("test-id-42".to_owned()));
    }

    #[test]
    fn dispatch_jsonrpc_version() {
        let req = make_request("nonexistent", serde_json::Value::Null);
        let resp = dispatch(&req);
        assert_eq!(resp.jsonrpc, JSONRPC_VERSION);
    }

    #[test]
    fn dispatch_configure_hooks_invalid_params() {
        let req = make_request(
            internal::HOST_CONFIGURE_HOOKS,
            serde_json::json!({"wrong_field": 42}),
        );
        let resp = dispatch(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }

    #[test]
    fn dispatch_step_invalid_params() {
        let req = make_request(internal::HOST_STEP, serde_json::json!({"wrong_field": 42}));
        let resp = dispatch(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }

    #[test]
    fn dispatch_update_probes_invalid_params() {
        let req = make_request(
            internal::HOST_UPDATE_PROBES,
            serde_json::json!({"wrong_field": 42}),
        );
        let resp = dispatch(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }
}
