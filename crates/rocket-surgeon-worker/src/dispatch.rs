use std::collections::HashMap;

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rocket_surgeon_protocol::jsonrpc::{
    INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND, Request, RequestId, Response, RpcError,
};
use rocket_surgeon_protocol::messages::internal;
use rocket_surgeon_protocol::messages::{
    CapturedTensor, HostConfigureHooksRequest, HostConfigureHooksResponse, HostDetachRequest,
    HostDetachResponse, HostInspectRequest, HostInspectResponse, HostStepRequest, HostStepResponse,
    HostUpdateProbesRequest, HostUpdateProbesResponse,
};
use rocket_surgeon_protocol::messages::{HostAttachRequest, HostAttachResponse};
use tracing::error;

use crate::adapter::ComponentMap;
use crate::bridge;
use crate::step_driver;
use crate::tick::TickState;

pub struct ForwardPassState {
    pub result_mailbox: pyo3::PyObject,
    pub resume_mailbox: pyo3::PyObject,
    pub sentinel_handles: Vec<pyo3::PyObject>,
    pub capture_handles: Vec<pyo3::PyObject>,
    #[allow(dead_code)]
    pub call_counts: pyo3::PyObject,
    pub forward_complete: bool,
}

pub struct WorkerState {
    pub component_map: Option<ComponentMap>,
    pub component_index: HashMap<(String, u32), usize>,
    pub module_paths: Vec<String>,
    pub model_handle: Option<u64>,
    pub rank: u32,
    pub tick_state: TickState,
    pub forward_pass: Option<ForwardPassState>,
    pub last_outputs: Option<pyo3::PyObject>,
}

impl WorkerState {
    pub fn new() -> Self {
        Self {
            component_map: None,
            component_index: HashMap::new(),
            module_paths: Vec::new(),
            model_handle: None,
            rank: 0,
            tick_state: TickState::new(0),
            forward_pass: None,
            last_outputs: None,
        }
    }

    fn build_component_index(map: &ComponentMap) -> HashMap<(String, u32), usize> {
        map.components
            .iter()
            .enumerate()
            .map(|(i, c)| ((c.module_path.clone(), c.call_index), i))
            .collect()
    }
}

pub fn dispatch(state: &mut WorkerState, request: &Request) -> Response {
    match request.method.as_str() {
        internal::HOST_ATTACH => handle_host_attach(state, request),
        internal::HOST_DETACH => handle_host_detach(state, request),
        internal::HOST_CONFIGURE_HOOKS => handle_host_configure_hooks(request),
        internal::HOST_STEP => handle_host_step(state, request),
        internal::HOST_UPDATE_PROBES => handle_host_update_probes(request),
        internal::HOST_INSPECT => handle_host_inspect(state, request),
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

fn handle_host_attach(state: &mut WorkerState, request: &Request) -> Response {
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

    state.component_index = WorkerState::build_component_index(&component_map);
    state.module_paths = component_map
        .components
        .iter()
        .map(|c| c.module_path.clone())
        .collect();
    state.model_handle = Some(info.handle);
    state.component_map = Some(component_map.clone());
    state.rank = req.rank;
    state.tick_state = TickState::new(req.rank);
    state.forward_pass = None;

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

fn handle_host_detach(state: &mut WorkerState, request: &Request) -> Response {
    let req: HostDetachRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    if let Some(fwd) = state.forward_pass.take() {
        Python::with_gil(|py| {
            if let Err(e) = fwd
                .resume_mailbox
                .bind(py)
                .call_method1("put", (py.None(),))
            {
                tracing::warn!("failed to signal resume mailbox during detach: {e}");
            }
            if let Err(e) = bridge::remove_hooks(py, &fwd.sentinel_handles) {
                tracing::warn!("failed to remove sentinel hooks during detach: {e}");
            }
            if let Err(e) = bridge::remove_hooks(py, &fwd.capture_handles) {
                tracing::warn!("failed to remove capture hooks during detach: {e}");
            }
        });
    }

    state.model_handle = None;
    state.component_map = None;
    state.component_index.clear();
    state.module_paths.clear();
    state.last_outputs = None;

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

const FORWARD_COMPLETE_SENTINEL: &str = "__forward_complete__";

fn ensure_forward_pass(py: Python<'_>, state: &mut WorkerState, handle: u64) -> anyhow::Result<()> {
    if state.forward_pass.is_some() {
        return Ok(());
    }

    state.last_outputs = Some(PyDict::new(py).unbind().into());

    let result_mb = bridge::create_mailbox(py)?;
    let resume_mb = bridge::create_mailbox(py)?;

    let sentinel_handles = bridge::install_sentinel_hooks(handle, &state.module_paths)?;

    let (capture_handles, call_counts) = bridge::install_capture_hooks(
        py,
        handle,
        &state.module_paths,
        result_mb.bind(py),
        resume_mb.bind(py),
        &state.module_paths,
    )?;

    let completion_mb = result_mb.clone_ref(py);
    let done_callback = pyo3::types::PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &pyo3::Bound<'_, PyTuple>,
              _kwargs: Option<&pyo3::Bound<'_, PyDict>>|
              -> pyo3::PyResult<()> {
            let py = args.py();
            let err_arg = args.get_item(0)?;
            if !err_arg.is_none() {
                tracing::error!("forward pass failed: {err_arg}");
            }
            let s: pyo3::Bound<'_, pyo3::types::PyAny> =
                FORWARD_COMPLETE_SENTINEL.into_pyobject(py)?.into_any();
            let z: pyo3::Bound<'_, pyo3::types::PyAny> = 0u32.into_pyobject(py)?.into_any();
            let sentinel = PyTuple::new(py, [s, z])?;
            completion_mb.bind(py).call_method1("put", (sentinel,))?;
            Ok(())
        },
    )?;

    bridge::run_forward(py, handle, done_callback.as_any())?;

    state.forward_pass = Some(ForwardPassState {
        result_mailbox: result_mb,
        resume_mailbox: resume_mb,
        sentinel_handles,
        capture_handles,
        call_counts: call_counts.into_py_any(py)?,
        forward_complete: false,
    });
    Ok(())
}

fn stash_tensor_output(
    py: Python<'_>,
    last_outputs: Option<&pyo3::PyObject>,
    path: &str,
    call_index: u32,
    tuple: &pyo3::Bound<'_, PyTuple>,
) -> anyhow::Result<()> {
    if tuple.len() > 2 {
        let output = tuple.get_item(2)?;
        if !output.is_none() {
            if let Some(lo) = last_outputs {
                let dict = lo
                    .bind(py)
                    .downcast::<PyDict>()
                    .map_err(|e| anyhow::anyhow!("last_outputs is not a dict: {e}"))?;
                let key = PyTuple::new(
                    py,
                    [
                        path.into_pyobject(py)?.into_any(),
                        call_index.into_pyobject(py)?.into_any(),
                    ],
                )?;
                dict.set_item(key, output)?;
            }
        }
    }
    Ok(())
}

fn handle_host_step(state: &mut WorkerState, request: &Request) -> Response {
    let req: HostStepRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let Some(handle) = state.model_handle else {
        return internal_error(request.id.clone(), "No model loaded".to_owned());
    };

    if req.model_handle != handle {
        return internal_error(
            request.id.clone(),
            format!(
                "model handle mismatch: expected {handle}, got {}",
                req.model_handle
            ),
        );
    }

    if state.component_map.is_none() {
        return internal_error(request.id.clone(), "No component map available".to_owned());
    }

    let plan = step_driver::plan_step(req.count, req.granularity);

    let resuming = state.forward_pass.is_some();

    match Python::with_gil(|py| -> anyhow::Result<HostStepResponse> {
        ensure_forward_pass(py, state, handle)?;

        let fwd = state
            .forward_pass
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("forward pass not initialized"))?;
        let result_mb = fwd.result_mailbox.bind(py);
        let resume_mb = fwd.resume_mailbox.bind(py);

        let mut ticks_consumed = 0u32;
        let current_layer = state.tick_state.to_tick_position().layer;
        let mut tracking_layer = if resuming { Some(current_layer) } else { None };

        if resuming {
            resume_mb.call_method1("put", (py.None(),))?;
        }

        let mut forward_complete = false;

        loop {
            let value = result_mb.call_method1("wait", (30.0,))?;

            let tuple = value
                .downcast::<PyTuple>()
                .map_err(|e| anyhow::anyhow!("expected tuple from mailbox, got: {e}"))?;
            let path: String = tuple.get_item(0)?.extract()?;
            let call_index: u32 = tuple.get_item(1)?.extract()?;

            stash_tensor_output(py, state.last_outputs.as_ref(), &path, call_index, tuple)?;

            result_mb.call_method0("restore")?;

            if path == FORWARD_COMPLETE_SENTINEL {
                forward_complete = true;
                break;
            }

            let (canonical, layer) =
                if let Some(&idx) = state.component_index.get(&(path.clone(), call_index)) {
                    let c = &state.component_map.as_ref().unwrap().components[idx];
                    (c.canonical.clone(), c.layer_index.unwrap_or(0))
                } else {
                    tracing::warn!(
                        path,
                        call_index,
                        "unrecognized module in forward pass, defaulting to layer 0"
                    );
                    (format!("_raw.{path}"), 0)
                };

            state.tick_state.advance(&canonical, layer, call_index);

            if plan.granularity == rocket_surgeon_protocol::types::TickGranularity::Layer {
                if step_driver::is_layer_boundary(tracking_layer, layer) {
                    ticks_consumed += 1;
                }
                tracking_layer = Some(layer);
            } else {
                ticks_consumed += 1;
            }

            if ticks_consumed >= plan.ticks_to_drain {
                break;
            }

            resume_mb.call_method1("put", (py.None(),))?;
        }

        if forward_complete {
            if let Some(fwd) = state.forward_pass.as_mut() {
                fwd.forward_complete = true;
            }
        }

        let position = state.tick_state.to_tick_position();

        Ok(HostStepResponse {
            position,
            events: vec![],
            forward_complete,
            events_truncated: false,
        })
    }) {
        Ok(resp) => match serde_json::to_value(resp) {
            Ok(value) => Response::success(request.id.clone(), value),
            Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
        },
        Err(e) => internal_error(request.id.clone(), format!("step failed: {e}")),
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

fn handle_host_inspect(state: &WorkerState, request: &Request) -> Response {
    let req: HostInspectRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let Some(handle) = state.model_handle else {
        return internal_error(request.id.clone(), "No model loaded".to_owned());
    };

    if req.model_handle != handle {
        return internal_error(
            request.id.clone(),
            format!(
                "model handle mismatch: expected {handle}, got {}",
                req.model_handle
            ),
        );
    }

    let Some(ref component_map) = state.component_map else {
        return internal_error(request.id.clone(), "No component map available".to_owned());
    };

    let matched_components: Vec<_> = component_map
        .components
        .iter()
        .filter(|c| crate::capture::probe_matches_target(&c.probe_point, &req.target))
        .collect();

    if matched_components.is_empty() {
        return Response::error(
            request.id.clone(),
            RpcError::from_error_data(rocket_surgeon_protocol::errors::ErrorData::new(
                rocket_surgeon_protocol::errors::ErrorCode::InvalidTarget,
                format!("No components match target '{}'", req.target),
            )),
        );
    }

    let tensors = match collect_tensors(state, &matched_components) {
        Ok(t) => t,
        Err(e) => {
            return internal_error(request.id.clone(), format!("inspect failed: {e}"));
        }
    };

    let resp = HostInspectResponse { tensors };

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}

fn collect_tensors(
    state: &WorkerState,
    matched_components: &[&crate::adapter::MappedComponent],
) -> anyhow::Result<Vec<CapturedTensor>> {
    use base64::Engine;

    Python::with_gil(|py| {
        let Some(ref lo) = state.last_outputs else {
            return Ok(vec![]);
        };
        let dict = lo
            .bind(py)
            .downcast::<PyDict>()
            .map_err(|e| anyhow::anyhow!("last_outputs is not a dict: {e}"))?;
        let mut result = Vec::new();

        for comp in matched_components {
            let key = PyTuple::new(
                py,
                [
                    comp.module_path.clone().into_pyobject(py)?.into_any(),
                    comp.call_index.into_pyobject(py)?.into_any(),
                ],
            )?;

            if let Some(tensor_obj) = dict.get_item(&key)? {
                let bytes_result = bridge::tensor_to_bytes(py, &tensor_obj)?;
                let shape = bridge::get_tensor_shape(py, &tensor_obj)?;
                let dtype = bridge::get_tensor_dtype(py, &tensor_obj)?;
                let device = bridge::get_tensor_device(py, &tensor_obj)?;

                let data_base64 = base64::engine::general_purpose::STANDARD.encode(&bytes_result);

                result.push(CapturedTensor {
                    module_path: comp.module_path.clone(),
                    canonical: comp.canonical.clone(),
                    layer: comp.layer_index.unwrap_or(0),
                    shape,
                    dtype,
                    device,
                    data_base64,
                });
            }
        }

        Ok(result)
    })
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

    fn make_state() -> WorkerState {
        WorkerState::new()
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        let mut state = make_state();
        let req = make_request("nonexistent/method", serde_json::Value::Null);
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, METHOD_NOT_FOUND);
    }

    #[test]
    fn host_attach_invalid_params_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_ATTACH,
            serde_json::json!({"wrong_field": 42}),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }

    #[test]
    fn host_detach_invalid_params_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_DETACH,
            serde_json::json!({"wrong_field": 42}),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }

    #[test]
    fn host_attach_unsupported_dtype_returns_error() {
        let mut state = make_state();
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
        let resp = dispatch(&mut state, &req);
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
        let mut state = make_state();
        let mut req = make_request("nonexistent", serde_json::Value::Null);
        req.id = RequestId::String("test-id-42".to_owned());
        let resp = dispatch(&mut state, &req);
        assert_eq!(resp.id, RequestId::String("test-id-42".to_owned()));
    }

    #[test]
    fn dispatch_jsonrpc_version() {
        let mut state = make_state();
        let req = make_request("nonexistent", serde_json::Value::Null);
        let resp = dispatch(&mut state, &req);
        assert_eq!(resp.jsonrpc, JSONRPC_VERSION);
    }

    #[test]
    fn dispatch_configure_hooks_invalid_params() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_CONFIGURE_HOOKS,
            serde_json::json!({"wrong_field": 42}),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }

    #[test]
    fn dispatch_step_invalid_params() {
        let mut state = make_state();
        let req = make_request(internal::HOST_STEP, serde_json::json!({"wrong_field": 42}));
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }

    #[test]
    fn dispatch_update_probes_invalid_params() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_UPDATE_PROBES,
            serde_json::json!({"wrong_field": 42}),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }

    #[test]
    fn dispatch_inspect_invalid_params() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_INSPECT,
            serde_json::json!({"wrong_field": 42}),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }

    #[test]
    fn dispatch_inspect_no_model_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_INSPECT,
            serde_json::json!({
                "model_handle": 1,
                "target": "model:0:0:q_proj:output",
                "detail": "summary"
            }),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert!(
            resp.error
                .as_ref()
                .unwrap()
                .message
                .contains("No model loaded")
        );
    }
}
