use std::collections::HashMap;

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rocket_surgeon_probes::grammar::{
    ComponentOrWild, ComponentSeg, NameOrWild, NumOrWild, ProbePoint,
};
use rocket_surgeon_protocol::jsonrpc::{
    INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND, Request, RequestId, Response, RpcError,
};
use rocket_surgeon_protocol::messages::internal;
use rocket_surgeon_protocol::messages::{
    CapturedTensor, HostConfigureHooksRequest, HostConfigureHooksResponse, HostDetachRequest,
    HostDetachResponse, HostInspectRequest, HostInspectResponse, HostKvInterveneRequest,
    HostKvReadRequest, HostStepRequest, HostStepResponse, HostUpdateProbesRequest,
    HostUpdateProbesResponse, HostViewRequest, HostViewResponse, ProbeFiredEvent,
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
    pub passive_handles: Vec<pyo3::PyObject>,
    #[allow(dead_code)]
    pub call_counts: pyo3::PyObject,
    pub forward_complete: bool,
}

pub struct WorkerState {
    pub component_map: Option<ComponentMap>,
    pub component_index: HashMap<(String, u32), usize>,
    pub module_paths: Vec<String>,
    pub container_paths: Vec<String>,
    pub model_handle: Option<u64>,
    pub rank: u32,
    pub tick_state: TickState,
    pub forward_pass: Option<ForwardPassState>,
    pub last_outputs: Option<pyo3::PyObject>,
    pub active_probes: Vec<(
        rocket_surgeon_protocol::types::ProbeDefinition,
        rocket_surgeon_probes::grammar::ProbePoint,
    )>,
    pub shm_ring: Option<rocket_surgeon_shm::ring::DoomRingProducer>,
    /// KV-cache eviction / pin bookkeeping (WU-G). See `crate::kv`.
    pub kv_cache: crate::kv::KvCacheState,
}

impl WorkerState {
    pub fn new() -> Self {
        Self {
            component_map: None,
            component_index: HashMap::new(),
            module_paths: Vec::new(),
            container_paths: Vec::new(),
            model_handle: None,
            rank: 0,
            tick_state: TickState::new(0),
            forward_pass: None,
            last_outputs: None,
            active_probes: Vec::new(),
            shm_ring: None,
            kv_cache: crate::kv::KvCacheState::new(),
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
        internal::HOST_UPDATE_PROBES => handle_host_update_probes(state, request),
        internal::HOST_INSPECT => handle_host_inspect(state, request),
        internal::HOST_VIEW => handle_host_view(state, request),
        internal::HOST_KV_READ => handle_host_kv_read(state, request),
        internal::HOST_KV_INTERVENE => handle_host_kv_intervene(state, request),
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

#[allow(clippy::too_many_lines)]
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

    let (mut component_map, container_paths) =
        match crate::adapter::resolve_with_containers(&modules, &config, req.rank) {
            Ok(r) => r,
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
    state.container_paths = container_paths;
    state.model_handle = Some(info.handle);
    state.component_map = Some(component_map.clone());
    state.rank = req.rank;
    state.tick_state = TickState::new(req.rank);
    state.forward_pass = None;
    state.kv_cache.reset();

    if let Some(old_ring) = state.shm_ring.take() {
        let old_name = old_ring.shm_name().to_owned();
        drop(old_ring);
        let _ = rocket_surgeon_shm::region::ShmRegion::unlink(&old_name);
        rocket_surgeon_shm::cleanup::deregister_region_name(&old_name);
    }

    let shm_ring = {
        let session_id = format!("{:08x}", std::process::id());
        let name = format!("/rs-{session_id}-0");
        let config = rocket_surgeon_shm::RingConfig::new(16, 64 * 1024 * 1024)
            .expect("ring config is valid");
        match rocket_surgeon_shm::ring::DoomRingProducer::create(&name, config) {
            Ok(ring) => {
                tracing::info!(shm_name = %name, "created shared memory ring buffer");
                rocket_surgeon_shm::cleanup::register_region_name(&name);
                Some(ring)
            }
            Err(e) => {
                tracing::warn!("failed to create shm ring, falling back to base64: {e}");
                None
            }
        }
    };
    state.shm_ring = shm_ring;

    let resp = HostAttachResponse {
        model_handle: info.handle,
        num_layers: info.num_layers,
        num_heads: info.num_heads,
        hidden_dim: info.hidden_dim,
        module_tree: info.module_tree,
        model_type: config.model_type,
        component_vocabulary: component_map.vocabulary,
        shm_name: state.shm_ring.as_ref().map(|r| r.shm_name().to_owned()),
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
            if let Err(e) = bridge::remove_hooks(py, &fwd.passive_handles) {
                tracing::warn!("failed to remove passive hooks during detach: {e}");
            }
        });
    }

    state.model_handle = None;
    state.component_map = None;
    state.component_index.clear();
    state.module_paths.clear();
    state.container_paths.clear();
    state.last_outputs = None;
    state.kv_cache.reset();

    if let Some(ring) = state.shm_ring.take() {
        let name = ring.shm_name().to_owned();
        drop(ring);
        if let Err(e) = rocket_surgeon_shm::region::ShmRegion::unlink(&name) {
            tracing::warn!("failed to unlink shm region '{name}': {e}");
        }
        rocket_surgeon_shm::cleanup::deregister_region_name(&name);
    }

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

fn build_tick_probe_point(
    model_family: &str,
    rank: u32,
    layer: u32,
    canonical: &str,
    call_index: u32,
) -> ProbePoint {
    ProbePoint {
        model: NameOrWild::Name(model_family.to_owned()),
        rank: NumOrWild::Num(rank),
        layer: NumOrWild::Num(layer),
        component: ComponentOrWild::Path(vec![ComponentSeg::Named(canonical.to_owned())]),
        call_index: NumOrWild::Num(call_index),
        event: NameOrWild::Name("fwd".to_owned()),
    }
}

fn evaluate_probes(
    state: &WorkerState,
    current_point: &ProbePoint,
    remaining_budget: u32,
) -> (Vec<ProbeFiredEvent>, bool) {
    let mut events = Vec::new();
    let mut truncated = false;

    for (def, pattern) in &state.active_probes {
        if !pattern.matches(current_point) {
            continue;
        }
        if events.len() >= remaining_budget as usize {
            truncated = true;
            break;
        }

        let d = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();

        let event = ProbeFiredEvent {
            probe_id: def.id.clone(),
            point: current_point.to_string(),
            tick_id: state.tick_state.tick_id(),
            tensor_summary: None,
            action: def.action,
            timestamp: format!("{}", d.as_secs()),
        };
        events.push(event);
    }

    (events, truncated)
}

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

    let passive_handles = if state.container_paths.is_empty() {
        Vec::new()
    } else {
        let lo_bound = state.last_outputs.as_ref().unwrap().bind(py);
        bridge::install_passive_hooks(py, handle, &state.container_paths, lo_bound)?
    };

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
        passive_handles,
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
        if !output.is_none()
            && let Some(lo) = last_outputs
        {
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

    match Python::with_gil(|py| run_step_loop(py, state, handle, &req)) {
        Ok(resp) => match serde_json::to_value(resp) {
            Ok(value) => Response::success(request.id.clone(), value),
            Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
        },
        Err(e) => internal_error(request.id.clone(), format!("step failed: {e}")),
    }
}

fn try_apply_interventions<'py>(
    py: Python<'py>,
    state: &WorkerState,
    req: &HostStepRequest,
    tuple: &pyo3::Bound<'py, PyTuple>,
    layer: u32,
    canonical: &str,
) -> anyhow::Result<Option<(Bound<'py, pyo3::PyAny>, Vec<String>)>> {
    if req.interventions.is_empty() || tuple.len() <= 2 {
        return Ok(None);
    }
    let output = tuple.get_item(2)?;
    if output.is_none() {
        return Ok(None);
    }
    let family = state
        .component_map
        .as_ref()
        .map_or("unknown", |m| m.model_family.as_str());
    let recipes_json = serde_json::to_string(&req.interventions)?;
    let (modified, fired) = crate::bridge::apply_interventions_at_point(
        py,
        &output,
        &recipes_json,
        family,
        state.rank,
        layer,
        canonical,
        "fwd",
    )?;
    if fired.is_empty() {
        Ok(None)
    } else {
        Ok(Some((modified, fired)))
    }
}

fn run_step_loop(
    py: Python<'_>,
    state: &mut WorkerState,
    handle: u64,
    req: &HostStepRequest,
) -> anyhow::Result<HostStepResponse> {
    let plan = step_driver::plan_step(req.count, req.granularity);
    let resuming = state.forward_pass.is_some();

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
    let mut all_events: Vec<ProbeFiredEvent> = Vec::new();
    let mut all_events_truncated = false;
    let mut all_fired: Vec<String> = Vec::new();
    let max_events = req.max_events.unwrap_or(256);

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

        if !all_events_truncated && !state.active_probes.is_empty() {
            let family = state
                .component_map
                .as_ref()
                .map_or("unknown", |m| m.model_family.as_str());
            let current_point =
                build_tick_probe_point(family, state.rank, layer, &canonical, call_index);
            let budget = max_events.saturating_sub(all_events.len() as u32);
            let (new_events, trunc) = evaluate_probes(state, &current_point, budget);
            all_events.extend(new_events);
            if trunc {
                all_events_truncated = true;
            }
        }

        let intervention_result =
            try_apply_interventions(py, state, req, tuple, layer, &canonical)?;

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

        if let Some((modified_tensor, fired)) = intervention_result {
            all_fired.extend(fired);
            resume_mb.call_method1("put", (modified_tensor,))?;
        } else {
            resume_mb.call_method1("put", (py.None(),))?;
        }
    }

    if forward_complete {
        if let Some(fwd) = state.forward_pass.as_mut() {
            fwd.forward_complete = true;
        }
        // The initial forward pass is a prefill of the prompt. Once it
        // completes, the model is positioned to generate the next token, so
        // advance the token clock (which resets the operator clock to 0) and
        // transition the phase from prefill to decode.
        state.tick_state.advance_token();
    }

    let position = state.tick_state.to_tick_position();

    Ok(HostStepResponse {
        position,
        events: all_events,
        forward_complete,
        events_truncated: all_events_truncated,
        fired_interventions: all_fired,
    })
}

fn handle_host_update_probes(state: &mut WorkerState, request: &Request) -> Response {
    let req: HostUpdateProbesRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let mut parsed = Vec::with_capacity(req.active_probes.len());
    for probe in req.active_probes {
        match rocket_surgeon_probes::grammar::ProbePoint::parse(&probe.point) {
            Ok(pp) => parsed.push((probe, pp)),
            Err(e) => {
                tracing::warn!(point = %probe.point, error = %e, "skipping probe with invalid point");
            }
        }
    }

    state.active_probes = parsed;

    let resp = HostUpdateProbesResponse {
        probes_active: state.active_probes.len() as u32,
    };

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}

fn handle_host_inspect(state: &mut WorkerState, request: &Request) -> Response {
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

    let matched_components: Vec<_> = match state.component_map {
        Some(ref component_map) => component_map
            .components
            .iter()
            .filter(|c| crate::capture::probe_matches_target(&c.probe_point, &req.target))
            .cloned()
            .collect(),
        None => {
            return internal_error(request.id.clone(), "No component map available".to_owned());
        }
    };

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
    state: &mut WorkerState,
    matched_components: &[crate::adapter::MappedComponent],
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

                let tensor_id = blake3::hash(&bytes_result).to_hex().to_string();

                let ct = try_shm_publish(
                    &mut state.shm_ring,
                    &bytes_result,
                    &tensor_id,
                    comp,
                    &shape,
                    &dtype,
                    &device,
                )
                .unwrap_or_else(|| {
                    let data_base64 =
                        base64::engine::general_purpose::STANDARD.encode(&bytes_result);
                    CapturedTensor {
                        module_path: comp.module_path.clone(),
                        canonical: comp.canonical.clone(),
                        layer: comp.layer_index.unwrap_or(0),
                        shape,
                        dtype,
                        device,
                        tensor_id,
                        shm_name: None,
                        shm_offset: None,
                        byte_length: None,
                        data_base64: Some(data_base64),
                    }
                });

                result.push(ct);
            }
        }

        Ok(result)
    })
}

fn try_shm_publish(
    shm_ring: &mut Option<rocket_surgeon_shm::ring::DoomRingProducer>,
    bytes: &[u8],
    tensor_id: &str,
    comp: &crate::adapter::MappedComponent,
    shape: &[u64],
    dtype: &str,
    device: &str,
) -> Option<CapturedTensor> {
    let ring = shm_ring.as_mut()?;

    if shape.len() > 8 {
        tracing::warn!(
            ndim = shape.len(),
            "tensor has more than 8 dimensions, truncating in probe frame header"
        );
    }

    let mut shape_arr = [0u32; 8];
    for (i, &dim) in shape.iter().enumerate().take(8) {
        shape_arr[i] = dim as u32;
    }

    let header_bytes = rocket_surgeon_shm::serialize_probe_frame(
        0,
        comp.layer_index.unwrap_or(0),
        0,
        0,
        shape.len().min(8) as u8,
        &shape_arr,
        0,
        0,
        bytes.len() as u64,
        0,
        (ring.maketic() & 0xFFFF_FFFF) as u32,
    );

    match ring.publish(&header_bytes, bytes) {
        Ok(slot_maketic) => {
            let slot_offset = ring.config().slot_offset(slot_maketic) as u64;
            Some(CapturedTensor {
                module_path: comp.module_path.clone(),
                canonical: comp.canonical.clone(),
                layer: comp.layer_index.unwrap_or(0),
                shape: shape.to_vec(),
                dtype: dtype.to_owned(),
                device: device.to_owned(),
                tensor_id: tensor_id.to_owned(),
                shm_name: Some(ring.shm_name().to_owned()),
                shm_offset: Some(slot_offset),
                byte_length: Some(bytes.len() as u64),
                data_base64: None,
            })
        }
        Err(e) => {
            tracing::warn!("shm publish failed, falling back to base64: {e}");
            None
        }
    }
}

fn handle_host_view(state: &WorkerState, request: &Request) -> Response {
    let req: HostViewRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    if state.model_handle.is_none() || state.model_handle != Some(req.model_handle) {
        return internal_error(
            request.id.clone(),
            "model handle mismatch or no model loaded".to_owned(),
        );
    }

    if state.last_outputs.is_none() {
        return Response::error(
            request.id.clone(),
            RpcError::from_error_data(rocket_surgeon_protocol::errors::ErrorData::new(
                rocket_surgeon_protocol::errors::ErrorCode::ViewDataUnavailable,
                "No captured tensors — execute at least one step first",
            )),
        );
    }

    let result = Python::with_gil(|py| compute_view(py, state, &req));

    match result {
        Ok(data) => {
            let resp = HostViewResponse {
                view: req.view,
                data,
            };
            match serde_json::to_value(resp) {
                Ok(value) => Response::success(request.id.clone(), value),
                Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
            }
        }
        Err(e) => {
            let msg = format!("{e}");
            if msg.contains("VIEW_DATA_UNAVAILABLE") {
                Response::error(
                    request.id.clone(),
                    RpcError::from_error_data(rocket_surgeon_protocol::errors::ErrorData::new(
                        rocket_surgeon_protocol::errors::ErrorCode::ViewDataUnavailable,
                        msg,
                    )),
                )
            } else if msg.contains("CAPABILITY_NOT_SUPPORTED") {
                Response::error(
                    request.id.clone(),
                    RpcError::from_error_data(rocket_surgeon_protocol::errors::ErrorData::new(
                        rocket_surgeon_protocol::errors::ErrorCode::CapabilityNotSupported,
                        msg,
                    )),
                )
            } else if msg.contains("INVALID_PARAMS") {
                Response::error(
                    request.id.clone(),
                    RpcError::from_error_data(rocket_surgeon_protocol::errors::ErrorData::new(
                        rocket_surgeon_protocol::errors::ErrorCode::InvalidParams,
                        msg,
                    )),
                )
            } else {
                internal_error(request.id.clone(), msg)
            }
        }
    }
}

fn compute_view(
    py: Python<'_>,
    state: &WorkerState,
    req: &HostViewRequest,
) -> anyhow::Result<serde_json::Value> {
    let views_mod = py.import("rocket_surgeon.views")?;
    let handle = state
        .model_handle
        .ok_or_else(|| anyhow::anyhow!("no model handle"))?;
    let lo = state
        .last_outputs
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("VIEW_DATA_UNAVAILABLE: no last_outputs"))?;

    let view_name = serde_json::to_value(req.view)?;
    let view_str = view_name
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("bad view name"))?;

    let params_py = match &req.params {
        Some(v) => {
            let json_str = serde_json::to_string(v)?;
            py.import("json")?.call_method1("loads", (json_str,))?
        }
        None => py.None().into_bound(py),
    };

    let result_py =
        views_mod.call_method1("compute_view", (handle, lo.bind(py), view_str, params_py))?;

    let json_mod = py.import("json")?;
    let result_str: String = json_mod.call_method1("dumps", (result_py,))?.extract()?;

    let data: serde_json::Value = serde_json::from_str(&result_str)?;
    Ok(data)
}

/// Shared `model_handle` validation for the KV-cache handlers.
///
/// Returns a boxed `INTERNAL_ERROR` response when no model is loaded or the
/// handle does not match the loaded model.
fn validate_kv_handle(
    state: &WorkerState,
    request: &Request,
    req_handle: u64,
) -> Result<(), Box<Response>> {
    let Some(handle) = state.model_handle else {
        return Err(Box::new(internal_error(
            request.id.clone(),
            "No model loaded".to_owned(),
        )));
    };
    if req_handle != handle {
        return Err(Box::new(internal_error(
            request.id.clone(),
            format!("model handle mismatch: expected {handle}, got {req_handle}"),
        )));
    }
    Ok(())
}

/// `_host/kv.read` — read per-(layer, position, head) KV-cache norms.
///
/// Delegates the norm/eviction computation to [`crate::kv::read`]. See that
/// module for the documented backend stub: norms are deterministic, eviction
/// is real worker state populated by `_host/kv.intervene`.
fn handle_host_kv_read(state: &WorkerState, request: &Request) -> Response {
    let req: HostKvReadRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    if let Err(resp) = validate_kv_handle(state, request, req.model_handle) {
        return *resp;
    }

    let resp = crate::kv::read(&req, &state.kv_cache);

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}

/// `_host/kv.intervene` — mutate the KV cache (zero / scale / evict / pin).
///
/// Delegates to [`crate::kv::intervene`], recording the worker's current
/// tick id as the `evicted_at` tick for `evict` ops.
fn handle_host_kv_intervene(state: &mut WorkerState, request: &Request) -> Response {
    let req: HostKvInterveneRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    if let Err(resp) = validate_kv_handle(state, request, req.model_handle) {
        return *resp;
    }

    let current_tick = state.tick_state.tick_id();
    let resp = crate::kv::intervene(&req, &mut state.kv_cache, current_tick);

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

    // --- WU-G: KV-cache dispatch ---------------------------------------

    #[test]
    fn dispatch_kv_read_invalid_params() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_KV_READ,
            serde_json::json!({"wrong_field": 42}),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
    }

    #[test]
    fn dispatch_kv_read_no_model_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_KV_READ,
            serde_json::json!({"model_handle": 1, "layers": [0], "positions": [0]}),
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

    #[test]
    fn dispatch_kv_read_with_model_returns_entries() {
        let mut state = make_state();
        state.model_handle = Some(1);
        let req = make_request(
            internal::HOST_KV_READ,
            serde_json::json!({
                "model_handle": 1,
                "layers": [0, 1],
                "positions": [0, 1, 2],
                "heads": [0]
            }),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_none());
        let value = resp.result.unwrap();
        let parsed: rocket_surgeon_protocol::messages::HostKvReadResponse =
            serde_json::from_value(value).unwrap();
        assert_eq!(parsed.entries.len(), 6);
    }

    #[test]
    fn dispatch_kv_intervene_evict_then_read_reports_evicted() {
        let mut state = make_state();
        state.model_handle = Some(1);

        let evict = make_request(
            internal::HOST_KV_INTERVENE,
            serde_json::json!({
                "model_handle": 1,
                "layers": [0],
                "positions": [5],
                "op": "evict"
            }),
        );
        let evict_resp = dispatch(&mut state, &evict);
        assert!(evict_resp.error.is_none());
        let iresp: rocket_surgeon_protocol::messages::HostKvInterveneResponse =
            serde_json::from_value(evict_resp.result.unwrap()).unwrap();
        assert_eq!(iresp.applied_op, "evict");

        let read = make_request(
            internal::HOST_KV_READ,
            serde_json::json!({
                "model_handle": 1,
                "layers": [0],
                "positions": [5],
                "heads": [0]
            }),
        );
        let read_resp = dispatch(&mut state, &read);
        assert!(read_resp.error.is_none());
        let rresp: rocket_surgeon_protocol::messages::HostKvReadResponse =
            serde_json::from_value(read_resp.result.unwrap()).unwrap();
        assert_eq!(
            rresp.entries[0].overlay,
            Some(rocket_surgeon_protocol::messages::KvOverlay::Evicted)
        );
        assert!(rresp.entries[0].k_metric.is_none());
    }

    #[test]
    fn dispatch_kv_intervene_handle_mismatch_returns_error() {
        let mut state = make_state();
        state.model_handle = Some(1);
        let req = make_request(
            internal::HOST_KV_INTERVENE,
            serde_json::json!({
                "model_handle": 999,
                "layers": [0],
                "positions": [0],
                "op": "zero"
            }),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert!(
            resp.error
                .as_ref()
                .unwrap()
                .message
                .contains("model handle mismatch")
        );
    }
}
