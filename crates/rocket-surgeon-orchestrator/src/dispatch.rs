use rocket_surgeon_protocol::jsonrpc::{
    INTERNAL_ERROR, METHOD_NOT_FOUND, Request, Response, RpcError,
};
use rocket_surgeon_protocol::messages::internal;
use tracing::{error, info};

use crate::worker_handle::WorkerHandle;

/// Orchestrator state — holds the (optional) live worker and its configuration.
pub struct OrchestratorState {
    pub worker: Option<WorkerHandle>,
    pub worker_bin: String,
    pub log_level: String,
}

/// Route a JSON-RPC request to the appropriate handler.
///
/// - `_host/attach` — spawn a worker and forward the request
/// - `_host/detach` — forward the request and tear down the worker
/// - anything else  — `METHOD_NOT_FOUND`
pub fn dispatch(state: &mut OrchestratorState, request: &Request) -> Response {
    match request.method.as_str() {
        internal::HOST_ATTACH => handle_host_attach(state, request),
        internal::HOST_DETACH => handle_host_detach(state, request),
        internal::HOST_STEP
        | internal::HOST_CONFIGURE_HOOKS
        | internal::HOST_UPDATE_PROBES
        | internal::HOST_INSPECT
        | internal::HOST_VIEW
        | internal::HOST_KV_READ
        | internal::HOST_KV_INTERVENE
        | internal::HOST_EXPORT_ENV
        | internal::HOST_CHECKPOINT
        | internal::HOST_REPLAY => forward_to_worker(state, request),
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

fn handle_host_attach(state: &mut OrchestratorState, request: &Request) -> Response {
    if state.worker.is_some() {
        return Response::error(
            request.id.clone(),
            RpcError {
                code: INTERNAL_ERROR,
                message: "Worker already running".to_owned(),
                data: None,
            },
        );
    }

    let mut worker = match WorkerHandle::spawn(&state.worker_bin, &state.log_level) {
        Ok(w) => w,
        Err(e) => {
            error!("failed to spawn worker: {e}");
            return Response::error(
                request.id.clone(),
                RpcError {
                    code: INTERNAL_ERROR,
                    message: format!("Failed to spawn worker: {e}"),
                    data: None,
                },
            );
        }
    };

    info!("worker spawned, forwarding _host/attach");

    if let Err(e) = worker.send_request(request) {
        error!("failed to send request to worker: {e}");
        worker.kill();
        return Response::error(
            request.id.clone(),
            RpcError {
                code: INTERNAL_ERROR,
                message: format!("Failed to send request to worker: {e}"),
                data: None,
            },
        );
    }

    let mut response = match worker.recv_response() {
        Ok(r) => r,
        Err(e) => {
            error!("failed to receive response from worker: {e}");
            worker.kill();
            return Response::error(
                request.id.clone(),
                RpcError {
                    code: INTERNAL_ERROR,
                    message: format!("Worker failed during attach: {e}"),
                    data: None,
                },
            );
        }
    };

    // Only keep the worker if the attach actually succeeded.
    if response.error.is_some() {
        worker.kill();
    } else {
        // Stamp the worker PID into the result so the daemon can declare a
        // Perfetto ProcessDescriptor for this rank. The worker doesn't know
        // its own PID (it would `getpid()` on its own process anyway); the
        // orchestrator is the one that just spawned it and holds the Child.
        if let Some(result) = response.result.as_mut()
            && let Some(obj) = result.as_object_mut()
        {
            obj.insert(
                "worker_pid".to_owned(),
                serde_json::Value::from(worker.pid()),
            );
        }
        state.worker = Some(worker);
    }

    response
}

fn handle_host_detach(state: &mut OrchestratorState, request: &Request) -> Response {
    let Some(mut worker) = state.worker.take() else {
        return Response::error(
            request.id.clone(),
            RpcError {
                code: INTERNAL_ERROR,
                message: "No worker running".to_owned(),
                data: None,
            },
        );
    };

    if !worker.is_alive() {
        error!("worker process died before detach could be sent");
        return Response::error(
            request.id.clone(),
            RpcError {
                code: INTERNAL_ERROR,
                message: "Worker process is no longer running".to_owned(),
                data: None,
            },
        );
    }

    if let Err(e) = worker.send_request(request) {
        error!("failed to send detach to worker: {e}");
        // Worker is already taken from state; drop will kill it.
        return Response::error(
            request.id.clone(),
            RpcError {
                code: INTERNAL_ERROR,
                message: format!("Failed to send detach to worker: {e}"),
                data: None,
            },
        );
    }

    // Worker is always torn down after detach — the drop impl kills it.
    match worker.recv_response() {
        Ok(r) => r,
        Err(e) => {
            error!("failed to receive detach response from worker: {e}");
            Response::error(
                request.id.clone(),
                RpcError {
                    code: INTERNAL_ERROR,
                    message: format!("Worker failed during detach: {e}"),
                    data: None,
                },
            )
        }
    }
}

fn forward_to_worker(state: &mut OrchestratorState, request: &Request) -> Response {
    let Some(worker) = state.worker.as_mut() else {
        return Response::error(
            request.id.clone(),
            RpcError {
                code: INTERNAL_ERROR,
                message: "No worker running".to_owned(),
                data: None,
            },
        );
    };

    if !worker.is_alive() {
        error!("worker process died before request could be sent");
        state.worker = None;
        return Response::error(
            request.id.clone(),
            RpcError {
                code: INTERNAL_ERROR,
                message: "Worker process is no longer running".to_owned(),
                data: None,
            },
        );
    }

    if let Err(e) = worker.send_request(request) {
        error!("failed to send request to worker: {e}");
        return Response::error(
            request.id.clone(),
            RpcError {
                code: INTERNAL_ERROR,
                message: format!("Failed to send request to worker: {e}"),
                data: None,
            },
        );
    }

    match worker.recv_response() {
        Ok(r) => r,
        Err(e) => {
            error!("failed to receive response from worker: {e}");
            Response::error(
                request.id.clone(),
                RpcError {
                    code: INTERNAL_ERROR,
                    message: format!("Worker failed: {e}"),
                    data: None,
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocket_surgeon_protocol::jsonrpc::{JSONRPC_VERSION, RequestId};

    fn make_state() -> OrchestratorState {
        OrchestratorState {
            worker: None,
            worker_bin: "/nonexistent/rs-worker".to_owned(),
            log_level: "info".to_owned(),
        }
    }

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
        let mut state = make_state();
        let req = make_request("nonexistent/method", serde_json::Value::Null);
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, METHOD_NOT_FOUND);
    }

    #[test]
    fn host_attach_spawn_failure_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_ATTACH,
            serde_json::json!({
                "model_source": "test",
                "model_family": "llama",
                "device": "cpu",
                "rank": 0
            }),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert!(
            resp.error
                .as_ref()
                .unwrap()
                .message
                .contains("Failed to spawn worker")
        );
        assert!(state.worker.is_none());
    }

    #[test]
    fn host_detach_without_worker_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_DETACH,
            serde_json::json!({"model_handle": 1}),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert!(
            resp.error
                .as_ref()
                .unwrap()
                .message
                .contains("No worker running")
        );
    }

    #[test]
    fn dispatch_preserves_request_id() {
        let mut state = make_state();
        let mut req = make_request("nonexistent", serde_json::Value::Null);
        req.id = RequestId::String("test-id-99".to_owned());
        let resp = dispatch(&mut state, &req);
        assert_eq!(resp.id, RequestId::String("test-id-99".to_owned()));
    }

    #[test]
    fn dispatch_jsonrpc_version() {
        let mut state = make_state();
        let req = make_request("nonexistent", serde_json::Value::Null);
        let resp = dispatch(&mut state, &req);
        assert_eq!(resp.jsonrpc, JSONRPC_VERSION);
    }

    #[test]
    fn host_step_without_worker_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_STEP,
            serde_json::json!({
                "model_handle": 1,
                "count": 1,
                "direction": "forward"
            }),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert!(
            resp.error
                .as_ref()
                .unwrap()
                .message
                .contains("No worker running")
        );
    }

    #[test]
    fn host_configure_hooks_without_worker_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_CONFIGURE_HOOKS,
            serde_json::json!({
                "model_handle": 1,
                "active_probes": []
            }),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
    }

    #[test]
    fn host_update_probes_without_worker_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_UPDATE_PROBES,
            serde_json::json!({
                "model_handle": 1,
                "active_probes": []
            }),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
    }

    #[test]
    fn host_inspect_without_worker_returns_error() {
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
                .contains("No worker running")
        );
    }

    #[test]
    fn host_kv_read_without_worker_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_KV_READ,
            serde_json::json!({
                "model_handle": 1,
                "layers": [0],
                "positions": [0]
            }),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert!(
            resp.error
                .as_ref()
                .unwrap()
                .message
                .contains("No worker running")
        );
    }

    #[test]
    fn host_kv_intervene_without_worker_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_KV_INTERVENE,
            serde_json::json!({
                "model_handle": 1,
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
                .contains("No worker running")
        );
    }

    #[test]
    fn host_checkpoint_without_worker_returns_error() {
        let mut state = make_state();
        let req = make_request(
            internal::HOST_CHECKPOINT,
            serde_json::json!({
                "Create": {
                    "model_handle": 1,
                    "checkpoint_id": "test-ckpt",
                    "tier": "Activation",
                    "tick_id": 0,
                    "layer_idx": 0
                }
            }),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert!(
            resp.error
                .as_ref()
                .unwrap()
                .message
                .contains("No worker running")
        );
    }

    #[test]
    fn host_attach_rejects_double_attach() {
        // We can't spawn a real worker, but we can simulate one by manually
        // inserting a WorkerHandle. Use `cat` as the worker binary so we get
        // a real child process.
        let worker = WorkerHandle::spawn("cat", "info").expect("cat should exist");
        let mut state = OrchestratorState {
            worker: Some(worker),
            worker_bin: "/nonexistent/rs-worker".to_owned(),
            log_level: "info".to_owned(),
        };
        let req = make_request(
            internal::HOST_ATTACH,
            serde_json::json!({
                "model_source": "test",
                "model_family": "llama",
                "device": "cpu",
                "rank": 0
            }),
        );
        let resp = dispatch(&mut state, &req);
        assert!(resp.error.is_some());
        assert!(
            resp.error
                .as_ref()
                .unwrap()
                .message
                .contains("already running")
        );
    }
}
