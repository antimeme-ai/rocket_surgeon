# Sub-project A: Bundle Completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the 5 missing artifacts (model-info.json, env.json, prompt.json, trace.perfetto-trace, bookmarks.json) to session bundle export so it meets the Phase 2 exit criterion of 9 required artifacts.

**Architecture:** The daemon's `handle_export` calls `_host/export_env` via the orchestrator to collect env/model data from the Python worker, reads the PerfettoSink trace file, and adds all 5 new artifacts alongside the existing 4. The orchestrator forwards `_host/export_env` to the worker the same way it forwards all other `_host/*` methods.

**Tech Stack:** Rust (daemon, orchestrator, worker, protocol), Python (bridge), PyO3, serde_json, tar/flate2

---

## File Structure

| File | Responsibility |
|------|---------------|
| `python/rocket_surgeon/bridge.py` | New `collect_export_env(handle)` — queries PyTorch/system for env + model info |
| `crates/rocket-surgeon-worker/src/bridge.rs` | New `collect_export_env()` — PyO3 call into Python bridge |
| `crates/rocket-surgeon-worker/src/dispatch.rs` | New `handle_host_export_env()` — worker dispatch handler |
| `crates/rocket-surgeon-orchestrator/src/dispatch.rs` | Route `_host/export_env` to `forward_to_worker` |
| `crates/rocket-surgeon/src/orchestrator_handle.rs` | New `export_env()` method — typed wrapper for daemon → orchestrator |
| `crates/rocket-surgeon/src/dispatch.rs` | Extend `handle_export` — accept orchestrator + perfetto_path, call export_env, add all 5 artifacts |
| `crates/rocket-surgeon/src/main.rs` | Update `handle_export` call site with new params |
| `tests/test_e2e_bundle.py` | Validate all 9 artifacts in the bundle |
| `tck/protocol/session-export.feature` | Add scenario for 9-artifact completeness |

---

### Task 1: Python bridge — collect_export_env

**Files:**
- Modify: `python/rocket_surgeon/bridge.py`
- Test: `python/tests/test_interventions.py` (add test at bottom, or inline verification via e2e)

- [ ] **Step 1: Write the Python function**

Add to `python/rocket_surgeon/bridge.py` after the existing `model_config()` function (after line 116):

```python
def collect_export_env(handle: int) -> dict[str, Any]:
    """Collect environment and model info for session bundle export."""
    import platform
    import sys

    model = _models[handle]
    config = model.config

    num_params = sum(p.numel() for p in model.parameters())
    param = next(model.parameters(), None)
    dtype_str = str(param.dtype).replace("torch.", "") if param is not None else "unknown"

    env = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()) if torch.cuda.is_available() else None,
        "python_version": sys.version.split()[0],
        "os": platform.platform(),
        "rocket_surgeon_version": "0.1.0",
    }

    model_info = {
        "model_family": getattr(config, "model_type", "unknown"),
        "model_path": getattr(config, "name_or_path", "unknown"),
        "num_layers": getattr(config, "num_hidden_layers", 0),
        "num_heads": getattr(config, "num_attention_heads", 0),
        "hidden_dim": getattr(config, "hidden_size", 0),
        "num_params": num_params,
        "dtype": dtype_str,
    }

    return {
        "env": env,
        "model_info": model_info,
        "prompt": None,
    }
```

- [ ] **Step 2: Verify ruff is happy**

Run: `ruff check python/rocket_surgeon/bridge.py && ruff format --check python/rocket_surgeon/bridge.py`
Expected: No errors. If the `nccl_version` line is too long, break it:
```python
        "nccl_version": (
            ".".join(str(v) for v in torch.cuda.nccl.version())
            if torch.cuda.is_available()
            else None
        ),
```

- [ ] **Step 3: Run mypy**

Run: `mypy python/rocket_surgeon/bridge.py`
Expected: PASS (no type errors — all args/returns are typed via dict[str, Any])

- [ ] **Step 4: Commit**

```bash
git add python/rocket_surgeon/bridge.py
git commit -m "feat(bridge): add collect_export_env for session bundle"
```

---

### Task 2: Rust bridge — PyO3 call into Python

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/bridge.rs`

- [ ] **Step 1: Add the Rust bridge function**

Add to `crates/rocket-surgeon-worker/src/bridge.rs` after the existing `model_config()` function (search for a function near the end that calls Python). Add before `run_forward`:

```rust
pub fn collect_export_env(handle: u64) -> anyhow::Result<serde_json::Value> {
    Python::with_gil(|py| {
        let bridge = py.import("rocket_surgeon.bridge")?;
        let result = bridge
            .getattr("collect_export_env")?
            .call1((handle,))?;
        let json_str = py
            .import("json")?
            .getattr("dumps")?
            .call1((result,))?
            .extract::<String>()?;
        let value: serde_json::Value = serde_json::from_str(&json_str)?;
        Ok(value)
    })
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p rocket-surgeon-worker`
Expected: Compiles cleanly

- [ ] **Step 3: Commit**

```bash
git add crates/rocket-surgeon-worker/src/bridge.rs
git commit -m "feat(worker/bridge): add collect_export_env PyO3 call"
```

---

### Task 3: Worker dispatch — handle_host_export_env

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/dispatch.rs`

- [ ] **Step 1: Add the import for HostExportEnvRequest and HostExportEnvResponse**

In `crates/rocket-surgeon-worker/src/dispatch.rs`, add `HostExportEnvRequest` and `HostExportEnvResponse` to the existing import from `rocket_surgeon_protocol::messages`:

```rust
use rocket_surgeon_protocol::messages::{
    CapturedTensor, HostConfigureHooksRequest, HostConfigureHooksResponse, HostDetachRequest,
    HostDetachResponse, HostExportEnvRequest, HostExportEnvResponse, HostInspectRequest,
    HostInspectResponse, HostKvInterveneRequest, HostKvReadRequest, HostStepRequest,
    HostStepResponse, HostUpdateProbesRequest, HostUpdateProbesResponse, HostViewRequest,
    HostViewResponse, ProbeFiredEvent,
};
```

- [ ] **Step 2: Add the handler function**

Add after the last handler function (after `handle_host_kv_intervene`):

```rust
fn handle_host_export_env(state: &WorkerState, request: &Request) -> Response {
    let req: HostExportEnvRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let result = bridge::collect_export_env(req.model_handle);
    match result {
        Ok(value) => {
            let env = value.get("env").cloned().unwrap_or(serde_json::Value::Null);
            let model_info = value
                .get("model_info")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let prompt = value.get("prompt").cloned().and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    Some(v)
                }
            });

            let resp = HostExportEnvResponse {
                env,
                model_info,
                prompt,
            };
            Response::success(request.id.clone(), serde_json::to_value(resp).unwrap())
        }
        Err(e) => internal_error(request.id.clone(), format!("collect_export_env failed: {e}")),
    }
}
```

- [ ] **Step 3: Add routing in the dispatch match**

In the `dispatch()` function, add a new arm before the `_` catch-all:

```rust
        internal::HOST_KV_INTERVENE => handle_host_kv_intervene(state, request),
        internal::HOST_EXPORT_ENV => handle_host_export_env(state, request),
        _ => Response::error(
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p rocket-surgeon-worker`
Expected: Compiles cleanly

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon-worker/src/dispatch.rs
git commit -m "feat(worker): handle _host/export_env dispatch"
```

---

### Task 4: Orchestrator — forward _host/export_env

**Files:**
- Modify: `crates/rocket-surgeon-orchestrator/src/dispatch.rs`

- [ ] **Step 1: Add HOST_EXPORT_ENV to the forward_to_worker match arm**

In `crates/rocket-surgeon-orchestrator/src/dispatch.rs`, add `internal::HOST_EXPORT_ENV` to the existing chain of methods that get forwarded:

```rust
        internal::HOST_STEP
        | internal::HOST_CONFIGURE_HOOKS
        | internal::HOST_UPDATE_PROBES
        | internal::HOST_INSPECT
        | internal::HOST_VIEW
        | internal::HOST_KV_READ
        | internal::HOST_KV_INTERVENE
        | internal::HOST_EXPORT_ENV => forward_to_worker(state, request),
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p rocket-surgeon-orchestrator`
Expected: Compiles cleanly

- [ ] **Step 3: Commit**

```bash
git add crates/rocket-surgeon-orchestrator/src/dispatch.rs
git commit -m "feat(orchestrator): route _host/export_env to worker"
```

---

### Task 5: OrchestratorHandle — typed export_env method

**Files:**
- Modify: `crates/rocket-surgeon/src/orchestrator_handle.rs`

- [ ] **Step 1: Add import for HostExportEnvRequest and HostExportEnvResponse**

Add to the import block at the top of `orchestrator_handle.rs`:

```rust
use rocket_surgeon_protocol::messages::{
    HostAttachRequest, HostAttachResponse, HostDetachRequest, HostExportEnvRequest,
    HostExportEnvResponse, HostInspectRequest, HostKvInterveneRequest, HostKvReadRequest,
    HostStepRequest, HostStepResponse, HostUpdateProbesRequest, HostUpdateProbesResponse,
    HostViewRequest, internal,
};
```

- [ ] **Step 2: Add the export_env method**

Add after the `update_probes` method (after line ~192), before `kill`:

```rust
    pub fn export_env(
        &mut self,
        req: &HostExportEnvRequest,
    ) -> anyhow::Result<HostExportEnvResponse> {
        let id = self.next_id();
        let params = serde_json::to_value(req)?;
        let request = Request::new(RequestId::Number(id), internal::HOST_EXPORT_ENV, params);

        self.send(&request)?;
        let response = self.recv()?;

        if let Some(err) = response.error {
            anyhow::bail!(
                "orchestrator export_env failed (code {}): {}",
                err.code,
                err.message
            );
        }

        let result = response
            .result
            .ok_or_else(|| anyhow::anyhow!("orchestrator export_env: missing result"))?;
        let resp: HostExportEnvResponse = serde_json::from_value(result)?;
        Ok(resp)
    }
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p rocket-surgeon`
Expected: Compiles cleanly

- [ ] **Step 4: Commit**

```bash
git add crates/rocket-surgeon/src/orchestrator_handle.rs
git commit -m "feat(daemon): OrchestratorHandle.export_env typed method"
```

---

### Task 6: Extend handle_export — 5 new artifacts

**Files:**
- Modify: `crates/rocket-surgeon/src/dispatch.rs`
- Modify: `crates/rocket-surgeon/src/main.rs`

- [ ] **Step 1: Update handle_export signature**

In `crates/rocket-surgeon/src/dispatch.rs`, change the `handle_export` signature from:

```rust
pub fn handle_export(
    session: &Session,
    request: &Request,
    trace_log: &TraceLog,
    tensor_store: &mut TensorStore,
) -> Response {
```

to:

```rust
pub fn handle_export(
    session: &Session,
    request: &Request,
    trace_log: &TraceLog,
    tensor_store: &mut TensorStore,
    orchestrator: &mut crate::orchestrator_handle::OrchestratorHandle,
    perfetto_path: Option<&std::path::Path>,
) -> Response {
```

- [ ] **Step 2: Add the 5 new artifacts to the function body**

After the existing tensor collection block (after the `if req.include_tensors { ... }` block, before the `let artifact_count = artifacts.len()` line), add:

```rust
    // Collect env/model info from worker
    let model_handle = session.state().model_handle.unwrap_or(0);
    match orchestrator.export_env(&HostExportEnvRequest { model_handle }) {
        Ok(env_resp) => {
            match serde_json::to_vec_pretty(&env_resp.env) {
                Ok(b) => artifacts.push(BundleArtifact {
                    name: "env.json".into(),
                    data: b,
                }),
                Err(e) => {
                    return Response::error(
                        request.id.clone(),
                        RpcError {
                            code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                            message: format!("env serialization failed: {e}"),
                            data: None,
                        },
                    );
                }
            }
            match serde_json::to_vec_pretty(&env_resp.model_info) {
                Ok(b) => artifacts.push(BundleArtifact {
                    name: "model-info.json".into(),
                    data: b,
                }),
                Err(e) => {
                    return Response::error(
                        request.id.clone(),
                        RpcError {
                            code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                            message: format!("model-info serialization failed: {e}"),
                            data: None,
                        },
                    );
                }
            }
            let prompt_data = match &env_resp.prompt {
                Some(p) => serde_json::to_vec_pretty(p).unwrap_or_else(|_| b"null".to_vec()),
                None => b"null".to_vec(),
            };
            artifacts.push(BundleArtifact {
                name: "prompt.json".into(),
                data: prompt_data,
            });
        }
        Err(e) => {
            return Response::error(
                request.id.clone(),
                RpcError {
                    code: rocket_surgeon_protocol::jsonrpc::INTERNAL_ERROR,
                    message: format!("export_env call failed: {e}"),
                    data: None,
                },
            );
        }
    }

    // Perfetto trace
    if let Some(pf_path) = perfetto_path {
        if let Ok(data) = std::fs::read(pf_path) {
            artifacts.push(BundleArtifact {
                name: "trace.perfetto-trace".into(),
                data,
            });
        }
    }

    // Bookmarks (empty for MVP, Phase 3 populates)
    artifacts.push(BundleArtifact {
        name: "bookmarks.json".into(),
        data: b"[]".to_vec(),
    });
```

- [ ] **Step 3: Add HostExportEnvRequest to imports**

At the top of `dispatch.rs`, add `HostExportEnvRequest` to the imports from `rocket_surgeon_protocol::messages`:

Find the existing import line that has `ExportRequest, ExportResponse` and add `HostExportEnvRequest`:

```rust
    ExportRequest, ExportResponse, HostExportEnvRequest,
```

- [ ] **Step 4: Update the call site in main.rs**

In `crates/rocket-surgeon/src/main.rs`, find the line:

```rust
            handle_export(&session, &request, &trace_log, &mut tensor_store)
```

Replace with:

```rust
            handle_export(
                &session,
                &request,
                &trace_log,
                &mut tensor_store,
                orchestrator.as_mut().expect("orchestrator required for export"),
                perfetto.as_ref().map(|p| p.path()),
            )
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check -p rocket-surgeon`
Expected: Compiles. If there's a borrow conflict with `orchestrator` (since it's `Option<OrchestratorHandle>`), the `as_mut().expect()` pattern handles it.

- [ ] **Step 6: Run all tests**

Run: `cargo test -p rocket-surgeon --quiet`
Expected: All tests pass. The unit tests for `handle_export` in dispatch.rs don't use the new params (they test the function in isolation), so they'll need the signature update. If tests call `handle_export` directly, update those call sites to pass dummy values:
- For unit tests that call `handle_export` directly, they won't have an orchestrator. If such tests exist, they'll need to be updated. Check first — the existing tests may not call `handle_export` directly since it requires a real orchestrator.

- [ ] **Step 7: Commit**

```bash
git add crates/rocket-surgeon/src/dispatch.rs crates/rocket-surgeon/src/main.rs
git commit -m "feat(daemon): bundle export includes all 9 artifacts"
```

---

### Task 7: Update e2e bundle test

**Files:**
- Modify: `tests/test_e2e_bundle.py`

- [ ] **Step 1: Update the artifact validation**

In `tests/test_e2e_bundle.py`, replace the existing "Step 5: validate bundle contents" block (lines 117-133) with:

```python
            # Validate tar.gz contents
            print("\n[test] Step 5: validate bundle contents")
            assert Path(bundle_path).is_file(), f"bundle file not found: {bundle_path}"
            with tarfile.open(bundle_path, "r:gz") as tar:
                names = tar.getnames()

                required = [
                    "manifest.json",
                    "interventions.json",
                    "protocol-trace.jsonl",
                    "env.json",
                    "model-info.json",
                    "prompt.json",
                    "trace.perfetto-trace",
                    "bookmarks.json",
                ]
                for artifact in required:
                    assert artifact in names, f"missing {artifact} in {names}"

                manifest_member = tar.getmember("manifest.json")
                manifest_file = tar.extractfile(manifest_member)
                assert manifest_file is not None
                manifest_data = manifest_file.read()
                manifest = json.loads(manifest_data)
                assert "session_id" in manifest
                assert "protocol_version" in manifest
                assert manifest["protocol_version"] == "0.1.0"

                env_file = tar.extractfile(tar.getmember("env.json"))
                assert env_file is not None
                env_data = json.loads(env_file.read())
                assert "torch_version" in env_data
                assert "python_version" in env_data

                model_file = tar.extractfile(tar.getmember("model-info.json"))
                assert model_file is not None
                model_data = json.loads(model_file.read())
                assert "model_family" in model_data
                assert "num_layers" in model_data
                assert model_data["num_layers"] > 0

                bookmarks_file = tar.extractfile(tar.getmember("bookmarks.json"))
                assert bookmarks_file is not None
                bookmarks_data = json.loads(bookmarks_file.read())
                assert bookmarks_data == []

            print(f"  {len(required)} required artifacts present")
            print("  PASS")
```

- [ ] **Step 2: Also update the artifact_count assertion**

Change line 112 from:

```python
            assert data["artifact_count"] >= 2
```

to:

```python
            assert data["artifact_count"] >= 8
```

(8 because `include_tensors: False` skips tensor files, but all other 8 artifacts should be present)

- [ ] **Step 3: Run ruff**

Run: `ruff check tests/test_e2e_bundle.py && ruff format --check tests/test_e2e_bundle.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_bundle.py
git commit -m "test(e2e): validate all 9 bundle artifacts"
```

---

### Task 8: TCK scenario for 9-artifact completeness

**Files:**
- Modify: `tck/protocol/session-export.feature`

- [ ] **Step 1: Add the scenario**

Add after the existing "Export with include_tensors defaults to true" scenario (after line 71):

```gherkin
  Scenario: Export produces all required artifacts
    When the client sends "rocket/session.export" with:
      """json
      {
        "path": "/tmp/full-bundle.tar.gz",
        "include_tensors": false
      }
      """
    Then the response status is "stopped"
    And the file "/tmp/full-bundle.tar.gz" is a valid gzip-compressed tar archive
    And the archive contains "manifest.json"
    And the archive contains "interventions.json"
    And the archive contains "protocol-trace.jsonl"
    And the archive contains "env.json"
    And the archive contains "model-info.json"
    And the archive contains "prompt.json"
    And the archive contains "trace.perfetto-trace"
    And the archive contains "bookmarks.json"
    And the response data field "artifact_count" is at least 8
```

- [ ] **Step 2: Commit**

```bash
git add tck/protocol/session-export.feature
git commit -m "tck: add 9-artifact completeness scenario for session export"
```

---

### Task 9: Run full e2e test and verify

- [ ] **Step 1: Run the e2e bundle test**

Run: `PYTHONPATH=python:tests python tests/test_e2e_bundle.py`
Expected: All steps pass, 8+ artifacts validated in bundle

- [ ] **Step 2: Run pre-commit hooks**

Run: `cargo clippy --workspace -- -D warnings && ruff check && mypy python/`
Expected: All clean

- [ ] **Step 3: Run the full test suite**

Run: `cargo test --workspace --quiet && PYTHONPATH=python pytest python/tests/ --quiet`
Expected: All pass

- [ ] **Step 4: Push**

```bash
git push origin HEAD
```
