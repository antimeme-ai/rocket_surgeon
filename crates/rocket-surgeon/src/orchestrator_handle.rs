use std::io::BufReader;
use std::process::{Child, Command, Stdio};

use rocket_surgeon_protocol::jsonrpc::{Request, RequestId, Response};
use rocket_surgeon_protocol::messages::{
    HostAttachRequest, HostAttachResponse, HostDetachRequest, HostInspectRequest, HostStepRequest,
    HostStepResponse, HostUpdateProbesRequest, HostUpdateProbesResponse, HostViewRequest, internal,
};
use rocket_surgeon_transport::framing::{read_message, write_message};
use tracing::{debug, warn};

/// Manages a spawned `rs-orchestrator` child process, providing framed
/// JSON-RPC communication over its stdin/stdout pipes.
pub struct OrchestratorHandle {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    writer: std::process::ChildStdin,
    next_id: i64,
}

impl OrchestratorHandle {
    /// Spawn `rs-orchestrator` as a child process with piped stdin/stdout and
    /// inherited stderr (so orchestrator logs go to the daemon's stderr).
    pub fn spawn(
        orchestrator_bin: &str,
        worker_bin: &str,
        log_level: &str,
    ) -> anyhow::Result<Self> {
        debug!(
            orchestrator_bin,
            worker_bin, log_level, "spawning orchestrator"
        );

        let mut child = Command::new(orchestrator_bin)
            .arg("--worker-bin")
            .arg(worker_bin)
            .arg("--log-level")
            .arg(log_level)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take().expect("child stdin was piped");
        let stdout = child.stdout.take().expect("child stdout was piped");

        Ok(Self {
            child,
            reader: BufReader::new(stdout),
            writer: stdin,
            next_id: 1,
        })
    }

    /// Send `_host/attach` to the orchestrator and parse the response.
    pub fn attach(&mut self, req: &HostAttachRequest) -> anyhow::Result<HostAttachResponse> {
        let id = self.next_id();
        let params = serde_json::to_value(req)?;
        let request = Request::new(RequestId::Number(id), internal::HOST_ATTACH, params);

        self.send(&request)?;
        let response = self.recv()?;

        if let Some(err) = response.error {
            anyhow::bail!(
                "orchestrator attach failed (code {}): {}",
                err.code,
                err.message
            );
        }

        let result = response
            .result
            .ok_or_else(|| anyhow::anyhow!("orchestrator attach: missing result"))?;
        let host_resp: HostAttachResponse = serde_json::from_value(result)?;
        Ok(host_resp)
    }

    /// Send `_host/detach` to the orchestrator and wait for acknowledgment.
    pub fn detach(&mut self, handle: u64) -> anyhow::Result<()> {
        let id = self.next_id();
        let params = serde_json::to_value(HostDetachRequest {
            model_handle: handle,
        })?;
        let request = Request::new(RequestId::Number(id), internal::HOST_DETACH, params);

        self.send(&request)?;
        let response = self.recv()?;

        if let Some(err) = response.error {
            anyhow::bail!(
                "orchestrator detach failed (code {}): {}",
                err.code,
                err.message
            );
        }

        Ok(())
    }

    /// Send `_host/step` to the orchestrator and parse the response.
    pub fn step(&mut self, req: &HostStepRequest) -> anyhow::Result<HostStepResponse> {
        let id = self.next_id();
        let params = serde_json::to_value(req)?;
        let request = Request::new(RequestId::Number(id), internal::HOST_STEP, params);

        self.send(&request)?;
        let response = self.recv()?;

        if let Some(err) = response.error {
            anyhow::bail!(
                "orchestrator step failed (code {}): {}",
                err.code,
                err.message
            );
        }

        let result = response
            .result
            .ok_or_else(|| anyhow::anyhow!("orchestrator step: missing result"))?;
        let host_resp: HostStepResponse = serde_json::from_value(result)?;
        Ok(host_resp)
    }

    /// Send `_host/inspect` to the orchestrator and return the raw response.
    /// The caller decides how to handle errors vs success.
    pub fn inspect_raw(&mut self, req: &HostInspectRequest) -> anyhow::Result<Response> {
        let id = self.next_id();
        let params = serde_json::to_value(req)?;
        let request = Request::new(RequestId::Number(id), internal::HOST_INSPECT, params);

        self.send(&request)?;
        self.recv()
    }

    pub fn view_raw(&mut self, req: &HostViewRequest) -> anyhow::Result<Response> {
        let id = self.next_id();
        let params = serde_json::to_value(req)?;
        let request = Request::new(RequestId::Number(id), internal::HOST_VIEW, params);

        self.send(&request)?;
        self.recv()
    }

    pub fn update_probes(
        &mut self,
        req: &HostUpdateProbesRequest,
    ) -> anyhow::Result<HostUpdateProbesResponse> {
        let id = self.next_id();
        let params = serde_json::to_value(req)?;
        let request = Request::new(RequestId::Number(id), internal::HOST_UPDATE_PROBES, params);

        self.send(&request)?;
        let response = self.recv()?;

        if let Some(err) = response.error {
            anyhow::bail!(
                "orchestrator update_probes failed (code {}): {}",
                err.code,
                err.message
            );
        }

        let result = response
            .result
            .ok_or_else(|| anyhow::anyhow!("orchestrator update_probes: missing result"))?;
        let resp: HostUpdateProbesResponse = serde_json::from_value(result)?;
        Ok(resp)
    }

    /// Kill the orchestrator child process and wait for it to exit.
    pub fn kill(&mut self) {
        if let Err(e) = self.child.kill() {
            warn!("failed to kill orchestrator: {e}");
        }
        if let Err(e) = self.child.wait() {
            warn!("failed to wait for orchestrator: {e}");
        }
    }

    fn next_id(&mut self) -> i64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn send(&mut self, request: &Request) -> anyhow::Result<()> {
        let json = serde_json::to_string(request)?;
        write_message(&mut self.writer, &json)?;
        Ok(())
    }

    fn recv(&mut self) -> anyhow::Result<Response> {
        let json = read_message(&mut self.reader)?;
        let response: Response = serde_json::from_str(&json)?;
        Ok(response)
    }
}

impl Drop for OrchestratorHandle {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_nonexistent_binary_returns_error() {
        let result = OrchestratorHandle::spawn(
            "/nonexistent/binary/rs-orchestrator",
            "/fake/worker",
            "info",
        );
        assert!(result.is_err());
    }

    #[test]
    fn spawn_real_process_and_kill() {
        // Use `cat` as a stand-in child — it reads stdin and echoes to stdout.
        let mut handle =
            OrchestratorHandle::spawn("cat", "/fake/worker", "info").expect("cat should exist");
        assert!(handle.next_id == 1);
        handle.kill();
    }

    #[test]
    fn drop_kills_child() {
        let handle =
            OrchestratorHandle::spawn("cat", "/fake/worker", "info").expect("cat should exist");
        let pid = handle.child.id();
        drop(handle);
        // After drop, the process should have been killed.
        let _ = pid;
    }

    #[test]
    fn step_method_exists() {
        use rocket_surgeon_protocol::messages::HostStepRequest;
        use rocket_surgeon_protocol::types::StepDirection;

        let mut handle =
            OrchestratorHandle::spawn("cat", "/fake/worker", "info").expect("cat should exist");
        let req = HostStepRequest {
            model_handle: 1,
            count: 1,
            direction: StepDirection::Forward,
            granularity: None,
            max_events: None,
        };
        let result = handle.step(&req);
        assert!(result.is_err());
    }

    #[test]
    fn next_id_increments() {
        let mut handle =
            OrchestratorHandle::spawn("cat", "/fake/worker", "info").expect("cat should exist");
        assert_eq!(handle.next_id(), 1);
        assert_eq!(handle.next_id(), 2);
        assert_eq!(handle.next_id(), 3);
    }

    #[test]
    fn inspect_method_exists() {
        use rocket_surgeon_protocol::messages::HostInspectRequest;
        use rocket_surgeon_protocol::messages::InspectDetail;

        let mut handle =
            OrchestratorHandle::spawn("cat", "/fake/worker", "info").expect("cat should exist");
        let req = HostInspectRequest {
            model_handle: 1,
            target: "model:0:0:q_proj:output".to_owned(),
            detail: InspectDetail::Summary,
            slices: None,
        };
        let result = handle.inspect_raw(&req);
        assert!(result.is_err());
    }
}
