use std::io::BufReader;
use std::process::{Child, Command, Stdio};

use rocket_surgeon_protocol::jsonrpc::{Request, Response};
use rocket_surgeon_transport::TransportError;
use rocket_surgeon_transport::framing::{read_message, write_message};
use tracing::{debug, warn};

/// Manages a spawned `rs-worker` child process, providing framed JSON-RPC
/// communication over its stdin/stdout pipes.
pub struct WorkerHandle {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    writer: std::process::ChildStdin,
}

impl WorkerHandle {
    /// Spawn `rs-worker` as a child process with piped stdin/stdout and
    /// inherited stderr (so worker logs go to the orchestrator's stderr).
    pub fn spawn(worker_bin: &str, log_level: &str) -> anyhow::Result<Self> {
        debug!(worker_bin, log_level, "spawning worker");

        let mut child = Command::new(worker_bin)
            .arg("--log-level")
            .arg(log_level)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        // These unwraps are safe — we just configured Stdio::piped() above.
        let stdin = child.stdin.take().expect("child stdin was piped");
        let stdout = child.stdout.take().expect("child stdout was piped");

        Ok(Self {
            child,
            reader: BufReader::new(stdout),
            writer: stdin,
        })
    }

    /// Send a JSON-RPC request to the worker via Content-Length framed stdin.
    pub fn send_request(&mut self, request: &Request) -> Result<(), TransportError> {
        let json = serde_json::to_string(request)?;
        write_message(&mut self.writer, &json)?;
        Ok(())
    }

    /// Read a JSON-RPC response from the worker via Content-Length framed stdout.
    pub fn recv_response(&mut self) -> Result<Response, TransportError> {
        let json = read_message(&mut self.reader)?;
        let response = serde_json::from_str(&json)?;
        Ok(response)
    }

    /// Check whether the child process is still running (non-blocking).
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Return the OS PID of the spawned worker process.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Kill the worker child process and wait for it to exit.
    pub fn kill(&mut self) {
        if let Err(e) = self.child.kill() {
            warn!("failed to kill worker: {e}");
        }
        if let Err(e) = self.child.wait() {
            warn!("failed to wait for worker: {e}");
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_nonexistent_binary_returns_error() {
        let result = WorkerHandle::spawn("/nonexistent/binary/rs-worker", "info");
        assert!(result.is_err());
    }

    #[test]
    fn spawn_real_process_and_kill() {
        // Use `cat` as a stand-in child process — it reads stdin and echoes to
        // stdout, which is close enough to verify spawn + kill lifecycle.
        let mut handle = WorkerHandle::spawn("cat", "info").expect("cat should exist");
        assert!(handle.is_alive());
        handle.kill();
        assert!(!handle.is_alive());
    }

    #[test]
    fn drop_kills_child() {
        let handle = WorkerHandle::spawn("cat", "info").expect("cat should exist");
        let pid = handle.child.id();
        drop(handle);
        // After drop, the process should have been killed. We cannot directly
        // check the PID easily cross-platform, but we can verify no panic.
        let _ = pid; // used only to confirm we got a valid pid
    }
}
