//! Stdio transport — Content-Length framed JSON-RPC over BufRead/Write pairs.
//!
//! Generic over reader (`R: BufRead`) and writer (`W: Write`) so it can wrap:
//! - `BufReader<ChildStdout>` + `ChildStdin`  (parent talking to spawned child)
//! - `BufReader<StdinLock>` + `StdoutLock`    (child talking to parent)
//! - `Cursor<Vec<u8>>` + `Vec<u8>`            (unit tests, no I/O)

use std::io::{BufRead, BufReader, Write};

use rocket_surgeon_protocol::jsonrpc::{Request, Response};

use crate::{
    Transport, TransportError,
    framing::{read_message, write_message},
};

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

pub struct StdioTransport<R: BufRead, W: Write> {
    reader: R,
    writer: W,
}

impl<R: BufRead, W: Write> StdioTransport<R, W> {
    pub fn new(reader: R, writer: W) -> Self {
        Self { reader, writer }
    }
}

// ---------------------------------------------------------------------------
// Transport impl
// ---------------------------------------------------------------------------

impl<R: BufRead, W: Write> Transport for StdioTransport<R, W> {
    fn send_request(&mut self, request: &Request) -> Result<(), TransportError> {
        let json = serde_json::to_string(request)?;
        write_message(&mut self.writer, &json)?;
        Ok(())
    }

    fn send_response(&mut self, response: &Response) -> Result<(), TransportError> {
        let json = serde_json::to_string(response)?;
        write_message(&mut self.writer, &json)?;
        Ok(())
    }

    fn recv_request(&mut self) -> Result<Request, TransportError> {
        let json = read_message(&mut self.reader)?;
        let request = serde_json::from_str(&json)?;
        Ok(request)
    }

    fn recv_response(&mut self) -> Result<Response, TransportError> {
        let json = read_message(&mut self.reader)?;
        let response = serde_json::from_str(&json)?;
        Ok(response)
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor for the parent-side (talking to a spawned child)
// ---------------------------------------------------------------------------

/// Wrap a child process's stdin/stdout pair in a `StdioTransport`.
///
/// # Arguments
/// * `stdin`  — writable end (`ChildStdin`) connected to the child's stdin
/// * `stdout` — readable end (`ChildStdout`) connected to the child's stdout
pub fn from_child_pipes(
    stdin: std::process::ChildStdin,
    stdout: std::process::ChildStdout,
) -> StdioTransport<BufReader<std::process::ChildStdout>, std::process::ChildStdin> {
    StdioTransport::new(BufReader::new(stdout), stdin)
}

// ---------------------------------------------------------------------------
// Tests (JSMNTL: written before implementation, run red first)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use rocket_surgeon_protocol::jsonrpc::{Request, RequestId, Response};
    use serde_json::json;

    use super::StdioTransport;
    use crate::{Transport, framing::write_message};

    // ---------------------------------------------------------------------------
    // 1. request_round_trip
    // ---------------------------------------------------------------------------

    /// Write a Request into the transport's writer, then read it back through
    /// its reader and verify method + id survive the round-trip.
    #[test]
    fn request_round_trip() {
        // Build the wire bytes first so we can seed the reader.
        let req = Request::new(
            RequestId::Number(42),
            "initialize",
            json!({"clientInfo": {"name": "test"}}),
        );
        let json = serde_json::to_string(&req).unwrap();
        let mut wire: Vec<u8> = Vec::new();
        write_message(&mut wire, &json).unwrap();

        // Create transport with the pre-written bytes as reader input.
        let mut transport = StdioTransport::new(Cursor::new(wire), Vec::<u8>::new());

        let received = transport.recv_request().unwrap();
        assert_eq!(received.method, "initialize");
        assert_eq!(received.id, RequestId::Number(42));
    }

    // ---------------------------------------------------------------------------
    // 2. response_round_trip
    // ---------------------------------------------------------------------------

    /// Write a Response into the transport's writer, then read it back through
    /// its reader and verify id + result survive the round-trip.
    #[test]
    fn response_round_trip() {
        let resp = Response::success(RequestId::Number(7), json!({"capabilities": {}}));
        let json = serde_json::to_string(&resp).unwrap();
        let mut wire: Vec<u8> = Vec::new();
        write_message(&mut wire, &json).unwrap();

        let mut transport = StdioTransport::new(Cursor::new(wire), Vec::<u8>::new());

        let received = transport.recv_response().unwrap();
        assert_eq!(received.id, RequestId::Number(7));
        assert_eq!(
            received.result.as_ref().unwrap(),
            &json!({"capabilities": {}})
        );
    }

    // ---------------------------------------------------------------------------
    // 3. send_then_recv_request
    // ---------------------------------------------------------------------------

    /// Use `framing::write_message` externally to write a framed request, then
    /// call `recv_request` on a transport backed by those bytes.
    #[test]
    fn send_then_recv_request() {
        let req = Request::new(
            RequestId::String("abc".to_owned()),
            "$/ping",
            serde_json::Value::Null,
        );
        let json = serde_json::to_string(&req).unwrap();
        let mut wire: Vec<u8> = Vec::new();
        write_message(&mut wire, &json).unwrap();

        let mut transport = StdioTransport::new(Cursor::new(wire), Vec::<u8>::new());
        let received = transport.recv_request().unwrap();

        assert_eq!(received.id, RequestId::String("abc".to_owned()));
        assert_eq!(received.method, "$/ping");
        // params is None because Request::new treats Null as absent
        assert!(received.params.is_none());
    }

    // ---------------------------------------------------------------------------
    // 4. send_request_writes_valid_framing
    // ---------------------------------------------------------------------------

    /// Call `send_request` on the transport and inspect the raw bytes that end
    /// up in the writer: must start with "Content-Length: " and must contain
    /// the method name somewhere in the body.
    #[test]
    fn send_request_writes_valid_framing() {
        let req = Request::new(
            RequestId::Number(1),
            "workspace/didChangeConfiguration",
            json!({"settings": {}}),
        );

        // Transport with an empty reader (we only care about what was written).
        let mut transport = StdioTransport::new(Cursor::new(Vec::<u8>::new()), Vec::<u8>::new());
        transport.send_request(&req).unwrap();

        // Destructure to get the writer out.
        let StdioTransport { writer, .. } = transport;
        let output = String::from_utf8(writer).expect("output must be valid UTF-8");

        assert!(
            output.starts_with("Content-Length: "),
            "output must open with Content-Length header; got: {output:?}"
        );
        assert!(
            output.contains("workspace/didChangeConfiguration"),
            "method name must appear in output; got: {output:?}"
        );
    }
}
