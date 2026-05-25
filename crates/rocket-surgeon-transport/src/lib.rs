#![forbid(unsafe_code)]

pub mod framing;
pub mod stdio;

use rocket_surgeon_protocol::jsonrpc::{Request, Response};

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("framing error: {0}")]
    Framing(#[from] framing::FramingError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("transport closed")]
    Closed,
}

pub trait Transport {
    fn send_request(&mut self, request: &Request) -> Result<(), TransportError>;
    fn send_response(&mut self, response: &Response) -> Result<(), TransportError>;
    fn recv_request(&mut self) -> Result<Request, TransportError>;
    fn recv_response(&mut self) -> Result<Response, TransportError>;
}
