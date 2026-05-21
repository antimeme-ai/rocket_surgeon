//! Async JSON-RPC client for the daemon link.
//!
//! `connection` is wired into the event loop via `daemon.rs` (BEAD-0015
//! slice 2). `ReconnectingClient` / `ConnectFn` and the whole `subscription`
//! module are retained for the reconnection slice and carry targeted
//! `dead_code` allowances until then.

pub mod connection;

#[allow(dead_code)]
pub mod subscription;
