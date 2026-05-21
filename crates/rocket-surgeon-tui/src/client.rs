// In-flight scaffolding: the async JSON-RPC client (`connection`,
// `subscription`) is fully unit-tested but not yet wired into the `main.rs`
// event loop — that integration is tracked as separate work. Every item here
// is exercised by this module's own `#[cfg(test)]` suites, so the bin-only
// `dead_code` lint is a false positive against intentional, tested API.
#![allow(dead_code)]

pub mod connection;
pub mod subscription;
