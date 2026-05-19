# ADR-0007: Breaking Wire Format Change for Shared-Memory Tensor Transport (WU 1.8)

## Status
Accepted

## Context
WU 1.8 introduces shared-memory tensor transport between the Python worker and the Rust daemon. When shared memory is available, raw tensor bytes live in a mapped region and never need to transit the JSON-RPC channel as base64. The protocol must express "the bytes are in shared memory, not inline" — which means the previously-required `data_base64` field on `CapturedTensor` must become optional.

Prior to WU 1.8, `CapturedTensor` carried:

```rust
pub data_base64: String,   // required — always present
```

Every daemon, orchestrator, and worker on `master` deserializes `CapturedTensor` with `data_base64` as a required `String`. Serde will reject any JSON object where that field is absent.

WU 1.8 changes `CapturedTensor` to:

```rust
pub tensor_id: String,                        // new — content-addressable BLAKE3 id
#[serde(skip_serializing_if = "Option::is_none")]
pub data_base64: Option<String>,              // was required String, now optional
#[serde(skip_serializing_if = "Option::is_none")]
pub shm_name: Option<String>,                 // new — shared-memory region name
#[serde(skip_serializing_if = "Option::is_none")]
pub shm_offset: Option<u64>,                  // new — byte offset within shm region
#[serde(skip_serializing_if = "Option::is_none")]
pub byte_length: Option<u64>,                 // new — tensor byte count in shm
```

Additionally, `Capabilities` gains a new required field:

```rust
pub shared_memory_supported: bool,            // new — required, non-optional
```

When shared memory is active, the worker omits `data_base64` entirely (serde's `skip_serializing_if` produces no key in the JSON). An old daemon (from `master`) receiving this response calls `serde_json::from_str` expecting a required `String` field and gets a deserialization error. The inverse also breaks: a new daemon receiving a response from an old worker will find `shared_memory_supported` missing from the capabilities handshake.

Options considered:
1. **Backward-compatible envelope**: keep `data_base64` required, set it to an empty string or sentinel value (`"SHM"`) when shared memory is used. Daemon code checks the sentinel and falls back to shm fields. This avoids the breaking change but pollutes the wire format with a meaningless required field and introduces a sentinel-value convention that must be documented and honored forever.
2. **Version negotiation**: add a protocol version field to the handshake. Daemon and worker agree on a version before exchanging tensors. Correct long-term, but WU 1.8 is the first time the wire format has changed — adding a full versioning scheme is premature when there are only two versions and no third-party consumers.
3. **Accept the break**: make `data_base64` optional, add the new fields, and require all three binaries (daemon, orchestrator, worker) to be deployed atomically from the same build. Document the rollback constraint.

## Decision
**Accept the breaking wire format change (option 3).**

`CapturedTensor.data_base64` becomes `Option<String>` with `skip_serializing_if = "Option::is_none"`. New fields (`tensor_id`, `shm_name`, `shm_offset`, `byte_length`) are added. `Capabilities` gains a required `shared_memory_supported: bool`.

Justification:
- rocket_surgeon has no third-party protocol consumers. All three binaries (daemon, orchestrator, worker) are built from the same repo and deployed together. There is no released stable API to preserve.
- The sentinel-value approach (option 1) adds permanent protocol debt to avoid a constraint that already exists in practice — the three binaries must be version-matched because they share Rust types.
- Full version negotiation (option 2) is the right call when we have external consumers or a stable release. We are not there yet. When we reach that point, we will introduce protocol versioning as a dedicated ADR.
- The shared-memory path is the primary transport for tensor data going forward. Making `data_base64` genuinely optional (absent from JSON, not present-but-empty) gives serde a clean model and avoids downstream code checking for sentinel strings.

## Consequences
- **Good**: Clean wire format. `data_base64` is present when the tensor travels inline; absent when it lives in shared memory. No sentinel values, no dead fields.
- **Good**: New fields (`tensor_id`, `shm_name`, `shm_offset`, `byte_length`) express shared-memory location directly in the type system. Serde enforces the invariant at deserialization time.
- **Good**: `Capabilities.shared_memory_supported` lets the daemon know at handshake time whether the worker can use shm, before any tensor exchange occurs.
- **Bad**: **Rollback requires atomic binary swap.** You cannot roll back to `master` daemon while keeping the WU 1.8 worker (or vice versa). All three binaries — daemon, orchestrator, worker — must be swapped together. A partial rollback (e.g., new worker + old daemon) causes deserialization failures on every inspect response containing a shared-memory tensor.
- **Bad**: Any future branch that cherry-picks commits across the WU 1.8 boundary must carry the full wire format change or nothing. Partial cherry-picks will produce binaries that cannot communicate.
- **Risk**: If we later need to support mixed-version deployments (e.g., rolling upgrades in a multi-node setup), we will need to retrofit protocol version negotiation. This ADR accepts that as future work, not a WU 1.8 requirement.
