# ADR-0006: Tensor Handling — Content-Addressable IDs and Summary-Then-Slice Protocol

## Status
Proposed

## Context
rocket_surgeon's protocol must transfer tensor data between the Python model host, the Rust daemon, and clients (TUI, LLM scripts). Tensors in transformer forward passes range from small (attention bias, a few KB) to enormous (attention matrices at 128K context, potentially hundreds of GB). Naive transfer of full tensors would be catastrophic for LLM clients (accidentally consuming 10 GB in a response), for TUI responsiveness (blocking on multi-second transfers), and for session bundle size (storing redundant copies of the same tensor). See design doc §9 for the full specification.

Key requirements:
1. **Identity**: two observations of the same tensor (e.g., the same residual stream captured at two different probe points) must be recognizable as identical without comparing all bytes at the client level.
2. **Lazy materialization**: clients should never receive full tensor bytes unless they explicitly ask for them. Summaries must be rich enough for most debugging workflows.
3. **Bounded responses**: no single protocol response should be unbounded. LLM clients have context windows; TUI rendering has frame budgets.
4. **Efficient handoff**: tensor bytes must move from GPU (Python host) to the daemon (Rust) with minimal copying and no serialization overhead.

Options considered for tensor identity:
1. **UUID per observation**: simple, unique, but two observations of the same tensor get different IDs. No dedup, no cache reuse.
2. **Content hash (BLAKE3)**: `tensor_id = blake3(raw_bytes)`. Same bytes = same ID regardless of where or when observed. Enables dedup in the tensor store, in session bundles, and in TUI display.
3. **Content hash (SHA-256)**: correct but 3-10x slower than BLAKE3 for large tensors. BLAKE3 achieves >10 GB/s on modern CPUs.

Options considered for data transfer protocol:
1. **Full tensor in every response**: simple but unbounded. A single `[1, 32, 2048, 2048]` fp16 attention matrix is 256 MB. Rejected.
2. **Summary only, full data on separate request**: two round-trips for drill-down. Acceptable latency for interactive use.
3. **Summary always, slice on demand with size cap**: summary (~200 bytes) returned with every inspection. Bounded slices (up to 64 KB) on explicit request. Full tensors via streaming for export only.

Options considered for Python-to-Rust tensor handoff:
1. **JSON-RPC with base64-encoded bytes**: simple but slow. Base64 encoding adds 33% overhead plus JSON parsing cost. Unacceptable for multi-megabyte tensors.
2. **Shared-memory ring buffer**: Python writes tensor bytes directly into a memory-mapped region. Rust reads zero-copy. Notification via a lightweight signal channel. Fast, but requires platform-specific shared memory management.
3. **Unix socket with sendmsg/SCM_RIGHTS**: pass file descriptors for memory-mapped tensors. More complex, less portable.

## Decision
**BLAKE3 content-addressable tensor IDs. Summary-then-slice protocol with 64 KB response cap. Shared-memory ring buffer with ProbeFrame format for Python-to-Rust handoff. Unix domain socket for frame notification.**

### Content-addressable identity

`tensor_id = blake3(raw_bytes)` — 32 bytes, hex-encoded (64 characters).

The hash is computed on the raw contiguous tensor bytes after `detach().contiguous()` and CPU transfer. The same tensor observed at two different probe points, or in two different ticks, produces the same `tensor_id`. This enables:
- **Dedup in the tensor store**: the daemon stores each unique tensor once, regardless of how many probes captured it.
- **Dedup in session bundles**: the `tensors/` directory in a bundle contains one safetensors file per unique tensor, not per observation.
- **Cache-friendly inspection**: the daemon caches summary statistics by `tensor_id`. Repeated inspections of the same tensor hit the cache.
- **TUI cross-referencing**: the TUI can highlight "this is the same tensor you saw at layer 10" when the same `tensor_id` appears at layer 15.

BLAKE3 is chosen over SHA-256 for speed: >10 GB/s on modern CPUs, with a Rust-native implementation (`blake3` crate) and Python bindings. The PyO3 bridge exposes `rs.blake3_hash(bytes)` for consistency between the Python host and the Rust daemon.

### Summary-then-slice protocol

Every tensor inspection follows a two-phase pattern:

**Phase 1 — Summary (always returned, ~200 bytes)**:
```json
{
  "tensor_id": "a1b2c3...",
  "shape": [1, 32, 2048, 128],
  "dtype": "float16",
  "device": "cuda:0",
  "sharding": null,
  "stats": {
    "mean": 0.0012,
    "std": 0.342,
    "min": -4.21,
    "max": 3.87,
    "abs_max": 4.21,
    "sparsity": 0.023,
    "l2_norm": 14.7,
    "histogram": { "bins": 32, "edges": ["..."], "counts": ["..."] }
  },
  "top_k": [
    { "index": [0, 7, 1024, 42], "value": 4.21 },
    { "index": [0, 3, 512, 99], "value": -3.98 }
  ]
}
```

Summary statistics (mean, std, min, max, abs_max, l2_norm, sparsity, 32-bin histogram, top-8-by-abs) are computed on-GPU before CPU transfer. These are single-reduction operations — cheap relative to the forward-pass computation that produced the tensor.

**Phase 2 — Slice (on demand, bounded)**:

Clients request specific index ranges. Response size is capped at 64 KB per slice request. Larger regions must be paginated. Requests exceeding the cap receive a `RESPONSE_TOO_LARGE` error with the actual size, allowing the client to subdivide.

Attention matrices at long contexts (128K) are never fully materialized. The protocol enforces row-only access for matrices exceeding 1 GB unless the client sets `--allow-large` explicitly.

### Shared-memory ring buffer (ProbeFrame format)

Tensor bytes move from Python host to Rust daemon via a shared-memory ring buffer, not through the JSON-RPC channel.

**Layout**: Each slot contains a ProbeFrame record — a 128-byte fixed header followed by raw tensor bytes:

```
ProbeFrame header (128 bytes):
┌─────────┬──────┬───────┬──────┬───────┬─────────┐
│ rank:u32│layer │comp_id│dtype │ndim:u8│shape    │
│         │:u32  │:u16   │:u8   │       │:[u32;8] │
├─────────┼──────┴───────┴──────┴───────┴─────────┤
│ tick_id │ offset:u64  │ size:u64  │ flags:u32   │
│ :u64    │             │           │             │
└─────────┴─────────────┴───────────┴─────────────┘
[raw tensor bytes at offset...]
```

**Allocation**: Python allocates the shared-memory region at host startup using `multiprocessing.shared_memory` (cross-platform: works on Linux, macOS, and Windows). Region name follows the convention `/rs-<session>-<rank>`.

**Write path** (Python host):
1. `tensor.detach().contiguous().to('cpu', non_blocking=True)`
2. `torch.cuda.Event` record + synchronize on the current stream (scoped, not global)
3. `memcpy` into ring slot
4. Publish slot index via notification channel

**Read path** (Rust daemon):
Rust reads the slot zero-copy via mmap, builds a `TensorRef` (metadata + `&[u8]` view), computes the BLAKE3 hash, and stores the result in the tensor handle store.

**Notification**: Frame availability is signaled via a Unix domain socket auxiliary channel (single byte write per frame). This replaces the Linux-only `eventfd` mechanism to ensure cross-platform operation — both macOS and Linux support Unix domain sockets. The notification is minimal (one byte per frame) and adds negligible latency.

**Ring buffer lifecycle**: Slots are reused after the Rust consumer acknowledges processing (read the frame, computed the hash, stored the handle). Acknowledgment flows back over the same Unix domain socket.

## Consequences
- **Good**: Content-addressable IDs eliminate redundant storage and transfer. A 32-layer model where layers 5 and 15 happen to produce identical residual norms stores the tensor once. Session bundles are smaller. Cache hit rates are higher.
- **Good**: Summary-then-slice means an LLM client can inspect every layer's residual stream in a single pass — 32 inspections, each returning ~200 bytes — without ever materializing a full tensor. Total cost: ~6.4 KB for a complete forward-pass overview.
- **Good**: The 64 KB response cap is a hard safety net against runaway responses. An LLM with a 128K context window cannot accidentally consume its entire context with one tensor.
- **Good**: Shared-memory ring buffer achieves near-memcpy throughput for tensor handoff. No serialization, no base64, no JSON encoding of bytes. For a 17 MB residual stream (Llama-3-8B), the overhead is dominated by GPU-to-CPU transfer (~2 ms), not by the IPC mechanism.
- **Good**: Unix domain socket notification is cross-platform (Linux + macOS) and adds negligible latency (~1 us per notification). No Linux-only dependencies in the data path.
- **Bad**: BLAKE3 hashing adds CPU cost proportional to tensor size. For a 17 MB tensor, BLAKE3 takes ~1.5 ms on a modern CPU. This is acceptable given that GPU-to-CPU transfer already takes ~2 ms, but it doubles the host-side latency per capture. Mitigated by computing the hash in a background thread (or via PyO3 with GIL released).
- **Bad**: Shared-memory ring buffer requires careful lifecycle management — the Python host must not crash between writing tensor bytes and publishing the notification, or the slot is leaked until the ring wraps. Mitigated by the daemon monitoring host process liveness and reclaiming the shared region on host crash.
- **Bad**: The 64 KB cap means inspecting a large tensor requires multiple round-trips. For TUI drill-down this is acceptable (progressive rendering); for bulk export, the session bundle path bypasses the cap by writing directly to safetensors files.
- **Risk**: BLAKE3 hash collisions. BLAKE3 produces 256-bit hashes; collision probability is negligible for any realistic number of tensors. Not a practical concern, but noted for completeness.
