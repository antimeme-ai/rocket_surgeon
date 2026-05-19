---
topic: "Tensor Encoding & Wire Format — SOTA approaches for moving tensor data between processes without base64/JSON overhead"
status: complete
created: 2026-05-19
sources:
  - "TZC: Efficient Inter-Process Communication for Robotics Middleware (Kronauer et al., 2018) — 1810.00556"
  - "Zerrow: True Zero-Copy Arrow Pipelines in Bauplan (Dai et al., 2025) — 2504.06151"
  - "Speeding up Model Loading with fastsafetensors (Yoshimura et al., 2025) — 2505.23072"
  - "Ray: A Distributed Framework for Emerging AI Applications (Moritz et al., 2018) — 1712.05889"
  - "Leveraging Apache Arrow for Zero-copy Cluster Shared Memory (Groet et al., 2024) — 2404.03030"
  - "Model CBOR Serialization for Federated Learning (Zandberg et al., 2024) — 2401.14056"
  - "Which Quantization Should I Use? GGUF Evaluation (Kurt, 2026) — 2601.14277"
  - "nnterp: A Standardized Interface for Mechanistic Interpretability (Dumas, 2025) — 2511.14465"
  - "safetensors Rust source (HuggingFace, 2024)"
  - "shmem-ipc Rust crate (Linux memfd + eventfd SPSC ring buffer)"
  - "Prior lit review: shared-memory-tensor-handoff.md (2026-05-18)"
related_wus: ["WU 1.8"]
---

# Tensor Encoding & Wire Format: Literature Review

## 1. Problem Statement

rocket_surgeon currently transfers tensor data via:

```
tensor.detach().contiguous().cpu().numpy().tobytes()
  → base64 encode (worker, Rust)
  → JSON string in Content-Length-framed JSON-RPC
  → base64 decode (daemon, Rust)
  → raw bytes → TensorStore (BLAKE3 content-addressable, LRU)
```

base64 inflates data by 33%. JSON framing adds further overhead. A 64 MB activation tensor becomes ~89 MB on the wire, plus JSON parsing and string allocation costs. The entire pipeline is synchronous — the daemon blocks on JSON parsing before it can access tensor bytes.

WU 1.8 replaces this with a shared-memory data plane: tensor bytes flow through shared memory (one memcpy), with only a small control message on the JSON-RPC channel.

This review surveys how the field handles this problem.


## 2. The TZC Pattern: Split Control and Data Planes

**Paper**: Kronauer et al., "TZC: Efficient Inter-Process Communication for Robotics Middleware" (2018)

**Core insight**: Split every IPC message into two parts:
1. **Control part** — small metadata (type, shape, offset, size) sent via the existing socket/RPC channel
2. **Data part** — bulk bytes placed directly in shared memory, never serialized

TZC achieves constant IPC latency regardless of message size. For a 4 MB image message, TZC reduces transfer from tens of ms to hundreds of μs. The control message is ~100 bytes regardless of data size.

**Key design decisions**:
- Shared memory region pre-allocated at session start, sized for worst-case message
- Producer writes data into shm slot, then sends control message with (offset, length) over the existing transport
- Consumer reads control message, then accesses data directly from shm via mmap — zero copy on the read side
- One memcpy on the write side (producer copies into shm) is unavoidable unless the producer can allocate directly in shm

**Relevance to WU 1.8**: This is exactly our architecture. The JSON-RPC channel carries `CapturedTensor` control messages (tensor_id, shape, dtype, device, shm_offset, byte_length). The shared memory ring buffer carries raw bytes. The daemon reads control metadata from JSON-RPC and tensor bytes from mmap.


## 3. Shared Memory Object Stores

### 3.1 Ray/Plasma (Moritz et al., 2018)

Ray's object store (Plasma) is the seminal implementation of shared-memory tensor transfer for ML:

- **Architecture**: Per-node object store using shared memory (Apache Arrow format). Immutable objects. Zero-copy between tasks on the same node via mmap.
- **Performance**: 15+ GB/s write throughput for large objects from a single client. 18K IOPS for small objects.
- **Key principle**: Data plane (object store, shared memory) is completely separated from control plane (Global Control Store, Redis-based). The scheduler never touches object data.
- **Immutability**: Once written, objects are immutable. This eliminates synchronization concerns — readers never see partial writes. Content is identified by ObjectID (hash).
- **Eviction**: LRU with pinning. Objects in use are pinned; eviction only targets unpinned objects.

**Relevance**: Our TensorStore already follows the same pattern (content-addressable by BLAKE3, immutable after insert, LRU eviction). The missing piece is the shared-memory transport — currently we serialize through JSON instead of using a Plasma-like data plane.

### 3.2 Zerrow (Dai et al., 2025)

Zerrow pushes beyond Plasma by eliminating even the write-side copy:

- **KernelZero**: Custom Linux kernel module that *de-anonymizes* memory — converts anonymous (malloc'd) memory regions to file-backed memory without copying. The process's page table entries are remapped to point to tmpfs files.
- **SIPC (Shared IPC)**: Extends Arrow IPC to write references (file_id, offset, length) instead of data. Reader reconstructs Arrow tables via mmap of the referenced files.
- **IPC inspection**: Detects when outputs overlap with inputs (e.g., column selection, slicing). Writes references to input files instead of copying data.
- **Results**: 2.8x throughput improvement for shared deserialization. Write-side copy elimination halves latency for Parquet→Arrow conversion.

**Relevance**: KernelZero requires a custom kernel module — too heavy for our use case and Linux-only. But the *architectural pattern* of sending references instead of data is exactly what TZC prescribes and what we'll implement. Our "reference" is (shm_name, offset, byte_length) in the JSON-RPC control message.

**Key takeaway from Zerrow**: "The term 'zero-copy', as commonly used in the data ecosystem, does not actually mean no copying." True zero-copy requires kernel support for de-anonymizing memory. For our purposes, one memcpy (tensor buffer → shared memory) is the practical floor.


## 4. Tensor File Formats as Design References

### 4.1 safetensors (HuggingFace)

Format layout:
```
[8 bytes: header_size as u64 LE]
[header_size bytes: JSON metadata {"tensor_name": {dtype, shape, data_offsets: [begin, end]}, ...}]
[padding to alignment boundary]
[raw tensor bytes, concatenated, at offsets specified in header]
```

- ~900 lines of Rust. No arbitrary code execution (unlike Pickle).
- Zero-copy loading via mmap: parse 8-byte size → parse JSON header → mmap body → slice into tensors by offset.
- Alignment: header padded so tensor data starts at page boundary. Individual tensors may not be aligned within the body.

**Relevance**: The safetensors header pattern (small structured metadata + raw bytes at known offsets) maps directly to our control message + shared memory design. Our "header" is the JSON-RPC `CapturedTensor` message; our "body" is the shared memory region.

### 4.2 fastsafetensors (Yoshimura et al., 2025)

Optimizes safetensors deserialization for GPU loading:

- **Aggregated tensor deserialization**: Instead of instantiating tensors one-by-one in host memory, transfers a large contiguous group of tensors from file to GPU memory in bulk, then instantiates via DLPack (wrapping raw GPU pointer as tensor object without copying).
- **GPU offloading**: Sharding, type conversion, and layout preprocessing performed on GPU (NVLink), not CPU.
- **GDS (GPUDirect Storage)**: Bypasses host CPU/memory entirely — NVMe SSD → GPU memory via DMA. 4.8–7.5x improvement over stock safetensors.
- **Key bottleneck identified**: Current safetensors instantiates tensors one-by-one in Python, hitting the GIL. fastsafetensors batches the I/O in C and uses DLPack to create tensor objects from raw pointers.

**Relevance**: The DLPack pattern is directly applicable. In our worker, we could use DLPack/`data_ptr()` to access tensor data from Rust without going through Python serialization. The prior shared-memory lit review already covers this via PyO3 `data_ptr()` access. The fastsafetensors insight about batching I/O to avoid per-tensor Python overhead is relevant if we ever batch multiple tensor captures per step.

### 4.3 GGUF (Kurt, 2026 — evaluation paper)

GGUF format:
```
[magic: "GGUF" (4 bytes)]
[version: u32]
[tensor_count: u64]
[metadata_kv_count: u64]
[metadata key-value pairs...]
[tensor info array: {name, n_dims, dims, type, offset}...]
[padding to alignment (default 32 bytes)]
[tensor data: raw bytes at specified offsets, 16-byte aligned]
```

- Memory-mapped loading: mmap the file, tensor data accessed via pointer arithmetic.
- Block-wise quantization metadata (scale, zero-point) stored inline with quantized data in custom GGML tensor types.
- 16-byte alignment for tensor data enables SIMD access.

**Relevance**: The alignment constraint is important. Our shared memory slots should align tensor data to at least 16 bytes (for SIMD operations on the daemon side) and preferably 64 bytes (cache line). GGUF's flat "header + aligned data" layout is another instance of the same pattern as safetensors and TZC.


## 5. Alternative Wire Formats

### 5.1 CBOR (Zandberg et al., 2024)

CBOR (RFC 8949) is a binary self-describing format, analogous to "binary JSON":

- Variable-length integer encoding: small values (≤23) in 1 byte, larger values use 2–9 bytes.
- Tagged typed arrays (RFC 8742): encode homogeneous float arrays as `tag + raw bytes`, avoiding per-element overhead.
- **Results**: Up to 75% smaller than JSON for ML model parameters. For a LeNet-5 model (~45K params), CBOR is ~24% the size of JSON.
- Designed for constrained devices (microcontrollers, IoT).

**Relevance**: CBOR is strictly better than JSON for encoding tensor metadata (shape arrays, dtype tags) in our control messages. However, for bulk tensor *data*, CBOR's tagged byte string is just a thin wrapper over raw bytes — it doesn't eliminate the serialization overhead, it only reduces framing. The real win is shared memory (eliminate serialization entirely), not a better serialization format. CBOR could be useful for the control plane (replacing JSON-RPC with CBOR-RPC), but that's a separate optimization and not WU 1.8 scope.

### 5.2 Apache Arrow IPC

Arrow IPC format:
```
[schema message: flatbuffer metadata (column names, types)]
[record batch message: flatbuffer metadata + 64-byte-aligned body]
  body = [buffer 0 (null bitmap)] [buffer 1 (values)] ...
```

- 64-byte alignment for SIMD/cache-line access.
- Zero-copy on read: receiver gets pointer into mmap'd buffer, creates Arrow arrays by pointer arithmetic.
- Designed for columnar tabular data, not individual tensors.

**Relevance**: Arrow IPC is over-engineered for our use case. We're transferring individual dense tensors, not columnar record batches. The alignment requirements (64-byte) and flatbuffer metadata overhead are heavier than needed. A simpler format (fixed-size header + aligned raw bytes, like safetensors) is more appropriate.

### 5.3 Arrow Cluster Shared Memory (Groet et al., 2024)

Extends Arrow IPC across nodes using ThymesisFlow (FPGA-based memory disaggregation over OpenCAPI):

- MAP_FIXED to map shared memory at identical virtual addresses across nodes — avoids pointer relocation.
- Only table descriptors (metadata) transferred over network; data stays in shared memory.
- Measured 300ms to initialize 1 GiB table in remote memory (180ms for the actual write, rest is gRPC + cache flushing overhead).

**Relevance**: The MAP_FIXED technique is interesting for multi-node scenarios (multi-GPU across machines) but irrelevant for our single-machine three-process architecture. The "only transfer metadata" principle reinforces the TZC pattern.


## 6. The nnterp Standardization Problem

**Paper**: Dumas, "nnterp: A Standardized Interface for Mechanistic Interpretability" (2025)

Not directly about tensor encoding, but highly relevant to rocket_surgeon's known limitation of hardcoded `model.layers.N` paths:

- nnterp wraps NNsight to provide standardized module naming across 21+ architecture families (Llama, GPT-2, Mistral, Gemma, Bloom, etc.)
- Canonical structure: `embed_tokens → layers[i] → {self_attn, mlp} → ln_final → lm_head`
- Each architecture has a `RenameConfig` mapping original module names to the standard names
- Attention probabilities require `enable_attention_probs=True` and eager attention (same constraint as our `attention_pattern` view)

**Relevance**: Our adapter system (WU 1.6–1.7) already does component mapping, but our views hardcode `model.layers.N`. nnterp's approach — a per-architecture rename config — is the right pattern for generalizing this. Not WU 1.8 scope, but validates the deferred I-4 finding from WU 1.13 CR.


## 7. Synthesis: What WU 1.8 Should Build

### 7.1 Architecture (TZC Pattern)

```
WORKER (Rust + Python)                    DAEMON (Rust)
┌──────────────────────┐                  ┌──────────────────────┐
│ Hook fires:          │                  │                      │
│  tensor.cpu()        │                  │ JSON-RPC recv:       │
│  data_ptr() → &[u8]  │                  │  CapturedTensor {    │
│  BLAKE3 hash         │                  │    tensor_id,        │
│  memcpy → shm slot   │ ──(shm)──────►  │    shape, dtype,     │
│                      │                  │    shm_offset,       │
│ JSON-RPC send:       │                  │    byte_length       │
│  CapturedTensor {    │ ──(socket)────►  │  }                   │
│    tensor_id,        │                  │                      │
│    shape, dtype,     │                  │ mmap read from shm:  │
│    shm_offset,       │                  │  &[u8] → TensorStore │
│    byte_length       │                  │  (zero-copy on read) │
│  }                   │                  │                      │
└──────────────────────┘                  └──────────────────────┘
```

- **Control plane**: Existing JSON-RPC channel. `CapturedTensor` gains `shm_offset` and `byte_length` fields; `data_base64` becomes optional (fallback for when shm is unavailable).
- **Data plane**: POSIX shared memory region (`shm_open` + `mmap`), pre-allocated at attach time.
- **Notification**: Existing orchestrator relay already notifies daemon of captured tensors — no additional notification mechanism needed.

### 7.2 Format Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Wire format for bulk data | Raw bytes in shared memory | All surveyed systems converge on this. No format overhead. |
| Wire format for metadata | JSON-RPC (existing) | Control messages are small (~200 bytes). JSON overhead is noise. |
| Alignment | 64-byte aligned slots | Cache-line aligned for SIMD (BLAKE3 AVX2) and future GPU DMA |
| Content addressing | BLAKE3 hash computed in worker | Hash computed on raw `data_ptr()` bytes before memcpy. Sent with control message. Daemon skips re-hashing. |
| Tensor data layout in shm | Contiguous, C-order, native dtype | Same as `tensor.contiguous().cpu()` layout. No transposition. |
| Fallback | base64 over JSON-RPC | For environments where shm is unavailable (rare). Keep existing path as fallback. |

### 7.3 What We Learn From Each Paper

| Paper | Key Takeaway for WU 1.8 |
|-------|------------------------|
| TZC | The split control/data plane architecture. Our blueprint. |
| Ray/Plasma | Content-addressable immutable object store over shm. We already have this (TensorStore). Add the shm transport. |
| Zerrow | True zero-copy requires kernel support. One memcpy is our practical floor. Don't over-engineer. |
| fastsafetensors | DLPack / `data_ptr()` for zero-copy tensor access within a process. We use this via PyO3 in the worker. |
| safetensors | Header + aligned raw bytes. Our control message + shm slot follows this pattern. |
| GGUF | 16-byte minimum alignment for SIMD. We should use 64-byte (cache line). |
| CBOR | Better than JSON for metadata, but irrelevant when data goes through shm. Not worth the protocol change. |
| Arrow IPC | Over-engineered for individual tensors. Our simpler format is correct. |
| Arrow cluster shm | MAP_FIXED for cross-node — not applicable to single-machine. Reinforces metadata-only transfer. |
| nnterp | Standardized module naming across architectures. Validates our adapter approach, not directly WU 1.8. |

### 7.4 Latency Budget (17 MB tensor, Llama-3-8B residual)

| Step | Current (base64) | With shm (WU 1.8) |
|------|------------------|--------------------|
| GPU → CPU DMA | ~2 ms | ~2 ms (unchanged) |
| Serialization | ~8 ms (tobytes + base64) | ~0.3 ms (memcpy to shm) |
| Wire transfer | ~12 ms (89 MB JSON parse) | ~0.002 ms (control msg only) |
| Deserialization | ~6 ms (base64 decode) | ~0 ms (mmap read) |
| BLAKE3 hash | ~6 ms (daemon, on decoded bytes) | ~6 ms (worker, on raw bytes — or overlap with DMA) |
| **Total per-tensor** | **~34 ms** | **~8 ms** |

**4x improvement**, dominated by the irreducible GPU→CPU transfer and BLAKE3 hash. The serialization/deserialization overhead drops from ~26 ms to ~0.3 ms.

### 7.5 Cross-Platform Considerations

| Aspect | Linux | macOS |
|--------|-------|-------|
| shm_open | /dev/shm (tmpfs) | Kernel-managed |
| Name length | 255 chars | 31 chars (PSHMNAMLEN) |
| Page size | 4 KiB | 16 KiB (Apple Silicon) |
| memfd_create | Yes (preferred) | No (use shm_open) |
| Cross-process atomics | Works (cache coherent) | Works (cache coherent) |

Use `shm_open` for cross-platform compatibility. Keep names ≤30 chars. Align to 16 KiB (Apple Silicon page size) for mmap granularity.


## 8. Papers Preserved

All PDFs in `../papers/wu-1.8-tensor-encoding/`:

```
arrow-zero-copy-cluster-shm-2404.03030.pdf    (5 pages, read in full)
cbor-federated-learning-2401.14056.pdf         (6 pages, read in full)
fastsafetensors-2505.23072.pdf                 (18 pages, read in full)
gguf-quantization-eval-2601.14277.pdf          (10 pages, read in full)
nnterp-standardized-interp-2511.14465.pdf      (7 pages, read in full)
ray-distributed-framework-1712.05889.pdf       (14 pages, read in full)
tzc-partial-serialization-1810.00556.pdf       (8 pages, read in full)
zerrow-arrow-pipelines-2504.06151.pdf          (12 pages, read in full)
```
