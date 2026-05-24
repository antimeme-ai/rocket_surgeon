# Checkpoint State Tier — Design Spec

Date: 2026-05-24
Phase: 3, Sub-project A

## Goal

Wire up worker-side tensor capture and storage so the daemon's existing
checkpoint metadata tier has actual activation data behind it. Zero-copy
arena architecture — Rust owns memory, CUDA DMA lands directly in Rust's
address space via `torch.frombuffer`.

## Scope

Sub-project A of three in Phase 3:

| Sub-project | Scope |
|-------------|-------|
| **A (this)** | Checkpoint arena, Python bridge, `_host/checkpoint` handlers, √L auto-checkpoint, NVMe spill |
| B | Forward replay, reverse step, divergence detection, determinism enforcement |
| C | Bundle extension with checkpoints, Tier 2 Python callbacks, TCK green sweep |

## Non-goals

- FullSnapshot tier (entire model state) — deferred, Activation tier only
- Forward replay from checkpoint (Sub-project B)
- Reverse step (Sub-project B)
- Divergence detection (Sub-project B)
- Op-level determinism pinning (Sub-project B) — we capture RNG state now, enforce later
- Tier 2 Python callback interventions (Sub-project C)
- Bundle extension (Sub-project C)

## Existing infrastructure

Already implemented and load-bearing — Sub-project A connects to these, does not rewrite them:

| Component | Location | Status |
|-----------|----------|--------|
| Daemon metadata tier | `session.rs:673-804` | Done: create/list/restore/delete/bookmark |
| `HostCheckpointRequest` wire type | `messages.rs:956-970` | Done: Create (with tier, tick_id, layer_idx) and Restore variants |
| `HostCheckpointResponse` wire type | `messages.rs:972-982` | Done: checkpoint_id, tier, restored_to, bytes_captured |
| Worker dispatch table | `dispatch.rs:85-104` | Missing: no `_host/checkpoint` match arm |
| `ShmRegion` mmap infrastructure | `rocket-surgeon-shm/region.rs` | Done: create, open, write, read, atomic ops |

## Architecture

### The zero-copy contract

```
GPU tensor
    │
    │  cudaMemcpyDeviceToHost (triggered by torch .copy_())
    │  + torch.cuda.synchronize() fence
    ▼
┌─────────────────────────────────────────┐
│  Checkpoint Arena                       │
│  mmap + cudaHostRegister (pinned DMA)   │
│  Rust-allocated, Rust-owned             │
│                                         │
│  ┌──────┬──────┬──────┬──────┬────────┐ │
│  │Slot 0│Slot 1│Slot 2│Slot 3│ free   │ │
│  └──────┴──────┴──────┴──────┴────────┘ │
│  ▲                                      │
│  │  Python sees this address via        │
│  │  torch.frombuffer(ctypes buffer)     │
│  │  CUDA DMA lands here at full PCIe BW │
└─────────────────────────────────────────┘
    │
    │  spill (Rust write() when arena > 80%)
    ▼
┌─────────────────────┐
│  NVMe file          │
│  [index][slot0][..] │
└─────────────────────┘
```

No intermediate CPU tensors. No Python-managed buffers. No serialization
format for in-memory data. One DMA transfer, one address space owner.

### Pinning: cudaHostRegister, not mlock

`mlock` prevents page-out to swap but does NOT register memory with the
CUDA runtime. GPU DMA to mlock'd memory goes through the slow pageable
path (staging buffer inside the driver), ~2x slower on PCIe Gen4/5.

`cudaHostRegister` registers existing host memory with CUDA, enabling
pinned DMA at full PCIe bandwidth. It also implicitly locks pages.
vLLM uses this pattern explicitly (`torch.cuda.cudart().cudaHostRegister`).

Flow:
1. Rust: anonymous `mmap` with `MAP_POPULATE` (pre-fault pages)
2. Rust: pass `(ptr, len)` to Python via PyO3
3. Python: `torch.cuda.cudart().cudaHostRegister(ptr, len, 0)` — register with CUDA
4. On arena drop: Python `cudaHostUnregister(ptr)`, then Rust `munmap`

Fallback: if `cudaHostRegister` returns non-zero (e.g., no GPU, test
environment), proceed with unpinned mmap. Correctness is unaffected,
DMA uses the pageable path.

### Slot layout

Each slot in the arena is a fixed 64-byte header followed by tensor bytes:

```
Offset  Size    Field
0       4       magic: 0x434B5054 ("CKPT")
4       1       dtype (enum: 0=f16, 1=bf16, 2=f32, 3=f64)
5       1       ndim (1-8)
6       2       reserved
8       48      shape[6] as u64 (max 6 dimensions)
56      8       byte_len (actual tensor bytes following header)
64      ...     tensor data (byte_len bytes, 64-byte aligned)
```

Total slot size = 64 + align_up(byte_len, 64).

### Components

#### 1. CheckpointArena (Rust, new module in worker crate)

Private mmap'd region (not shared memory — worker-local). Uses anonymous
mmap + `cudaHostRegister` for CUDA-pinned pages, not `shm_open`. The
arena is not cross-process; DoomRing handles IPC.

Fixed-size slot pool — all slots are the same size (determined at
construction from `hidden_dim * max_seq_len * dtype_size`). Free list
instead of bump allocator. No fragmentation, O(1) alloc/free.

```
pub struct CheckpointArena {
    ptr: *mut u8,
    capacity: usize,
    slot_size: usize,                // fixed, computed from model dims
    num_slots: usize,
    free_list: Vec<usize>,           // indices of available slots
    index: HashMap<(String, u32), usize>,  // (checkpoint_id, layer_idx) -> slot index
    checkpoint_slots: HashMap<String, Vec<usize>>,  // checkpoint_id -> owned slot indices
}
```

Operations:
- `new(slot_size: usize, num_slots: usize) -> Result<Self>` — anonymous mmap, pre-fault with MAP_POPULATE on Linux (`#[cfg(target_os = "linux")]`; no-op on macOS)
- `register_cuda(py, ptr, len)` — call `cudaHostRegister` from Python side
- `alloc_slot(checkpoint_id, layer_idx) -> Result<(*mut u8, &SlotHeader)>` — pop from free list, write header
- `get_slot(checkpoint_id, layer_idx) -> Option<(*const u8, &SlotHeader)>` — lookup for restore
- `free_checkpoint(checkpoint_id)` — return all slots to free list
- `available() -> usize` — free_list.len()
- `spill_oldest(dir: &Path) -> io::Result<String>` — write oldest checkpoint to NVMe, free slots
- `load_spilled(path: &Path, checkpoint_id: &str) -> io::Result<()>` — read back into slots

Sizing: `slot_size = 64 + align_up(hidden_dim * max_seq_len * dtype_size, 64)`.
Number of slots = `sqrt_l_count * 2` (two concurrent checkpoints before
spill). Total arena = `slot_size * num_slots`. Configurable override via
`RS_CHECKPOINT_ARENA_MB`. Default max_seq_len from model config's
`max_position_embeddings`.

Transactional creation: save free_list length before starting a
checkpoint capture. On failure (any layer capture errors), restore
free_list and remove index entries — no partial checkpoints in the arena.

#### 2. Python bridge (4 functions in `python/rocket_surgeon/checkpoint.py`)

```python
import ctypes
import torch

def capture_activation(
    layer_idx: int,
    dst_ptr: int,
    dst_len: int,
) -> tuple[str, list[int]]:
    tensor = _get_residual_stream(layer_idx)
    t = tensor.detach().contiguous()
    nbytes = t.nelement() * t.element_size()
    assert nbytes <= dst_len
    buf = (ctypes.c_byte * dst_len).from_address(dst_ptr)
    cpu_view = torch.frombuffer(buf, dtype=t.dtype).reshape(t.shape)
    cpu_view.copy_(t)  # CUDA DMA -> arena
    torch.cuda.synchronize()
    del cpu_view
    return (str(t.dtype), list(t.shape))


def restore_activation(
    layer_idx: int,
    src_ptr: int,
    src_len: int,
    dtype: str,
    shape: list[int],
) -> None:
    torch_dtype = getattr(torch, dtype.replace("torch.", ""))
    buf = (ctypes.c_byte * src_len).from_address(src_ptr)
    cpu_view = torch.frombuffer(buf, dtype=torch_dtype).reshape(shape)
    target = _get_residual_stream(layer_idx)
    target.copy_(cpu_view)  # CPU arena -> GPU
    torch.cuda.synchronize()
    del cpu_view


def capture_rng_state() -> bytes:
    """Length-prefixed raw bytes: [num_devices: u32][device_id: u32, len: u32, data: bytes] * N"""
    import struct
    parts = []
    device_count = torch.cuda.device_count()
    parts.append(struct.pack("<I", device_count))
    for i in range(device_count):
        rng_bytes = torch.cuda.get_rng_state(i).numpy().tobytes()
        parts.append(struct.pack("<II", i, len(rng_bytes)))
        parts.append(rng_bytes)
    return b"".join(parts)


def restore_rng_state(state: bytes) -> None:
    import struct
    offset = 0
    (device_count,) = struct.unpack_from("<I", state, offset); offset += 4
    for _ in range(device_count):
        device_id, length = struct.unpack_from("<II", state, offset); offset += 8
        rng_bytes = state[offset:offset + length]; offset += length
        t = torch.frombuffer(bytearray(rng_bytes), dtype=torch.uint8)
        torch.cuda.set_rng_state(t, device_id)
```

`_get_residual_stream(layer_idx)` accesses the model's residual stream
at the given layer. It reads from the forward pass result mailbox — the
same mailbox that `inspect` and `view` already read from after a tick
completes. The residual stream is the tensor at the output of
`model.layers[layer_idx]` (the post-norm residual). This function is
called while the forward pass is stopped (between ticks), so the mailbox
is populated and stable.

#### 3. √L boundary selector (Rust, in checkpoint module)

```rust
pub fn checkpoint_layers(num_layers: u32) -> Vec<u32> {
    let sqrt_l = (num_layers as f64).sqrt().ceil() as u32;
    let interval = num_layers as f64 / sqrt_l as f64;
    (1..sqrt_l)
        .map(|i| (i as f64 * interval).floor() as u32)
        .collect()
}
```

For 32 layers: [5, 10, 16, 21, 26] — 5 boundaries, evenly spaced.
For 80 layers: [8, 17, 26, 35, 44, 53, 62, 71] — 8 boundaries.

The last layer is excluded — checkpointing the final layer's output is
useless for replay because there's nothing after it. Layer 0 is also
excluded: the first segment replays from the embedding output, which is
always available from the model input (token IDs are small and the daemon
retains them). Sub-project B's replay code must know to start the first
segment from the embedding, not from a checkpoint.

#### 4. `_host/checkpoint` dispatch handler (Rust, in dispatch.rs)

Wires into the existing dispatch table:

```rust
internal::HOST_CHECKPOINT => handle_host_checkpoint(state, request),
```

The existing `HostCheckpointRequest::Create` has a single `layer_idx`
field. The handler ignores it and captures all √L layers in one call —
the daemon does not need per-layer control. Consider removing `layer_idx`
from the wire type and adding `layers: Option<Vec<u32>>` if per-layer
control is ever needed. For now, single-call multi-layer capture.

**Create flow:**
1. Parse `HostCheckpointRequest::Create { checkpoint_id, tier, tick_id, .. }`
2. Save free_list state for rollback
3. For each layer in `checkpoint_layers(num_layers)`:
   a. `arena.alloc_slot(checkpoint_id, layer, byte_len)` → get `(ptr, header)`
   b. Call Python bridge `capture_activation(layer, ptr, byte_len)` via PyO3 → get `(dtype, shape)`
   c. Write dtype/shape into slot header
3. Capture RNG state via bridge → store as a dedicated slot (layer_idx = u32::MAX sentinel)
4. On failure at any layer: rollback free_list, remove index entries, return error
5. If arena utilization > 80%, spill oldest checkpoint to NVMe
6. Return `HostCheckpointResponse { checkpoint_id, tier, bytes_captured }`

**Restore flow:**
1. Parse `HostCheckpointRequest::Restore { checkpoint_id }`
2. If checkpoint is spilled, load from NVMe back into arena
3. For each slot belonging to checkpoint_id:
   a. `arena.get_slot(checkpoint_id, layer)` → get `(ptr, header)`
   b. Call Python bridge `restore_activation(layer, ptr, byte_len, dtype, shape)` via PyO3
4. Restore RNG state from the sentinel slot
5. Return `HostCheckpointResponse { checkpoint_id, restored_to }`

#### 5. Auto-checkpoint integration

The daemon's step handler already calls `_host/step`. After each step
response, if the new layer index is a √L boundary:

1. Daemon calls `session.checkpoint_create(Some(Activation))` — registers metadata
2. Daemon sends `_host/checkpoint Create` to worker via orchestrator — captures tensors

This is "always-on" — every √L crossing gets a checkpoint. Old auto-checkpoints
are freed when the arena spills. Explicit user-created checkpoints via
`rocket/checkpoint create` are pinned and never auto-spilled.

The distinction between auto-checkpoints and user-checkpoints:
- Auto: created implicitly at √L boundaries, spill-eligible, no bookmark
- User: created explicitly via `rocket/checkpoint`, spill-eligible only when arena is critically full, can be bookmarked

#### 6. NVMe spill format

No serialization library. Raw bytes with a fixed-size index:

```
File layout:
[8 bytes]   magic: "CKPTSPIL"
[4 bytes]   version: 1
[4 bytes]   num_slots
[num_slots × 80 bytes]  slot index entries
[...]       slot data (concatenated, 64-byte aligned)

Slot index entry (80 bytes):
[4 bytes]   layer_idx
[1 byte]    dtype enum
[1 byte]    ndim
[2 bytes]   reserved
[48 bytes]  shape[6] as u64
[8 bytes]   data_offset (from file start)
[8 bytes]   data_len
[4 bytes]   crc32 (of slot data bytes)
[4 bytes]   reserved
```

Read and write via standard `File::read_exact` / `File::write_all`.
No mmap for files — the arena is the working set, files are cold storage.

## What this unblocks

After Sub-project A, the checkpoint TCK scenarios (9 in `checkpoint.feature`,
currently passing against the metadata tier) will have real tensor data behind
them. The `bytes_captured` field in responses will be non-zero.

Sub-project B builds on top: replay reads checkpoint data from the arena,
replays forward, compares tensors for divergence.

## TCK impact

The 9 existing `checkpoint.feature` scenarios pass today against the metadata
tier. Sub-project A should not break them — the worker handler is additive.
No new TCK scenarios are needed for Sub-project A; the existing ones cover
the protocol surface. Worker-side tensor correctness is verified by the
Sub-project B replay scenarios (divergence detection proves the checkpoint
data is valid by reproducing the same activations).

Unit tests for the arena (alloc, free, spill, load) and the √L boundary
selector are implementation-level, not protocol-level.

## Multi-GPU considerations

Sub-project A targets single-GPU and data-parallel. Each rank's worker
has its own arena. For data-parallel, all ranks cross √L boundaries at
the same tick, so checkpoint IDs stay coordinated via the daemon.

Pipeline-parallel (different ranks at different layers/micro-batches)
requires a checkpoint barrier or consensus protocol across ranks. This
is flagged for Sub-project B or later — not a Sub-project A concern.

## Risk

| Risk | Mitigation |
|------|------------|
| `cudaHostRegister` fails (no GPU, test env) | Fall back to unpinned mmap, log warning. Correctness unaffected, slower DMA. |
| `torch.frombuffer` rejects arena pointer | Guard with integration test on real torch; fallback to `ctypes.memmove` (one extra memcpy). |
| Arena sizing wrong for large models | `RS_CHECKPOINT_ARENA_MB` env override. Spill policy prevents OOM. |
| RNG state restore insufficient for determinism | Known limitation — full determinism is Sub-project B. RNG capture is necessary-but-not-sufficient. |
| Partial capture failure (OOM on `.contiguous()`) | Transactional rollback — free_list restored, no partial checkpoints. |
| NVMe spill corruption (interrupted write) | CRC32 per slot validated on load. |
