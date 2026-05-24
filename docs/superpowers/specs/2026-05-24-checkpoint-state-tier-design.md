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
    ▼
┌─────────────────────────────────────────┐
│  Checkpoint Arena (mmap + mlock)        │
│  Rust-allocated, Rust-owned             │
│                                         │
│  ┌─────┬─────┬─────┬─────┬───────────┐ │
│  │Hdr 0│Data0│Hdr 1│Data1│  free ... │ │
│  └─────┴─────┴─────┴─────┴───────────┘ │
│  ▲                                      │
│  │  Python sees this address via        │
│  │  torch.frombuffer(ctypes buffer)     │
│  │  CUDA DMA lands here directly        │
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
mmap + mlock for pinned pages, not `shm_open`. The arena is not
cross-process; DoomRing handles IPC.

```
pub struct CheckpointArena {
    ptr: *mut u8,
    capacity: usize,
    watermark: usize,               // bump allocator head
    slots: Vec<SlotDescriptor>,      // (offset, byte_len, checkpoint_id, layer_idx)
    index: HashMap<(String, u32), usize>,  // (checkpoint_id, layer_idx) -> slot index
}
```

Operations:
- `new(capacity_bytes: usize) -> Result<Self>` — anonymous mmap + mlock
- `alloc_slot(checkpoint_id, layer_idx, byte_len) -> (*mut u8, &SlotHeader)` — bump allocate, write header, return data pointer
- `get_slot(checkpoint_id, layer_idx) -> Option<(*const u8, &SlotHeader)>` — lookup for restore
- `free_checkpoint(checkpoint_id)` — mark slots as reclaimable (compaction is deferred)
- `utilization() -> f64` — watermark / capacity
- `spill_oldest(dir: &Path) -> io::Result<String>` — write oldest checkpoint to NVMe, free arena slots
- `load_spilled(path: &Path, checkpoint_id: &str) -> io::Result<()>` — read back into arena

Sizing heuristic at construction: `max(256 MB, num_layers * hidden_dim * seq_len * dtype_size * sqrt_l_count * 2)`. The `* 2` accommodates two concurrent checkpoints before spill. Configurable via environment variable `RS_CHECKPOINT_ARENA_MB`.

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


def capture_rng_state() -> bytes:
    states = {}
    for i in range(torch.cuda.device_count()):
        states[i] = torch.cuda.get_rng_state(i).numpy().tobytes()
    return pickle.dumps(states)


def restore_rng_state(state: bytes) -> None:
    states = pickle.loads(state)
    for device_id, rng_bytes in states.items():
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
    (1..=sqrt_l)
        .map(|i| ((i as f64 * (num_layers as f64).sqrt()).floor() as u32).min(num_layers - 1))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
```

For 32 layers: [5, 11, 16, 22, 27, 31] — 6 boundaries.
For 80 layers: [8, 17, 26, 35, 44, 53, 62, 71, 79] — 9 boundaries.

#### 4. `_host/checkpoint` dispatch handler (Rust, in dispatch.rs)

Wires into the existing dispatch table:

```rust
internal::HOST_CHECKPOINT => handle_host_checkpoint(state, request),
```

**Create flow:**
1. Parse `HostCheckpointRequest::Create { checkpoint_id, tier, tick_id, layer_idx }`
2. For each layer in `checkpoint_layers(num_layers)`:
   a. `arena.alloc_slot(checkpoint_id, layer, byte_len)` → get `(ptr, header)`
   b. Call Python bridge `capture_activation(layer, ptr, byte_len)` via PyO3 → get `(dtype, shape)`
   c. Write dtype/shape into slot header
3. Capture RNG state via bridge → store as a dedicated slot (layer_idx = u32::MAX sentinel)
4. If arena utilization > 80%, spill oldest checkpoint to NVMe
5. Return `HostCheckpointResponse { checkpoint_id, tier, bytes_captured }`

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
[8 bytes]   reserved
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

## Risk

| Risk | Mitigation |
|------|------------|
| `mlock` fails (ulimit) | Fall back to unpinned mmap, log warning. Correctness unaffected, just slower DMA. |
| `torch.frombuffer` rejects arena pointer | Guard with integration test on real torch; fallback path uses `ctypes.memmove` (one extra memcpy). |
| Arena sizing wrong for large models | `RS_CHECKPOINT_ARENA_MB` env override. Spill policy prevents OOM. |
| RNG state restore insufficient for determinism | Known limitation — full determinism is Sub-project B. RNG capture is necessary-but-not-sufficient. |
