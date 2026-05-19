# WU 1.8: Shared Memory Data Plane — Design Spec

## Goal

Replace base64-over-JSON-RPC tensor transfer with a shared-memory ring buffer (the TZC split control/data plane pattern). Tensor bytes flow through POSIX shared memory with one `memcpy`; only a small control message traverses the JSON-RPC channel. Target: 4x latency improvement (~34ms → ~8ms per 17MB tensor).

The ring buffer uses Doom's `d_net.c` naming conventions: `maketic` (write cursor), `nettics` (read cursor), `BACKUPTICS` (capacity). Index arithmetic follows kfifo (monotonic `AtomicU64`, power-of-2 masking). Memory layout follows PortAudio (caller-allocated mmap'd region). Scatter/gather reads follow JACK (for wraparound handling on the daemon side).

## Dependencies

- WU 1.11 (inspect integration): `collect_tensors`, `CapturedTensor`, `TensorStore` — done
- WU 1.12 (probe events): capture hooks, `last_outputs` population — done
- ADR-0006: content-addressable BLAKE3 IDs, summary-then-slice, ring buffer with ProbeFrame — done
- ProbeFrame header (`probe_frame.rs`): 128-byte fixed header — done, needs alignment fix (this WU)

## TCK Contract

WU 1.8 is an internal optimization — the client-facing protocol does not change. Existing TCK scenarios for `rocket/inspect` serve as regression tests. New TCK scenarios verify the shm capability advertisement and fallback behavior:

1. Capabilities include `shared_memory_supported` flag
2. Inspect returns valid tensor summary (regression — same behavior, shm transport)
3. Inspect returns valid tensor slice (regression)
4. Inspect with shm unavailable falls back gracefully (no error to client)

---

## 1. Architecture: The TZC Split

Every surveyed system (TZC, Ray/Plasma, Zerrow, fastsafetensors, safetensors, GGUF) converges on the same pattern: **split the control plane from the data plane.**

```
WORKER (Rust + Python)                     DAEMON (Rust)
┌───────────────────────────┐              ┌───────────────────────────┐
│ Hook fires:               │              │                           │
│  tensor.cpu()             │              │ JSON-RPC recv:            │
│  data_ptr() → &[u8]      │              │  CapturedTensor {         │
│  blake3::hash() (GIL-free)│              │    tensor_id,             │
│  memcpy → ring slot       │──(shm)────►│    shape, dtype, device,  │
│                           │              │    shm_name, shm_offset,  │
│ JSON-RPC send:            │              │    byte_length            │
│  CapturedTensor {         │──(socket)──►│  }                        │
│    tensor_id,             │              │                           │
│    shm_name, shm_offset,  │              │ mmap read from shm:      │
│    byte_length            │              │  &[u8] → TensorStore     │
│  }                        │              │  (pre-computed hash, no   │
│                           │              │   re-hash on daemon side) │
└───────────────────────────┘              └───────────────────────────┘
```

- **Control plane**: Existing JSON-RPC channel. `CapturedTensor` gains `shm_name`, `shm_offset`, `byte_length`; `data_base64` becomes optional (fallback).
- **Data plane**: POSIX shared memory ring buffer, pre-allocated at attach time.
- **Notification**: Not needed as a separate mechanism — the JSON-RPC control message (`CapturedTensor` within `HostInspectResponse`) already tells the daemon where to read.

---

## 2. Ring Buffer Design

### 2.1 Doom Naming Convention

| Doom name | Role | Type |
|-----------|------|------|
| `maketic` | Write cursor — monotonic byte offset, never wraps | `AtomicU64` |
| `nettics` | Read cursor — monotonic byte offset, never wraps | `AtomicU64` |
| `BACKUPTICS` | Ring buffer capacity in bytes (power of 2) | `u64` |
| `netcmds` | The ring buffer data region (mmap'd shared memory) | `&[u8]` |
| `doomcom` | Unix domain socket for wakeup notification | UDS |

### 2.2 Index Arithmetic (kfifo style)

Cursors are monotonic `u64` — they increment forever, never reset. The physical position within the ring is computed by masking:

```rust
let mask = BACKUPTICS - 1; // power-of-2 capacity enables bitwise mask
let physical_offset = maketic & mask;
```

Available space: `BACKUPTICS - (maketic - nettics)`. The ring is empty when `maketic == nettics`, full when `maketic - nettics == BACKUPTICS`. Natural unsigned arithmetic handles the (theoretical) u64 overflow correctly — the subtraction wraps correctly modulo 2^64.

### 2.3 Fixed-Size Slots

The ring is divided into fixed-size slots. Each slot holds one ProbeFrame (128-byte header + tensor bytes up to `slot_data_capacity`).

| Parameter | Default | Notes |
|-----------|---------|-------|
| `BACKUPTICS` (slot count) | 16 | Power of 2 |
| `slot_size` | 64 MiB | Header + max tensor bytes |
| Total ring size | 1 GiB | `BACKUPTICS × slot_size` |

Slot-based (not streaming) for simplicity: no wraparound handling, no scatter/gather reads, each slot is self-contained. The slot index is `maketic & (BACKUPTICS - 1)` where `maketic` counts slots (not bytes).

Both `BACKUPTICS` and `slot_size` are configurable at attach time. The Python worker creates the shm region sized to `CONTROL_SIZE + BACKUPTICS × slot_size`.

### 2.4 Oversized Tensors

Tensors whose byte count exceeds `slot_size - HEADER_SIZE` (default: 64MiB - 128B) **fall back to the existing base64-over-JSON-RPC path**. The existing `data_base64` field on `CapturedTensor` is preserved for this purpose. The daemon checks: if `shm_offset` is present, read from shm; if `data_base64` is present, decode base64. Both may be absent (error) but never both present.

For Phase 1 debugging scenarios, slot_size = 64MiB covers all single-module outputs for Llama-3-8B at seq_len ≤ 2048. Attention matrices (captured by views, not inspect) are computed in-place in the worker and returned as JSON — they don't traverse the ring buffer.

---

## 3. Control Block

The control block occupies the first `CONTROL_SIZE = 4096` bytes of the shared memory region. Slots start at offset 4096.

```
Control Block (4096 bytes):

  ┌─ Config section (immutable after init) ──────────────────────┐
  │ offset 0:   magic        [u8; 8]  = b"DOOMRING"             │
  │ offset 8:   version      u32      = 1                       │
  │ offset 12:  slot_count   u32      = BACKUPTICS (power of 2) │
  │ offset 16:  slot_size    u64      (bytes per slot)           │
  │ offset 24:  region_size  u64      (total shm bytes)          │
  │ offset 32-127: reserved (zeroed)                             │
  └──────────────────────────────────────────────────────────────┘

  ┌─ Producer cursor (own cache line) ───────────────────────────┐
  │ offset 128: maketic      AtomicU64 (slot index, monotonic)   │
  │ offset 136-255: padding                                      │
  └──────────────────────────────────────────────────────────────┘

  ┌─ Consumer cursor (own cache line) ───────────────────────────┐
  │ offset 256: nettics      AtomicU64 (slot index, monotonic)   │
  │ offset 264-383: padding                                      │
  └──────────────────────────────────────────────────────────────┘

  offset 384-4095: reserved
```

**False sharing avoidance**: `maketic` and `nettics` are separated by 128 bytes — each on its own cache line on both x86 (64B lines) and Apple Silicon (128B lines). The producer only writes `maketic`; the consumer only writes `nettics`.

### 3.1 Initialization Handshake (Linus fix #3)

The `magic` field is the initialization barrier:

1. Python zeros the entire control block
2. Python writes `version`, `slot_count`, `slot_size`, `region_size`
3. Python writes `maketic = 0`, `nettics = 0`
4. **Last**: Python writes `magic = b"DOOMRING"` with a Release store (via PyO3 bridge — see §5)
5. Python sends the shm region name to the daemon via JSON-RPC (part of the attach response)

The daemon, upon receiving the shm name:

1. Opens the region via `shm_open` + `mmap`
2. Reads `magic` with an Acquire load
3. If `magic != b"DOOMRING"`, the control block is not yet initialized — retry or error
4. Reads `version`, `slot_count`, `slot_size`, `region_size`
5. Validates: `version == 1`, `slot_count` is power of 2, `region_size == CONTROL_SIZE + slot_count × slot_size`

**Documented invariant**: the shm region name is sent via JSON-RPC only after the control block is fully initialized. The magic field is defense-in-depth.

---

## 4. ProbeFrame Header v2 (Alignment Fix)

The existing ProbeFrame header has all `u64` fields at +4 alignment (offset 44, 52, 60) due to the `u16 + u8 + u8` at offset 8 shifting everything by 4 bytes after the shape array. This crosses cache line boundaries on unaligned loads.

**Fix**: Insert 4 bytes of explicit padding after `shape`, and add a `generation` field (stale-data detection) after `flags`.

```
ProbeFrame Header v2 (128 bytes):

  offset 0:   rank       u32
  offset 4:   layer      u32
  offset 8:   comp_id    u16
  offset 10:  dtype      u8
  offset 11:  ndim       u8
  offset 12:  shape      [u32; 8]  (32 bytes)
  offset 44:  _pad0      u32       ← NEW (alignment padding)
  offset 48:  tick_id    u64       ← NOW 8-byte aligned
  offset 56:  data_off   u64       ← NOW 8-byte aligned (renamed from 'offset')
  offset 64:  size       u64       ← NOW 8-byte aligned
  offset 72:  flags      u32
  offset 76:  generation u32       ← NEW (low bits of maketic at write time)
  offset 80-127: reserved (48 bytes, zeroed)
```

Changes from v1:
- `_pad0` (4 bytes) at offset 44 pushes `tick_id` to offset 48 (8-byte aligned)
- All `u64` fields now at 8-byte aligned offsets (48, 56, 64)
- `offset` renamed to `data_off` — byte offset of tensor data within the shm region (absolute from region start). For slot `i`: `data_off = CONTROL_SIZE + i × slot_size + HEADER_SIZE`.
- `generation` (u32 at offset 76) — low 32 bits of `maketic` at write time. Consumer verifies `header.generation == (nettics & 0xFFFF_FFFF)` to detect stale/corrupted data. Defense-in-depth for SPSC correctness.
- Reserved space shrinks from 56 to 48 bytes (still generous).

The existing `probe_frame.rs` is updated in-place — nothing external depends on the v1 layout yet.

---

## 5. Python-Side Atomics via PyO3 Bridge (Linus fix #1)

**This is a show-stopper on ARM/Apple Silicon.** A plain `struct.pack_into` on a `memoryview` is NOT an atomic store. On x86, every aligned store is implicitly release-ordered, so it works by accident. On ARM (Apple Silicon), plain stores have no ordering guarantees — a consumer on another core can see a stale value or a partial write. The ring buffer will silently corrupt data on M-series Macs.

**Solution**: Expose atomic operations from Rust to Python via the PyO3 bridge.

New functions in the worker's PyO3 bridge (`crates/rocket-surgeon-worker/src/shm_ops.rs`):

```rust
/// Atomic store with Release ordering into a shared memory region.
/// `shm_ptr` is the base address of the mmap'd region.
/// `offset` is the byte offset of the u64 to write.
#[pyfunction]
fn shm_atomic_store_u64(shm_ptr: usize, offset: usize, value: u64) {
    let ptr = (shm_ptr + offset) as *const AtomicU64;
    // Safety: caller ensures ptr is valid, aligned, and within the shm region
    unsafe { &*ptr }.store(value, Ordering::Release);
}

/// Atomic load with Acquire ordering from a shared memory region.
#[pyfunction]
fn shm_atomic_load_u64(shm_ptr: usize, offset: usize) -> u64 {
    let ptr = (shm_ptr + offset) as *const AtomicU64;
    unsafe { &*ptr }.load(Ordering::Acquire)
}

/// Write raw bytes into shared memory at the given offset. Not atomic,
/// but used for bulk data that is fenced by the cursor atomic store.
#[pyfunction]
fn shm_write_bytes(shm_ptr: usize, offset: usize, data: &[u8]) {
    let dst = (shm_ptr + offset) as *mut u8;
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
}

/// Write the magic bytes (final initialization step).
/// Uses Release ordering to ensure all prior writes are visible.
#[pyfunction]
fn shm_write_magic(shm_ptr: usize) {
    let dst = shm_ptr as *mut u8;
    unsafe { std::ptr::copy_nonoverlapping(b"DOOMRING".as_ptr(), dst, 8) };
    // Release fence ensures all prior control block writes are visible
    // before any consumer reads magic and proceeds
    std::sync::atomic::fence(Ordering::Release);
}
```

**Python write protocol** (in `python/rocket_surgeon/ring_buffer.py`):

```python
from rocket_surgeon._worker import shm_atomic_store_u64, shm_atomic_load_u64, shm_write_bytes

def publish_frame(shm_ptr, maketic, header_bytes, tensor_bytes):
    slot_idx = maketic & (BACKUPTICS - 1)
    slot_offset = CONTROL_SIZE + slot_idx * slot_size

    # Write header + data (non-atomic bulk writes)
    shm_write_bytes(shm_ptr, slot_offset, header_bytes)
    shm_write_bytes(shm_ptr, slot_offset + HEADER_SIZE, tensor_bytes)

    # Advance write cursor with Release ordering
    # This is the publication barrier — all prior writes become visible
    shm_atomic_store_u64(shm_ptr, MAKETIC_OFFSET, maketic + 1)
```

The Release store on `maketic` is the single synchronization point. No separate `fence(Release)` needed — the atomic store with `Ordering::Release` subsumes it (Linus fix #6).

---

## 6. Write Protocol (Producer — Worker)

Executed in the worker process after a barrier hook captures a tensor:

```
1. Get raw pointer: data_ptr() → &[u8] (zero-copy, GIL held, Py<PyAny> pins tensor)
2. Release GIL
3. Compute BLAKE3 hash: blake3::hash(raw_slice) → tensor_id  (~5.7ms / 17MB)
4. Check slot availability:
     nettics = shm_atomic_load_u64(shm_ptr, NETTICS_OFFSET)     // Acquire
     if (maketic - nettics) >= BACKUPTICS:
       → ring full: fall back to base64
5. Compute slot offset: CONTROL_SIZE + (maketic & (BACKUPTICS - 1)) × slot_size
6. Check tensor fits: if tensor_bytes > slot_size - HEADER_SIZE:
     → too large: fall back to base64
7. Serialize ProbeFrame header (128 bytes, includes generation = maketic & 0xFFFFFFFF)
8. shm_write_bytes(header, 128 bytes)
9. shm_write_bytes(tensor_data, size bytes)
10. shm_atomic_store_u64(maketic + 1)                            // Release
11. Write 1 byte to doomcom (UDS) — wakeup notification
12. Build CapturedTensor with shm_name, shm_offset, byte_length, tensor_id
    (data_base64 omitted — shm path)
```

Step 10 (Release store on `maketic`) is the publication barrier. All header + data writes from steps 8-9 are guaranteed visible to the consumer before the cursor advance is visible. No redundant `fence(Release)` (Linus fix #6).

**Fallback**: Steps 4 and 6 check for ring-full and oversized conditions. On fallback, the worker uses the existing `tensor_to_bytes()` → base64 → `data_base64` path. The daemon handles both transparently.

---

## 7. Read Protocol (Consumer — Daemon)

Executed in the daemon's async event loop (tokio):

```
1. Receive CapturedTensor via JSON-RPC (from orchestrator relay)
2. If data_base64 is present: decode base64, insert into TensorStore (existing path)
3. If shm_offset is present:
   a. Look up mmap'd region by shm_name (cached from attach)
   b. Read ProbeFrame header at shm_offset (128 bytes)
   c. Validate generation: header.generation == expected
   d. Read tensor bytes: &shm[shm_offset + 128 .. shm_offset + 128 + byte_length]
      (zero-copy mmap view)
   e. Insert into TensorStore with PRE-COMPUTED tensor_id from CapturedTensor
      (no re-hash — Linus fix #9)
   f. Advance nettics: atomic store with Release ordering
```

Step (e) is critical: the daemon trusts the worker's BLAKE3 hash. The worker computed it on the raw bytes before writing to shm. The daemon skips re-hashing, saving ~5.7ms per 17MB tensor.

Step (f): The `nettics` advance tells the producer the slot is free. No UDS acknowledgment needed — the producer checks `nettics` before writing (Linus fix #10).

### 7.1 Notification (doomcom)

The `doomcom` Unix domain socket carries one-direction wakeup bytes (worker → daemon). One byte per frame published. The daemon's tokio event loop polls `doomcom` for readability; a byte arrival triggers processing of the next JSON-RPC response containing `CapturedTensor` messages.

No acknowledgment bytes flow on `doomcom`. The producer checks slot availability by reading `nettics` directly from shared memory (Acquire load). This eliminates the bidirectional UDS complexity.

---

## 8. Shared Memory Lifecycle

### 8.1 Creation (Worker)

During `_host/attach`, after model load succeeds:

```python
import multiprocessing.shared_memory as shm

region_size = CONTROL_SIZE + BACKUPTICS * slot_size
name = f"/rs-{session_id[:8]}-{rank}"   # ≤ 30 chars (macOS PSHMNAMLEN = 31)
mem = shm.SharedMemory(name=name, create=True, size=region_size, track=False)
```

`track=False` (Python 3.13+) prevents the resource tracker from unlinking the region out from under the daemon. For Python < 3.13, monkey-patch the resource tracker to skip our region.

After creation, the worker initializes the control block (§3.1) and returns the shm name in the `_host/attach` response.

### 8.2 Opening (Daemon)

On receiving the attach response with `shm_name`:

```rust
let fd = shm_open(shm_name, OFlag::O_RDWR, Mode::empty())?;
let ptr = mmap(None, region_size.try_into()?, 
    ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
    MapFlags::MAP_SHARED, &fd, 0)?;
// Validate magic with Acquire load
assert_eq!(&shm[0..8], b"DOOMRING");
```

The daemon caches the mmap pointer for the session lifetime.

### 8.3 Destruction (Clean Shutdown)

On `_host/detach` or session end:

1. Daemon: `munmap` the region, close fd
2. Worker: `mem.close()`, then `mem.unlink()` → calls `shm_unlink()`

### 8.4 Cleanup on Worker Crash

If the worker dies without unlinking:

1. Orchestrator detects worker death (process exit)
2. Orchestrator reports to daemon via JSON-RPC error
3. Daemon calls `shm_unlink(shm_name)` for the dead worker's region
4. Daemon drops the cached mmap pointer

### 8.5 Cleanup on Daemon Crash — Stale Region Sweep (Linus fix #5)

On daemon startup, sweep for stale regions:

**Linux**: Enumerate `/dev/shm/` for files matching `rs-*`. Unlink any that belong to dead sessions (cross-reference with active session state, which is empty on fresh startup → unlink all).

**macOS**: No filesystem interface for POSIX shm objects. Convention: the daemon writes active region names to `~/.local/state/rocket_surgeon/shm_regions.json`. On startup, read the file, attempt to open each listed region, unlink it, then clear the file.

### 8.6 macOS Name Length

`PSHMNAMLEN = 31` on macOS (including the leading `/`). Our convention `/rs-XXXXXXXX-R` uses 15 characters (8-hex session ID, single-digit rank). Multi-digit ranks (0-99) use at most 16 characters. Safe.

Runtime check: if `name.len() > 30`, return an error at region creation time.

### 8.7 macOS Page Size

Apple Silicon uses 16 KiB pages. `region_size` is rounded up to the nearest 16 KiB boundary by the OS. Our default (4096 + 16 × 64MiB = 1 GiB + 4 KiB) is already page-aligned on both platforms. No special handling needed.

---

## 9. CapturedTensor Wire Format Changes

Current (`messages.rs:539`):

```rust
pub struct CapturedTensor {
    pub module_path: String,
    pub canonical: String,
    pub layer: u32,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub device: String,
    pub data_base64: String,
}
```

New:

```rust
pub struct CapturedTensor {
    pub module_path: String,
    pub canonical: String,
    pub layer: u32,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub device: String,
    pub tensor_id: String,                          // NEW: pre-computed BLAKE3 hex hash
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shm_name: Option<String>,                   // NEW: shm region name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shm_offset: Option<u64>,                    // NEW: byte offset within shm region
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_length: Option<u64>,                   // NEW: tensor byte count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_base64: Option<String>,                // CHANGED: now optional (fallback)
}
```

The daemon dispatches on presence:
- `shm_offset.is_some()` → read from shared memory
- `data_base64.is_some()` → decode base64 (fallback)
- Neither → error

`tensor_id` is always present — computed by the worker regardless of transport. The daemon uses it directly for TensorStore insertion without re-hashing (Linus fix #9).

---

## 10. TensorStore Improvements

### 10.1 Accept Pre-Computed Hash (Linus fix #9)

Current `TensorStore::insert()` (`tensor_store.rs:74`) computes `blake3::hash(&data)` on every insert. With shm, the worker already computed the hash. New signature:

```rust
pub fn insert_with_id(
    &mut self,
    tensor_id: String,          // pre-computed BLAKE3 hex
    data: Vec<u8>,
    shape: Vec<u64>,
    dtype: DType,
    device: String,
) -> TensorHandle
```

The existing `insert()` (which computes the hash) is preserved for the base64 fallback path.

Dedup check: if `tensor_id` already exists in `entries`, skip insertion, update access order, return existing handle.

### 10.2 O(1) LRU Touch (Linus fix #8)

Current `touch_access_order()` (`tensor_store.rs:208-211`) does `self.access_order.retain(|id| id != tensor_id)` — O(n) scan on every access.

**Fix**: Replace `VecDeque<String>` access order with a generation counter:

```rust
pub struct TensorStore {
    entries: HashMap<String, StoredTensor>,
    max_entries: usize,
    max_bytes: usize,
    current_bytes: usize,
    access_generation: u64,     // NEW: monotonic counter
}

pub struct StoredTensor {
    // ... existing fields ...
    last_access_gen: u64,       // NEW: replaces position in VecDeque
}
```

- `touch()`: `entry.last_access_gen = self.access_generation; self.access_generation += 1;` — O(1)
- `evict_oldest()`: find entry with minimum `last_access_gen` — O(n) but only during eviction, which is rare and bounded by `max_entries`
- Total: every `get`/`summarize`/`slice` goes from O(n) to O(1). Eviction stays O(n) but runs infrequently.

---

## 11. Daemon-Side Data Flow Changes

### 11.1 `ingest_and_respond()` (`dispatch.rs:258-343`)

Current flow:
1. Base64 decode each `CapturedTensor`
2. `store.insert(bytes, ...)` — hashes bytes
3. `store.summarize(tensor_id)`

New flow:
1. For each `CapturedTensor`:
   - If `shm_offset` present: read bytes from mmap view (zero-copy slice), call `store.insert_with_id(tensor_id, bytes, ...)`
   - If `data_base64` present: decode base64, call `store.insert(bytes, ...)` (computes hash)
2. `store.summarize(tensor_id)` — unchanged

The mmap view (`&[u8]`) is zero-copy — no allocation until `TensorStore` copies the bytes into its owned `Vec<u8>`. A future optimization (Phase 2+) could store a reference to the shm region instead of copying, but that couples store lifetime to shm lifetime.

### 11.2 `collect_tensors()` (`worker/dispatch.rs:667-714`)

Current flow:
1. `bridge::tensor_to_bytes(py, &tensor)` → raw bytes
2. `base64::encode(&bytes)` → `data_base64`

New flow:
1. `data_ptr()` → `&[u8]` (zero-copy via PyO3, tensor pinned by `Py<PyAny>`)
2. `blake3::hash(raw_slice)` → `tensor_id` (GIL released)
3. Check ring availability and tensor size
4. If shm path: `shm_write_bytes(header + data)`, advance `maketic`, populate `shm_offset`/`byte_length`
5. If fallback: `tensor_to_bytes()` → base64 → `data_base64`
6. Build `CapturedTensor` with appropriate fields

---

## 12. New Crate: `rocket-surgeon-ring` (or module within `rocket-surgeon-python`)

The ring buffer implementation lives in a focused module:

```
crates/rocket-surgeon-python/src/
  ring_buffer.rs        — Rust-side ring buffer types and operations
  shm_ops.rs            — PyO3 functions for Python-side atomic access
```

Plus the Python-side wrapper:

```
python/rocket_surgeon/
  ring_buffer.py        — Python ring buffer producer (calls shm_ops via PyO3)
```

### Key Types (Rust)

```rust
pub const CONTROL_SIZE: usize = 4096;
pub const MAGIC: &[u8; 8] = b"DOOMRING";
pub const MAKETIC_OFFSET: usize = 128;
pub const NETTICS_OFFSET: usize = 256;

pub struct RingConfig {
    pub backuptics: u32,        // slot count, power of 2
    pub slot_size: u64,         // bytes per slot
}

pub struct DoomRing {
    shm_ptr: *mut u8,
    shm_len: usize,
    config: RingConfig,
}
```

### Key Types (Python)

```python
class DoomRing:
    def __init__(self, name: str, backuptics: int, slot_size: int): ...
    def publish(self, header: bytes, tensor_data: bytes) -> int: ...
    def is_full(self) -> bool: ...
    def maketic(self) -> int: ...
```

---

## 13. Error Handling

### 13.1 New Error Conditions

| Condition | Handling | Error to Client |
|-----------|----------|-----------------|
| Ring full (`maketic - nettics >= BACKUPTICS`) | Fall back to base64 | None (transparent) |
| Tensor exceeds slot size | Fall back to base64 | None (transparent) |
| shm_open fails (platform issue) | Fall back to base64 for entire session | Warning log, no error |
| Stale magic on daemon open | Retry 3× with 10ms backoff, then error | `INTERNAL_ERROR` |
| Generation mismatch on read | Log warning, skip frame | Log only |
| Worker crash mid-write | Daemon detects via orchestrator, unlinks shm | Session terminates |

### 13.2 Existing Error Codes (Unchanged)

The client-facing error codes from `rocket/inspect` do not change. Shm failures are handled transparently via base64 fallback. The client never knows which transport was used.

---

## 14. Latency Budget

### 14.1 Per-Tensor (17 MB, Llama-3-8B residual stream)

| Step | Current (base64) | With shm (WU 1.8) |
|------|------------------|--------------------|
| GPU → CPU DMA | ~2 ms | ~2 ms |
| data_ptr() zero-copy access | N/A | ~0 ns |
| BLAKE3 hash | ~6 ms (daemon) | ~6 ms (worker, GIL released) |
| Serialization | ~8 ms (tobytes + base64) | ~0.3 ms (memcpy to shm) |
| Wire transfer | ~12 ms (89 MB JSON parse) | ~0.002 ms (control msg only) |
| Deserialization | ~6 ms (base64 decode) | ~0 ms (mmap read, no re-hash) |
| **Total** | **~34 ms** | **~8 ms** |
| **Speedup** | | **4.25×** |

The BLAKE3 hash moves from daemon to worker but the cost is unchanged. The win comes from eliminating base64 encode/decode (~14ms) and JSON parsing of the inflated payload (~12ms).

### 14.2 Per-Tick (32-layer model, all components captured)

Assuming ~64 module outputs captured per tick (conservative — many are small):

| Scenario | Current | With shm |
|----------|---------|----------|
| All small tensors (< 1MB each) | ~2.2s | ~0.5s |
| Mixed (mostly small, few 17MB) | ~4s | ~1s |

---

## 15. Testing Strategy

### 15.1 Unit Tests (Rust)

**Ring buffer (`ring_buffer.rs`):**
- Control block serialize/parse round-trip
- Magic validation (correct, incorrect, partially initialized)
- Slot offset computation for various `maketic` values
- Ring full detection (`maketic - nettics == BACKUPTICS`)
- Monotonic cursor wraparound (values near `u64::MAX`)
- Power-of-2 masking produces correct slot indices

**ProbeFrame v2 (`probe_frame.rs`):**
- Round-trip with new alignment (all existing tests updated)
- Generation field preserved
- `_pad0` field is zeroed
- u64 fields at correct 8-byte-aligned offsets

**PyO3 shm_ops:**
- Atomic store/load round-trip on a heap-allocated buffer (simulates shm)
- Release/Acquire ordering verified via concurrent test (worker thread writes, reader thread reads)

**TensorStore improvements:**
- `insert_with_id()` accepts pre-computed hash
- `insert_with_id()` dedup returns existing handle
- Generation-based LRU: touch is O(1), eviction picks minimum generation
- Regression: all existing TensorStore tests still pass

### 15.2 Integration Tests

**Shared memory lifecycle (`tests/test_shm_lifecycle.rs`):**
- Python creates region → Rust opens by name → both see same bytes
- Python writes control block → Rust validates magic
- Worker crash → daemon unlinks region → shm_open fails (cleaned up)
- macOS name length validation (reject names > 30 chars)

### 15.3 E2E Test (`tests/test_e2e_shm.py`)

Against tiny-random-LlamaForCausalLM (2 layers, 4 heads):

1. Initialize + attach (verify shm region created)
2. Step forward (tensors flow through shm)
3. Inspect — verify tensor summary matches expected (same as non-shm)
4. Inspect slice — verify bytes match (same as non-shm)
5. Inspect large tensor — if exceeds slot, verify base64 fallback works
6. Detach — verify shm region cleaned up

### 15.4 TCK Scenarios (`tck/protocol/shm.feature`)

4 scenarios as listed in the TCK Contract section.

### 15.5 Benchmark (`tests/bench_tensor_transfer.py`)

Measure per-tensor latency for:
- base64 path (existing)
- shm path (new)
- Various tensor sizes: 1KB, 1MB, 16MB, 64MB

Report mean, p50, p95, p99. Target: shm path ≤ 10ms for 17MB tensor.

---

## 16. What This Does NOT Include

- **GPU-side summary stats forwarding**: Stats are still computed on the daemon side after reading from shm. Forwarding GPU-computed stats through ProbeFrame reserved bytes is a future optimization.
- **Zero-copy TensorStore**: The daemon still copies tensor bytes from the mmap view into an owned `Vec<u8>` in TensorStore. Storing references to the shm region would couple store lifetime to shm lifetime — deferred.
- **Multi-threaded BLAKE3**: Single-threaded AVX2 for now (~5.7ms/17MB). Rayon-based tree hashing could cut this to ~1.5ms. Deferred — not the bottleneck.
- **Streaming ring buffer**: Fixed-size slots for Phase 1. Streaming (variable-size frames) avoids wasted space but adds wraparound complexity. Deferred.
- **Multi-GPU shm regions**: One region per rank. Cross-rank aggregation is not addressed (single-rank capture only in Phase 1).
- **Binary control messages**: Control plane stays JSON-RPC. CBOR or flatbuffers for control messages is a separate optimization and not worth the protocol change.
- **The double-mmap trick**: Mapping the same pages twice for wraparound-free reads. Does not work reliably on macOS. Not needed with fixed-size slots.
