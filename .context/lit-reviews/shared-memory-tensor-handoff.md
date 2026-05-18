---
topic: Shared-Memory Tensor Handoff — cross-process and in-process patterns for moving tensor data between Python (PyTorch) and Rust in a three-process architecture (daemon, orchestrator, worker)
status: draft
created: 2026-05-18
sources: Python multiprocessing.shared_memory (CPython 3.13+), POSIX shm_open(3), nix crate (Rust), PyO3, pyo3-dlpack, DLPack protocol, BLAKE3, iceoryx2, PyTorch internals (TensorImpl, data_ptr, share_memory_), torch.histc, Welford 1962, Chan/Golub/LeVeque 1979
---

# Shared-Memory Tensor Handoff: Lit Review

Patterns for moving captured tensor data from PyTorch (in the worker process) through the Rust daemon for inspection, summarization, and slicing. Covers cross-process shared memory, in-process PyO3 access, ring buffer design, BLAKE3 hashing, and GPU-side summary statistics.

## 1. Python multiprocessing.shared_memory

### 1.1 API

```python
SharedMemory(name=None, create=False, size=0, *, track=True)
```

| Parameter | Meaning |
|-----------|---------|
| `name` | String identifier. On POSIX, Python prepends `/` if missing. Auto-generated names use prefix `/psm_` + `secrets.token_hex()`. |
| `create` | `True` = create new (`O_CREAT | O_EXCL | O_RDWR`). `False` = attach to existing (`O_RDWR`). |
| `size` | Bytes. Only meaningful when `create=True`. Platform may round up to page size. |
| `track` | Python 3.13+. When `True`, registers with resource_tracker for auto-cleanup. |

The `.buf` property returns a `memoryview` over the mmap'd region -- direct read/write, no copy.

`close()` releases this process's file descriptor. `unlink()` destroys the underlying shared memory object. On POSIX, `unlink()` calls `shm_unlink()`. On Windows, `unlink()` is a no-op (Windows auto-deletes when all handles close).

### 1.2 Implementation (CPython Source)

The CPython implementation in `Lib/multiprocessing/shared_memory.py`:

1. On POSIX: calls `_posixshmem.shm_open(name, flags, mode)` (a C extension wrapping `shm_open(3)`).
2. Then `os.ftruncate(fd, size)` to set the size.
3. Then `mmap.mmap(fd, size)` to map into the address space.
4. The `.buf` property returns a `memoryview` over the mmap.

Name handling: POSIX names must start with `/`. Python's `_prepend_leading_slash = True` on POSIX adds the slash automatically. The `.name` property strips it back off when reporting to user code. Auto-generated names use `"/psm_" + secrets.token_hex()` truncated to 14 characters.

### 1.3 Can Rust Open the Same Region by Name?

Yes. The POSIX shared memory object created by Python's `SharedMemory` is a standard POSIX shm object. Any process that knows the name can open it via `shm_open(3)` and `mmap(2)`.

In Rust, via the `nix` crate:

```rust
use nix::sys::mman::{shm_open, mmap, MapFlags, ProtFlags};
use nix::fcntl::OFlag;
use nix::sys::stat::Mode;

// Open the same shared memory region Python created
let fd = shm_open(
    "/rs-session123-rank0",  // same name Python used
    OFlag::O_RDWR,           // read-write
    Mode::empty(),           // mode ignored for open (not create)
)?;

// mmap it into this process's address space
let ptr = unsafe {
    mmap(
        None,                // let OS choose address
        size.try_into()?,    // same size Python allocated
        ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
        MapFlags::MAP_SHARED,
        &fd,
        0,                   // offset
    )?
};

// ptr is now a *mut c_void pointing to the same bytes Python sees
let slice: &[u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
```

The `nix::sys::mman::shm_open` function signature:

```rust
pub fn shm_open<P: ?Sized + NixPath>(
    name: &P,
    flag: OFlag,
    mode: Mode,
) -> Result<OwnedFd>
```

This returns an `OwnedFd` which auto-closes on drop.

### 1.4 Lifecycle: Who Creates, Who Destroys?

**ADR-0004 convention**: `/rs-<session>-<rank>` naming.

**Creator**: The Python host (worker process) creates the shared memory on startup. It knows the required size (based on the largest expected tensor for the ring buffer).

**Consumer**: The Rust daemon opens the same region by name via `shm_open` + `mmap`.

**Destruction**: The Python host calls `unlink()` on detach/shutdown. If the host crashes, the daemon must clean up: open the known name, call `shm_unlink()`. On Linux, orphaned POSIX shm objects persist in `/dev/shm/` until explicitly unlinked or system reboot. On macOS, they persist until unlink or reboot (no `/dev/shm` filesystem, but the kernel tracks them).

**Critical gotcha -- resource_tracker (Python < 3.13)**: Python's `multiprocessing` module spawns a resource_tracker process that auto-unlinks shared memory when the creating process exits. If the worker uses `subprocess` or standalone processes (not `multiprocessing.Process`), each gets its own resource_tracker, and the first to exit unlinks the shared memory out from under the others. **Fix**: Use `track=False` (Python 3.13+) or monkey-patch the resource tracker (Python < 3.13). Since the daemon (Rust) is responsible for lifecycle management and will call `shm_unlink()` on cleanup, Python should always use `track=False`.

### 1.5 Platform Differences

| Aspect | Linux | macOS |
|--------|-------|-------|
| Backing | tmpfs mounted at `/dev/shm/` | Kernel-managed, no filesystem exposure |
| Visibility | `ls /dev/shm/` shows shared memory objects | No filesystem interface; `ipcs -m` shows SysV only, not POSIX |
| Default size limit | `/dev/shm` is typically 50% of RAM (tmpfs default) | POSIX shm: **not** constrained by `kern.sysv.shmmax` (that's SysV only). Practical limit is available RAM. |
| Name length | 255 characters (PATH_MAX for shm) | 31 characters (PSHMNAMLEN on macOS). **This is a hard constraint.** Names like `/rs-<session>-<rank>` must be kept short. |
| Cleanup on crash | Objects persist in `/dev/shm/` until `shm_unlink()` or reboot | Objects persist in kernel until `shm_unlink()` or reboot |
| Page size | 4 KiB (default), 2 MiB (huge pages) | 16 KiB on Apple Silicon, 4 KiB on Intel |

**macOS name length gotcha**: PSHMNAMLEN is 31 characters including the leading `/`. Names must be at most 30 characters after the slash. The convention `/rs-<session>-<rank>` must use short session IDs (e.g., 8 hex chars, not full UUIDs).

**macOS POSIX shm size**: The commonly cited 4 MB limit on macOS (`kern.sysv.shmmax`) applies only to System V shared memory (`shmget()`), NOT to POSIX shared memory (`shm_open()`). POSIX shm on macOS is limited only by available virtual memory. This is confirmed by multiple sources: "System V and POSIX shared memory segments can coexist in a system. However, they are totally independent and have no interoperability. The config and limits of System V does not apply to POSIX and vice versa."

### 1.6 Performance: Is It Zero-Copy?

From the creating process's perspective: `SharedMemory.buf` is a `memoryview` over an mmap'd region. Writing to it is a direct memory store -- no kernel copy, no serialization.

From the consuming process's perspective: `shm_open` + `mmap` maps the same physical pages into the consumer's virtual address space. Reading is a direct memory load -- zero copy.

**The only copies in the pipeline are**:
1. GPU-to-CPU: `tensor.cpu()` -- unavoidable, involves DMA transfer.
2. CPU tensor data into shared memory ring buffer: `memcpy` from the tensor's data buffer into the mmap'd slot. **This is one copy.** Can potentially be eliminated (see Section 2.4).

### 1.7 Writing a PyTorch Tensor into Shared Memory

**Standard approach (one copy)**:

```python
# In the hook, after GPU->CPU transfer:
cpu_tensor = tensor.detach().contiguous().cpu()
raw_bytes = cpu_tensor.numpy().tobytes()  # copies into a new bytes object
shm.buf[offset:offset+len(raw_bytes)] = raw_bytes  # copies into shm
# Two copies total: tensor -> bytes -> shm
```

**Better approach (one copy)**:

```python
cpu_tensor = tensor.detach().contiguous().cpu()
# numpy() shares memory with the tensor -- zero copy
np_array = cpu_tensor.numpy()
# Write directly from numpy's buffer into shared memory
shm.buf[offset:offset+np_array.nbytes] = np_array.data  # memcpy via buffer protocol
# One copy: tensor buffer -> shm (via memoryview assignment)
```

**Best approach (zero extra copy -- tensor lives in shared memory)**:

```python
import numpy as np
# Create a numpy array backed by shared memory
np_arr = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf, offset=slot_offset)
# Tell PyTorch to write the CPU copy directly into this array
cpu_tensor = tensor.detach().contiguous()
# torch.Tensor.numpy() on a CPU contiguous tensor shares storage
# But we need to go the other direction: write INTO the shm-backed array
np.copyto(np_arr, cpu_tensor.numpy())  # single memcpy into shm
```

Or using `torch.frombuffer`:

```python
# Create a tensor whose storage IS the shared memory region
shm_tensor = torch.frombuffer(shm.buf, dtype=tensor.dtype, count=numel, offset=slot_offset)
# Copy the captured tensor directly into the shm-backed tensor
shm_tensor.copy_(cpu_tensor)  # single memcpy into shm
```

**Verdict**: One `memcpy` from CPU tensor buffer into shared memory is the practical minimum. The GPU-to-CPU DMA transfer (~2ms for 17MB) dominates; the memcpy into shm (~0.3ms for 17MB at ~50 GB/s memory bandwidth) is noise.


## 2. PyO3 In-Process Approach (Worker: Rust + Python Same Address Space)

### 2.1 Architecture Recap

The worker binary embeds Python via PyO3 (`pyo3::prepare_freethreaded_python()`). Rust and Python share the same virtual address space. This means:

- Rust can call Python functions and receive Python objects.
- Python objects (including tensors) live in the same heap.
- Raw pointers from Python are valid in Rust (same address space).
- **No IPC needed between Rust and Python within the worker.**

### 2.2 Getting a Raw Pointer to Tensor Data via PyO3

`tensor.data_ptr()` returns the address of the first element of the tensor's data buffer. For CPU tensors, this is a host memory address. For CUDA tensors, this is a **device pointer** (not dereferenceable from CPU code).

From Rust via PyO3:

```rust
Python::with_gil(|py| {
    let tensor: &Bound<'_, PyAny> = /* ... */;

    // Ensure contiguous CPU tensor
    let cpu_tensor = tensor.call_method0("contiguous")?
        .call_method1("to", ("cpu",))?;

    // Get raw pointer
    let data_ptr: usize = cpu_tensor
        .call_method0("data_ptr")?
        .extract::<usize>()?;

    // Get size in bytes
    let numel: usize = cpu_tensor.call_method0("numel")?.extract()?;
    let element_size: usize = cpu_tensor.call_method0("element_size")?.extract()?;
    let byte_size = numel * element_size;

    // Create a Rust slice from the raw pointer -- ZERO COPY
    let slice: &[u8] = unsafe {
        std::slice::from_raw_parts(data_ptr as *const u8, byte_size)
    };

    // Now we can hash it, copy it, whatever -- no copy yet
    let hash = blake3::hash(slice).to_hex().to_string();
    // ...
})
```

**This is truly zero-copy within the worker process.** The `data_ptr()` gives direct access to the tensor's underlying storage. No serialization, no memcpy.

### 2.3 GPU vs CPU Tensors

| Tensor location | `data_ptr()` returns | Dereferenceable from CPU? | Action needed |
|----------------|---------------------|--------------------------|---------------|
| CPU | Host virtual address | Yes | Direct access |
| CUDA | Device pointer (GPU VRAM address) | **No** -- segfault | Must call `.cpu()` first |
| MPS (Apple) | Metal buffer pointer | **No** | Must call `.cpu()` first |

**There is no way to read GPU tensor data from CPU code without a GPU-to-CPU transfer.** This is fundamental to how GPUs work -- the tensor data lives in GPU VRAM, which is not mapped into the CPU's virtual address space. The `.cpu()` call triggers a DMA transfer.

### 2.4 GIL Considerations

**Do you need the GIL to read tensor data via `data_ptr()`?**

Strictly: you need the GIL to *call* `data_ptr()` (it's a Python method call). But once you have the raw pointer, the underlying memory is just bytes in the process's address space. You could release the GIL and read from the pointer.

**However**, this is dangerous without ensuring the tensor stays alive:

```rust
Python::with_gil(|py| {
    let tensor = /* get tensor */;
    let ptr = tensor.call_method0("data_ptr")?.extract::<usize>()?;
    let size = /* compute size */;

    // DANGER: if we release GIL here and Python GC collects the tensor,
    // the memory at ptr becomes invalid

    // SAFE pattern: hold a Py<PyAny> reference to prevent GC
    let tensor_ref: Py<PyAny> = tensor.clone().unbind();

    py.allow_threads(|| {
        // GIL released -- other Python threads can run
        // But tensor_ref prevents GC of the tensor object
        // (Py<T> prevents GC by holding a strong reference)
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
        let hash = blake3::hash(slice);
        // ... process the data ...
        hash
    });

    // Drop tensor_ref -- now GC can collect if refcount drops to 0
    drop(tensor_ref);
})
```

**Key safety rule**: Hold a `Py<PyAny>` (or `PyObject`) reference to the tensor for the entire duration of the raw pointer access. `Py<T>` increments the Python reference count, preventing garbage collection even without the GIL.

### 2.5 What If Python GCs the Tensor While Rust Reads?

If the tensor's Python refcount drops to zero while Rust holds a raw pointer to its data:

1. Python's GC calls the tensor's `__del__` / C-level `tp_dealloc`.
2. PyTorch's `TensorImpl` destructor frees the storage.
3. The allocator (CPU allocator, or CUDA caching allocator) marks the memory as available.
4. Rust's raw pointer now points to freed memory -- **use-after-free**.

This is a hard crash (segfault) or silent data corruption. The fix is always: hold a `Py<PyAny>` reference.

### 2.6 Implication for the Architecture

Since Rust and Python share an address space in the worker, **within the worker process there is no need for shared memory for tensor access**. The Rust side of the worker can:

1. Call the Python hook capture function.
2. Get `data_ptr()` and read tensor bytes directly.
3. Compute BLAKE3 hash on the raw bytes (GIL released, tensor pinned via `Py<PyAny>`).
4. Compute summary stats on the raw bytes (GIL released).
5. Copy bytes into the shared-memory ring buffer for the daemon.

The shared memory is only needed for **cross-process** communication: worker-to-daemon.


## 3. Ring Buffer Design for Tensor Handoff

### 3.1 The Problem

Tensors captured during a forward pass must flow from the worker process to the daemon process. Tensors have variable sizes (a bias vector might be 4 KB; an attention matrix might be 256 MB). The flow is strictly one-directional for data (worker -> daemon) with a reverse acknowledgment channel.

### 3.2 SPSC Ring Buffer Basics

Single-Producer, Single-Consumer (SPSC) ring buffers are the simplest correct lock-free data structure. One thread/process writes, one reads. No locks needed -- just atomic read/write cursors.

**Classic layout**:
```
┌──────────────────────────────────────────────────┐
│  slot 0  │  slot 1  │  slot 2  │  ...  │  slot N │
└──────────────────────────────────────────────────┘
     ↑ read_cursor          ↑ write_cursor
```

Producer advances `write_cursor` after writing. Consumer advances `read_cursor` after reading. Buffer is empty when `read == write`, full when `write - read == N`.

### 3.3 Variable-Size Messages: Slot-Based vs Streaming

**Slot-based (fixed slots, variable fill)**:
- Pre-allocate N slots of fixed maximum size (e.g., 64 MB each).
- Each slot holds one ProbeFrame (128-byte header + tensor bytes up to slot size).
- Wastes space when tensors are small, but simple.
- **Pro**: No fragmentation. Reader always reads from aligned offsets.
- **Con**: Slot size limits maximum tensor size. Wastes memory for small tensors.

**Streaming (byte-oriented)**:
- Write frames sequentially into a flat circular buffer.
- Frame header includes length field so reader knows where the next frame starts.
- **Pro**: No wasted space. Arbitrarily large tensors (up to buffer size).
- **Con**: Frames can wrap around the buffer boundary, requiring split reads. More complex.

**Recommendation for rocket_surgeon**: **Slot-based with tiered slot sizes.** Most tensors in a transformer forward pass fall into predictable size categories (bias: ~KB, weight projections: ~MB, activations: ~tens of MB). Use a small number of slot size tiers. Or simpler: use the streaming approach with the existing ProbeFrame header (which already contains a `size` field at offset 60), and handle wraparound in the reader.

ADR-0006 already specifies slot-based with ProbeFrame headers. The `size` field in the 128-byte header tells the daemon how many bytes follow.

### 3.4 Notification Mechanisms

The ring buffer itself handles data transfer. But the consumer needs to know when new data is available. Options:

| Mechanism | Linux | macOS | Latency | Complexity |
|-----------|-------|-------|---------|------------|
| `eventfd` | Yes | **No** | ~1-2 us | Low |
| `kqueue` | No (use epoll) | Yes | ~1-2 us | Medium |
| Unix domain socket (1-byte write) | Yes | Yes | ~2-5 us | Low |
| Pipe (1-byte write) | Yes | Yes | ~2-5 us | Low |
| Futex | Yes | **No** (macOS has `os_unfair_lock`) | ~0.5 us | High |
| Polling (busy-wait) | Yes | Yes | ~0 ns | Burns CPU |

ADR-0004 already decided: **Unix domain socket, 1-byte write per frame**. This is cross-platform (Linux + macOS), integrates naturally with tokio's async runtime (the daemon is async), and adds negligible latency relative to the GPU-to-CPU transfer.

The daemon's async event loop (tokio) can `poll` the Unix socket for readability. When a byte arrives, it reads the next frame from the shared memory ring buffer.

### 3.5 Existing Rust Crates

**For in-process SPSC (reference, not direct use)**:
- `rtrb` -- wait-free SPSC ring buffer by mgeier. Clean API, well-tested. But in-process only (uses `Arc<SharedRb>`).
- `ringbuf` -- lock-free SPSC FIFO with direct access to inner data. In-process.
- `ringbuffer-spsc` -- `#[no_std]` SPSC. In-process.

**For cross-process shared memory IPC**:
- `iceoryx2` -- Eclipse project. True zero-copy cross-process IPC. Rust core. Supports Linux (Tier 1), macOS (Tier 2), Windows (Tier 2). Uses POSIX shared memory internally. SPSC queues with completion queues for back-pressure. ~100ns latency on some systems. Apache-2.0 / MIT dual license.
- `ipmpsc` -- Inter-process MPSC channels. Older (2021), less maintained.
- `mmap-sync` (Cloudflare) -- Concurrent data access using mmap'd files. Wait-free synchronization. Not specifically a ring buffer.

**Recommendation**: Do not take a dependency on iceoryx2 or any external IPC crate (per project principle: "reimplement everything; prior art is reference, never a dependency"). Study iceoryx2's SPSC queue design as reference, implement the ring buffer from scratch using `nix` crate's `shm_open` + `mmap`. The ring buffer is simple enough (fixed slots, atomic cursors, ProbeFrame headers) that a custom implementation is straightforward and avoids dragging in a large framework.

### 3.6 Proposed Ring Buffer Layout

```
Shared Memory Region: /rs-<session>-<rank>
┌─────────────────────────────────────────────────┐
│ Control Block (cache-line aligned, 128 bytes)    │
│  write_cursor: AtomicU64  (slot index)           │
│  read_cursor:  AtomicU64  (slot index)           │
│  slot_count:   u64                               │
│  slot_size:    u64                               │
│  _padding: [u8; 96]                              │
├─────────────────────────────────────────────────┤
│ Slot 0: [ProbeFrame header (128B) | data (up to slot_size - 128)] │
│ Slot 1: [ProbeFrame header (128B) | data ...]                     │
│ ...                                                                │
│ Slot N-1: [...]                                                    │
└─────────────────────────────────────────────────┘
```

**Write protocol** (Python worker):
1. Read `write_cursor` and `read_cursor`. If `(write_cursor - read_cursor) == slot_count`, buffer is full -- block or drop.
2. Compute slot offset: `128 + (write_cursor % slot_count) * slot_size`.
3. Write ProbeFrame header at slot offset.
4. Write tensor bytes at slot offset + 128.
5. Memory fence (`atomic::fence(Release)`).
6. Increment `write_cursor` (atomic store with Release ordering).
7. Write 1 byte to notification Unix socket.

**Read protocol** (Rust daemon):
1. Receive notification byte on Unix socket.
2. Read `write_cursor` (atomic load with Acquire ordering).
3. While `read_cursor < write_cursor`:
   a. Parse ProbeFrame header at current slot.
   b. Process tensor bytes (BLAKE3 hash, store in TensorStore).
   c. Increment `read_cursor` (atomic store with Release ordering).

The atomic ordering (Release on write, Acquire on read) ensures the consumer sees all bytes written by the producer before the cursor advances. This is the standard SPSC lock-free protocol.

**Cross-process atomics**: On POSIX systems, `AtomicU64` in shared memory works correctly because:
- The shared memory is backed by the same physical pages.
- Modern CPUs provide cache coherence across all cores (MESI protocol).
- Atomic operations on shared-memory-mapped regions are guaranteed coherent on x86 and ARM.
- **Caveat**: This assumes the atomic variables are properly aligned (naturally aligned to 8 bytes for u64).


## 4. BLAKE3 Hashing for Content-Addressable Storage

### 4.1 Performance on Typical Tensor Sizes

BLAKE3 throughput on modern CPUs:

| Configuration | Throughput | Source |
|--------------|-----------|--------|
| Single-threaded, AVX2 | ~2.5-3 GB/s | BLAKE3 benchmarks |
| Single-threaded, AVX-512 | ~4.5 GB/s | BLAKE3 benchmarks |
| Single-threaded, no SIMD | ~1 GB/s | Conservative estimate |
| Multi-threaded (Rayon), 4 cores | ~10-12 GB/s | BLAKE3 tree hashing |
| Multi-threaded, 16 cores | ~90 GB/s | Linear scaling |

**For a 4096x4096 float32 tensor (64 MB)**:

| Configuration | Time |
|--------------|------|
| Single-threaded, AVX2 | ~21 ms |
| Single-threaded, AVX-512 | ~14 ms |
| Multi-threaded, 4 cores | ~5 ms |

**For a typical Llama-3-8B residual stream (4096 x batch x fp16, ~17 MB at batch=1)**:

| Configuration | Time |
|--------------|------|
| Single-threaded, AVX2 | ~5.7 ms |
| Multi-threaded, 4 cores | ~1.5 ms |

Compared to the GPU-to-CPU transfer time (~2 ms for 17 MB), BLAKE3 hashing is comparable in cost. The hash can be computed concurrently with other work (e.g., while the barrier gate is waiting for daemon commands).

### 4.2 Hashing from mmap'd Regions

Yes. `blake3::hash()` takes `&[u8]`. An mmap'd region accessed as a `&[u8]` slice works identically to any other byte slice. The BLAKE3 crate also provides `Hasher::update_mmap()` for file-backed mmaps, with a heuristic (file >= 16 KiB) to decide whether mmap is faster than read. For shared-memory-backed mmaps, just use `blake3::hash(slice)` directly -- the data is already in memory.

```rust
// Hashing directly from shared memory -- zero copy
let shm_slice: &[u8] = /* mmap'd shared memory region, tensor bytes portion */;
let hash = blake3::hash(shm_slice);
```

### 4.3 Hashing GPU Tensors

**Cannot hash a GPU tensor without CPU transfer.** BLAKE3 runs on CPU. GPU tensor data lives in VRAM, which is not accessible from CPU code. The tensor must be transferred to CPU first (`.cpu()` or `.to('cpu')`), then hashed.

This is unavoidable with any CPU-based hash function. A theoretical alternative would be a GPU-based hash kernel, but:
- BLAKE3's tree structure could parallelize on GPU, but no production implementation exists.
- The GPU-to-CPU transfer is needed anyway for the daemon (which runs on CPU).
- Hashing on GPU and transferring only the 32-byte hash would save bandwidth, but the daemon also needs the raw bytes for slicing.

**Conclusion**: Transfer to CPU, then hash. The GPU-to-CPU transfer is the bottleneck, not the hash computation.

### 4.4 Incremental / Streaming Hashing

The `blake3::Hasher` supports incremental hashing via `update()`:

```rust
let mut hasher = blake3::Hasher::new();
hasher.update(chunk1);
hasher.update(chunk2);
// ...
let hash = hasher.finalize();
```

`finalize()` is idempotent -- calling it does not consume the hasher. You can call `update()` again after `finalize()` to extend the hash (though the hash value changes).

This is useful if tensor data arrives in chunks (e.g., streaming from multiple ring buffer slots), but for rocket_surgeon's design where each ProbeFrame contains a complete tensor, the one-shot `blake3::hash(data)` is simpler.

The `Hasher` also implements `std::io::Write`, so you can use `std::io::copy()` from any `Read` source.


## 5. Summary Statistics on GPU

### 5.1 Relevant PyTorch Ops

For computing summary stats on GPU before CPU transfer:

| Statistic | PyTorch op | GPU support | Notes |
|-----------|-----------|-------------|-------|
| Mean | `tensor.mean()` | Yes | Single reduction kernel |
| Std | `tensor.std()` | Yes | Uses Welford internally for numerical stability |
| Min | `tensor.min()` | Yes | Single reduction |
| Max | `tensor.max()` | Yes | Single reduction |
| Abs max | `tensor.abs().max()` | Yes | Two ops, likely fused by torch.compile |
| L2 norm | `tensor.norm(2)` | Yes | Uses scaled accumulation internally |
| Sparsity | `(tensor.abs() < eps).float().mean()` | Yes | Comparison + reduction |
| Histogram | `torch.histc(tensor, bins, min, max)` | **Partial** | See below |
| Top-k | `tensor.abs().topk(k)` | Yes | GPU-optimized radix-based selection |

### 5.2 Single Pass vs Multiple Passes on GPU

**On GPU, individual reduction ops are already highly parallel and fast.** A single `tensor.mean()` call launches a GPU kernel that reduces millions of elements in microseconds. The bottleneck is kernel launch overhead (~5-10 us per kernel), not the computation itself.

**Can all stats be computed in a single fused kernel?** Not with stock PyTorch ops. Each op (`mean()`, `std()`, `min()`, `max()`) launches a separate kernel. However:

1. **torch.compile can fuse pointwise + reduction ops** in some cases, but not arbitrary multi-output reductions.
2. **Triton custom kernel**: Could write a single kernel that computes mean/std/min/max/abs_max/sparsity/l2_norm in one pass. This would save ~7 kernel launches (~50 us total). Whether this is worth the complexity depends on profiling.
3. **For the skeleton (WU 1.5)**: Use individual PyTorch ops. Optimize later if kernel launch overhead is measurable relative to the forward pass computation time (typically hundreds of ms).

**Practical approach for rocket_surgeon**:

```python
# Phase 1: ops that can run independently (could be overlapped)
mean_val = tensor.mean().item()
std_val = tensor.std().item()
min_val = tensor.min().item()
max_val = tensor.max().item()
abs_max_val = tensor.abs().max().item()
l2_norm = tensor.norm(2).item()
sparsity = (tensor.abs() < 1e-8).float().mean().item()

# Phase 2: histogram needs min/max (already computed, but on CPU now)
hist = torch.histc(tensor.float(), bins=32, min=min_val, max=max_val)
hist_counts = hist.tolist()

# Phase 3: top-k
abs_tensor = tensor.abs().view(-1)
top_k_values, top_k_indices = abs_tensor.topk(10)
```

**Key insight**: Computing stats on GPU and transferring only the summary (~200 bytes of scalars) is massively cheaper than transferring the full tensor to CPU and computing stats there. For a 64 MB tensor, GPU-to-CPU transfer takes ~8ms; the GPU reductions take ~0.1ms combined.

### 5.3 torch.histc Details and Edge Cases

```python
torch.histc(input, bins=100, min=0, max=0) -> Tensor
```

Computes a histogram of `input` with `bins` equal-width bins in the range `[min, max]`.

**Edge cases**:
- **Empty tensor**: Returns a tensor of zeros with `bins` elements.
- **min == max**: All values fall into a single bin. PyTorch handles this by effectively putting everything in one bin.
- **Values outside [min, max]**: Excluded from the histogram (not counted).
- **NaN values**: Behavior is undefined/platform-dependent. **Must filter NaNs before calling histc.**
- **Inf values**: If min/max are finite, Inf values are excluded (outside range). If min or max is Inf, behavior is undefined.
- **dtype**: `torch.histc` works on float tensors. Integer tensors must be cast to float first.

**GPU support**: `torch.histc` works on CUDA tensors (the early bug from 2017 where it didn't work on CUDA was fixed long ago). However, `torch.histogram` (the more featured version with edge computation) historically lacked CUDA support (GitHub issue #69519). Use `torch.histc` for GPU, `torch.histogram` for CPU.

**Memory overhead of 32-bin histogram on GPU**: Negligible. The output tensor is 32 float32 values = 128 bytes. The kernel uses minimal workspace.

### 5.4 Welford's Algorithm -- Applicability

Welford's algorithm is primarily relevant for **CPU-side** streaming computation (what tensor_stats.rs already implements). On GPU:

- PyTorch's `tensor.std()` / `tensor.var()` already use Welford internally for numerical stability (visible in the ATen C++ source).
- PyTorch's LayerNorm CUDA kernel uses Welford with warp-level reductions for computing mean/variance.
- For rocket_surgeon's GPU-side stats: just call PyTorch ops, which use Welford internally.

Welford is directly applicable for:
1. **CPU-side stats in tensor_stats.rs** (already implemented).
2. **Multi-GPU merge via Chan/Golub/LeVeque** (already implemented in tensor_stats.rs).
3. **Streaming stats if we ever compute stats incrementally** (e.g., across batches).


## 6. End-to-End Data Flow Assessment

### 6.1 The Complete Picture

Given the three-process architecture (daemon, orchestrator, worker) where the worker embeds Python via PyO3:

```
┌──────────────────────────────────────────────────────────────┐
│  WORKER PROCESS (Rust binary + embedded Python via PyO3)      │
│                                                                │
│  ┌─────────────────────────────────────┐                      │
│  │ Python: PyTorch hook fires          │                      │
│  │  tensor is on GPU (CUDA)            │                      │
│  │                                     │                      │
│  │  Step 1: Compute GPU summary stats  │  ← GPU reductions   │
│  │    mean, std, min, max, abs_max,    │    ~0.1 ms           │
│  │    l2_norm, sparsity, histc, topk   │                      │
│  │                                     │                      │
│  │  Step 2: CPU transfer               │  ← DMA transfer     │
│  │    cpu_tensor = tensor.detach()     │    ~2 ms (17 MB)     │
│  │      .contiguous().cpu()            │                      │
│  │    torch.cuda.Event sync            │                      │
│  └─────────┬───────────────────────────┘                      │
│            │ (same address space -- function call, not IPC)    │
│  ┌─────────▼───────────────────────────┐                      │
│  │ Rust: In-process tensor processing  │                      │
│  │                                     │                      │
│  │  Step 3: Get raw pointer via PyO3   │  ← zero copy         │
│  │    data_ptr() → &[u8]              │    ~0 ns              │
│  │    (hold Py<PyAny> to pin tensor)   │                      │
│  │                                     │                      │
│  │  Step 4: BLAKE3 hash (GIL released) │  ← ~5.7 ms (17 MB)  │
│  │    blake3::hash(raw_slice)          │    AVX2, single-thr  │
│  │                                     │                      │
│  │  Step 5: Copy into shared memory    │  ← memcpy            │
│  │    ring buffer slot                 │    ~0.3 ms (17 MB)   │
│  │    Write ProbeFrame header (128B)   │                      │
│  │    Write tensor bytes               │                      │
│  │                                     │                      │
│  │  Step 6: Notify daemon              │  ← 1-byte write      │
│  │    Unix domain socket               │    ~2 us              │
│  └─────────────────────────────────────┘                      │
└──────────────────────────────────────────────────────────────┘
            │
            │  (cross-process, via shared memory + Unix socket)
            │
┌───────────▼──────────────────────────────────────────────────┐
│  DAEMON PROCESS (Rust binary)                                 │
│                                                                │
│  Step 7: Receive notification byte     ← tokio async read     │
│                                           ~2 us                │
│                                                                │
│  Step 8: Read from shared memory       ← mmap'd, zero copy    │
│    Parse ProbeFrame header                ~0 ns (page fault    │
│    Access tensor bytes via &[u8] view     on first access)     │
│                                                                │
│  Step 9: Store in TensorStore          ← already have hash    │
│    Content-addressable by BLAKE3 ID       from worker          │
│    Dedup check, LRU insert                                     │
│                                                                │
│  Step 10: Compute CPU-side stats       ← ~8 ms (17 MB)        │
│    tensor_stats::compute_summary()        (or use GPU stats    │
│    (or cache GPU-computed stats          sent with ProbeFrame) │
│     forwarded from worker)                                     │
│                                                                │
│  Step 11: Serve to clients via         ← JSON-RPC response    │
│    inspect → TensorSummary (~200 B)       ~50 us               │
│    slice → bounded byte range             per request          │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Total Latency Budget (17 MB tensor, Llama-3-8B residual)

| Step | Time | Location |
|------|------|----------|
| GPU summary stats | ~0.1 ms | Worker (GPU) |
| GPU-to-CPU DMA transfer | ~2 ms | Worker |
| BLAKE3 hash (single-thread AVX2) | ~5.7 ms | Worker (CPU, GIL released) |
| memcpy into shared memory | ~0.3 ms | Worker |
| Unix socket notification | ~0.002 ms | Worker -> Daemon |
| Daemon reads from shm | ~0 ms (already paged in) | Daemon |
| CPU-side summary stats | ~8 ms | Daemon |
| **Total** | **~16 ms** | |

**Optimization opportunities**:
- BLAKE3 with multi-threading (4 cores): 5.7 ms -> ~1.5 ms. Saves ~4 ms.
- Send GPU-computed summary stats with the ProbeFrame (avoid CPU-side recomputation): saves ~8 ms.
- Overlap BLAKE3 hashing with GPU-to-CPU transfer of the next tensor: pipeline savings.

With optimizations, per-tensor overhead could be ~4 ms, dominated by the GPU-to-CPU transfer.

### 6.3 Is Shared Memory Needed?

**Between worker's Rust and Python code**: No. They share an address space. `data_ptr()` gives zero-copy access.

**Between worker process and daemon process**: **Yes**, shared memory is the right choice. Alternatives:
- JSON-RPC with base64: adds 33% overhead + JSON parsing. Unacceptable for multi-MB tensors.
- Unix socket with raw bytes: possible, but involves kernel copies (write syscall copies from user space to kernel buffer, read syscall copies from kernel buffer to user space). Two extra copies per tensor.
- Shared memory + notification: one memcpy (into shm), then daemon reads zero-copy from mmap. **One copy total.**

**Between daemon and orchestrator**: Not needed. The orchestrator is a thin relay. Tensor data goes directly from worker to daemon via shared memory (the orchestrator is not in the data path).

### 6.4 Alternative Considered: Daemon Also Embeds Python

If the daemon embedded Python, it could share tensors via `torch.multiprocessing` or `share_memory_()`. But this violates the architecture: the daemon is pure Rust, no Python dependency, no GIL, no PyTorch. The daemon must outlive worker crashes and serve multiple clients concurrently. Embedding Python would couple it to the worker's failure domain.

### 6.5 Optimization: Forward GPU Stats Through ProbeFrame

A refinement to ADR-0006's data flow: compute summary stats on GPU *before* the CPU transfer, and send them alongside the tensor bytes in the ProbeFrame. This avoids the daemon recomputing stats on CPU (saving ~8 ms per tensor).

The ProbeFrame header has a `flags` field and 56 bytes of reserved space. Options:
1. **Extend ProbeFrame header** to include stats inline (would need more than 56 bytes for full stats + histogram).
2. **Send stats as a separate small message** over the JSON-RPC control channel (stats are ~200 bytes, fits easily in JSON).
3. **Append stats after tensor bytes in the ring buffer slot** (variable-length trailer after the raw tensor bytes, size indicated in ProbeFrame header).

Option 2 is cleanest: stats go over the control channel (small, structured), raw bytes go over the data channel (large, unstructured). The daemon matches them by tick_id + component_id.


## Key Gotchas Summary

1. **macOS shm name length**: 31 chars max including leading `/`. Keep session IDs short.
2. **Python resource_tracker**: Always use `track=False` when creating shared memory that Rust will also access. Let Rust handle cleanup.
3. **GIL + raw pointer**: Hold `Py<PyAny>` reference to prevent GC while reading tensor data without GIL.
4. **GPU tensor data_ptr()**: Returns device pointer, not CPU-addressable. Must `.cpu()` first.
5. **torch.histc + NaN**: Filter NaNs before computing histogram. Behavior with NaN is undefined.
6. **Cross-process atomics**: Work correctly on shared memory regions on x86 and ARM, but variables must be naturally aligned.
7. **BLAKE3 dominates CPU budget**: For large tensors, BLAKE3 hashing takes longer than the memcpy. Multi-threaded hashing (Rayon) can help.
8. **Shared memory cleanup on worker crash**: Daemon must detect worker death (orchestrator reports it) and call `shm_unlink()` for the dead worker's region.


## Key References

- Python multiprocessing.shared_memory documentation: https://docs.python.org/3/library/multiprocessing.shared_memory.html
- CPython shared_memory.py source: https://github.com/python/cpython/blob/main/Lib/multiprocessing/shared_memory.py
- CPython resource_tracker bug: https://github.com/python/cpython/issues/82300
- nix crate shm_open: https://docs.rs/nix/latest/nix/sys/mman/fn.shm_open.html
- pyo3-dlpack (zero-copy DLPack for PyO3): https://docs.rs/pyo3-dlpack/latest/pyo3_dlpack/
- BLAKE3 Rust crate: https://docs.rs/blake3/latest/blake3/
- BLAKE3 Hasher API: https://docs.rs/blake3/latest/blake3/struct.Hasher.html
- iceoryx2 (reference implementation for zero-copy IPC): https://github.com/eclipse-iceoryx/iceoryx2
- PyTorch TorchStore RFC: https://github.com/pytorch/pytorch/issues/64932
- PyTorch data_ptr() documentation: https://docs.pytorch.org/docs/stable/tensors.html
- torch.histc documentation: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.histogram.html
- POSIX shared memory on macOS vs Linux: https://www.deepanseeralan.com/tech/playing-with-shared-memory/
- Welford's algorithm for LayerNorm CUDA kernel: https://oneflow2020.medium.com/how-to-implement-an-efficient-layernorm-cuda-kernel-oneflow-performance-optimization-731e91a285b8
- BLAKE3 performance benchmarks: https://github.com/BLAKE3-team/BLAKE3
- Shared memory IPC ring buffer (Rust forum): https://users.rust-lang.org/t/safe-shared-memory-ipc-with-ring-buffer/123725
