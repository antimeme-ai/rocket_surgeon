# Checkpoint State Tier Design Spec — Verification Report

Date: 2026-05-24
Spec under review: `docs/superpowers/specs/2026-05-24-checkpoint-state-tier-design.md`

---

## Claim 1: cudaHostRegister vs mlock

**Claim**: `mlock` prevents page-out but does NOT register memory with CUDA. GPU DMA
to mlock'd memory goes through the slow pageable path (~2x slower). `cudaHostRegister`
registers existing host memory with CUDA for pinned DMA at full PCIe bandwidth. It
also implicitly locks pages. vLLM uses `torch.cuda.cudart().cudaHostRegister(ptr, len, 0)`.

**Evidence**:

- `quarantine/vllm/vllm/v1/kv_offload/cpu/gpu_worker.py:89-108` — The function
  `pin_mmap_region()` is documented as "Register the entire mmap as CUDA pinned
  memory via cudaHostRegister." The exact call at line 94:
  ```python
  result = torch.cuda.cudart().cudaHostRegister(base_ptr, region.total_size_bytes, 0)
  ```
  This matches the spec's API exactly: `torch.cuda.cudart().cudaHostRegister(ptr, len, 0)`.

- `quarantine/vllm/vllm/v1/simple_kv_offload/cuda_mem_ops.py:17-26` — Another vLLM
  module with the same pattern:
  ```python
  def pin_tensor(tensor: torch.Tensor) -> None:
      """Pin a CPU tensor via cudaHostRegister."""
      err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
  ```
  The docstring at line 18-22 explicitly states this bypasses PyTorch's
  CUDACachingHostAllocator.

- The distinction between mlock and cudaHostRegister is well-established in CUDA
  documentation: mlock only prevents swap-out at the OS level; cudaHostRegister
  additionally registers the memory region with the CUDA driver so that DMA
  transfers can use the pinned (page-locked) path instead of the staging-buffer
  pageable path.

**Verdict**: CONFIRMED

**Notes**: The 3rd argument `0` corresponds to `cudaHostRegisterDefault`. vLLM also
supports `cudaHostRegisterPortable` via flag 1 in some contexts, but the spec's
default-flag approach is the common pattern.

---

## Claim 2: torch.frombuffer with ctypes.from_address

**Claim**: `torch.frombuffer` creates a tensor wrapping external memory (no copy).
`.copy_()` from a GPU tensor triggers CUDA DMA. The tensor does not own the memory.

**Evidence**:

- `quarantine/vllm/vllm/v1/kv_offload/cpu/shared_offload_region.py:113`:
  ```python
  self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
  ```
  vLLM uses `torch.frombuffer` to wrap an mmap'd region as a tensor. The cleanup
  code at lines 170-191 explicitly releases the tensor (`self._base = None`) before
  closing the mmap — confirming the tensor does NOT own the underlying memory.

- `quarantine/vllm/vllm/v1/serial_utils.py:416`:
  ```python
  arr = torch.frombuffer(buffer, dtype=torch.uint8)
  ```
  Another usage confirming frombuffer wraps existing buffer-protocol objects.

- `quarantine/vllm/tests/v1/kv_connector/unit/test_hf3fs_client.py:47`:
  ```
  .buf      -- memoryview / buffer-protocol object consumed by torch.frombuffer
  ```
  Documentation confirms frombuffer consumes buffer-protocol objects.

- The `ctypes.from_address` pattern in the spec (`(ctypes.c_byte * dst_len).from_address(dst_ptr)`)
  creates a ctypes array backed by the given address. This is a standard Python/ctypes
  pattern. The resulting object implements the buffer protocol, which `torch.frombuffer`
  accepts.

- Regarding `.copy_()` triggering CUDA DMA: when the source tensor is on GPU and the
  destination tensor is in pinned host memory (registered via cudaHostRegister), PyTorch
  dispatches a `cudaMemcpyDeviceToHost` which uses DMA at full PCIe bandwidth. This is
  the fundamental mechanism behind `pin_memory()` and the entire pinned-memory transfer
  paradigm in PyTorch/CUDA.

**Verdict**: CONFIRMED

**Notes**: The PyTorch quarantine directory has only a bare `.git` (no checked-out
source tree), so verification relied on vLLM's usage of the same APIs and
PyTorch's documented behavior.

---

## Claim 3: torch.cuda.synchronize()

**Claim**: `copy_()` to pinned memory can be async. `synchronize()` is needed to
ensure completion before reading the data.

**Evidence**:

- `quarantine/vllm/vllm/v1/kv_offload/cpu/gpu_worker.py:182,311-335` — vLLM's
  `transfer_async` method explicitly uses CUDA streams for GPU<->CPU transfers.
  At line 377, `event.synchronize()` is called to wait for transfer completion.
  At line 382, `transfer.end_event.synchronize()` during shutdown.

- The vLLM architecture demonstrates that GPU-to-CPU copies via pinned memory are
  indeed asynchronous when launched on a non-default stream. The `wait()` method
  at line 373-377 calls `event.synchronize()` to block until the transfer completes.

- For the spec's simpler pattern (copy on the default stream followed by
  `torch.cuda.synchronize()`), this is a well-known PyTorch pattern. The default
  stream's `copy_()` to pinned memory is async from the CPU's perspective — the
  CPU thread returns immediately while the DMA engine handles the transfer.
  `torch.cuda.synchronize()` blocks the CPU thread until all operations on the
  current device are complete.

**Verdict**: CONFIRMED

**Notes**: The spec uses the simpler pattern (`copy_()` + `synchronize()` on the
default stream) rather than vLLM's multi-stream approach. Both are correct. The
simpler approach has higher latency (synchronize blocks on ALL pending operations)
but is appropriate for checkpoint capture where simplicity matters.

---

## Claim 4: torch.cuda.get_rng_state / set_rng_state

**Claim**: The API exists, returns a ByteTensor, and can be round-tripped.

**Evidence**:

- The PyTorch quarantine is a bare `.git` clone with no source tree checked out,
  so direct source verification was not possible.

- `torch.cuda.get_rng_state(device)` and `torch.cuda.set_rng_state(new_state, device)`
  are documented PyTorch APIs. `get_rng_state` returns a `torch.ByteTensor` (uint8)
  containing the CUDA RNG state for the specified device. `set_rng_state` accepts a
  ByteTensor and restores the RNG state.

- The spec's code at lines 206-208:
  ```python
  rng_bytes = torch.cuda.get_rng_state(i).numpy().tobytes()
  ```
  This correctly chains: get ByteTensor -> numpy array -> raw bytes.

- The restore code at lines 220-221:
  ```python
  t = torch.frombuffer(bytearray(rng_bytes), dtype=torch.uint8)
  torch.cuda.set_rng_state(t, device_id)
  ```
  This correctly wraps bytes back into a ByteTensor for set_rng_state.

- PyTorch's `torch/utils/checkpoint.py` (the gradient checkpointing module) uses
  these same APIs to save/restore RNG state during recomputation, providing indirect
  confirmation that the round-trip pattern works.

**Verdict**: CONFIRMED (via documented API; no local source to verify directly)

**Notes**: The `bytearray()` wrapper in the restore code is important — `torch.frombuffer`
requires a mutable buffer, and `bytes` is immutable. The spec handles this correctly.

---

## Claim 5: mmap with MAP_POPULATE

**Claim**: `MAP_POPULATE` pre-faults pages (avoids page faults on first access).

**Evidence**:

- `quarantine/vllm/vllm/v1/kv_offload/cpu/shared_offload_region.py:85-111` — vLLM
  uses `MADV_POPULATE_WRITE` (the madvise equivalent) to pre-fault pages after mmap.
  Lines 85-86:
  ```python
  # MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
  _MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
  ```
  This confirms the concept: pre-faulting pages is important for avoiding page faults
  on first access.

- `MAP_POPULATE` is a Linux mmap flag (value 0x008000) that causes the kernel to
  pre-fault (populate) page tables for the mapping at mmap time, reading in any
  file-backed pages or zeroing anonymous pages. This avoids soft page faults on
  first access to each page.

- The spec uses `MAP_POPULATE` for anonymous mmap (Rust-side), while vLLM uses the
  `madvise(MADV_POPULATE_WRITE)` approach (Python mmap doesn't expose MAP_POPULATE
  directly). Both achieve the same goal: pre-faulted pages.

**Verdict**: CONFIRMED

**Notes**: `MAP_POPULATE` is Linux-specific. On macOS, it is not available (macOS
does not define this flag). The spec should note that MAP_POPULATE will be a no-op
or conditionally compiled on non-Linux platforms. vLLM sidesteps this by using
`madvise` which has broader support. For the Rust implementation, conditional
compilation with `#[cfg(target_os = "linux")]` for MAP_POPULATE is standard practice.

---

## Claim 6: Fixed-size slot pool vs bump allocator — vLLM uses block pool / free list

**Claim**: vLLM uses a block pool / free list pattern for KV cache, which the spec's
fixed-size slot pool with free list is modeled after.

**Evidence**:

- `quarantine/vllm/vllm/v1/core/block_pool.py` — File exists, confirming vLLM has
  a dedicated `BlockPool` concept.

- `quarantine/vllm/vllm/v1/core/kv_cache_utils.py:116-161` — `KVCacheBlock` class
  with fixed-size blocks (identified by `block_id`), reference counting (`ref_cnt`),
  and doubly-linked list pointers (`prev_free_block`, `next_free_block`) for O(1)
  free-list operations.

- `quarantine/vllm/vllm/v1/core/kv_cache_utils.py:164-327` — `FreeKVCacheBlockQueue`
  class: a custom doubly-linked-list free-block queue with O(1) `popleft()` (allocate),
  `append()` (free), and `remove()` (evict from middle). The docstring at line 165
  states: "This class organizes a list of KVCacheBlock objects to a doubly linked
  list of free blocks."

- This is exactly the pattern the spec describes: fixed-size blocks, free list for
  O(1) alloc/free, no fragmentation. The spec's `Vec<usize>` free list is simpler
  than vLLM's doubly-linked list (which supports O(1) removal from the middle for
  LRU eviction), but the core pattern matches.

**Verdict**: CONFIRMED

---

## Claim 7: sqrt(L) checkpoint strategy — Chen et al. 2016

**Claim**: The sqrt(L) checkpoint boundary strategy matches gradient checkpointing
literature (Chen et al. 2016 "Training Deep Nets with Sublinear Memory Cost").

**Evidence**:

- `papers/rocket-surgeon/checkpointing/Chen2016_Sublinear_Memory_Cost.pdf` — Section
  4.3 "An O(sqrt(n)) Memory Cost Algorithm" (page 5):
  > "Assume we divide the n network into k segments... Setting k = sqrt(n),
  > we get the cost of O(2*sqrt(n)). This algorithm only requires an additional
  > forward pass during training."
  
  The paper proves that dividing an n-layer network into sqrt(n) segments and
  keeping only the segment boundary activations yields O(sqrt(n)) memory cost
  with only one additional forward pass.

- The spec's `checkpoint_layers()` function computes `sqrt_l = ceil(sqrt(num_layers))`
  and places boundaries at evenly-spaced intervals — this directly implements
  Chen et al.'s sqrt(n) segmentation strategy.

- The spec correctly adapts the strategy for debugging (not training): instead of
  recomputing dropped activations during backprop, the checkpoint boundaries define
  where activations are saved so that forward replay (Sub-project B) can restart
  from the nearest checkpoint. The mathematical foundation (sqrt(n) boundaries
  minimizes max-replay-distance * num-checkpoints) is the same optimization.

**Verdict**: CONFIRMED

**Notes**: The spec's exclusion of layer 0 and the last layer is a sensible
adaptation: layer 0's input is always available (token IDs), and the last layer's
output has nothing after it to replay. Chen et al. don't discuss these edge cases
because their context is automatic differentiation, not interactive debugging.

---

## Claim 8: session.rs checkpoint methods

**Claim**: `checkpoint_create`, `checkpoint_restore`, `checkpoint_list`,
`checkpoint_delete`, `checkpoint_bookmark` exist at lines 673-804.

**Evidence**:

- `crates/rocket-surgeon/src/session.rs:673-804` — All five methods confirmed:
  - `checkpoint_create()` at line 673 — creates a checkpoint with UUID, tier,
    pushes to state.checkpoints, returns CheckpointResponse
  - `checkpoint_list()` at line 702 — returns all checkpoints
  - `checkpoint_restore()` at line 713 — finds checkpoint by ID, restores
    position, returns CheckpointResponse with restored_to
  - `checkpoint_delete()` at line 755 — removes by ID, returns updated list
  - `checkpoint_bookmark()` at line 777 — attaches name to checkpoint at tick_id,
    creates marker entry if none exists

**Verdict**: CONFIRMED

**Notes**: The spec says lines "673-804" and this matches exactly. All methods
work as described: create registers metadata, restore updates position, delete
retains others, bookmark either updates existing or creates a ProbeLog marker.

---

## Claim 9: HostCheckpointRequest/Response wire types

**Claim**: `HostCheckpointRequest` has Create (with tier, tick_id, layer_idx) and
Restore variants at `messages.rs:956-970`. `HostCheckpointResponse` has
checkpoint_id, tier, restored_to, bytes_captured at `messages.rs:972-982`.

**Evidence**:

- `crates/rocket-surgeon-protocol/src/messages.rs:958-970`:
  ```rust
  pub enum HostCheckpointRequest {
      Create {
          model_handle: u64,
          checkpoint_id: String,
          tier: CreateCheckpointTier,
          tick_id: u64,
          layer_idx: u32,
      },
      Restore {
          model_handle: u64,
          checkpoint_id: String,
      },
  }
  ```
  Create has `tier`, `tick_id`, `layer_idx` as claimed. Also has `model_handle`
  and `checkpoint_id` (not mentioned in spec's table but present).

- `crates/rocket-surgeon-protocol/src/messages.rs:972-982`:
  ```rust
  pub struct HostCheckpointResponse {
      pub checkpoint_id: String,
      pub tier: CheckpointTier,
      pub restored_to: Option<TickPosition>,
      pub bytes_captured: Option<u64>,
  }
  ```
  Fields match exactly: checkpoint_id, tier, restored_to, bytes_captured.

**Verdict**: CONFIRMED

**Notes**: The spec's line numbers (956-970 and 972-982) match exactly. The spec's
table says "Create (with tier, tick_id, layer_idx)" which is accurate. The Create
variant also has `model_handle` and `checkpoint_id` which the spec doesn't
mention in the table but does reference in the handler flow description.

---

## Claim 10: dispatch.rs has NO _host/checkpoint handler

**Claim**: The worker dispatch table at `dispatch.rs:85-104` is missing a
`_host/checkpoint` match arm.

**Evidence**:

- `crates/rocket-surgeon-worker/src/dispatch.rs:85-106` — The `dispatch()` function
  matches on:
  ```
  HOST_ATTACH, HOST_DETACH, HOST_CONFIGURE_HOOKS, HOST_STEP,
  HOST_UPDATE_PROBES, HOST_INSPECT, HOST_VIEW, HOST_KV_READ,
  HOST_KV_INTERVENE, HOST_EXPORT_ENV
  ```
  There is NO `HOST_CHECKPOINT` match arm. The fallthrough at line 97-104 returns
  `METHOD_NOT_FOUND`.

- `crates/rocket-surgeon-protocol/src/messages.rs:62` confirms the constant exists:
  ```rust
  pub const HOST_CHECKPOINT: &str = "_host/checkpoint";
  ```
  The constant is defined but not used in dispatch.

**Verdict**: CONFIRMED

**Notes**: The spec's line range "85-104" matches the dispatch function location.
The constant `HOST_CHECKPOINT` exists in the protocol crate but is simply not
referenced in the worker's dispatch table — exactly what the spec says needs to
be added.

---

## Claim 11: ShmRegion mmap pattern

**Claim**: The existing mmap pattern in `crates/rocket-surgeon-shm/src/region.rs`
uses MAP_SHARED mmap, and the spec's approach is consistent (but uses anonymous
mmap rather than shm_open).

**Evidence**:

- `crates/rocket-surgeon-shm/src/region.rs:31-96` — `ShmRegion::create()` does:
  1. `shm_open(O_CREAT | O_RDWR | O_EXCL, 0o600)` — POSIX shared memory
  2. `ftruncate(fd, size)` — set size
  3. `mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)` — map it

- `crates/rocket-surgeon-shm/src/region.rs:272-280` — Drop impl does:
  ```rust
  libc::munmap(self.ptr.cast(), self.len);
  libc::close(self.fd);
  ```

- The spec's CheckpointArena uses anonymous mmap (not shm_open) because the arena
  is worker-local, not cross-process. This is a deliberate design choice documented
  in the spec: "Private mmap'd region (not shared memory -- worker-local). Uses
  anonymous mmap + cudaHostRegister for CUDA-pinned pages, not shm_open."

- The mmap/munmap lifecycle pattern is consistent: allocate via mmap, cleanup via
  munmap. The spec adds cudaHostUnregister before munmap, which is the correct
  ordering (unregister from CUDA before unmapping the memory).

**Verdict**: CONFIRMED

**Notes**: The spec does NOT use MAP_POPULATE in the existing ShmRegion (ShmRegion
uses MAP_SHARED without MAP_POPULATE). The spec's arena adds MAP_POPULATE for
anonymous mmap, which is a new addition. This is consistent — ShmRegion is for
IPC and doesn't need pre-faulting as urgently as a DMA target region.

---

## Claim 12: Spill format — raw bytes + CRC32 vs safetensors JSON headers

**Claim**: Safetensors uses JSON headers; our raw approach is different/simpler.

**Evidence**:

- `quarantine/safetensors/safetensors/src/tensor.rs:251`:
  ```rust
  let mut metadata_buf = serde_json::to_string(&metadata)?.into_bytes();
  ```
  Safetensors serializes metadata (tensor names, dtypes, shapes, offsets) as a
  JSON string.

- `quarantine/safetensors/safetensors/src/tensor.rs:395-417` — Deserialization reads
  an 8-byte little-endian header size, then parses the header bytes as UTF-8 JSON
  via `serde_json::from_str`.

- `quarantine/safetensors/safetensors/src/tensor.rs:799-806` — `TensorInfo` contains
  dtype, shape, and data_offsets — all serialized as JSON.

- The spec's spill format uses a fixed-size binary index (80 bytes per slot) with
  direct struct fields (layer_idx, dtype enum, ndim, shape as u64 array, offset,
  length, CRC32). No JSON parsing, no string allocation, no variable-length headers.
  This is deliberately simpler and faster for the checkpoint use case where tensor
  names are irrelevant (slots are identified by layer_idx).

**Verdict**: CONFIRMED

**Notes**: The spec's format is appropriate for its use case. Safetensors optimizes
for named tensor collections with arbitrary metadata; the checkpoint spill format
optimizes for fast sequential read/write of known-schema slot data. The CRC32
per-slot provides corruption detection that safetensors does not have (safetensors
relies on header consistency checks but not per-tensor checksums).

---

## Claim 13: cudaHostUnregister

**Claim**: `cudaHostUnregister` exists and should be called before munmap.

**Evidence**:

- `quarantine/vllm/vllm/v1/kv_offload/cpu/shared_offload_region.py:170-191` —
  The `cleanup()` method calls cudaHostUnregister BEFORE releasing the mmap:
  ```python
  def cleanup(self) -> None:
      if self.is_pinned and self._base is not None:
          base_ptr = self._base.data_ptr()
          result = torch.cuda.cudart().cudaHostUnregister(base_ptr)
          ...
      self._base = None        # release tensor view
      if self.mmap_obj:
          self.mmap_obj.close() # close mmap
      if self.fd is not None:
          os.close(self.fd)     # close fd
  ```
  The ordering is: (1) cudaHostUnregister, (2) release tensor, (3) close mmap,
  (4) close fd. This confirms the spec's claim that Unregister must precede munmap.

- The API is `torch.cuda.cudart().cudaHostUnregister(ptr)` — takes only the pointer,
  no size argument (unlike cudaHostRegister which takes ptr + size + flags).

**Verdict**: CONFIRMED

**Notes**: The spec's ordering at line 93-94 ("Python cudaHostUnregister(ptr), then
Rust munmap") exactly matches vLLM's cleanup sequence. Calling munmap on memory
still registered with CUDA would cause undefined behavior (the CUDA driver would
hold stale page table entries for unmapped virtual addresses).

---

## Claim 14: O_DIRECT / F_NOCACHE — safetensors uses F_NOCACHE on macOS

**Claim**: Safetensors uses F_NOCACHE on macOS.

**Evidence**:

- `quarantine/safetensors/safetensors/src/tensor.rs:317-324`:
  ```rust
  // Serialize tensors to a file using direct I/O (bypassing page cache) using F_NOCACHE.
  // This yields ~30% performance improvement.
  #[cfg(target_os = "macos")]
  unsafe {
      use std::os::fd::AsRawFd;
      libc::fcntl(temp.as_file().as_raw_fd(), libc::F_NOCACHE, 1);
  }
  ```
  Confirmed: safetensors uses `F_NOCACHE` on macOS, conditionally compiled with
  `#[cfg(target_os = "macos")]`. The comment documents a ~30% performance improvement
  from bypassing the page cache.

- `F_NOCACHE` is the macOS equivalent of Linux's `O_DIRECT` — it instructs the
  kernel to bypass the buffer cache for the file descriptor. On macOS, `O_DIRECT`
  does not exist; `F_NOCACHE` via `fcntl()` is the standard alternative.

**Verdict**: CONFIRMED

**Notes**: The spec mentions both O_DIRECT and F_NOCACHE. For the NVMe spill
format, the spec uses standard `File::read_exact` / `File::write_all` without
mentioning whether it will use direct I/O. This is appropriate for the initial
implementation — direct I/O adds alignment constraints and complexity that can
be optimized later.

---

## Additional Context: Papers

The `papers/rocket-surgeon/checkpointing/` directory contains three relevant papers:

1. **Chen2016_Sublinear_Memory_Cost.pdf** — The primary reference for the sqrt(L)
   strategy. Verified above in Claim 7.
2. **PhoenixOS2024_GPU_Checkpoint_Restore.pdf** — GPU checkpoint/restore at the
   OS level (different scope — full GPU context, not activation-level).
3. **2502.16631_CRIUgpu.pdf** — CRIU-based GPU checkpoint (process-level, not
   activation-level).

Papers 2 and 3 are relevant background but address process-level GPU checkpointing,
not the activation-level checkpointing in this spec. The spec's approach (capturing
specific activation tensors at layer boundaries) is the correct granularity for
a debugger — it's more selective and efficient than process-level snapshots.

---

## Summary

| # | Claim | Verdict |
|---|-------|---------|
| 1 | cudaHostRegister vs mlock | CONFIRMED |
| 2 | torch.frombuffer with ctypes.from_address | CONFIRMED |
| 3 | torch.cuda.synchronize() for async copy | CONFIRMED |
| 4 | torch.cuda.get_rng_state / set_rng_state | CONFIRMED (API docs, no local source) |
| 5 | mmap with MAP_POPULATE | CONFIRMED |
| 6 | Fixed-size slot pool / free list (vLLM pattern) | CONFIRMED |
| 7 | sqrt(L) checkpoint strategy (Chen 2016) | CONFIRMED |
| 8 | session.rs checkpoint methods | CONFIRMED |
| 9 | HostCheckpointRequest/Response wire types | CONFIRMED |
| 10 | No _host/checkpoint handler in dispatch.rs | CONFIRMED |
| 11 | ShmRegion mmap pattern consistency | CONFIRMED |
| 12 | Spill format vs safetensors JSON headers | CONFIRMED |
| 13 | cudaHostUnregister before munmap | CONFIRMED |
| 14 | F_NOCACHE on macOS in safetensors | CONFIRMED |

**All 14 technical claims are CONFIRMED.**

### Corrections / Caveats Noted

1. **MAP_POPULATE portability** (Claim 5): MAP_POPULATE is Linux-only. The Rust
   implementation will need `#[cfg(target_os = "linux")]` conditional compilation.
   On macOS (the current dev platform), this flag doesn't exist. vLLM uses
   `madvise(MADV_POPULATE_WRITE)` which is also Linux 5.14+. The spec should
   document the fallback behavior on non-Linux platforms.

2. **PyTorch source unavailable** (Claim 4): The quarantine pytorch directory is a
   bare git clone with no checked-out source. Claims about PyTorch APIs were verified
   via vLLM's usage of those APIs and documented behavior. Consider checking out the
   pytorch source if future verification needs direct source references.

3. **RNG state device argument** (Claim 4): The spec uses `torch.cuda.get_rng_state(i)`
   where `i` is the device index. The PyTorch API accepts either a device index (int)
   or a device object. The spec's usage is correct.
