# Checkpoint State Tier Design Critique

Date: 2026-05-24
Reviewer: Claude (deep research agent)
Spec reviewed: `docs/superpowers/specs/2026-05-24-checkpoint-state-tier-design.md`

Reference implementations examined:
- PyTorch `torch/utils/checkpoint.py` (gradient checkpointing, RNG state handling)
- CUDA checkpoint API (`cuda-checkpoint/src/r580-migration-api.c`)
- TransformerLens `ActivationCache` + `HookPoint` system
- nnsight intervention hooks (`intervention/hooks.py`)
- pyvene intervention framework
- OpenAI transformer-debugger activation records + hooks
- vLLM v1 `BlockPool`, `SharedOffloadRegion`, `CpuGpuOffloadingHandlers`, `cuda_mem_ops`
- safetensors Rust crate (`tensor.rs`)
- rr record-and-replay (`ReplaySession.cc`, `ExportImportCheckpoints.cc`)
- Existing codebase: `ShmRegion`, `dispatch.rs`, `session.rs`, `messages.rs`, `bridge.py`

---

## 1. Memory Management

### 1.1 mlock vs cudaHostRegister — wrong pinning primitive

**What the spec says:** Anonymous mmap + `mlock` for pinned pages. Fallback to unpinned
mmap if ulimit fails.

**What the references say:** vLLM explicitly uses `cudaHostRegister` (via
`torch.cuda.cudart().cudaHostRegister`) on mmap'd regions, NOT `mlock`. Their
`pin_mmap_region()` function (in `vllm/v1/kv_offload/cpu/gpu_worker.py:89-105`)
and `pin_tensor()` (in `vllm/v1/simple_kv_offload/cuda_mem_ops.py:18-26`) both
call `cudaHostRegister`. vLLM's comment explains: "This bypasses PyTorch's
CUDACachingHostAllocator which rounds every `pin_memory=True` allocation up
to the next power of 2."

`mlock` and `cudaHostRegister` do different things:
- `mlock`: prevents page-out to swap. Does NOT register with CUDA. The GPU
  cannot DMA to mlock'd memory any faster than to regular memory.
- `cudaHostRegister`: registers existing host memory with CUDA, making it
  pinned from the GPU's perspective. This enables DMA transfers at full PCIe
  bandwidth. Internally it also calls the equivalent of mlock.

Using `mlock` alone means the CUDA `copy_()` call goes through the pageable
transfer path (staging buffer inside the driver), which is roughly 2x slower
than pinned DMA on PCIe Gen4/5.

**Risk level:** CRITICAL

**Recommendation:** Replace `mlock` with `cudaHostRegister`. Call it from
Python (via `torch.cuda.cudart().cudaHostRegister(ptr, size, 0)`) after the
Rust side creates the anonymous mmap and passes the pointer. The Rust side
should still do `MAP_POPULATE` to pre-fault pages. Fall back to unpinned if
`cudaHostRegister` fails (code != 0), exactly as vLLM does.

### 1.2 Bump allocator with no compaction — arena fragmentation

**What the spec says:** Bump allocator with a watermark. `free_checkpoint`
marks slots as reclaimable but "compaction is deferred."

**What the references say:** vLLM's `BlockPool` uses a `FreeKVCacheBlockQueue`
with an eviction order queue — blocks are returned to a free list, not a bump
watermark. This avoids the fragmentation problem entirely because all blocks
are the same size. The rr debugger uses `clone()` (fork) for checkpoints,
sidestepping memory management entirely via copy-on-write.

With a bump allocator and no compaction, after freeing checkpoint A (whose
slots are scattered among B and C), the freed space is inaccessible until the
watermark resets. This means the arena can report 50% utilization while having
no contiguous space for a new checkpoint, forcing a premature spill.

**Risk level:** IMPORTANT

**Recommendation:** Two options:
1. Fixed-size slots (round up to power-of-2 slot sizes) with a free list, like
   vLLM's block pool. This is simpler and eliminates fragmentation.
2. If variable-size slots are needed, implement a simple compaction pass: when
   utilization is below threshold but watermark is high, memcpy live slots
   down. This is acceptable because checkpoint creation/restore already involves
   large memcpy operations.

Option 1 is strongly preferred. The per-layer activation size is constant for a
given model, so fixed-size slots waste almost nothing.

### 1.3 Arena sizing heuristic is backwards

**What the spec says:** `max(256 MB, num_layers * hidden_dim * seq_len * dtype_size * sqrt_l_count * 2)`

**What the references say:** This formula includes `num_layers`, but the whole
point of sqrt(L) checkpointing is that you store only sqrt(L) checkpoints, not
all L layers. The formula should be:
`sqrt_l_count * (hidden_dim * seq_len * dtype_size) * 2`

As written, the formula allocates `num_layers * sqrt_l_count` worth of space —
far more than needed. For a 32-layer model with hidden_dim=4096, seq_len=2048,
bf16: the spec formula gives `32 * 4096 * 2048 * 2 * 6 * 2 = ~6.4 GB`, while
the correct formula gives `6 * 4096 * 2048 * 2 * 2 = ~192 MB`.

Also, `seq_len` is not known at construction time — it depends on the input.
The arena should be sized for the maximum expected sequence length (a config
parameter) or resized on first checkpoint when the actual tensor size is known.

**Risk level:** CRITICAL

**Recommendation:** Fix the formula: `sqrt_l_count * per_layer_bytes * 2` where
`per_layer_bytes = batch_size * seq_len * hidden_dim * dtype_size`. Add
`RS_CHECKPOINT_MAX_SEQ_LEN` as a config parameter (default: model's
`max_position_embeddings`).

### 1.4 No CUDA stream synchronization

**What the spec says:** `cpu_view.copy_(t)` triggers CUDA DMA. No mention of
stream synchronization.

**What the references say:** vLLM's `SingleDirectionOffloadingHandler` uses
dedicated CUDA streams for GPU-to-CPU transfers, with explicit `start_event` and
`end_event` synchronization. Each transfer records events: `start_event.record()`,
then `end_event.record()` after copy, and `end_event.synchronize()` to confirm
completion. They chain streams with inter-stream dependencies to ensure ordering.

PyTorch's `copy_()` on the default stream is synchronous with respect to
subsequent host code on that stream, but if the model's forward pass uses
non-default streams (which it does under CUDA graphs, torch.compile, or FSDP),
the copy may see stale data.

More critically: after `cpu_view.copy_(t)`, the spec immediately returns control.
If the copy is async (CUDA default behavior for pinned memory), the arena
memory may not be fully written when Rust reads it. The spec needs an explicit
`torch.cuda.synchronize()` or stream event fence after the copy.

**Risk level:** CRITICAL

**Recommendation:** After `cpu_view.copy_(t)`, add `torch.cuda.synchronize()` or
use `torch.cuda.current_stream().synchronize()`. For performance, consider using
a dedicated transfer stream with events (like vLLM does), so checkpoint capture
doesn't block the compute stream. However, since rocket_surgeon operates between
ticks (forward pass is stopped), a simple `torch.cuda.synchronize()` is
acceptable and much simpler.

---

## 2. Python Bridge

### 2.1 torch.frombuffer lifetime and ctypes.from_address safety

**What the spec says:** `buf = (ctypes.c_byte * dst_len).from_address(dst_ptr)`,
then `torch.frombuffer(buf, dtype=t.dtype)`.

**What the references say:** Verified empirically: `torch.frombuffer` does hold
a reference to the ctypes buffer object (refcount goes from 2 to 3). However,
`ctypes.from_address()` creates a ctypes array that does NOT own the underlying
memory — it's a view into the raw address. If the Rust mmap is unmapped while
any torch tensor still references the ctypes buffer, the tensor becomes a
dangling pointer with no way to detect it.

TransformerLens and nnsight both avoid raw pointer handoff entirely — they store
activations as normal PyTorch tensors in Python dicts.

Additionally, tested: `mmap.close()` fails with `BufferError: cannot close
exported pointers exist` when a ctypes buffer created via `from_buffer()` still
exists. But `from_address()` has no such protection — the caller is responsible
for lifetime management.

**Risk level:** IMPORTANT

**Recommendation:** This is acceptable given the architecture (Rust owns the
arena, Python bridge functions are called while the arena is live, and the
tensors created by `frombuffer` are ephemeral — used only for the copy_ call
and then dropped). BUT: ensure that `cpu_view` does not escape the function.
The current spec code is safe because `cpu_view` is a local variable. Add a
comment documenting the invariant: "cpu_view must not be stored or returned;
it aliases arena memory that Rust controls."

Also add a guard: after `copy_()`, explicitly `del cpu_view` before returning,
to make the invariant visible.

### 2.2 .contiguous() may allocate a copy

**What the spec says:** `t = tensor.detach().contiguous()` then
`cpu_view.copy_(t)`.

**What the references say:** `.contiguous()` is a no-op if the tensor is already
contiguous (which residual stream outputs almost always are). But if it IS
non-contiguous (e.g., after a transpose or slice), it allocates a new GPU
tensor. This is a hidden GPU memory allocation that could OOM on large models.

**Risk level:** MINOR

**Recommendation:** Check `t.is_contiguous()` first. If not contiguous, either
reshape in-place or use `t.contiguous()` but log a warning about the extra GPU
allocation. In practice, residual stream outputs from HuggingFace models are
always contiguous, so this is defensive.

### 2.3 Missing batch dimension handling

**What the spec says:** The bridge captures `_get_residual_stream(layer_idx)` and
copies the full tensor. The slot header stores `shape[6]` with `ndim` up to 8.

**What the references say:** TransformerLens's `ActivationCache` explicitly
handles `remove_batch_dim` as a parameter, and stores shape metadata. The
residual stream tensor is typically `[batch, seq_len, hidden_dim]` — 3D. The
spec's header supports this fine (ndim=3, shape[0..2]).

However, the spec doesn't discuss what happens when batch_size > 1. The arena
stores the full `[batch, seq_len, hidden_dim]` tensor. For batch_size=1 this
is fine. For batch_size > 1, the checkpoint grows linearly with batch size,
and the arena sizing heuristic doesn't account for it.

**Risk level:** MINOR (batch_size > 1 is unusual for debugger use cases)

**Recommendation:** Add `batch_size` to the arena sizing formula. Document that
the initial implementation targets batch_size=1 and that batch_size > 1 may
require a larger arena.

### 2.4 dtype string format inconsistency

**What the spec says:** `capture_activation` returns `str(t.dtype)` which gives
`"torch.float16"`, `"torch.bfloat16"`, etc. `restore_activation` does
`dtype.replace("torch.", "")` to convert back.

**What the references say:** This is fragile. The slot header uses a numeric enum
(0=f16, 1=bf16, 2=f32, 3=f64) but the Python bridge passes dtype as a string.
There's a mismatch: the Rust side has a dtype enum in the header, but the
Python bridge ignores it and passes dtype as a string through PyO3.

**Risk level:** MINOR

**Recommendation:** Use the slot header's dtype enum for the Rust-to-Python
interface. Convert to `torch.dtype` from the enum value on the Python side,
rather than round-tripping through strings. This eliminates the string parsing
and makes the Python bridge match the on-disk format.

---

## 3. sqrt(L) Strategy

### 3.1 Missing layer 0 checkpoint — replay gap

**What the spec says:** `checkpoint_layers(32)` returns `[5, 11, 16, 22, 28, 31]`.
Checkpoints capture the post-norm residual (output of `model.layers[layer_idx]`).

**What the references say:** PyTorch's `checkpoint_sequential` divides the model
into segments and checkpoints the INPUT to each segment. The classic sqrt(N)
strategy from the gradient checkpointing literature (Chen et al. 2016, "Training
Deep Nets with Sublinear Memory Cost") stores activations at evenly-spaced
boundaries INCLUDING the input.

The spec's formula starts at `i=1`, producing boundaries starting at layer 5.
This means layers 0-4 have no checkpoint to replay from. Sub-project B (replay)
needs the input activations to replay a segment. For layers 0-4, the only
source is the original model input (embedding output).

If the model input (token IDs + embedding) is always available (it should be —
it's small and the daemon has it), then this is fine: replay of layers 0-4
re-runs from the embedding. But this needs to be explicit.

**Risk level:** IMPORTANT

**Recommendation:** Either:
1. Always include layer 0 in the checkpoint set (modify the formula to start
   from `i=0` or prepend 0).
2. Document that replay of the first segment requires re-running from the
   embedding, and ensure the embedding output / model input is always available
   in the tick system.

Option 2 is correct and matches the classic approach (the "input" to the model
is implicitly available). But it must be explicitly documented and the replay
code in Sub-project B must know about it.

### 3.2 Uneven spacing

**What the spec says:** The formula `floor(i * sqrt(L))` produces spacing like
`[6, 5, 6, 6, 3]` for 32 layers and `[9, 9, 9, 9, 9, 9, 9, 8]` for 80 layers.

**What the references say:** The classic evenly-spaced approach gives uniform
intervals: `[5, 5, 5, 5, 5, 5]` for 32 layers. The spec's approach is slightly
uneven but produces approximately the same number of checkpoints (6 vs 7 for
32 layers). The worst-case replay cost is the same: max(gap) layers to replay.

The last boundary (31 for 32 layers, 79 for 80 layers) is the last layer. This
is wasteful — checkpointing the LAST layer's output is useless for replay
because there's nothing after it to replay. The classic approach doesn't include
the last layer.

**Risk level:** MINOR

**Recommendation:** Change the formula to exclude the last layer. The last
checkpoint should be at approximately `L - sqrt(L)`, not at `L - 1`. Simpler
formula: `[i * interval for i in range(sqrt_l_count)]` where
`interval = L / sqrt_l_count`, truncating the last value to avoid the final
layer.

### 3.3 The strategy is correct for the use case

**What the spec says:** sqrt(L) boundaries, auto-checkpoint at each boundary
crossing during stepping.

**What the references say:** This is well-established. Chen et al. 2016 proved
that sqrt(N) checkpoints are optimal for single-pass replay with O(sqrt(N))
memory and O(N*sqrt(N)) recomputation. For rocket_surgeon's interactive
debugger use case (step forward, occasionally step backward), this is exactly
right.

PyTorch's `checkpoint_sequential` uses a user-specified `segments` count
rather than auto-computing sqrt(L), but the principle is identical.

The "always-on" approach (checkpoint at every crossing) is appropriate for a
debugger where backward stepping is the common case.

**Risk level:** INFORMATIONAL — the overall strategy is sound.

---

## 4. Spill Format

### 4.1 Raw bytes is the right call — safetensors is wrong for this

**What the spec says:** Raw bytes with a fixed-size index. No serialization
library.

**What the references say:** Safetensors uses a JSON header followed by
concatenated raw tensor bytes, with alignment to 8 bytes. Its `prepare()`
function sorts tensors by dtype alignment. The format is designed for
interchange between frameworks and includes security protections (header
size limit, offset validation).

For rocket_surgeon's spill files, these safetensors features are unnecessary:
- **Security:** The spill file is written and read by the same process. There's
  no untrusted input.
- **JSON header:** Adds parsing overhead and variable-length header. The spec's
  fixed-size index (80 bytes per slot) is O(1) to parse.
- **Cross-framework compat:** Not needed. The spill file is ephemeral.
- **Dtype sorting:** The spec uses a simpler approach (concatenated, aligned).

Safetensors also uses mmap for loading (via `MmapOptions::new().map(&file)`),
which the spec explicitly avoids for spill files ("No mmap for files — the
arena is the working set, files are cold storage"). This is correct: mmapping
cold storage would pollute the page cache and compete with the arena for
physical memory.

**Risk level:** INFORMATIONAL — the decision to not use safetensors is correct.

### 4.2 Spill file lacks a checksum

**What the spec says:** Magic + version + index + data. No checksum.

**What the references say:** Safetensors validates tensor offsets and sizes
against the buffer. rr's checkpoint export uses socket-based transfer with
explicit length framing. Neither uses checksums for local storage, but both
have integrity validation.

For NVMe spill, bit rot is unlikely but silent corruption during async writes
is possible (e.g., if the write is interrupted). A CRC32 per slot or per file
would catch this cheaply.

**Risk level:** MINOR

**Recommendation:** Add a CRC32 field to the slot index entry (use the 8-byte
reserved field). Validate on load. Cost is negligible and it prevents subtle
data corruption from turning into silent divergence in replay.

### 4.3 No O_DIRECT / DIO for NVMe writes

**What the spec says:** `File::write_all` / `File::read_exact` for spill.

**What the references say:** Safetensors uses `F_NOCACHE` on macOS for direct
I/O, noting "~30% performance improvement." For NVMe spill of potentially
gigabyte-scale checkpoint data, going through the page cache is wasteful —
it pollutes the cache with data that will only be read back once (on restore).

**Risk level:** MINOR (spill is a cold path)

**Recommendation:** Use `O_DIRECT` (Linux) or `F_NOCACHE` (macOS) for spill
writes to avoid polluting the page cache. This is a performance optimization,
not a correctness issue.

---

## 5. Architecture

### 5.1 Checkpoint creation blocks the step loop

**What the spec says:** Auto-checkpoint is triggered after each step response
when crossing a sqrt(L) boundary. The daemon calls `_host/checkpoint Create`
to the worker.

**What the references say:** vLLM's KV offloading uses dedicated CUDA streams
and async transfers to avoid blocking the compute path. The transfer runs
concurrently with the next batch of computation.

In rocket_surgeon's case, the forward pass is stopped between ticks, so
blocking is less of a concern. However, for models with many layers (80+),
capturing 9 checkpoints × large tensors could take significant time. If the
user is stepping quickly (repeated "step layer"), each step would block on
checkpoint capture.

**Risk level:** MINOR (acceptable for initial implementation)

**Recommendation:** For Sub-project A, synchronous capture is fine. Note in
the design that Sub-project B or C may want to pipeline checkpoint capture
with the next step using async transfers.

### 5.2 No checkpoint eviction ordering by utility

**What the spec says:** `spill_oldest(dir)` — spill the oldest checkpoint.

**What the references say:** rr's replay session uses a more sophisticated
checkpoint strategy: checkpoints closer to the current execution point are
more valuable (less replay cost to reach). Oldest-first eviction is simple
but suboptimal — the oldest checkpoint might be the most valuable if the
user is stepping backward.

The optimal eviction strategy for a reverse debugger is to keep checkpoints
that minimize the worst-case replay distance. This is the "binary-search
style" placement: keep checkpoints at exponentially-spaced intervals from the
current position.

**Risk level:** IMPORTANT (for Sub-project B, not Sub-project A)

**Recommendation:** For Sub-project A, oldest-first is acceptable. Flag for
Sub-project B: implement utility-based eviction that considers distance from
the current tick position. The eviction policy should be pluggable.

### 5.3 No coordination with existing ShmRegion / DoomRing

**What the spec says:** "Private mmap'd region (not shared memory —
worker-local). Uses anonymous mmap + mlock for pinned pages, not shm_open."

**What the references say:** The existing `ShmRegion` in `region.rs` already
handles mmap with proper error handling, bounds checking, atomic ops, and
cleanup (munmap/close in Drop). The checkpoint arena re-implements mmap
setup from scratch.

The spec is correct that the checkpoint arena should NOT be shared memory
(it's worker-local), but the code could reuse the mmap/munmap patterns from
`ShmRegion` without using `shm_open`.

**Risk level:** MINOR

**Recommendation:** Consider extracting the mmap/munmap/bounds-check patterns
from `ShmRegion` into a shared utility (e.g., `MmapRegion` base) that both
`ShmRegion` and `CheckpointArena` can use. This avoids duplicating unsafe code.

### 5.4 The HostCheckpointRequest wire type needs layer list

**What the spec says:** The existing `HostCheckpointRequest::Create` has a
single `layer_idx: u32`. The create flow iterates over `checkpoint_layers()`.

**What the references say:** The wire type has a single `layer_idx` but the
handler captures ALL sqrt(L) layers in one call. This means either:
1. The handler ignores `layer_idx` and always captures all sqrt(L) layers, or
2. The daemon sends multiple `_host/checkpoint` requests, one per layer.

Option 1 wastes the `layer_idx` field. Option 2 is N round-trips for N layers.

**Risk level:** MINOR

**Recommendation:** Either remove `layer_idx` from `Create` (capture all
sqrt(L) layers in one call), or add a `layers: Option<Vec<u32>>` field and
remove `layer_idx`. The current design implies option 1 but the wire type
suggests option 2.

---

## 6. Things the Spec Missed

### 6.1 No torch.cuda.synchronize() in capture path

Already covered in 1.4 above, but worth repeating: the spec has no explicit
CUDA synchronization after `copy_()`. This is the single most likely source
of data corruption bugs.

### 6.2 RNG state: pickle is a risk

**What the spec says:** `pickle.dumps(states)` / `pickle.loads(state)` for RNG
state serialization.

**What the references say:** PyTorch's own `get_device_states` / `set_device_states`
in `torch/utils/checkpoint.py` returns raw tensors, not pickled bytes. The
tensors are saved via `ctx.save_for_backward()`.

Using pickle for RNG state has two problems:
1. Pickle is not safe for untrusted input (though in this case, the producer
   and consumer are the same process, so this is moot).
2. Pickle format is version-dependent and can change between Python versions.
   For ephemeral in-memory state this doesn't matter, but if RNG state is
   ever spilled to disk and loaded in a different session, it could break.

**Risk level:** MINOR

**Recommendation:** Store RNG state as raw bytes (the torch tensor's underlying
buffer) rather than pickle. `torch.cuda.get_rng_state()` returns a `ByteTensor`;
call `.numpy().tobytes()` as the spec already does, but concatenate them with a
simple length-prefix format instead of pickle: `[num_devices: u32][device_id: u32,
len: u32, bytes: [u8; len]] * num_devices`.

### 6.3 No error recovery for partial checkpoint capture

**What the spec says:** The create flow iterates over layers, capturing each one.
If any capture fails, the response is presumably an error.

**What the references say:** vLLM's offloading handlers track individual transfer
completions and can retry failed transfers. rr's checkpoint creation is atomic
(fork-based).

If capture of layer 5 succeeds but layer 11 fails (e.g., OOM on `.contiguous()`),
the arena has a partial checkpoint with some slots allocated but no valid
checkpoint. The `index` map may have stale entries.

**Risk level:** IMPORTANT

**Recommendation:** Make checkpoint creation transactional. Save the watermark
before starting, and reset it on failure (rolling back all allocations for
this checkpoint). Remove any index entries added. Return the error to the
daemon so it doesn't register the checkpoint in the metadata tier.

### 6.4 Thread safety of the arena

**What the spec says:** The arena is a struct with mutable fields (watermark,
slots, index). No mention of synchronization.

**What the references say:** The spec says the arena is worker-local, and the
worker processes one request at a time (dispatch loop is serial). So no
concurrent access is expected.

However, if the Python bridge's `copy_()` is async (see 1.4), the arena memory
could be written by CUDA DMA while Rust reads the header. This is a data race.

**Risk level:** IMPORTANT (contingent on 1.4)

**Recommendation:** After resolving 1.4 (adding synchronize), this is a non-issue.
But add a comment documenting the single-threaded assumption.

### 6.5 Multi-GPU: no cross-device coordination

**What the spec says:** `capture_rng_state` iterates over
`torch.cuda.device_count()` devices. No other multi-GPU handling.

**What the references say:** PyTorch's checkpoint code in `CheckpointFunction.forward`
captures device states for all devices that have tensor arguments
(`get_device_states(*args)`). vLLM's `SharedOffloadRegion` uses interleaved
layouts for multi-worker coordination with explicit rank-based offsets.

For multi-GPU (DDP/FSDP/tensor-parallel), each rank's worker has its own arena.
But the checkpoint_id must be coordinated across ranks — all ranks must
checkpoint at the same tick. The spec's auto-checkpoint (triggered by sqrt(L)
boundary crossing) assumes all ranks cross boundaries at the same tick, which
is true for data-parallel but NOT true for pipeline-parallel (different ranks
process different micro-batches at different layers).

**Risk level:** IMPORTANT (for multi-GPU, which is a project design principle)

**Recommendation:** Document that Sub-project A targets single-GPU and
data-parallel. For pipeline-parallel, checkpoint coordination requires a
barrier or consensus protocol across ranks. Flag this as a Sub-project B or
later concern.

### 6.6 No mention of model weight checkpointing path

**What the spec says:** "FullSnapshot tier (entire model state) — deferred."

**What the references say:** The spec correctly defers this. But the arena
architecture should be designed so that weight checkpoints (if ever needed)
don't go through the same bump-allocator arena. Weights are much larger
(billions of parameters) and have different access patterns.

**Risk level:** INFORMATIONAL

**Recommendation:** No action needed for Sub-project A. Note that the arena is
activation-only and the FullSnapshot tier will need a different storage strategy.

---

## Summary Table

| # | Finding | Risk | Section |
|---|---------|------|---------|
| 1.1 | mlock is wrong — need cudaHostRegister for pinned DMA | CRITICAL | Memory |
| 1.2 | Bump allocator fragments without compaction | IMPORTANT | Memory |
| 1.3 | Arena sizing formula is wrong (num_layers should not be in formula) | CRITICAL | Memory |
| 1.4 | No CUDA stream sync after copy_() | CRITICAL | Memory |
| 2.1 | torch.frombuffer + from_address lifetime is safe but fragile | IMPORTANT | Python |
| 2.2 | .contiguous() may allocate GPU copy | MINOR | Python |
| 2.3 | No batch dimension in arena sizing | MINOR | Python |
| 2.4 | dtype string round-trip is fragile | MINOR | Python |
| 3.1 | Layer 0 not in checkpoint set — replay gap for first segment | IMPORTANT | sqrt(L) |
| 3.2 | Last layer checkpoint is wasteful | MINOR | sqrt(L) |
| 3.3 | Overall strategy is correct and well-grounded | INFO | sqrt(L) |
| 4.1 | Not using safetensors is correct | INFO | Spill |
| 4.2 | No checksum on spill files | MINOR | Spill |
| 4.3 | No O_DIRECT for NVMe writes | MINOR | Spill |
| 5.1 | Checkpoint creation blocks step loop | MINOR | Arch |
| 5.2 | Oldest-first eviction is suboptimal for reverse debugging | IMPORTANT | Arch |
| 5.3 | Could reuse mmap patterns from ShmRegion | MINOR | Arch |
| 5.4 | Wire type layer_idx vs multi-layer capture mismatch | MINOR | Arch |
| 6.1 | CUDA sync missing (dup of 1.4) | CRITICAL | Missing |
| 6.2 | RNG pickle is unnecessary fragility | MINOR | Missing |
| 6.3 | No transactional rollback on partial capture failure | IMPORTANT | Missing |
| 6.4 | Thread safety assumption undocumented | IMPORTANT | Missing |
| 6.5 | Multi-GPU checkpoint coordination unspecified | IMPORTANT | Missing |
| 6.6 | FullSnapshot tier needs different architecture | INFO | Missing |

---

## Overall Assessment

The design is **structurally sound** — the zero-copy arena architecture, the
sqrt(L) strategy, and the raw spill format are all well-chosen for the use
case. The separation between daemon metadata (session.rs) and worker tensor
data is clean.

There are **three critical issues** that will cause data corruption or severe
performance degradation if not fixed before implementation:
1. **mlock vs cudaHostRegister** — using the wrong pinning primitive means DMA
   goes through the slow pageable path, defeating the zero-copy goal.
2. **Arena sizing formula** — multiplying by `num_layers` makes the arena 5-30x
   larger than needed.
3. **Missing CUDA synchronization** — without `torch.cuda.synchronize()` after
   `copy_()`, the arena may contain partially-written data.

These are all straightforward fixes. The design does not need rearchitecting;
it needs three targeted corrections before implementation begins.

The **important issues** (fragmentation, layer 0 gap, partial capture rollback,
multi-GPU coordination, eviction policy) should be addressed in the
implementation plan but are not blockers for starting Sub-project A on
single-GPU.
