# Session Handoff: Phase 3A — Checkpoint State Tier

**Date:** 2026-05-24
**Branch:** `phase3/subproject-a-checkpoint-state-tier` (merged to master via PR #42)
**Commits:** 10 (9 implementation + 1 CR remediation)
**Scope:** 12 files changed, 1,858 lines added, 29 removed
**Tests:** 877 workspace tests passing, zero clippy warnings

---

## What was built

The entire checkpoint data tier — from raw mmap'd memory through CUDA DMA
to NVMe spill files. This is the storage foundation that makes reverse
stepping, replay, and bookmarks possible in Phase 3B/C.

Before this session, the daemon had checkpoint *metadata* (session.rs could
create/list/restore/delete/bookmark checkpoint entries and track their tick
positions) but zero actual tensor data behind them. Now the worker owns a
real arena, captures real activations at √L layer boundaries, stores real
RNG state, spills to NVMe with CRC32 integrity, and restores everything
back into PyTorch's live tensors.

---

## Architecture

### The three-layer stack

```
┌──────────────────────────────────────────────────────────┐
│ Daemon (session.rs)                                       │
│ Metadata tier — checkpoint registry, position tracking,   │
│ bookmark annotations. Stateless w.r.t. tensor data.       │
│ Routes rocket/checkpoint → orchestrator → worker.          │
│ Auto-checkpoint: fires after step if tick is on √L layer.  │
├──────────────────────────────────────────────────────────┤
│ Worker (dispatch.rs, checkpoint.rs)                        │
│ _host/checkpoint Create: iterate √L layers, call Python    │
│ capture for each, store RNG state in sentinel slot.        │
│ _host/checkpoint Restore: reload activations + RNG.        │
│ Spill oldest auto-checkpoint to NVMe when arena > 80%.     │
├──────────────────────────────────────────────────────────┤
│ Python bridge (checkpoint.py)                              │
│ capture_activation: torch.frombuffer → copy_ → sync        │
│ restore_activation: frombuffer → copy_ into last_outputs   │
│ capture/restore_rng_state: per-device CUDA RNG bytes       │
│ register/unregister_cuda_pinned: cudaHostRegister           │
└──────────────────────────────────────────────────────────┘
```

### The zero-copy contract

No intermediate CPU tensors. No Python-managed buffers. No serialization
format for in-memory data.

1. Rust `mmap(MAP_ANONYMOUS | MAP_POPULATE)` — pre-faulted pages
2. Python `cudaHostRegister(ptr, len, 0)` — CUDA sees it as pinned
3. Python `torch.frombuffer(ctypes.from_address(ptr))` — zero-copy view
4. PyTorch `.copy_()` triggers CUDA DMA at full PCIe bandwidth
5. `torch.cuda.synchronize()` — fence before Rust reads the bytes

One DMA transfer. One address space owner (Rust). CUDA just uses the
pages directly.

### √L checkpoint strategy (Chen et al. 2016)

For a model with N layers, checkpoint at √N evenly-spaced boundaries.
This minimizes peak arena memory: instead of storing all N layers (O(N)),
store √N layers and replay at most √N layers to reconstruct any
intermediate state.

`checkpoint_layers()` lives in the protocol crate (not the worker) because
the daemon also needs it for auto-checkpoint layer matching. It's pure math,
no runtime dependencies.

---

## File inventory

### New files

| File | Lines | Role |
|------|-------|------|
| `crates/rocket-surgeon-worker/src/checkpoint.rs` | 939 | Arena, slot headers, spill/load, tests |
| `python/rocket_surgeon/checkpoint.py` | 131 | PyO3 bridge — capture/restore/RNG/CUDA pinning |
| `crates/rocket-surgeon-worker/src/bridge.rs` | +87 | Rust wrappers calling Python checkpoint functions |

### Modified files

| File | Delta | What changed |
|------|-------|--------------|
| `crates/rocket-surgeon-worker/src/dispatch.rs` | +522 | `handle_host_checkpoint` Create/Restore, arena init in attach, `layer_index_from_path`, sanitization |
| `crates/rocket-surgeon-worker/src/main.rs` | +1 | `mod checkpoint;` |
| `crates/rocket-surgeon-worker/Cargo.toml` | +38 | `crc32fast`, `libc` deps |
| `crates/rocket-surgeon/src/main.rs` | +37 | Checkpoint routing branch, auto-checkpoint after step |
| `crates/rocket-surgeon/src/dispatch.rs` | +63 | `handle_checkpoint` widened for orchestrator forwarding |
| `crates/rocket-surgeon/src/orchestrator_handle.rs` | +34 | `checkpoint()` method |
| `crates/rocket-surgeon/src/session.rs` | +23 | `checkpoint_create_with_id`, `auto_checkpoint_layers` |
| `crates/rocket-surgeon-protocol/src/lib.rs` | +11 | `checkpoint_layers()` function |
| `Cargo.toml` | +1 | workspace `crc32fast` |

---

## Key design decisions and invariants

### 1. Arena is worker-local, not shared memory

The checkpoint arena uses anonymous `mmap`, NOT `shm_open`. Cross-process
access goes through the existing DoomRing IPC channel. This avoids the
complexity of managing shared mmap lifecycle across daemon/worker crashes.

### 2. Fixed-size slots, no fragmentation

Every slot is the same size: `SLOT_HEADER_SIZE + align_up(hidden_dim * max_seq_len * dtype_size, 64)`.
This means a free-list allocator with O(1) alloc/free and zero
fragmentation. The tradeoff is wasted space when a layer's activation is
smaller than the slot — acceptable because the dominant cost is the full
hidden state.

### 3. Deterministic eviction ordering

`checkpoint_order: Vec<String>` tracks insertion order. `oldest_checkpoint()`
returns the first element, not a random HashMap key. This is critical for
the spill policy (spill oldest auto-checkpoint when arena > 80%).

### 4. Transactional creation with rollback

`arena.snapshot()` before starting a checkpoint capture. If any layer
capture fails, `arena.rollback(snap, checkpoint_id)` frees all allocated
slots and removes index entries. No partial checkpoints survive.

### 5. Single-threaded dispatch — the safety invariant

`unsafe impl Send for CheckpointArena` is sound ONLY because the worker
dispatch loop is serial. The arena uses `RefCell` (not `Mutex`) for interior
mutability. If the worker is ever made multi-threaded, `RefCell` must be
replaced with `Mutex` and the `unsafe impl Send` must be re-evaluated.

The Python bridge functions receive raw arena pointers. They are safe
because the dispatch loop that calls them via PyO3 owns the arena reference,
so the arena is alive for the duration of each call. This is documented in
checkpoint.py's module docstring.

### 6. checkpoint_id is a filename component

The spill path is `{dir}/{checkpoint_id}.ckpt`. The `is_safe_checkpoint_id()`
function rejects `/`, `\`, `..`, NUL bytes, and empty strings. Any code
that constructs checkpoint IDs must use this validator or avoid
path-sensitive characters.

### 7. RNG state stored in sentinel slot

CUDA RNG state (per-device, variable length) is stored in the arena at
layer index `u32::MAX` — a sentinel value that real layers can never have.
This keeps RNG storage inside the arena lifecycle (freed with the
checkpoint, spilled/loaded with it) without a separate data structure.

### 8. SlotHeader and SpillIndexEntry validate on read

Both `SlotHeader::read_from()` and `SpillIndexEntry::read_from()` return
`Option`, validating magic bytes and dtype tags. Callers propagate `None`
as errors. This prevents silent corruption from mangled arena memory or
truncated spill files.

---

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `RS_CHECKPOINT_ARENA_MB` | auto-computed | Override arena size in MB. Overrides the √L-based slot count. |
| `RS_MAX_SEQ_LEN` | 2048 | Maximum sequence length for slot sizing. Affects arena memory footprint. |

---

## CR remediation: what was found and fixed

Two Opus subagents reviewed the entire checkpoint implementation. Combined
findings after deduplication: 3 Critical, 7 High, 11 Medium, 10 Low.
All 21 unique findings were fixed in commit `143df1d`.

### Critical fixes

| ID | Finding | Fix |
|----|---------|-----|
| C1 | `slot_info_for_checkpoint` did O(N) linear scan of all slots | Rewrote to use `checkpoint_slots` map → O(slots-per-checkpoint) |
| C2 | `free_checkpoint` could double-free (no early return guard) | Added `let Some(...) else { return }` guard |
| C3 | No documentation on why arena can't be dropped during Python call | Added safety comment in checkpoint.py module docstring |

### High fixes

| ID | Finding | Fix |
|----|---------|-----|
| H1 | `oldest_checkpoint` used HashMap iteration (non-deterministic) | Added `checkpoint_order: Vec<String>` for insertion ordering |
| H2 | Arena capacity `slot_size * num_slots` could overflow | Changed to `checked_mul` with error message |
| H3 | `checkpoint_id` used directly in spill filenames — path traversal | Added `is_safe_checkpoint_id()` validator |
| H5 | `restore_activation` silently no-op'd when key missing from last_outputs | Now raises `KeyError` |
| H6 | `SlotHeader::read_from` returned `Self`, no validation | Returns `Option<Self>`, validates magic + dtype |
| H7 | `load_spilled_checkpoint` didn't bounds-check data_len vs slot capacity | Added `anyhow::ensure!` check |

### Medium fixes (selected)

- **M1:** Shape dimensions validated ≥ 0 before byte_len product
- **M2:** RNG alloc failure now logged (was silently ignored)
- **M3:** `num_layers` stored from attach metadata (eliminated runtime fallback)
- **M4:** `max_seq_len` overridable via `RS_MAX_SEQ_LEN`
- **M8:** `flush()` → `sync_all()` for spill durability
- **M10:** `layer_index_from_path` now searches after "layers" segment

### Low fixes (selected)

- **L1:** `align_up` debug_assert power-of-two
- **L3:** `base_ptr` visibility narrowed to `pub(crate)`
- **L6:** Reusable `[0u8; 64]` pad buffer instead of `vec![0u8; pad]`
- **L7:** SLOT_MAGIC endianness comment
- **L8:** Element size lookup table instead of `torch.tensor([]).element_size()`

---

## Test coverage

### checkpoint.rs (28 tests)

- Slot header roundtrip, bad magic rejection, bad dtype rejection
- Arena lifecycle: new, alloc, free, exhaustion, transactional rollback
- `get_slot` read-back, missing returns None
- `checkpoint_slots` ownership tracking
- Spill/load roundtrip with multi-slot checkpoint
- CRC32 corruption detection
- Double-free is no-op
- Oldest checkpoint insertion ordering
- Pre-existing checkpoint additive load
- `checkpoint_layers()`: specific values for 32/80 layers, edge cases (1, 2), excludes 0 and last

### dispatch.rs (8 checkpoint tests + 2 new)

- Invalid params returns error
- No model returns error
- No arena returns error
- Restore not-found returns error
- `layer_index_from_path` extraction (including non-"layers" path)
- `is_safe_checkpoint_id` sanitization

### Other crates

- Protocol serde roundtrips cover `HostCheckpointRequest/Response`
- Daemon dispatch and session tests cover metadata tier operations

---

## What's next: Phase 3B and 3C

### Sub-project B: Forward replay and reverse step

This is where the checkpoint tier gets *used*. Requires:

1. **`rocket/replay` verb** — restore checkpoint, re-run forward pass from
   that point, re-fire probes, collect divergences
2. **Divergence detection** — compare replayed activations against original
   captured values. The arena's slot data IS the ground truth.
3. **Determinism enforcement** — the RNG state capture/restore is already
   implemented. Op-level pinning (`torch.use_deterministic_algorithms`)
   is deferred to 3B.
4. **Reverse step** — conceptually `checkpoint_restore(nearest_before) +
   replay_forward(to_target - 1)`. The pieces exist, the verb doesn't.

Key API surface the checkpoint tier provides to 3B:
- `arena.get_slot(checkpoint_id, layer_idx)` — read back captured data
- `checkpoint_layers(num_layers)` — know which layers have checkpoints
- `bridge::restore_rng_state()` — RNG determinism
- `session.checkpoint_restore()` — position tracking
- `load_spilled_checkpoint()` — bring spilled data back for replay

### Sub-project C: Bundle extension and Tier 2 interventions

1. **Bundle extension** — session bundles (`.rsb` files) need to include
   checkpoint data and bookmark annotations alongside the existing
   session state export
2. **Tier 2 Python callbacks** — interventions that fire during replay
   (modify activations mid-replay to test counterfactuals)
3. **TCK green sweep** — 161 deferred scenarios across 37 feature files.
   checkpoint.feature has 0 deferred (already green). replay.feature has
   8 deferred that 3B should unlock. Many others in inspection, kv-cache,
   and session-export depend on checkpoint/replay infrastructure.

---

## Deferred debt and known limitations

1. **FullSnapshot tier** — only Activation tier is implemented. Full model
   state capture (weights + optimizer state) is out of scope for Phase 3.
2. **Multi-GPU checkpoint coordination** — the arena is per-worker. In
   multi-GPU setups, each worker checkpoints independently. No cross-rank
   coordination or synchronized checkpoint IDs yet. The architecture
   supports it (each worker has its own arena, daemon routes to the right
   worker) but it's untested.
3. **Spill file cleanup** — spill files in `/tmp/rocket-surgeon-spill/`
   are not automatically cleaned up on daemon shutdown. They survive
   crashes (which is good for recovery) but accumulate (which is bad
   for disk).
4. **Arena resize** — the arena is fixed-size at construction time. If the
   model's activation sizes change (e.g., variable sequence length beyond
   `max_seq_len`), the arena can't grow. The env var override
   (`RS_CHECKPOINT_ARENA_MB`) is the escape hatch.
5. **161 deferred TCK scenarios** — total across all feature files. This
   number has been stable since Phase 2. Phase 3B should start cutting
   into it significantly.

---

## Session arc

This session ran the full JSMNTL cycle across one complete sub-project:

1. **Brainstorming** — decomposed Phase 3 (12 tasks from the plan.md) into
   three sub-projects. Spec'd Sub-project A with zero-copy arena
   architecture, slot layout, √L strategy, NVMe spill format.
2. **Plan** — 12-task implementation plan with TDD steps, exact file paths,
   test commands. Saved to `docs/superpowers/plans/2026-05-24-checkpoint-state-tier.md`.
3. **Execution** — Tasks 1-10 implemented in order (√L selector → slot
   headers → arena → spill/load → Python bridge → Rust bridge → dispatch
   handler → daemon wiring → auto-checkpoint). Each task committed
   atomically. Two mid-plan corrections required sub-plans (tasks 8+9
   combined due to dead_code deps, task 10 moved `checkpoint_layers()` to
   protocol crate).
4. **Code review** — Dual Opus subagent CR + red team. 21 unique findings
   after dedup.
5. **Remediation** — All 21 findings fixed, tested, committed, pushed.
6. **Ship** — PR #42 created and merged to master.

The user invoked JSMNTL-HNVC discipline twice during execution ("plan,
don't patch") which caught me about to jump into code without proper
sub-plans. Those corrections were load-bearing — the code that came out
of proper sub-planning was significantly cleaner than what I was about
to write.
