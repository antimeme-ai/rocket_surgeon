# ADR-0008: Checkpoint Metadata/State Tier Split and Storage Strategy (WU-C)

## Status
Accepted

## Context
WU-C implements `rocket/checkpoint` — the keystone verb of the v0.3.0
roadmap. Three downstream work units depend on it: `rocket/replay`,
`branch.fork/drop/compare`, and `sweep`. All three reference checkpoints by
id (`from_checkpoint`, `baseline_checkpoint`), so the guarantees a checkpoint
makes are load-bearing for the rest of the roadmap.

A checkpoint has two conceptually separate halves:

1. **Metadata** — the `CheckpointRef` (`checkpoint_id`, `tick_id`,
   `layer_idx`, `tier`, `bookmark`, `created_at`). Small, JSON-serialisable,
   needed to answer `list`, to validate `restore`/`delete` ids, and to
   populate the `checkpoints` array in every `SessionState` envelope.
2. **State** — the actual model tensors: residual streams + RNG state +
   input ids for the `activation` tier, or a full `state_dict` for the
   `full_snapshot` tier. Large, lives as live torch objects, only
   meaningful inside the worker process that owns the `nn.Module`.

Every scenario in `tck/protocol/checkpoint.feature` is satisfiable by
metadata alone — the feature asserts `tick_id`/`layer_idx`/`tier`/`bookmark`
and that `restore` moves the logical tick position. It never asserts that
restored tensors are bit-identical. The genuinely hard part (capturing and
re-seating torch state against a live, possibly mid-forward-pass model, with
RNG round-trip and multi-GPU sharding) is therefore separable from the
behavioural contract.

Options considered for *where* checkpoint state lives:
1. **Single owner in the worker, daemon holds nothing.** The daemon would
   round-trip to the worker even for `list`/`delete`, and the
   `SessionState.checkpoints` envelope could not be populated without a
   worker call on every response.
2. **shm ring.** The existing `DoomRing` shared-memory transport is a fixed
   16-slot × 64 MB ring sized for streaming probe frames. Retained
   multi-MB (or model-sized) snapshots do not fit its lifecycle model.
3. **Disk spill.** Correct eventually for large `full_snapshot`s, but a
   storage-tiering concern that belongs to the `branch` WU's `Spilled`
   tier, not to WU-C.

## Decision
**Split checkpoints into a daemon-owned metadata tier and a worker-owned
state tier.**

- The daemon `Session` owns the `CheckpointRef` registry as the single
  source of truth. It lives directly in `state.checkpoints`, so the
  `SessionState` envelope is always consistent with no projection step.
  `list`, `delete`, and `bookmark` are pure daemon bookkeeping and never
  round-trip to the worker. `create` and `restore` additionally move the
  logical tick position and *may* round-trip for state capture.
- The internal `_host/checkpoint` channel (new in WU-C:
  `HostCheckpointRequest` / `HostCheckpointResponse`) carries only the two
  state-affecting actions, `create` and `restore`. The daemon mints the
  `checkpoint_id` so the worker can key its snapshot store under the same id.
- Checkpoint *state*, when implemented, lives in **worker-process memory**
  as live torch objects keyed by `checkpoint_id` — not in the shm ring
  (wrong shape/lifecycle) and not on disk (deferred to branch-tiering).
- WU-C ships the **metadata tier**: the verb is fully functional per
  `checkpoint.feature`, `Capabilities.supports_checkpointing` is advertised
  `true`, and `restore` moves the logical position in `SessionState`.
  Worker-side tensor capture/restore is a follow-up slice over the
  `_host/checkpoint` channel that this ADR's protocol already provides for.

## Consequences
- **Good**: `list`/`delete`/`bookmark` are zero-latency, worker-free, and
  work even with no orchestrator attached. The `checkpoints` envelope is
  populated for free on every response.
- **Good**: The downstream WUs (replay/branch/sweep) can build against the
  daemon-side registry immediately — a valid `checkpoint_id` is all they
  need to begin, regardless of whether the state tier has landed.
- **Good**: The `_host/checkpoint` wire types exist now, so the state-tier
  slice wires them in without a second protocol change.
- **Trade-off**: A checkpoint does **not** guarantee bit-identical replay
  from metadata alone. `restore` moves the logical tick position;
  tensor-level fidelity is the state tier's job, and divergence checking
  belongs to `rocket/replay` with `verify`. Downstream WUs must not assume
  a bare checkpoint reproduces activations.
- **Trade-off**: `full_snapshot` state is N × model size resident in worker
  memory with no eviction policy. The `VramExhausted` error code exists for
  this; an eviction/spill policy is explicitly deferred to the `branch` WU.
- **Trade-off**: a `bookmark` for a tick with no existing checkpoint mints a
  `ProbeLog`-tier `CheckpointRef`. Its id is a valid `restore` target —
  `restore` moves the logical position to the bookmarked tick — but it
  captures no tensor state and no full `TickPosition`. Downstream WUs
  (`replay`, `branch`) MUST inspect `tier` before treating a checkpoint id
  as state-bearing: a `ProbeLog` id has nothing to replay. The daemon
  captures the full `TickPosition` (not just `tick_id`/`layer_idx`) for
  `activation`/`full_snapshot` checkpoints so `restore` preserves
  `direction`/`component`/`phase` — the wire `CheckpointRef` cannot.
- **Limitation**: WU-C implements and tests single-rank. A checkpoint is
  inherently distributed — one shard per worker process — and a correct
  multi-rank `restore` for FSDP/TP requires a collective barrier (a
  half-restored model is corrupt). Multi-rank checkpoint is a documented
  follow-up, tracked as a bead.
