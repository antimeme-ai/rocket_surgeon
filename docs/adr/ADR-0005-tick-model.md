# ADR-0005: Tick Model — Granularity, Synchronization, and Identity

## Status
Proposed

## Context
rocket_surgeon steps through transformer forward passes "one tick at a time." The tick model defines what a tick is, how many granularity levels exist, how the forward pass is paused at tick boundaries, and how ticks are identified across steps and replays. See design doc §7 for the full specification.

Key tensions:
1. **Granularity vs. overhead**: finer ticks (per-head, per-expert) give more surgical precision but require more barriers and potentially unfused execution. Coarser ticks (per-layer) are cheap but hide internal structure.
2. **Synchronization cost**: the mechanism that pauses the forward pass at tick boundaries must be cheap when not at a boundary and correct when it is. The naive approach (`cudaDeviceSynchronize`) blocks ALL streams on the device, which is catastrophically expensive on multi-stream workloads and pipeline-parallel setups.
3. **Tick identity**: ticks must be uniquely and stably identifiable across the session, including replayed ticks that revisit the same model position.
4. **MoE complexity**: MoE layers have internal structure (router, dispatch, per-expert compute, combine) that dense tick granularities cannot express.

Options considered for synchronization:
1. **`cudaDeviceSynchronize`**: blocks all CUDA streams on the device. Simple but expensive — it serializes all GPU work, not just the stream we care about. Unacceptable for multi-stream, pipeline-parallel, or any setup with concurrent GPU work.
2. **CUDA events scoped to relevant stream**: record an event on the current compute stream, synchronize only that event. Blocks the host thread for that stream only; other streams continue. This is the standard pattern for targeted synchronization.
3. **Host-only barrier (no GPU sync)**: fastest, but tensor data may not be ready on CPU when the barrier fires. Useless for inspection.

Options considered for MoE granularity:
1. **Treat MoE layers as opaque components**: layer-level ticks only. Loses the ability to inspect routing decisions or individual experts.
2. **Four sub-granularities within MoE layers**: router_pre_topk, router_post_topk, expert, moe_layer. Designed into the tick model from day one, implemented in Phase 6.

## Decision
**CUDA events scoped to the relevant stream for synchronization. Three dense granularities plus four MoE granularities. Tick scoping for per-region granularity control. Monotonic u64 tick_id with replay_of references.**

### Synchronization

Tick boundaries use CUDA events, not `cudaDeviceSynchronize`:

```python
event = torch.cuda.Event()
event.record(torch.cuda.current_stream())
event.synchronize()  # blocks host, not other streams
```

For pipeline parallelism, each rank synchronizes its own stream independently. For tensor parallelism, all ranks synchronize before the daemon declares STOPPED.

### Dense granularities

| Level | What fires | Approx. count (32-layer model) | Use case |
|-------|-----------|--------------------------------|----------|
| `layer` | Between transformer blocks | ~32 | Coarse navigation |
| `component` | Between attn/MLP/norm within a block | ~192 | Standard debugging (default) |
| `head` | Per attention head | ~1024 | Fine attention analysis |

Head granularity requires unfused execution. In standard (fused) execution, all heads run as a single batched matmul. You cannot pause "between head 3 and head 4" without splitting the computation. The protocol's capabilities response states this honestly: `"head_granularity": "requires_unfused"`. For MVP, head data is accessible via tensor slicing in `inspect` (no per-head stepping); per-head stepping ships in Phase 7 alongside FlashAttention shadow replay.

### MoE granularities

Four additional sub-granularities within each MoE layer:

| Level | What fires | Use case |
|-------|-----------|----------|
| `router_pre_topk` | After router emits logits, before top-k selection | Routing inspection and override |
| `router_post_topk` | After top-k selection, before expert dispatch | Assignment inspection and override |
| `expert` | Inside a specific expert, post-dispatch | Per-expert tensor inspection |
| `moe_layer` | After combine (post-expert weighted sum) | Layer-level MoE inspection |

Expert granularity has the same unfused-execution caveat as head granularity: fused grouped GEMM (e.g., Megablocks) computes all experts simultaneously. Per-expert stepping requires unfused dispatch. Designed into the protocol from day one; implemented in Phase 6.

### Tick scoping

Users can set different granularities for different regions of the model:

```json
{
  "method": "rocket/probe",
  "params": {
    "action": "set_granularity",
    "scopes": [
      { "match": "layers[12]", "granularity": "component" },
      { "match": "layers[*]", "granularity": "layer" }
    ]
  }
}
```

Specific scopes override general ones. This avoids stepping through 192 ticks when you only care about layer 12's internals.

### Tick identity

- `tick_id` is a monotonic `u64`, never reused, never reset within a session. It is the primary key for checkpoints, probe firings, intervention attachment, and session bundle references.
- Replayed ticks get fresh `tick_id` values with a `replay_of: Option<u64>` field referencing the original tick. This preserves the invariant that `tick_id` is unique while maintaining the causal link to the original execution.
- Bookmarks are named references to tick_ids: `bookmark("before_ablation") -> tick_id 42`.

### Backward-tick schema

The tick model is symmetric forward/backward in the schema, even though backward-pass support is deferred to Phase 8+:

```
TickPosition {
    tick_id:    u64,
    direction:  forward | backward,
    rank:       Option<u32>,
    layer:      u32,
    component:  String,
    event:      pre | post,
}
```

Including `direction` now means the protocol does not need breaking changes when backward-pass support ships.

## Consequences
- **Good**: CUDA event synchronization is the correct primitive. It blocks only the host thread waiting on one stream, not all GPU work. This is essential for pipeline parallelism (multiple stages on different streams) and for not destroying throughput when concurrent GPU work is happening.
- **Good**: Three dense + four MoE granularities cover the full range of debugging needs from coarse navigation to fine surgical intervention.
- **Good**: Tick scoping means users pay the overhead of fine granularity only where they need it. Stepping through a 32-layer model at layer granularity is 32 ticks; zooming into one layer at component granularity adds ~6 ticks for that layer only.
- **Good**: Monotonic u64 tick_id is simple, efficient, and unambiguous. No collision risk, no coordination needed, no UUID overhead.
- **Good**: replay_of preserves causality without violating tick_id uniqueness. A client can trace replayed ticks back to their originals for divergence analysis.
- **Good**: Forward/backward symmetry in the schema is cheap insurance against future protocol breakage.
- **Bad**: Head and expert granularity require unfused execution, which is slower than fused kernels. This is an honest limitation, not a design flaw — the protocol surfaces it via capability negotiation rather than hiding it.
- **Bad**: Seven granularity levels add complexity to the barrier logic. The hook manager must check the active granularity scope for each component at each tick boundary. Mitigated by fast-path no-op checks when no fine-grained scopes are active.
- **Risk**: MoE granularity is designed but not implemented until Phase 6. The schema may need refinement when implementation reveals edge cases (e.g., expert parallelism across ranks with all-to-all dispatch). Mitigated by the protocol versioning strategy.
