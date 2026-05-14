# ADR-0003: Probe Model — DTrace-Inspired Naming with Composable Hooks

## Status
Proposed

## Context
rocket_surgeon needs a unified abstraction for:
1. Observing tensor state at arbitrary points in the forward pass
2. Triggering checkpoints at strategic locations
3. Applying interventions (surgery) at specific components
4. Running interpretability analyses (logit lens, SAE decomposition, attention patterns)
5. Collecting aggregate statistics across ticks

These are all instances of the same pattern: WHAT to do at WHERE under WHEN conditions. Systems observability (DTrace, eBPF, tracepoints) solved this decades ago.

Options considered:
- **PyTorch hooks directly**: simple, no abstraction. But no lifecycle management, no filtering, no composition, no naming convention, no discovery. Every consumer reimplements the same patterns.
- **Breakpoint model (DAP-style)**: familiar to developers. But breakpoints are boolean (fire/don't fire) — we need typed hooks with parameters, filtering, and aggregation.
- **DTrace-inspired probe model**: named points with composable hooks, lifecycle management, zero-cost-when-off, wildcard queries, self-describing metadata. Proven at scale in production systems.

## Decision
**DTrace-inspired probe model: named probe points with a registry of composable hooks.**

### Probe Point Naming

Four-level hierarchical: `model:layer:component:event`

- `model`: architecture identifier (llama, mixtral, pythia)
- `layer`: layer number or `*` wildcard
- `component`: attention, mlp, norm, residual, router, expert.N
- `event`: input, output, weight, decision

Wildcards at any level: `llama:*:attention:output` matches all layers.

### Probe = Point + Hook + Filter

```
{
  point:    "llama:12:attention:output",
  hook:     "inspect",
  filter:   "norm > 50.0",
  enabled:  true,
  priority: 0
}
```

### Hook Registry

Built-in hooks: inspect, checkpoint, trace, aggregate, assert, intervene, sae_decompose. User-defined hooks via Python callables registered at runtime.

### Lifecycle

register → arm → fire → disarm → deregister

Same as DTrace/eBPF. Probes exist but are NOPs until armed. Arming patches the hook into the execution path. Disarming reverts to NOP. Deregistration removes entirely.

### Zero-Cost When Off

Unarmed probe points are unconditional branches over empty blocks. The PyTorch hook is only registered when at least one probe on that point is armed. When all probes on a point are disarmed, the hook is removed.

## Consequences
- **Good**: single abstraction covers observation, checkpointing, intervention, and analysis. No separate systems for each.
- **Good**: LLM clients discover available probes via `probe list` with wildcards. Self-documenting.
- **Good**: composable — multiple hooks on one point, same hook on multiple points. Multiplexing and fan-out.
- **Good**: zero-cost-when-off means probes can be compiled into the model wrapper without affecting performance until activated.
- **Bad**: abstraction overhead — simple cases (just inspect one tensor) require creating a probe. Mitigation: shorthand `inspect llama:12:attention:output` auto-creates ephemeral probe.
- **Bad**: DTrace naming convention is unfamiliar to ML researchers. Mitigation: aliases for common patterns (e.g., `layer 12 attention` maps to `llama:12:attention:output`).
- **Risk**: wildcard queries on large models could create thousands of probes. Need guard rails (max active probes, warn on > 100).
