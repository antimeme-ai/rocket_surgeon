# Work Unit 0.4 — Capability Negotiation Spec

## Scope

Define the capability negotiation protocol: what the server advertises at `initialize`, how
capabilities evolve across phases, and how clients adapt. This is a **narrative document** plus
verification that the `Capabilities` type in `common.json` is complete.

## Deliverables

1. `protocol/capabilities.md` — Narrative doc explaining:
   - Every capability flag and what it controls
   - Which phase introduces each capability
   - How clients should degrade when a capability is absent
   - Forward-compatibility: unknown capabilities are ignored
   - The `head_granularity: "requires_unfused"` honesty pattern

2. Audit `protocol/schema/v0.1.0/common.json` `Capabilities` type:
   - Verify every phase-gated feature has a flag
   - Verify all flags are boolean or enum (no free-form strings)
   - Add any missing flags discovered during narrative writing

## Acceptance Criteria (from plan doc)

- [ ] Every phase-gated feature has a corresponding capability flag
- [ ] Capability flags are boolean or enum (not free-form strings)
- [ ] Client can determine available tick granularities, intervention types, built-in views,
      execution mode, parallelism mode from capabilities alone
- [ ] `head_granularity` explicitly states `"requires_unfused"`
- [ ] Unknown capabilities are ignored (forward-compatible) — documented
- [ ] TCK targets identified: capability-gated verb rejection, unknown capability handling

## Approach

1. Read design doc phases (§18) to inventory all phase-gated features
2. Cross-reference against current `Capabilities` type in common.json
3. Write `protocol/capabilities.md` with per-flag documentation
4. Update common.json if gaps found
5. Subagent review
6. Fix findings

## Phase-gated features to inventory

| Phase | Feature | Expected capability flag |
|-------|---------|------------------------|
| MVP (0-2) | eager execution, single-GPU, 5 interventions, component/layer granularity | execution_mode, parallelism, intervention_types, tick_granularities |
| 3 | checkpointing, reverse step | supports_checkpointing, supports_reverse_step |
| 4 | protobuf wire format | wire_format (stretch) |
| 5 | TCP transport, remote debugging | transports |
| 6 | MoE support, route_override | supports_moe, moe tick granularities |
| 7 | head granularity, FlashAttention unfusing | head_granularity |
| 8+ | backward pass, SAE, MCP | supports_backward, supports_sae |
