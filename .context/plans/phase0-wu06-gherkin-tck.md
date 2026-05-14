# Work Unit 0.6 — Gherkin TCK Feature Files

## Scope

Write `.feature` files covering the protocol's behavioral contract. These are the **specification**
that will drive implementation — tests fail red until each feature is built. Split by verb group.

## Deliverables

16 `.feature` files in `tck/`:

### Protocol verbs (9 files)
1. `tck/protocol/lifecycle.feature`
2. `tck/protocol/stepping.feature`
3. `tck/protocol/inspection.feature`
4. `tck/protocol/intervention.feature`
5. `tck/protocol/probes.feature`
6. `tck/protocol/checkpoint.feature` (`@phase3`)
7. `tck/protocol/replay.feature` (`@phase3`)
8. `tck/protocol/subscribe.feature`
9. `tck/protocol/errors.feature`

### Protocol envelope (1 file)
10. `tck/protocol/state-envelope.feature`

### Model adapter (2 files)
11. `tck/model/adapter.feature`
12. `tck/model/hooks.feature`

### Tensor handling (1 file)
13. `tck/tensor/handles.feature`

### MoE (1 file)
14. `tck/moe/tick-granularity.feature` (`@phase6`)

### Session (1 file)
15. `tck/session/bundle.feature`

### Capability negotiation (1 file — not in plan but needed per AC)
16. `tck/protocol/capabilities.feature`

## Coverage Matrix: State Machine Transitions

Every arrow in design doc §3 must have at least one scenario.

| From | To | Trigger | Feature file |
|------|----|---------|-------------|
| UNINITIALIZED | INITIALIZED | initialize | lifecycle |
| INITIALIZED | ATTACHING | attach | lifecycle |
| ATTACHING | STOPPED | attached (internal) | lifecycle |
| STOPPED | STEPPING | step | stepping |
| STEPPING | STOPPED | step complete | stepping |
| STOPPED | INSPECTING | inspect | inspection |
| INSPECTING | STOPPED | inspect complete | inspection |
| STOPPED | MODIFYING | intervene | intervention |
| MODIFYING | STOPPED | intervene complete | intervention |
| STOPPED | DETACHING | detach | lifecycle |
| DETACHING | INITIALIZED | detach complete | lifecycle |

Invalid transitions (→ INVALID_STATE error):
| From | Attempted | Feature file |
|------|-----------|-------------|
| UNINITIALIZED | attach | errors |
| UNINITIALIZED | step | errors |
| INITIALIZED | step | errors |
| INITIALIZED | inspect | errors |
| INITIALIZED | detach (no model) | errors |
| ATTACHING | step | errors |
| ATTACHING | inspect | errors |
| STEPPING | inspect | errors |
| STEPPING | intervene | errors |
| INSPECTING | step | errors |
| MODIFYING | step | errors |

## Coverage Matrix: Error Codes (18 codes)

Every error code in errors.json must have at least one trigger scenario in `errors.feature`.

| Code | Trigger scenario |
|------|-----------------|
| INVALID_STATE | step while not stopped |
| INVALID_TARGET | inspect non-existent component |
| INVALID_RECIPE | intervene with malformed recipe |
| MODEL_NOT_ATTACHED | step before attach |
| TENSOR_NOT_FOUND | inspect with bad tensor_id |
| CHECKPOINT_NOT_FOUND | restore non-existent checkpoint |
| PROBE_NOT_FOUND | enable non-existent probe_id |
| CAPABILITY_NOT_SUPPORTED | checkpoint before Phase 3 |
| SLICE_OUT_OF_BOUNDS | inspect slice beyond shape |
| RESPONSE_TOO_LARGE | inspect full on large tensor |
| HOST_ERROR | Python host crash (simulated) |
| GPU_OOM | allocate beyond GPU memory (simulated) |
| NCCL_TIMEOUT | NCCL timeout (simulated, @phase5) |
| REPLAY_DIVERGENCE | replay with mutation, verify divergence (@phase3) |
| UNSUPPORTED_MODEL | attach unsupported architecture |
| COMPILED_MODEL | attach torch.compile model |
| MODEL_ALREADY_ATTACHED | double attach |
| INVALID_PARAMS | malformed JSON-RPC params |

## Coverage Matrix: Response Envelope Fields

Every field in SessionState must be asserted in at least one scenario in `state-envelope.feature`.

| Field | Assertion |
|-------|-----------|
| session_id | present, uuid format, stable across responses |
| model_id | null before attach, populated after |
| status | matches expected state |
| position | null before step, populated after |
| tick_id | null before step, monotonically increasing |
| active_probes | empty initially, populated after probe define |
| checkpoints | empty initially, populated after checkpoint create |
| available_actions | matches state machine per status |

## Approach

1. Write feature files in groups, dispatching to parallel agents:
   - Group A: lifecycle, stepping, inspection, state-envelope (core verbs)
   - Group B: intervention, probes, subscribe, capabilities
   - Group C: checkpoint, replay, errors (error coverage matrix)
   - Group D: adapter, hooks, handles, moe, bundle (domain-specific)
2. Self-review: verify coverage matrices above are satisfied
3. Subagent code review
4. Fix findings

## Tags

- `@phase3` — checkpoint/replay scenarios (schema frozen, implementation deferred)
- `@phase5` — multi-GPU scenarios (NCCL timeout)
- `@phase6` — MoE tick granularity scenarios
- `@wip` — scenarios that need step definition stubs in 0.7

## Acceptance Criteria (from plan doc)

- [ ] Every state transition in design doc §3 has at least one scenario
- [ ] Every error code in registry (0.5) has at least one trigger scenario
- [ ] Every response-envelope field from design doc §6 is asserted in at least one scenario
- [ ] `@phase3`, `@phase6` tags mark future-phase scenarios
- [ ] Total scenario count secondary to transition coverage
