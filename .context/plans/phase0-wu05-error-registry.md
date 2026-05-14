# Work Unit 0.5 — Error Code Registry

## Scope

Define the complete enumeration of structured error codes. Every protocol error must use a
registered code — no ad-hoc invention during implementation. This is a **schema file** plus
narrative documentation, plus verification that `common.json` ErrorCode/ErrorData are aligned.

## Deliverables

1. `protocol/schema/v0.1.0/errors.json` — Error code registry:
   - Each code: string identifier, numeric JSON-RPC code, description, suggestion template,
     which verbs can raise it, severity (fatal vs. recoverable)

2. Audit `protocol/schema/v0.1.0/common.json`:
   - Verify `ErrorCode` enum matches registry exactly
   - Verify `ErrorData` type has all required fields
   - Add any missing codes from the plan doc's minimum set

3. Update `protocol/README.md` error section if needed

## Acceptance Criteria (from plan doc)

- [ ] Every error code has a unique string identifier, numeric code, and description
- [ ] Every error includes `suggestion` field with recovery guidance
- [ ] Error codes are referenced by all verb schemas' error response definitions
- [ ] No verb implementation may invent error codes not in this registry
- [ ] TCK targets identified: error contract scenarios use registry codes

## Minimum error code set (from plan doc)

| Code | Numeric | Description |
|------|---------|-------------|
| INVALID_STATE | -32001 | Verb not valid in current state |
| INVALID_TARGET | -32002 | Probe point doesn't match any component |
| INVALID_RECIPE | -32003 | Intervention recipe malformed or unsupported |
| MODEL_NOT_ATTACHED | -32004 | Operation requires attached model |
| TENSOR_NOT_FOUND | -32005 | tensor_id doesn't exist in store |
| CHECKPOINT_NOT_FOUND | -32006 | Checkpoint name/id doesn't exist |
| PROBE_NOT_FOUND | -32007 | probe_id doesn't exist |
| CAPABILITY_NOT_SUPPORTED | -32008 | Verb requires a capability the session lacks |
| SLICE_OUT_OF_BOUNDS | -32009 | Tensor slice indices exceed shape |
| RESPONSE_TOO_LARGE | -32010 | Requested data exceeds 64 KB cap |
| HOST_ERROR | -32011 | Python host process error |
| GPU_OOM | -32012 | GPU out of memory |
| NCCL_TIMEOUT | -32013 | NCCL collective timed out |
| REPLAY_DIVERGENCE | -32014 | Replay exceeded tolerance (informational) |
| UNSUPPORTED_MODEL | -32015 | Model architecture not in support matrix |
| COMPILED_MODEL | -32016 | Model uses torch.compile, Tier A cannot attach |

## Approach

1. Cross-reference plan doc minimum set against current common.json `ErrorCode` enum
2. Identify gaps (plan doc has 16 codes, common.json currently has 14)
3. Write `protocol/schema/v0.1.0/errors.json` with full registry
4. Update common.json `ErrorCode` enum to match
5. Write per-code suggestion templates and verb applicability
6. Subagent review
7. Fix findings

## Numeric code range

JSON-RPC 2.0 reserves:
- -32700 to -32600: parse errors
- -32603 to -32600: standard errors
- -32099 to -32000: server errors (implementation-defined)

We use -32001 through -32099 for rocket_surgeon domain errors. Codes below -32099 are available
for future extension.
