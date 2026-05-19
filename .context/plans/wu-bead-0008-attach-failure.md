# Sub-plan: BEAD-0008 — daemon attach response reflects real backend state

**Bead**: [[BEAD-0008-daemon-silent-attach-failure]]
**Branch**: `wu-bead-0008-attach-failure`
**Base**: `master` (post-PR #4 and PR #5 merges)

## Problem

`Session::attach` calls `stub_model_info(&req.model_family)` to fabricate
`num_layers`, `num_heads`, `hidden_dim`. The orchestrator's real attach
happens *after* the response is built in `main.rs`, and its failure is
only logged as a warning — never propagated. Three e2e tests pass even
when the backend is broken (BEAD-0008 description).

## Approach

Flip the attach flow so the backend round-trip happens *before* the
client-facing response is built, and use real metadata from
`HostAttachResponse` to populate `AttachResponse`. Failed backend → error
response with a dedicated error code.

### Order today (broken)

```
client → ATTACH
       → session.attach()        # builds response with stub values
       → write response          # client sees fabricated "32 layers"
       → spawn_and_attach()      # fires, failure just logged
```

### Target order

```
client → ATTACH
       → validate request        # parse params, check session state
       → spawn_and_attach()      # block on real backend attach
         ↳ failure → BACKEND_ATTACH_FAILED response, no session mutation
       → session.attach(req, &host_resp)  # use real metadata
       → write response          # client sees real "2 layers"
```

## Files to change

| File | Change |
|------|--------|
| `crates/rocket-surgeon-protocol/src/errors.rs` | Add `BackendAttachFailed` variant |
| `crates/rocket-surgeon/src/session.rs` | `attach` takes `&HostAttachResponse`, uses real metadata; delete `stub_model_info` |
| `crates/rocket-surgeon/src/dispatch.rs` | `handle_attach` takes `Option<&HostAttachResponse>`, returns error if None |
| `crates/rocket-surgeon/src/main.rs` | For ATTACH: call `spawn_and_attach` first; pass response into dispatch; rollback session on failure |
| `tck/protocol/lifecycle.feature` | Add 2 scenarios: backend failure → error, success → real metadata |
| `tests/test_e2e_lifecycle.py` | Assert `num_layers` matches the tiny-llama (2, not 32) |

## TCK scenarios (RED first)

1. **Attach succeeds with backend metadata** — the response's `num_layers`,
   `num_heads`, `hidden_dim` equal whatever the worker reports (not a
   per-family stub).
2. **Attach with broken backend returns BACKEND_ATTACH_FAILED** — when the
   orchestrator/worker cannot load the model, the daemon's `attach`
   returns an error response carrying the backend's error message.

## Acceptance

- [ ] `BackendAttachFailed` exists with numeric_code and message template
- [ ] `stub_model_info` removed
- [ ] All e2e tests still pass; lifecycle test asserts real layer count
- [ ] New TCK scenarios written and visible in `lefthook run pre-commit`
- [ ] `cargo xtask ci` green
- [ ] Code-reviewer subagent run, all findings addressed
- [ ] Branch merges cleanly into master

## Out of scope

- The `BackendAttachFailed` error response includes the worker's error
  message in `data`. We do NOT introduce a separate machine-readable
  taxonomy for backend failure modes (out-of-memory vs missing module vs
  config error) — that's a future WU.
- Standalone-daemon mode (no orchestrator binary) is rejected with the
  same error. If anyone needs that mode for testing, they can fake
  the orchestrator over stdio.
