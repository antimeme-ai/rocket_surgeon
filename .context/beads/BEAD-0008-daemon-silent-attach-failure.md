---
id: BEAD-0008
title: Daemon returns synthesized success on failed backend attach
status: closed
priority: high
created: 2026-05-19
resolved: 2026-05-19
---

## Description

The daemon's `attach` handler returns a successful response with synthesized
model metadata (`num_layers`, `num_heads`, `hidden_dim`, etc.) **regardless
of whether the backend orchestrator and worker actually succeeded in loading
the model**. The orchestrator is only spawned *after* the response is
constructed, and its failure is logged as a warning but never propagated
back to the client.

This was discovered while running the e2e suite under the new uv-based
bootstrap (PR #4). The worker subprocess could not import torch (different
root cause, fixed separately); the orchestrator attach failed; the daemon
nevertheless returned `attach` success to the client with believable but
wholly fabricated model metadata.

## Repro

1. Configure environment so the worker subprocess cannot import torch
   (e.g. clear the venv's site-packages from PYTHONPATH).
2. Send `initialize` → `attach` to the daemon over stdio.
3. Observe `attach` returns 200 with `num_layers=32`, `num_heads=32`,
   `hidden_dim=4096` — values not derived from any real model.
4. Send any subsequent `inspect` / `view` request.
5. Daemon returns `TENSOR_NOT_FOUND` / `VIEW_DATA_UNAVAILABLE` with
   message `"No orchestrator available …"`.

## Impact

- E2E tests that don't exercise the worker beyond attach (lifecycle,
  subscribe, probes) appear green even when the entire backend is broken.
- Real failures surface only at the first inspect/view/step that needs
  worker computation, making root-causing harder.
- Client-side code (TUI, LLM driver) cannot distinguish "model loaded" from
  "backend silently dead" without making a follow-up backend-dependent call.

## Suggested fix

Restructure `crates/rocket-surgeon/src/main.rs` so the orchestrator is
spawned and `_host/attach` round-trips **before** the client-facing
attach response is built. If the backend fails, return an `attach` error
(reuse `BACKEND_ATTACH_FAILED` or add a new error code) with the
orchestrator's error message in `data`.

Today's flow (broken):

    handle_attach(req)          # synthesizes success response
    write_message(response)     # client sees "success"
    spawn_and_attach(req)       # fires async, failure only logged

Target flow:

    let backend = spawn_and_attach(req)?;   # block on real attach
    let response = handle_attach(req, &backend);  # consult backend metadata
    write_message(response);

This also lets `handle_attach` return *real* model metadata (`num_layers`
etc.) instead of hardcoded placeholders.

## Acceptance

- [x] `attach` returns an error response when backend orchestrator fails
      → dispatch.rs: handle_attach(Err) returns BACKEND_ATTACH_FAILED
- [x] `attach` success response contains real model metadata from the worker,
      not synthesized placeholders
      → main.rs: spawn_and_attach() blocks, real HostAttachResponse flows through
- [x] New TCK scenario in `tck/lifecycle.feature` covering backend failures
      → lifecycle.feature:129-181: five BEAD-0008 scenarios (real metadata,
        broken backend, model_family override, zero metadata, duplicate precheck)
- [x] Existing passing e2e tests continue to pass
      → cargo xtask ci green

## Related

- [[wu-1.13-built-in-views]] — views were the first feature to surface this
  bug, since they're the first verbs that *can't* succeed via synthesis.
- PR #4 — discovered while running e2e tests under the new bootstrap.
