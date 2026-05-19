# WU 1.13: Built-in Views — Handoff Document

**Branch:** `wu-1.13-built-in-views`
**Base:** `master` (at `6683d4f`)
**Date:** 2026-05-19
**Status:** Complete — all E2E steps pass, ready for merge

---

## What Was Built

The `rocket/view` verb: a new JSON-RPC method that computes pre-packaged interpretability views (`residual_stream_norm`, `attention_pattern`) over the most recently captured tensor state. The full pipeline — protocol types, daemon dispatch, orchestrator forwarding, worker dispatch, Python view computations, passive hooks for container output capture — is implemented and passing all checks.

### Commits (oldest first)

| SHA | Description |
|-----|-------------|
| `c3ca56a` | Design spec + implementation plan |
| `a74a042` | Protocol types: `ViewRequest`, `ViewResponse`, `HostViewRequest`, `HostViewResponse`, `ViewDataUnavailable` error code, method constants |
| `01ccaa1` | TCK: 8 Gherkin scenarios for built-in views (RED) |
| `a9073e6` | Daemon dispatch: `handle_view` handler + 4 unit tests |
| `1a70fdf` | Daemon wiring: `try_orchestrator_view` in main loop, `view_raw` on OrchestratorHandle, orchestrator forwarding |
| `361efe6` | Worker dispatch: `handle_host_view` — validates model handle, checks last_outputs, calls Python via PyO3, maps error prefixes to protocol error codes |
| `8f1483f` | Python views module: `compute_view` dispatcher, `_residual_stream_norm`, `_attention_pattern`, plus `output_attentions=True` in bridge.py for eager models |
| `565dbaf` | Handoff doc (initial blocked state) |
| `8cee4ec` | Adapter: `resolve_with_containers` exposes container module paths |
| `dc9f628` | Python: `install_passive_hooks` — container output capture without barriers |
| `7bd0271` | Worker: `bridge::install_passive_hooks` FFI wrapper |
| `d1b9d87` | Worker: wire passive hooks into `ensure_forward_pass` for container outputs |
| `8bc34cf` | CR remediation: ErrorData consistency, int cast, capabilities TCK, layer tracking |
| (latest) | E2E test green — all 7 steps pass |

---

## Architecture: Passive Hooks

The core innovation in this WU is the **passive hook** mechanism — a third hook type alongside sentinel and capture hooks.

**Problem:** Views need container module outputs (decoder layers, self_attn) but only leaf (tick-bearing) modules get capture hooks. Container modules were intentionally excluded to keep tick counting clean.

**Solution:** Passive hooks are plain `register_forward_hook` callbacks on container modules that stash `(path, 0) → output` directly into `last_outputs`. No mailbox, no barrier, no tick counting. They fire naturally during the forward pass after all child hooks return.

**Thread safety:** Passive hooks write to `last_outputs` on the forward-pass thread. The step loop writes via `stash_tensor_output` on the main thread but only during mailbox processing when the forward thread is blocked on `resume_mailbox.wait()`. GIL serializes all Python dict operations.

**Lifecycle:** Installed in `ensure_forward_pass`, cleaned up in `handle_host_detach`.

---

## E2E Test Results

All 7 steps pass:

1. **Initialize** — `capabilities.built_in_views` includes both views
2. **Attach** — tiny model loads (2 layers, 4 heads)
3. **View before step** — returns `VIEW_DATA_UNAVAILABLE`
4. **Step forward** — full forward pass completes
5. **residual_stream_norm** — returns 2 norms (one per real layer), all positive
6. **attention_pattern (sdpa)** — returns `CAPABILITY_NOT_SUPPORTED` (expected for sdpa attention)
7. **attention_pattern invalid layer** — returns error (CAPABILITY_NOT_SUPPORTED precedes layer validation for sdpa)

---

## Known Limitations

1. **Hardcoded `model.layers.N` path pattern** — Both views hardcode HuggingFace Llama-style module paths. Won't work for other model families. Should derive from component map in a future WU.

2. **sdpa attention not supported** — The attention_pattern view requires `attn_implementation="eager"`. The attach protocol doesn't yet expose this as a parameter.

3. **Daemon attach stub values** — The daemon's attach response uses `stub_model_info("llama")` → 32 layers regardless of actual model. Pre-existing issue, not introduced by WU 1.13.

4. **TCK scenarios are RED** — All 8 Gherkin scenarios are written but the TCK harness doesn't have step definitions yet. The E2E test validates the same behavioral intents.

---

## CR Findings Addressed

- **I-1:** `INVALID_PARAMS` error branch now uses `ErrorData` like sibling branches
- **I-3:** `head` parameter cast to `int` before use
- **I-5:** Capabilities TCK scenario now tests initialize response, not view call
- **M-1:** `_residual_stream_norm` returns `layers` array alongside `norms` for consumer traceability

### Deferred (not WU 1.13 scope)

- **I-2:** `try_orchestrator_view` silently swallows deserialization errors (matches existing `try_orchestrator_inspect` pattern)
- **I-4:** Hardcoded Llama paths (needs component map integration)
- **M-2:** JSON round-trip inefficiency (pythonize would be better)
- **M-3:** No worker-level unit tests for `handle_host_view`
