# WU 1.13: Built-in Views — Handoff Document

**Branch:** `wu-1.13-built-in-views`
**Base:** `master` (at `6683d4f`)
**Date:** 2026-05-19
**Status:** Blocked — E2E test reveals fundamental architecture gap

---

## What Was Built

The `rocket/view` verb: a new JSON-RPC method that computes pre-packaged interpretability views (`residual_stream_norm`, `attention_pattern`) over the most recently captured tensor state. The full pipeline — protocol types, daemon dispatch, orchestrator forwarding, worker dispatch, Python view computations — is implemented and passing all pre-commit checks (clippy, fmt, ruff, mypy, cargo test).

### Commits (7, oldest first)

| SHA | Description |
|-----|-------------|
| `c3ca56a` | Design spec + implementation plan |
| `a74a042` | Protocol types: `ViewRequest`, `ViewResponse`, `HostViewRequest`, `HostViewResponse`, `ViewDataUnavailable` error code, method constants |
| `01ccaa1` | TCK: 8 Gherkin scenarios for built-in views (RED) |
| `a9073e6` | Daemon dispatch: `handle_view` handler + 4 unit tests |
| `1a70fdf` | Daemon wiring: `try_orchestrator_view` in main loop, `view_raw` on OrchestratorHandle, orchestrator forwarding |
| `361efe6` | Worker dispatch: `handle_host_view` — validates model handle, checks last_outputs, calls Python via PyO3, maps error prefixes to protocol error codes |
| `8f1483f` | Python views module: `compute_view` dispatcher, `_residual_stream_norm`, `_attention_pattern`, plus `output_attentions=True` in bridge.py for eager models |

### Files Changed (13 files, +2264/-12)

| File | What |
|------|------|
| `crates/rocket-surgeon-protocol/src/errors.rs` | `ViewDataUnavailable` variant, numeric code -32020 |
| `crates/rocket-surgeon-protocol/src/messages.rs` | `ViewRequest`, `ViewResponse`, `HostViewRequest`, `HostViewResponse`, `method::VIEW`, `internal::HOST_VIEW` |
| `crates/rocket-surgeon-protocol/tests/serde_roundtrip.rs` | 7 roundtrip tests for new types |
| `tck/protocol/view.feature` | 8 behavioral scenarios |
| `crates/rocket-surgeon/src/dispatch.rs` | `handle_view` + 4 unit tests |
| `crates/rocket-surgeon/src/main.rs` | `try_orchestrator_view`, wired into main loop |
| `crates/rocket-surgeon/src/orchestrator_handle.rs` | `view_raw` method |
| `crates/rocket-surgeon-orchestrator/src/dispatch.rs` | `HOST_VIEW` added to `forward_to_worker` match |
| `crates/rocket-surgeon-worker/src/dispatch.rs` | `handle_host_view`, `compute_view` Python bridge call |
| `python/rocket_surgeon/bridge.py` | `output_attentions=True` for eager attention models at load time |
| `python/rocket_surgeon/views.py` | New file: `compute_view`, `_residual_stream_norm`, `_attention_pattern` |
| `docs/specs/2026-05-19-built-in-views-design.md` | Design spec |
| `docs/superpowers/plans/2026-05-19-wu-1.13-built-in-views.md` | Implementation plan |

### Uncommitted Work

- `tests/test_e2e_view.py` — Draft E2E test (untracked, not committed). This is the file that exposed the blocking issues. It's a 7-step lifecycle test following the existing E2E test patterns. Currently fails at Step 5 (residual_stream_norm) due to the container hook gap described below.

---

## What Works

Everything above the Python views layer works correctly:

1. **Protocol types** — `ViewRequest`, `ViewResponse`, `HostViewRequest`, `HostViewResponse` all serialize/deserialize correctly. 7 serde roundtrip tests pass.

2. **Daemon dispatch** — `handle_view` validates session state (requires stopped), parses params, constructs response envelope. 4 unit tests pass.

3. **Main loop wiring** — `try_orchestrator_view` follows the exact pattern of `try_orchestrator_inspect`. Orchestrator forwarding works (single match arm addition).

4. **Worker dispatch** — `handle_host_view` validates model handle, checks `last_outputs` exists, calls Python `compute_view` via PyO3, maps error message prefixes (`VIEW_DATA_UNAVAILABLE`, `CAPABILITY_NOT_SUPPORTED`, `INVALID_PARAMS`) to protocol error codes.

5. **Python views module** — Clean dispatcher + two view functions. Lints clean (ruff, mypy). Logic is correct for computing L2 norms and extracting attention weight matrices.

6. **E2E test Steps 1-4** — Initialize (capabilities include views), attach, view-before-step returns `VIEW_DATA_UNAVAILABLE`, step forward — all pass.

7. **All pre-commit checks pass** — clippy, fmt, ruff, ruff format, mypy, cargo test (145 daemon tests, 51 protocol tests, 44 worker tests, 115 python tests).

---

## What Doesn't Work — The Blocking Issues

### Issue 1: Container Module Outputs Not in `last_outputs`

This is the showstopper. Both views assume `last_outputs` contains outputs keyed by container module paths:

- `_residual_stream_norm` looks for `("model.layers.0", 0)`, `("model.layers.1", 0)`, etc.
- `_attention_pattern` looks for `("model.layers.N.self_attn", 0)`

**These keys never exist.** Here's why:

The adapter (`crates/rocket-surgeon-worker/src/adapter.rs`) classifies modules during attach:
- `LlamaDecoderLayer` → `Container` (skipped)
- `LlamaAttention`/`LlamaSdpaAttention` → `Container` (skipped)
- `LlamaMLP` → `Container` (skipped)
- Leaf modules (`q_proj`, `o_proj`, `down_proj`, etc.) → `Direct` (included)

Only `Direct`-mapped modules end up in `state.module_paths`. Only `module_paths` modules get capture hooks. Only hooked modules produce mailbox messages. Only mailbox messages get stashed in `last_outputs`.

Container modules are intentionally excluded because they would break the tick protocol — every container hook firing would count as an extra tick, inflating `tick_id` and corrupting step granularity. The step loop has no concept of a "passive" hook that captures data without advancing the tick counter.

**The residual stream tensor** (the thing `residual_stream_norm` needs) is computed INSIDE the container's `forward()` method via residual addition. No leaf module captures it. `down_proj` outputs the MLP contribution before addition; `o_proj` outputs the attention contribution before addition. The sum doesn't appear in any leaf.

**The attention weight matrix** (the thing `attention_pattern` needs) is internal to the self_attn container's forward computation. When `output_attentions=True`, it appears as element 1 of the container's output tuple. No leaf module sees it.

**Root cause:** The design spec (written during brainstorming) assumed `last_outputs` would have container module outputs. This assumption was never validated against the actual hook infrastructure. The hook infrastructure was built for the debugger's tick-stepping use case, not for passive data collection.

### Issue 2: HuggingFace `attn_implementation` Detection

The public `config.attn_implementation` attribute is NOT SET on HuggingFace models loaded with default settings. The actual runtime attention backend is stored in the private `config._attn_implementation` attribute (e.g., `"sdpa"` for modern models, `"eager"` only when explicitly requested).

**Fixed in commit `8f1483f`:** Both `bridge.py` and `views.py` use `_attn_implementation` with a fallback to `"eager"` for old models that lack the attribute. This is correct but relies on a private HF API.

**Implication:** The tiny test model (`hf-internal-testing/tiny-random-LlamaForCausalLM`) defaults to sdpa. With the fix, `load_model` correctly does NOT set `output_attentions=True` for sdpa models. The attention_pattern view correctly returns `CAPABILITY_NOT_SUPPORTED` for sdpa models. Testing the attention_pattern happy path requires loading a model with `attn_implementation="eager"`, which the current attach protocol doesn't support (no field for it).

### Issue 3: Stub Model Info in Attach Response (Pre-existing)

The daemon's attach response uses `stub_model_info("llama")` → `(32, 32, 4096)` regardless of actual model. The real values from the worker (`num_layers=2, num_heads=4, hidden_dim=16` for the tiny model) are logged but never propagated to the client response. This is because the daemon builds the client response BEFORE spawning the orchestrator.

This is pre-existing and not introduced by WU 1.13, but it means:
- The E2E test's `num_layers` assertion would check against 32 (stub), not 2 (real)
- The `_attention_pattern` view validates `layer < num_layers` using the REAL config value from the loaded model, not the stub — so they would be consistent within the Python side, but inconsistent with what the client sees

---

## Code Review Findings

Full top-to-bottom review was performed. Findings by severity:

### Important

1. **Worker INVALID_PARAMS error response missing `ErrorData`** — `dispatch.rs:753-761`: The `INVALID_PARAMS` branch constructs `RpcError { data: None }` while `VIEW_DATA_UNAVAILABLE` and `CAPABILITY_NOT_SUPPORTED` use `RpcError::from_error_data(ErrorData::new(...))`. Inconsistent error envelope.

2. **`try_orchestrator_view` silently swallows deserialization errors** — `main.rs:254-261`: If params fail to parse, returns `Ok(None)`, causing `handle_view(session, request, None)` to produce `VIEW_DATA_UNAVAILABLE` instead of `INVALID_PARAMS`. Matches the existing `try_orchestrator_inspect` pattern (same masked bug), but the TCK scenario for "unknown view" still passes because `handle_view` catches it at `parse_params`. Latent logical inconsistency.

3. **`head` parameter not cast to `int`** — `views.py:34` casts `layer` to `int()` but `head` (line 102, 130) is used as-is. Asymmetric treatment. Low real-world risk since JSON integers → Python int, but inconsistent.

4. **Hardcoded `model.layers.N` path pattern** — `views.py:47-54`: Both views hardcode HuggingFace Llama-style module paths. Won't work for other model families. Should document as limitation or derive from component map.

5. **TCK "capabilities" scenario doesn't test capabilities** — `view.feature:100-107`: Titled "Available views are reported in capabilities at initialize" but just sends a view request and checks the response. Never inspects the initialize response's `capabilities.built_in_views`.

### Minor

6. **Silent layer-skipping** — `views.py:60-61`: If a layer's key is missing from `last_outputs`, the function silently skips it. The returned `norms` array could have gaps (layer 0, layer 2, missing layer 1) with no way for the consumer to detect which layers are represented.

7. **JSON round-trip for view results** — `dispatch.rs:796-803`: `compute_view` result goes Python dict → `json.dumps` → string → `serde_json::from_str` → `Value`. Works but wasteful; `pythonize` crate would be more direct.

8. **Misleading test name** — `dispatch.rs:1155`: `handle_view_from_stopped_returns_view_response` tests the error path (no orchestrator → error), not the success path.

9. **No worker-level unit tests for `handle_host_view`** — Other handlers (`host_attach`, `host_detach`, `host_step`, `host_inspect`) have worker dispatch unit tests. `host_view` doesn't.

10. **`view_data_unavailable_error_code` test placement** — In `serde_roundtrip.rs` instead of `errors.rs` unit tests where similar tests live.

### Nitpick

11. Magic number extractions (`layer_path_depth = 3`, `min_attn_tuple_len = 2`) are over-engineered; the surrounding code makes the intent obvious.

---

## Proposed Path Forward

### The Core Problem to Solve

The hook infrastructure needs a mechanism to capture container module outputs without participating in the tick protocol. Two approaches:

**Approach A: Passive Hooks (Recommended)**

Add a separate set of plain forward hooks on container modules during `ensure_forward_pass`. These hooks stash `(path, 0) → output` directly into `last_outputs` — no mailbox, no barrier, no tick counting. They fire naturally during the forward pass (after all child hooks return for that container).

Thread safety: The GIL protects dict operations. Passive hooks fire on the forward-pass thread. The step loop (on the main thread) only writes to `last_outputs` via `stash_tensor_output` during mailbox processing, and the mailbox barrier serializes access — the forward-pass thread blocks on `resume_mailbox.wait()` while the step loop processes each tick. Container hooks fire AFTER child hooks return, so by the time a container hook fires, its children's mailbox messages have already been processed.

Changes needed:
- `bridge.py`: New `install_passive_hooks(handle, paths, storage_dict)` function
- Worker `ensure_forward_pass`: Determine container paths (from the adapter's Container classifications), install passive hooks alongside capture hooks
- Worker cleanup: Remove passive hooks when forward pass tears down
- Views module: No changes — already looks for the right keys

**Approach B: Observer Flag on MappedComponent**

Add a `tick_bearing: bool` flag to `MappedComponent`. Container modules get `tick_bearing: false`. The step loop checks this flag and skips `tick_state.advance()` for non-tick modules while still stashing their output.

More invasive (touches adapter, component map, step loop) but more principled.

### Recommended Execution Order

1. **Fix CR findings** from the committed code (Items I-1, I-3, I-5, M-1/M-4, M-3)
2. **Implement passive hooks** (Approach A) — new bridge function + worker changes
3. **Update views module** if passive hook key format differs
4. **Finalize E2E test** — residual_stream_norm will work once passive hooks capture container outputs; attention_pattern tests the CAPABILITY_NOT_SUPPORTED path for sdpa models
5. **Commit and push**

### Separately (Not WU 1.13)

- **Attach response stub values** — The daemon needs to propagate real model metadata from the worker instead of using `stub_model_info()`. This is a pre-existing issue that affects all E2E tests, not just views.
- **`attn_implementation` parameter on attach** — To test attention_pattern happy path, the attach request needs to support passing HF config overrides (like `attn_implementation="eager"`). The protocol already has `config: Option<Value>` on `HostAttachRequest` but it's not wired to `from_pretrained` kwargs.

---

## Key Architectural Insight

The design spec assumed `last_outputs` was a passive capture of all module outputs. In reality, `last_outputs` is populated by the *capture hook / mailbox / tick* machinery, which is designed for the debugger's step-through use case. Only tick-bearing modules (leaf components) participate. Container modules are intentionally excluded to keep the tick protocol clean.

Views need a different kind of data access — passive observation of the forward pass without affecting the debugger's stepping behavior. This is a new capability the system doesn't have yet. The passive hooks approach adds it with minimal disruption to the existing architecture.

This is a good example of why the JSMNTL methodology matters: the E2E test (TCK red) caught the gap before any user ever saw it. The Rust pipeline, Python views, and protocol types are all correct — the issue is in the assumptions about the data layer, which only become visible under real execution.

---

## Quick Reference

**Branch:** `wu-1.13-built-in-views`
**All pre-commit checks:** PASS
**All existing tests:** PASS (145 daemon + 51 protocol + 44 worker + 115 python)
**E2E test:** Steps 1-4 PASS, Step 5 BLOCKED (container hook gap)
**Uncommitted:** `tests/test_e2e_view.py` (draft, not added to git)
