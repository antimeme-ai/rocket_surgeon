# WU 1.13: Built-in Views — Design Spec

## Goal

Add `rocket/view` as a distinct verb that computes pre-packaged interpretability analyses over the most recently captured tensor state. Two views for Phase 1: `residual_stream_norm` (L2 norms per layer) and `attention_pattern` (attention weight matrix for a specific layer). Views are pure functions over existing `last_outputs` data — no new hooks, no coupling to the step flow.

## Dependencies

- WU 1.11 (inspect integration): `last_outputs` dict, `collect_tensors`, component map — done
- WU 1.12 (probe events): probe point grammar, capture hooks — done
- WU 1.10 (step integration): barrier-driven stepping populates `last_outputs` — done

## TCK Contract

8 scenarios in `tck/protocol/view.feature`:

1. `residual_stream_norm` returns norms array with correct layer count
2. `attention_pattern` for a specific layer returns all heads
3. `attention_pattern` for a specific layer+head returns single head
4. View before any step returns `VIEW_DATA_UNAVAILABLE`
5. View without attached model returns `MODEL_NOT_ATTACHED`
6. View with invalid layer index returns `INVALID_PARAMS`
7. View with unknown view name returns `INVALID_PARAMS`
8. Available views reported in capabilities at initialize

---

## 1. Protocol

### New Verb: `rocket/view`

Distinct from `rocket/inspect`. Inspect is raw tensor access; views are pre-packaged analyses that organize captured data into something viewable.

**Method constants:**

```rust
pub mod method {
    pub const VIEW: &str = "rocket/view";
}

pub mod internal {
    pub const HOST_VIEW: &str = "_host/view";
}
```

### Request

```rust
pub struct ViewRequest {
    pub view: BuiltInView,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}
```

`params` is view-specific:

- `residual_stream_norm` — no params required (empty object or omitted)
- `attention_pattern` — `{ "layer": u32 }` required, `{ "layer": u32, "head": u32 }` optional

### Response

```rust
pub struct ViewResponse {
    pub view: BuiltInView,
    pub data: serde_json::Value,
}
```

Generic JSON value. Each view defines its own data shape. Wrapped in the standard `ResponseEnvelope` with session state.

### View Data Shapes

**`residual_stream_norm`:**

```json
{
  "norms": [0.42, 0.38, 0.51],
  "num_layers": 3,
  "norm_type": "l2"
}
```

`norms` is indexed by layer (0-based), one f32 per layer.

**`attention_pattern`:**

```json
{
  "layer": 3,
  "heads": [
    { "head": 0, "weights": [[0.1, 0.9], [0.5, 0.5]] },
    { "head": 1, "weights": [[0.3, 0.7], [0.6, 0.4]] }
  ],
  "seq_len": 2
}
```

If `head` was specified in params, `heads` contains one entry. Otherwise, all heads for the requested layer. `weights` is `[seq_len, seq_len]` — row `i` is the attention distribution over keys for query position `i`.

### Error Codes

New error code:

- `VIEW_DATA_UNAVAILABLE` (numeric: -32020) — required tensors not present in `last_outputs`. Could mean: no step executed yet, required probe points not active, or model architecture doesn't expose the needed tensors.

Reused existing codes:

- `MODEL_NOT_ATTACHED` — not in stopped state
- `CAPABILITY_NOT_SUPPORTED` — FlashAttention prevents attention weight materialization
- `INVALID_PARAMS` — bad layer/head index, unknown view name, malformed params

### Capabilities

Already advertised in `Capabilities::phase1_defaults()`:

```rust
built_in_views: vec![
    BuiltInView::ResidualStreamNorm,
    BuiltInView::AttentionPattern,
],
```

No changes needed — the views are already declared, they just need working computation.

---

## 2. Architecture

### Data Flow

```
Client                 Daemon              Orchestrator         Worker (Python)
  |                      |                      |                    |
  |-- rocket/view ------>|                      |                    |
  |                      |-- validate stopped   |                    |
  |                      |-- parse ViewRequest  |                    |
  |                      |-- _host/view ------->|                    |
  |                      |                      |-- forward -------->|
  |                      |                      |                    |-- read last_outputs
  |                      |                      |                    |-- compute view (PyTorch)
  |                      |                      |<-- HostViewResp ---|
  |                      |<-- HostViewResp -----|                    |
  |<-- envelope(ViewResponse)                   |                    |
```

### Daemon (`crates/rocket-surgeon/src/dispatch.rs`)

New `handle_view` function:

1. `parse_params::<ViewRequest>(request)`
2. `session.require_stopped("rocket/view")`
3. Forward to orchestrator as `_host/view` with the same params
4. Deserialize `HostViewResponse` from orchestrator
5. Wrap in `ViewResponse`, return via `session.envelope()`

Follows the exact same pattern as `handle_inspect` → `try_orchestrator_inspect`.

### Orchestrator (`crates/rocket-surgeon-orchestrator/src/dispatch.rs`)

Pure forward. Add `internal::HOST_VIEW` to the `forward_to_worker` match arm alongside `HOST_INSPECT`.

### Worker (`crates/rocket-surgeon-worker/src/dispatch.rs`)

New `handle_host_view` function:

1. Parse `HostViewRequest { view, params, model_handle }`
2. Validate model handle
3. Dispatch to view-specific Python computation based on `view` variant
4. Return `HostViewResponse { view, data }` or error

### Python Bridge (`python/rocket_surgeon/views.py`)

New module — keeps view logic separate from the core bridge.

Two functions, plus a model-config helper:

**`compute_residual_stream_norm(model_handle, last_outputs_dict, component_map)`**

The residual stream at layer `i` is the output of `model.layers.i` (the full transformer block, which includes the residual connection). The barrier hooks already capture module outputs keyed by `(module_path, call_index)` in `last_outputs`.

- Walk `component_map` for layer container modules (module paths matching `model.layers.N`)
- For each matched layer: look up `(module_path, 0)` in `last_outputs`
- If the output is a tuple (some models return `(hidden_state, ...)`), take element 0
- Compute `torch.norm(tensor.float(), p=2).item()`
- Sort by layer index
- Return `{ "norms": [...], "num_layers": N, "norm_type": "l2" }`
- If no layer outputs found in `last_outputs`: raise `ViewDataUnavailable`

Note: the component map currently tracks leaf modules (q_proj, k_proj, etc.) but not layer containers. The worker will need to resolve layer container paths from the model directly — `dict(model.named_modules())` filtered by pattern `model.layers.N` or equivalent for the model family.

**`compute_attention_pattern(model_handle, last_outputs_dict, component_map, layer, head=None)`**

Attention weights are NOT module outputs — they're internal to `self_attn.forward()`. To access them, we use HuggingFace's `output_attentions=True` config flag, which makes self_attn return `(attn_output, attn_weights, ...)` instead of just `attn_output`.

- Check model config: if `attn_implementation` is not `"eager"`, return `CAPABILITY_NOT_SUPPORTED`
- Set `model.config.output_attentions = True` (the flag is already respected by HF attention implementations; this must be set before the forward pass that populates `last_outputs`)
- Look up self_attn output for the requested layer in `last_outputs`: key `(model.layers.{layer}.self_attn, 0)`
- The output is now `(attn_output, attn_weights)` — extract `attn_weights` at index 1
- Shape: `[batch, num_heads, seq_len, seq_len]`
- If `head` specified: index `tensor[0, head]`, return single-head entry
- If `head` omitted: return all heads
- Convert to nested lists via `.tolist()`
- Return `{ "layer": N, "heads": [...], "seq_len": S }`

**Lifecycle note**: `output_attentions = True` must be set before the step that populates `last_outputs`. Two options: (a) set it at subscribe/view-enable time, or (b) set it at attach time always. Option (b) is simpler and the overhead is negligible for eager attention. We go with (b): set `output_attentions = True` during model load when `attn_implementation == "eager"`. This ensures attention weights are always available in `last_outputs` for eager models.

**`detect_attention_impl(model_handle) -> str`**

- Return `model.config.attn_implementation` or `"eager"` if not set (HF default for older models)

### Internal Wire Types

```rust
// Worker receives
pub struct HostViewRequest {
    pub model_handle: u64,
    pub view: BuiltInView,
    pub params: Option<serde_json::Value>,
}

// Worker returns
pub struct HostViewResponse {
    pub view: BuiltInView,
    pub data: serde_json::Value,
}
```

---

## 3. Preconditions and Errors

Views operate over `last_outputs` — the tensors captured during the most recent `_host/step` execution. Precondition failures:

| Condition | Error Code | Message |
|-----------|-----------|---------|
| No model attached (not stopped) | `MODEL_NOT_ATTACHED` | "Attach a model before calling this method" |
| No step executed yet (`last_outputs` is None) | `VIEW_DATA_UNAVAILABLE` | "No captured tensors — execute at least one step first" |
| `residual_stream_norm`: no layer container outputs in `last_outputs` | `VIEW_DATA_UNAVAILABLE` | "Layer outputs not captured — ensure barrier hooks cover layer container modules" |
| `attention_pattern`: non-eager attention implementation | `CAPABILITY_NOT_SUPPORTED` | "Attention weights not materialized — model uses fused attention (FlashAttention/SDPA). Set attn_implementation='eager' at attach." |
| `attention_pattern`: requested layer out of range | `INVALID_PARAMS` | "Layer N out of range (model has M layers)" |
| `attention_pattern`: requested head out of range | `INVALID_PARAMS` | "Head N out of range (layer has M heads)" |
| Unknown view type | `INVALID_PARAMS` | "Unknown view: <name>" |

FlashAttention detection: check `model.config.attn_implementation`. If it's `"flash_attention_2"`, `"sdpa"`, or any non-eager value, return `CAPABILITY_NOT_SUPPORTED`. For eager models, `output_attentions = True` is set at load time so attention weights are always in `last_outputs` as element 1 of the self_attn output tuple.

---

## 4. Testing Strategy

### Unit Tests (Rust)

**Daemon dispatch (`dispatch.rs`):**
- `handle_view` from stopped state with valid params → success
- `handle_view` when not stopped → `MODEL_NOT_ATTACHED`
- `handle_view` with invalid params → `INVALID_PARAMS`
- `handle_view` with unknown view name → `INVALID_PARAMS`

**Protocol (`serde_roundtrip.rs`):**
- `ViewRequest` roundtrip
- `ViewResponse` roundtrip with both view types
- `HostViewRequest` / `HostViewResponse` roundtrip

### E2E Test (`tests/test_e2e_view.py`)

Against tiny-random-LlamaForCausalLM (2 layers, 4 heads, hidden_dim=16):

1. Initialize + attach + step (populate `last_outputs`)
2. `rocket/view` with `residual_stream_norm`:
   - Verify `norms` is an array of length 2 (2 layers)
   - Verify each norm is a positive float
   - Verify `norm_type == "l2"`
3. `rocket/view` with `attention_pattern`, layer=0:
   - Verify response has `heads` array with 4 entries (4 heads)
   - Verify each head has `weights` as a 2D array
   - Verify `seq_len` is a positive integer
   - Verify attention weights approximately sum to 1.0 per row (softmax output)
4. `rocket/view` with `attention_pattern`, layer=0, head=2:
   - Verify single head returned
5. `rocket/view` before step → `VIEW_DATA_UNAVAILABLE`
6. `rocket/view` with invalid layer → `INVALID_PARAMS`
7. Wire format: response has `view` field matching request, `data` is object

### TCK Scenarios

8 Gherkin scenarios in `tck/protocol/view.feature` as listed in the TCK Contract section above.

---

## 5. What This Does NOT Include

- **New hook infrastructure**: views operate over existing `last_outputs` data. No dedicated view hooks.
- **Binary tensor encoding**: view data is JSON. Known to be inefficient for large attention matrices. Flagged for future optimization (WU 1.8 shared memory data plane may help).
- **Views beyond Phase 1**: `HeadOutput`, `LogitLens`, `RoutingDecision`, `RoutingEntropy`, `FeatureAttribution`, `SaeActivation` are defined in the `BuiltInView` enum but not computed. They return `INVALID_PARAMS` ("view not yet implemented") until their respective phases.
- **Caching**: no memoization of view results. Each `rocket/view` call recomputes from `last_outputs`. Acceptable for Phase 1 model sizes.
- **Multi-GPU aggregation**: views return data from rank 0 only. Multi-rank view aggregation is a future concern.
