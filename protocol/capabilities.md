# Capability Negotiation

The `initialize` handshake returns a `Capabilities` object describing what the server supports.
Clients adapt to available capabilities — they never assume features exist without checking.

## Forward Compatibility

`additionalProperties: true` on the Capabilities schema. Clients **must ignore** unknown fields.
A Phase 2 client talking to a Phase 6 server will see `supports_moe: true` and MoE-specific
fields it doesn't recognize — it ignores them and continues operating at its own feature level.

A server **must always** populate all required fields. Fields it doesn't support are set to
`false`, empty arrays, or the most restrictive enum value.

## Capability Flags

### Boolean Feature Flags

| Flag | Phase | Default | Description |
|------|-------|---------|-------------|
| `supports_reverse_step` | 3+ | `false` | Backward stepping via checkpoint restore + forward replay. Requires `supports_checkpointing`. |
| `supports_checkpointing` | 3+ | `false` | `rocket/checkpoint` and `rocket/replay` verbs are functional. Pre-Phase 3: verbs exist in schema but return `CAPABILITY_NOT_SUPPORTED`. |
| `supports_moe` | 6+ | `false` | MoE tick granularities (`router_pre_topk`, `router_post_topk`, `expert`, `moe_layer`) and `route_override` intervention are available. |
| `supports_backward` | 8+ | `false` | Backward-pass ticks (gradient ablation, backward patching). |
| `supports_sae` | 8+ | `false` | SAE feature-level surgery and `sae_activation` built-in view. |

### Enum Flags

| Flag | Values | Phase progression |
|------|--------|-------------------|
| `execution_mode` | `eager`, `compiled`, `mixed` | Phase 0-6: `eager`. Phase 7+: `compiled` (Dynamo backend) or `mixed` (eager+compiled). |
| `parallelism` | `single_gpu`, `ddp`, `fsdp`, `tensor_parallel`, `pipeline_parallel` | Phase 0-4: `single_gpu`. Phase 5+: others. |
| `head_granularity` | `native`, `requires_unfused`, `unavailable` | Pre-Phase 7: `unavailable`. Phase 7+: `requires_unfused` (FlashAttention shadow replay) or `native` (eager attention). |

### Array Flags

| Flag | Type | Phase progression |
|------|------|-------------------|
| `tick_granularities` | `TickGranularity[]` | Phase 0-2: `["layer", "component"]`. Phase 6+: adds `["router_pre_topk", "router_post_topk", "expert", "moe_layer"]`. Phase 7+: adds `["head"]`. |
| `intervention_types` | `InterventionType[]` | Phase 2: `["ablate", "scale", "add", "patch", "clamp"]`. Phase 6+: adds `["route_override"]`. |
| `built_in_views` | `BuiltInView[]` | Grows across phases. Phase 1-2 (MVP): `["residual_stream_norm", "attention_pattern"]`. Phase 4+ (TUI dogfood): adds `["head_output", "logit_lens"]`. Phase 6+: adds `["routing_decision", "routing_entropy"]`. Phase 8+: adds `["feature_attribution", "sae_activation"]`. |
| `transports` | string[] | Phase 0-4: `["stdio", "unix_socket"]`. Phase 5+: adds `["tcp"]`. Phase 8+: adds `["websocket"]`. |
| `wire_formats` | string[] | Phase 0-3: `["json"]`. Phase 4+: adds `["protobuf"]`. |

### Scalar Flags

| Flag | Type | Description |
|------|------|-------------|
| `max_response_bytes` | integer | Maximum response payload. Default 65536 (64 KB). Clients requesting data beyond this get `RESPONSE_TOO_LARGE`. |

### Model Metadata (populated after `attach`)

These fields are absent or null before a model is attached. Clients must check for their presence.

| Field | Type | Description |
|-------|------|-------------|
| `model_family` | string | Architecture family: `"llama"`, `"mixtral"`, `"gpt-neox"`, etc. |
| `model_id` | string | Content hash of model files (`sha256:...`). |
| `num_layers` | integer | Transformer layer count. |
| `num_heads` | integer | Attention head count. |
| `hidden_dim` | integer | Hidden dimension size. |
| `num_ranks` | integer | GPU rank count. |
| `num_experts` | integer or null | Experts per MoE layer. Null for dense models. |
| `top_k_experts` | integer or null | Experts selected per token. Null for dense models. |

## Client Adaptation Patterns

### Degraded operation when capabilities are absent

```
if not capabilities.supports_checkpointing:
    # Hide checkpoint/replay UI elements
    # Disable reverse-step commands
    # Inform user: "Checkpointing available in Phase 3+"

if capabilities.head_granularity == "unavailable":
    # Don't offer head-level stepping
    # Offer component-level as finest granularity

if "route_override" not in capabilities.intervention_types:
    # Hide MoE routing override from intervention menu
```

### Capability-gated verb rejection

If a client sends a verb that requires an unsupported capability, the server returns:

```json
{
  "error": {
    "code": -32008,
    "message": "Checkpointing is not supported in this session",
    "data": {
      "error_code": "CAPABILITY_NOT_SUPPORTED",
      "numeric_code": -32008,
      "severity": "recoverable",
      "suggestion": "Upgrade to Phase 3+ for checkpoint support",
      "context": {
        "required_capability": "supports_checkpointing"
      }
    }
  }
}
```

The `CAPABILITY_NOT_SUPPORTED` error code (see error registry) is the canonical response.
Clients can detect this programmatically and adapt.

### Unknown capability handling

Servers may return fields not in the client's schema version. Clients **must not** reject
unknown fields. JSON Schema `additionalProperties: true` enforces this at the schema level.

An LLM client seeing an unknown capability should note it in its context and not attempt to
use features it doesn't understand. The protocol guarantees that unknown capabilities never
change the semantics of known verbs.

## MVP Capabilities (Phase 0-2)

A compliant MVP server returns:

```json
{
  "protocol_version": "0.1.0",
  "supports_reverse_step": false,
  "supports_checkpointing": false,
  "supports_moe": false,
  "supports_backward": false,
  "supports_sae": false,
  "execution_mode": "eager",
  "parallelism": "single_gpu",
  "tick_granularities": ["layer", "component"],
  "intervention_types": ["ablate", "scale", "add", "patch", "clamp"],
  "built_in_views": ["residual_stream_norm", "attention_pattern"],
  "head_granularity": "unavailable",
  "transports": ["stdio", "unix_socket"],
  "wire_formats": ["json"],
  "max_response_bytes": 65536,
  "model_family": "llama",
  "model_id": "sha256:abc123...",
  "num_layers": 32,
  "num_heads": 32,
  "hidden_dim": 4096,
  "num_ranks": 1,
  "num_experts": null,
  "top_k_experts": null
}
```

## Phase Progression Summary

| Phase | New capabilities |
|-------|-----------------|
| 0-2 (MVP) | Base: eager, single-GPU, 5 interventions, 2 views, component/layer granularity |
| 3 | `supports_checkpointing: true`, `supports_reverse_step: true` |
| 4 | `wire_formats` adds `"protobuf"`, TUI-driven view additions |
| 5 | `parallelism` beyond `single_gpu`, `transports` adds `"tcp"`, `num_ranks > 1` |
| 6 | `supports_moe: true`, MoE tick granularities, `route_override` intervention, `num_experts`/`top_k_experts` populated |
| 7 | `execution_mode` beyond `eager`, `head_granularity: "requires_unfused"` or `"native"`, `tick_granularities` adds `"head"` |
| 8+ | `supports_backward: true`, `supports_sae: true`, `transports` adds `"websocket"` |
