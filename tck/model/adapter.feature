@adapter
Feature: Model adapter contract — registration, vocabulary, name mapping
  A model adapter bridges between a concrete model architecture (e.g.,
  LlamaForCausalLM) and the canonical component vocabulary used by the
  protocol. Every adapter must register the model families it supports,
  expose a canonical component vocabulary matching the protocol schema,
  map model-specific parameter names to canonical names, and report model
  metadata. Unsupported model families are rejected with UNSUPPORTED_MODEL.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.1.0"

  # ── Registration ──────────────────────────────────────────────────

  Scenario: Adapter registers supported model family
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response status is "stopped"
    And the response "data.model_family" is "llama"
    And the response "data.adapter" is a non-empty string

  Scenario: Adapter provides component vocabulary
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response status is "stopped"
    And the response "data.component_vocabulary" is a non-empty array
    And each entry in "data.component_vocabulary" is a non-empty string

  Scenario: Component vocabulary matches canonical names from components.json
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response "data.component_vocabulary" contains at least:
      | component        |
      | attn.q_proj      |
      | attn.k_proj      |
      | attn.v_proj      |
      | attn.o_proj      |
      | mlp.gate_proj    |
      | mlp.up_proj      |
      | mlp.down_proj    |
      | ln1              |
      | ln2              |
      | residual_pre     |
      | residual_post    |
    And every entry in "data.component_vocabulary" is a valid canonical component name

  # ── Name mapping ──────────────────────────────────────────────────

  Scenario: Adapter maps model-specific layer names to canonical names
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response status is "stopped"
    When the client sends "rocket/inspect" with:
      | target | llama:0:12:attn.q_proj:output |
      | detail | summary                       |
    Then the response status is "stopped"
    And the response "data.tensors" is an array with at least 1 element
    And the response "data.resolved_target" is "llama:0:12:attn.q_proj:output"
    And the response "data.native_name" is "model.layers.12.self_attn.q_proj"

  Scenario: Adapter maps multiple components within the same layer
    Given a model "llama-7b" is attached
    When the client sends "rocket/inspect" with:
      | target | llama:0:12:mlp.gate_proj:output |
      | detail | summary                         |
    Then the response status is "stopped"
    And the response "data.native_name" is "model.layers.12.mlp.gate_proj"
    When the client sends "rocket/inspect" with:
      | target | llama:0:12:ln1:output |
      | detail | summary               |
    Then the response status is "stopped"
    And the response "data.native_name" is "model.layers.12.input_layernorm"

  # ── Model metadata ───────────────────────────────────────────────

  Scenario: Adapter reports model metadata on attach
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response status is "stopped"
    And the response "data.num_layers" is a positive integer
    And the response "data.num_heads" is a positive integer
    And the response "data.hidden_dim" is a positive integer
    And the response "data.num_layers" is 32
    And the response "data.num_heads" is 32
    And the response "data.hidden_dim" is 4096

  # ── Error paths ──────────────────────────────────────────────────

  Scenario: Unsupported model family returns UNSUPPORTED_MODEL error
    When the client sends "attach" with:
      | model_path   | /models/mystery-model |
      | model_family | unknown_arch          |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "UNSUPPORTED_MODEL"
    And the error "data.severity" is "recoverable"
    And the error "data.suggestion" is a non-empty string

  # ── MoE adapter ──────────────────────────────────────────────────

  @phase6
  Scenario: Adapter handles MoE models with expert metadata
    When the client sends "attach" with:
      | model_path   | /models/mixtral-8x7b |
      | model_family | mixtral              |
    Then the response status is "stopped"
    And the response "data.model_family" is "mixtral"
    And the response "data.num_layers" is a positive integer
    And the response "data.num_heads" is a positive integer
    And the response "data.hidden_dim" is a positive integer
    And the response "data.num_experts" is a positive integer
    And the response "data.num_experts" is 8
    And the response "data.top_k_experts" is a positive integer
    And the response "data.top_k_experts" is 2

  @phase6
  Scenario: MoE adapter component vocabulary includes router and expert components
    When the client sends "attach" with:
      | model_path   | /models/mixtral-8x7b |
      | model_family | mixtral              |
    Then the response "data.component_vocabulary" contains at least:
      | component         |
      | router            |
      | router.logits     |
      | router.decision   |
      | experts[j]        |
      | attn.q_proj       |
      | attn.k_proj       |
      | attn.v_proj       |
      | attn.o_proj       |

  @phase6
  Scenario: Dense model attach does not populate num_experts
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response status is "stopped"
    And the response "data.num_experts" is null
    And the response "data.top_k_experts" is null
