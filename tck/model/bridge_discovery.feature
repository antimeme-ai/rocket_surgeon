@bridge_discovery
Feature: Bridge discovery functions and tensor utilities
  The Python bridge exposes model introspection (module discovery,
  config extraction, execution order tracing) and tensor utilities
  (fp32-accurate stats, raw byte serialization, fused output splitting).

  Background:
    Given a tiny llama model is loaded on CPU

  # ── Module discovery ─────────────────────────────────────────────

  Scenario: discover_modules returns module inventory
    When discover_modules is called
    Then the result is a non-empty list of module dicts
    And each module dict has keys "path", "type_name", "attr_name"
    And at least one module path contains "self_attn.q_proj"

  Scenario: model_config returns architecture metadata
    When model_config is called
    Then the result contains "model_type" as "llama"
    And the result contains "num_layers" as a positive integer
    And the result contains "num_heads" as a positive integer
    And the result contains "hidden_size" as a positive integer

  Scenario: discover_execution_order returns ordered module firings
    When discover_execution_order is called
    Then the result is a non-empty list of (path, call_index) tuples
    And the first entry's call_index is 0
    And module paths appear in forward-pass order

  # ── Tensor stats ─────────────────────────────────────────────────

  Scenario: compute_tensor_stats casts to fp32 before reduction
    Given a tensor of dtype float16
    When compute_tensor_stats is called on the tensor
    Then the result dtype field is "float16"
    And the result mean is a finite float
    And the result std is a non-negative float
    And the result shape matches the tensor's shape

  Scenario: compute_tensor_stats reports sparsity
    Given a tensor where half the elements are zero
    When compute_tensor_stats is called on the tensor
    Then the result sparsity is approximately 0.5

  # ── Tensor serialization ─────────────────────────────────────────

  Scenario: tensor_to_bytes preserves dtype
    Given a float32 tensor with known values
    When tensor_to_bytes is called
    Then the byte length equals numel * 4

  # ── Fused output splitting ───────────────────────────────────────

  Scenario: split_fused_output splits along given dimension
    Given a tensor of shape [1, 6]
    When split_fused_output is called with dim=1 and sizes [2, 2, 2]
    Then the result is 3 tensors each of shape [1, 2]
