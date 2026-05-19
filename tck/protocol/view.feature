@view
Feature: Built-in interpretability views
  The rocket/view verb computes pre-packaged analyses over the most
  recently captured tensor state. Distinct from rocket/inspect — views
  organize data into something viewable, inspect gives raw tensor access.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.1.0"
    And a model "llama-7b" is attached
    And the session has been stepped to tick 0 at layer 0

  # ── residual_stream_norm ──────────────────────────────────────────

  Scenario: Residual stream norm returns per-layer L2 norms
    When the client sends "rocket/view" with:
      """json
      {"view": "residual_stream_norm"}
      """
    Then the response status is "stopped"
    And the response data field "data" contains "norms" as an array
    And the "norms" array length equals the model's num_layers
    And each element of "norms" is a positive float
    And the response data field "data" contains "norm_type" equal to "l2"

  # ── attention_pattern (all heads) ─────────────────────────────────

  Scenario: Attention pattern for a layer returns all heads
    When the client sends "rocket/view" with:
      """json
      {"view": "attention_pattern", "params": {"layer": 0}}
      """
    Then the response status is "stopped"
    And the response data field "data" contains "layer" equal to 0
    And the response data field "data" contains "heads" as an array
    And the "heads" array length equals the model's num_heads
    And each head entry contains "head" as a non-negative integer
    And each head entry contains "weights" as a 2D array
    And the response data field "data" contains "seq_len" as a positive integer

  # ── attention_pattern (single head) ───────────────────────────────

  Scenario: Attention pattern for a specific head returns single entry
    When the client sends "rocket/view" with:
      """json
      {"view": "attention_pattern", "params": {"layer": 0, "head": 0}}
      """
    Then the response status is "stopped"
    And the "heads" array length equals 1
    And the first head entry has "head" equal to 0

  # ── View before step ──────────────────────────────────────────────
  # Background steps the model, so detach+reattach to get pre-step state.

  Scenario: View before any step returns view-data-unavailable error
    When the client sends "rocket/detach" with:
      """json
      {}
      """
    And the client sends "attach" with model "llama-7b"
    And the client sends "rocket/view" with:
      """json
      {"view": "residual_stream_norm"}
      """
    Then the response contains an error with code "VIEW_DATA_UNAVAILABLE"

  # ── View without model ────────────────────────────────────────────

  Scenario: View without attached model returns model-not-attached error
    When the client sends "rocket/detach" with:
      """json
      {}
      """
    And the client sends "rocket/view" with:
      """json
      {"view": "residual_stream_norm"}
      """
    Then the response contains an error with code "MODEL_NOT_ATTACHED"

  # ── Invalid layer ─────────────────────────────────────────────────

  Scenario: Attention pattern with out-of-range layer returns invalid params
    When the client sends "rocket/view" with:
      """json
      {"view": "attention_pattern", "params": {"layer": 9999}}
      """
    Then the response contains an error with code "INVALID_PARAMS"

  # ── Unknown view ──────────────────────────────────────────────────

  Scenario: Unknown view name returns invalid params
    When the client sends "rocket/view" with:
      """json
      {"view": "nonexistent_view"}
      """
    Then the response contains an error with code "INVALID_PARAMS"

  # ── Capabilities ──────────────────────────────────────────────────

  Scenario: Available views are reported in capabilities at initialize
    When the client sends "initialize" with:
      """json
      {"client_name": "test", "protocol_version": "0.1.0"}
      """
    Then the response data field "capabilities" contains "built_in_views" as an array
    And the "built_in_views" array contains "residual_stream_norm"
    And the "built_in_views" array contains "attention_pattern"
