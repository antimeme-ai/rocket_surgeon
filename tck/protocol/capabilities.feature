@capabilities
Feature: Capability negotiation at session initialization
  The initialize handshake returns a Capabilities object describing
  server features. Clients adapt to available capabilities and must
  tolerate unknown fields. Unsupported verbs return
  CAPABILITY_NOT_SUPPORTED.

  Background:
    Given a rocket_surgeon server is running

  # ── Initialize response shape ─────────────────────────────────────

  Scenario: Initialize response contains capabilities object
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.3.0"
      }
      """
    Then the response status is "initialized"
    And the response data contains a "capabilities" object

  Scenario: Capabilities includes protocol_version 0.3.0
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.3.0"
      }
      """
    Then the capabilities field "protocol_version" is "0.3.0"

  Scenario: Capabilities lists tick_granularities including layer and component
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.3.0"
      }
      """
    Then the capabilities field "tick_granularities" is an array
    And the capabilities field "tick_granularities" contains "layer"
    And the capabilities field "tick_granularities" contains "component"

  Scenario: Capabilities lists all 8 intervention types
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.3.0"
      }
      """
    Then the capabilities field "intervention_types" is an array with 8 entries
    And the capabilities field "intervention_types" contains "ablate"
    And the capabilities field "intervention_types" contains "scale"
    And the capabilities field "intervention_types" contains "add"
    And the capabilities field "intervention_types" contains "patch"
    And the capabilities field "intervention_types" contains "clamp"
    And the capabilities field "intervention_types" contains "attention_mask"
    And the capabilities field "intervention_types" contains "embed_swap"
    And the capabilities field "intervention_types" contains "embed_noise"

  Scenario: Capabilities lists v0.3.0 built-in views including lens and KV views
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.3.0"
      }
      """
    Then the capabilities field "built_in_views" is an array
    And the capabilities field "built_in_views" contains "tuned_lens"
    And the capabilities field "built_in_views" contains "kv_cache_ribbon"
    And the capabilities field "built_in_views" contains "worldline_dag"

  Scenario: Capabilities lists websocket transport in v0.3.0
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.3.0"
      }
      """
    Then the capabilities field "transports" contains "websocket"

  # ── MVP boolean flags ─────────────────────────────────────────────

  Scenario: supports_checkpointing is false in MVP
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.3.0"
      }
      """
    Then the capabilities field "supports_checkpointing" is false

  Scenario: supports_moe is false in MVP
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.3.0"
      }
      """
    Then the capabilities field "supports_moe" is false

  Scenario: head_granularity is unavailable in MVP
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.3.0"
      }
      """
    Then the capabilities field "head_granularity" is "unavailable"

  # ── Capability-gated verb rejection ───────────────────────────────

  Scenario: Checkpoint create succeeds in stopped state
    Given the session is initialized with protocol_version "0.3.0"
    And a model "llama-7b" is attached
    When the client sends "rocket/checkpoint" with:
      """json
      {
        "action": "create"
      }
      """
    Then the response status is "stopped"
    And the response "data.checkpoint_id" is a non-empty string

  # ── Forward compatibility ─────────────────────────────────────────

  Scenario: Unknown capability fields in client request are ignored by server
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.3.0",
        "client_capabilities": {
          "supports_quantum_tunneling": true,
          "max_entanglement_depth": 42
        }
      }
      """
    Then the response status is "initialized"
    And the response data contains a "capabilities" object
    And the server does not return an error
