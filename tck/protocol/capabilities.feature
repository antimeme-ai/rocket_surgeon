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
        "protocol_version": "0.1.0"
      }
      """
    Then the response status is "initialized"
    And the response data contains a "capabilities" object

  Scenario: Capabilities includes protocol_version 0.1.0
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.1.0"
      }
      """
    Then the capabilities field "protocol_version" is "0.1.0"

  Scenario: Capabilities lists tick_granularities including layer and component
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.1.0"
      }
      """
    Then the capabilities field "tick_granularities" is an array
    And the capabilities field "tick_granularities" contains "layer"
    And the capabilities field "tick_granularities" contains "component"

  Scenario: Capabilities lists all 5 MVP intervention types
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.1.0"
      }
      """
    Then the capabilities field "intervention_types" is an array with 5 entries
    And the capabilities field "intervention_types" contains "ablate"
    And the capabilities field "intervention_types" contains "scale"
    And the capabilities field "intervention_types" contains "add"
    And the capabilities field "intervention_types" contains "patch"
    And the capabilities field "intervention_types" contains "clamp"

  # ── MVP boolean flags ─────────────────────────────────────────────

  Scenario: supports_checkpointing is false in MVP
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.1.0"
      }
      """
    Then the capabilities field "supports_checkpointing" is false

  Scenario: supports_moe is false in MVP
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.1.0"
      }
      """
    Then the capabilities field "supports_moe" is false

  Scenario: head_granularity is unavailable in MVP
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.1.0"
      }
      """
    Then the capabilities field "head_granularity" is "unavailable"

  # ── Capability-gated verb rejection ───────────────────────────────

  Scenario: Checkpoint verb when supports_checkpointing is false returns CAPABILITY_NOT_SUPPORTED
    Given the session is initialized with protocol_version "0.1.0"
    And a model "llama-7b" is attached
    And the session has been stepped to tick 0 at layer 0
    When the client sends "rocket/checkpoint" with:
      """json
      {
        "action": "create"
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "CAPABILITY_NOT_SUPPORTED"
    And the error "data.severity" is "recoverable"
    And the error "data.context.required_capability" is "supports_checkpointing"

  # ── Forward compatibility ─────────────────────────────────────────

  Scenario: Unknown capability fields in client request are ignored by server
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.1.0",
        "client_capabilities": {
          "supports_quantum_tunneling": true,
          "max_entanglement_depth": 42
        }
      }
      """
    Then the response status is "initialized"
    And the response data contains a "capabilities" object
    And the server does not return an error
