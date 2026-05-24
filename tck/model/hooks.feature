@hooks
Feature: Hook registration and firing on model components
  Hooks are the low-level mechanism that captures tensor data at specific
  positions in the forward pass. A hook is registered against a target
  pattern (using the probe-point grammar), fires when execution reaches
  a matching component, and produces a tensor capture with a
  content-addressable tensor_id (BLAKE3 hash). Hooks respect registration
  order, can be removed, and support wildcard matching.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.3.0"
    And a model "llama-7b" is attached
    And the session has been stepped to tick 0 at layer 0

  # ── Basic capture ─────────────────────────────────────────────────

  Scenario: Forward hook registered on a component captures output tensor
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "hook-cap-1",
          "point": "llama:0:12:attn.o_proj:output",
          "action": "capture",
          "config": {"summary": true, "capture_tensor": true}
        }
      }
      """
    Then the response status is "stopped"
    And the response data field "probe_id" is "hook-cap-1"
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | layer     |
    And the forward pass reaches layer 12
    Then the client receives a "probe.fired" notification for probe "hook-cap-1"
    And the notification includes a tensor summary with field "shape" of type array
    And the notification includes a tensor summary with field "dtype" of type string
    And the notification includes a tensor summary with field "tensor_id" of type string

  # ── Correct tick position ─────────────────────────────────────────

  Scenario: Hook fires at correct tick position (layer, component, event)
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "hook-pos-1",
          "point": "llama:0:8:mlp.down_proj:output",
          "action": "capture",
          "config": {"summary": true}
        }
      }
      """
    And the client steps forward until the forward pass reaches layer 8
    Then the client receives a "probe.fired" notification for probe "hook-pos-1"
    And the notification "params.position.layer" is 8
    And the notification "params.position.component" is "mlp.down_proj"
    And the notification "params.position.event" is "post"

  # ── Registration order ────────────────────────────────────────────

  Scenario: Multiple hooks on same component fire in registration order
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "hook-first",
          "point": "llama:0:12:attn.o_proj:output",
          "action": "capture",
          "config": {"summary": true},
          "priority": 0
        }
      }
      """
    And the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "hook-second",
          "point": "llama:0:12:attn.o_proj:output",
          "action": "capture",
          "config": {"summary": true},
          "priority": 1
        }
      }
      """
    And the client subscribes to "probe.fired" events
    And the client steps forward until the forward pass reaches layer 12
    Then probe "hook-first" fires before probe "hook-second" at point "llama:0:12:attn.o_proj:output"

  # ── Hook removal ──────────────────────────────────────────────────

  Scenario: Hook removal stops future captures
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "hook-rm-1",
          "point": "llama:0:12:attn.o_proj:output",
          "action": "capture",
          "config": {"summary": true}
        }
      }
      """
    Then the response data field "probe_id" is "hook-rm-1"
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "remove",
        "probe_id": "hook-rm-1"
      }
      """
    Then the response status is "stopped"
    And the response data field "probes" does not contain an entry with id "hook-rm-1"
    When the client subscribes to "probe.fired" events
    And the client steps forward until the forward pass reaches layer 12
    Then the client does not receive a "probe.fired" notification for probe "hook-rm-1"

  # ── Wildcard matching ─────────────────────────────────────────────

  Scenario: Hook on wildcard target fires for all matching components
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "hook-wild-1",
          "point": "llama:0:12:*:output",
          "action": "capture",
          "config": {"summary": true}
        }
      }
      """
    And the client subscribes to "probe.fired" events
    And the client steps forward until the forward pass completes layer 12
    Then probe "hook-wild-1" fires for at least 2 distinct components at layer 12
    And every "probe.fired" notification for "hook-wild-1" has position layer 12

  # ── tensor_id contract ────────────────────────────────────────────

  Scenario: Hook captures include tensor_id as BLAKE3 hash (64 hex chars)
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "hook-hash-1",
          "point": "llama:0:12:attn.o_proj:output",
          "action": "capture",
          "config": {"summary": true, "capture_tensor": true}
        }
      }
      """
    And the client subscribes to "probe.fired" events
    And the client steps forward until the forward pass reaches layer 12
    Then the client receives a "probe.fired" notification for probe "hook-hash-1"
    And the notification tensor "tensor_id" matches the pattern "^[0-9a-f]{64}$"

  # ── Non-matching suppression ──────────────────────────────────────

  Scenario: Hook does not fire on components that do not match the target pattern
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "hook-narrow-1",
          "point": "llama:0:12:attn.o_proj:output",
          "action": "capture",
          "config": {"summary": true}
        }
      }
      """
    And the client subscribes to "probe.fired" events
    And the client steps forward until the forward pass completes layer 12
    Then every "probe.fired" notification for "hook-narrow-1" has position component "attn.o_proj"
    And the client does not receive a "probe.fired" notification for "hook-narrow-1" at component "mlp.down_proj"
    And the client does not receive a "probe.fired" notification for "hook-narrow-1" at layer 8

  # ── Compiled model error ──────────────────────────────────────────

  Scenario: Hook on compiled model returns COMPILED_MODEL error
    Given the session is initialized with protocol_version "0.3.0"
    When the client sends "attach" with:
      | model_path     | /models/llama-7b-compiled |
      | model_family   | llama                     |
      | execution_mode | compiled                  |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "COMPILED_MODEL"
    And the error "data.severity" is "recoverable"
