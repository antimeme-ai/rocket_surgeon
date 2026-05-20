@intervention
Feature: Surgical interventions on the forward pass
  The rocket/intervene verb allows clients to set, clear, and list
  declarative intervention recipes that modify tensor values at
  probe points during the forward pass.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.1.0"
    And a model "llama-7b" is attached
    And the session has been stepped to tick 0 at layer 0

  # ── Set interventions (one per type) ──────────────────────────────

  Scenario: Set ablate intervention on attention output projection
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-ablate-1",
          "type": "ablate",
          "target": "llama:0:12:attn.o_proj:output",
          "params": {}
        }
      }
      """
    Then the response status is "stopped"
    And the response data field "applied" is true
    And the response data field "active_interventions" contains an entry with id "iv-ablate-1"

  Scenario: Set scale intervention with factor 0.5
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-scale-1",
          "type": "scale",
          "target": "llama:0:12:attn.o_proj:output",
          "params": {"factor": 0.5}
        }
      }
      """
    Then the response status is "stopped"
    And the response data field "applied" is true
    And the response data field "active_interventions" contains an entry with id "iv-scale-1"
    And the entry "iv-scale-1" has type "scale"
    And the entry "iv-scale-1" has params.factor equal to 0.5

  Scenario: Set add intervention with inline vector
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-add-1",
          "type": "add",
          "target": "llama:0:12:attn.o_proj:output",
          "params": {"vector": [1.0, 0.0, -1.0, 0.5]}
        }
      }
      """
    Then the response status is "stopped"
    And the response data field "applied" is true
    And the response data field "active_interventions" contains an entry with id "iv-add-1"

  Scenario: Set patch intervention with source tensor ID
    Given a tensor with id "a1b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1" exists in the tensor store
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-patch-1",
          "type": "patch",
          "target": "llama:0:12:attn.o_proj:output",
          "params": {"source_tensor_id": "a1b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1"}
        }
      }
      """
    Then the response status is "stopped"
    And the response data field "applied" is true
    And the response data field "active_interventions" contains an entry with id "iv-patch-1"

  Scenario: Set clamp intervention with min and max
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-clamp-1",
          "type": "clamp",
          "target": "llama:0:12:attn.o_proj:output",
          "params": {"min": -1.0, "max": 1.0}
        }
      }
      """
    Then the response status is "stopped"
    And the response data field "applied" is true
    And the response data field "active_interventions" contains an entry with id "iv-clamp-1"

  # ── Clear and list ────────────────────────────────────────────────

  Scenario: Clear intervention by ID removes it from active interventions
    Given an active intervention "iv-ablate-1" of type "ablate" on "llama:0:12:attn.o_proj:output"
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "clear",
        "intervention_id": "iv-ablate-1"
      }
      """
    Then the response status is "stopped"
    And the response data field "active_interventions" does not contain an entry with id "iv-ablate-1"

  Scenario: List interventions returns all active interventions
    Given an active intervention "iv-scale-1" of type "scale" on "llama:0:12:attn.o_proj:output"
    And an active intervention "iv-clamp-1" of type "clamp" on "llama:0:8:mlp:output"
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "list"
      }
      """
    Then the response status is "stopped"
    And the response data field "active_interventions" has 2 entries
    And the response data field "active_interventions" contains an entry with id "iv-scale-1"
    And the response data field "active_interventions" contains an entry with id "iv-clamp-1"

  # ── Composition semantics ─────────────────────────────────────────

  Scenario: Two interventions at same point execute in priority order
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-scale-lo",
          "type": "scale",
          "target": "llama:0:12:attn.o_proj:output",
          "params": {"factor": 0.5},
          "priority": 0
        }
      }
      """
    And the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-clamp-hi",
          "type": "clamp",
          "target": "llama:0:12:attn.o_proj:output",
          "params": {"min": -1.0, "max": 1.0},
          "priority": 10
        }
      }
      """
    Then the response data field "active_interventions" has 2 entries
    And the entry "iv-scale-lo" has priority 0
    And the entry "iv-clamp-hi" has priority 10
    When the client sends "rocket/step" with direction "forward"
    Then intervention "iv-scale-lo" executes before "iv-clamp-hi" at point "llama:0:12:attn.o_proj:output"

  Scenario: Intervention with mode replace overrides prior interventions
    Given an active intervention "iv-add-base" of type "add" on "llama:0:12:attn.o_proj:output" with mode "additive" and priority 0
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-scale-replace",
          "type": "scale",
          "target": "llama:0:12:attn.o_proj:output",
          "params": {"factor": 2.0},
          "priority": 5,
          "mode": "replace"
        }
      }
      """
    Then the response data field "applied" is true
    When the client sends "rocket/step" with direction "forward"
    Then only intervention "iv-scale-replace" takes effect at point "llama:0:12:attn.o_proj:output"
    And the prior additive intervention "iv-add-base" is overridden for this tick

  # ── Persistence across steps ──────────────────────────────────────

  Scenario: Intervention persists across multiple steps
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-persist",
          "type": "ablate",
          "target": "llama:0:12:attn.o_proj:output",
          "params": {}
        }
      }
      """
    And the client sends "rocket/step" with direction "forward"
    And the client sends "rocket/step" with direction "forward"
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "list"
      }
      """
    Then the response data field "active_interventions" contains an entry with id "iv-persist"

  # ── Error cases ───────────────────────────────────────────────────

  Scenario: Intervention on invalid target returns INVALID_TARGET
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-bad-target",
          "type": "ablate",
          "target": "llama:0:999:nonexistent.component:output",
          "params": {}
        }
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_TARGET"
    And the error "data.severity" is "recoverable"

  Scenario: Malformed recipe returns INVALID_RECIPE
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-bad-recipe",
          "type": "scale",
          "target": "llama:0:12:attn.o_proj:output",
          "params": {}
        }
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_RECIPE"
    And the error "data.severity" is "recoverable"

  Scenario: Intervene while session is not in stopped state returns INVALID_STATE
    Given the session is in "stepping" state
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-wrong-state",
          "type": "ablate",
          "target": "llama:0:12:attn.o_proj:output",
          "params": {}
        }
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_STATE"
    And the error "data.current_state" is "stepping"
    And the error "data.severity" is "recoverable"

  # ── Extended activation patching ─────────────────────────────────

  Scenario: Ablate with mode zero (default)
    Given an intervention recipe with type "ablate" and params {"mode": "zero"}
    Then the intervention deserializes successfully
    And mode is AblateMode::Zero

  Scenario: Ablate with mode mean
    Given an intervention recipe with type "ablate" and params {"mode": "mean", "reference_run": "ckpt-baseline"}
    Then the intervention deserializes successfully
    And mode is AblateMode::Mean

  Scenario: AttentionMask intervention
    Given an intervention recipe with type "attention_mask"
    And params {"source_positions": [0, 3], "target_positions": [5], "mask_value": -10000.0}
    Then the intervention deserializes successfully

  Scenario: EmbedSwap intervention
    Given an intervention recipe with type "embed_swap"
    And params {"position": 5, "new_token_id": 1234}
    Then the intervention deserializes successfully

  Scenario: EmbedNoise intervention
    Given an intervention recipe with type "embed_noise"
    And params {"position": 5, "std": 0.1, "seed": 42}
    Then the intervention deserializes successfully
