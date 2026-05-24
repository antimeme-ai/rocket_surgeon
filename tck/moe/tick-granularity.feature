@moe @phase6
Feature: MoE tick granularity — router and expert-level stepping
  Mixture-of-Experts models expose additional tick granularities beyond
  the standard layer and component levels. The MoE granularities —
  router_pre_topk, router_post_topk, expert, and moe_layer — allow
  clients to stop at specific stages of the MoE dispatch cycle: before
  top-k selection, after top-k selection, at individual expert execution,
  and at the full MoE layer boundary. These granularities require the
  supports_moe capability.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.3.0"
    And the server capability "supports_moe" is true
    And a model "mixtral-8x7b" is attached with model_family "mixtral"
    And the session is in "stopped" state

  # ── Granularity exposure ──────────────────────────────────────────

  @deferred
  @deferred
  Scenario: MoE model exposes all 4 MoE tick granularities
    When the client sends "rocket/status" with no parameters
    Then the response "data.capabilities.tick_granularities" contains "router_pre_topk"
    And the response "data.capabilities.tick_granularities" contains "router_post_topk"
    And the response "data.capabilities.tick_granularities" contains "expert"
    And the response "data.capabilities.tick_granularities" contains "moe_layer"
    And the response "data.capabilities.tick_granularities" contains "layer"
    And the response "data.capabilities.tick_granularities" contains "component"

  # ── router_pre_topk ──────────────────────────────────────────────

  @deferred
  @deferred
  Scenario: Step with granularity=router_pre_topk stops before top-k selection
    When the client sends "rocket/step" with:
      | direction   | forward          |
      | count       | 1                |
      | granularity | router_pre_topk  |
    Then the response status is "stopped"
    And the response "data.ticks_executed" is 1
    And the response "data.stopped_at.component" is "router.logits"
    And the response "data.stopped_at.event" is "output"
    And the response "data.stopped_at.moe_phase" is "router_pre_topk"

  @deferred
  @deferred
  Scenario: Inspect at router_pre_topk shows raw router logits
    Given the client has stepped forward 1 tick at "router_pre_topk" granularity
    When the client sends "rocket/inspect" with:
      """json
      {
        "target": "mixtral:0:0:router.logits:output",
        "detail": "summary"
      }
      """
    Then the response status is "stopped"
    And the response "data.tensors" is an array with at least 1 element
    And the first tensor in "data.tensors" has field "shape" of type array
    And the first tensor "shape" has 2 dimensions
    And the first tensor "shape[1]" equals the number of experts (8)
    And the first tensor in "data.tensors" has field "dtype" of type string
    And the first tensor in "data.tensors" has field "stats" of type object
    And the first tensor "stats" includes "mean", "std", "min", "max"

  # ── router_post_topk ─────────────────────────────────────────────

  @deferred
  @deferred
  Scenario: Step with granularity=router_post_topk stops after top-k selection
    When the client sends "rocket/step" with:
      | direction   | forward           |
      | count       | 1                 |
      | granularity | router_post_topk  |
    Then the response status is "stopped"
    And the response "data.ticks_executed" is 1
    And the response "data.stopped_at.component" is "router.decision"
    And the response "data.stopped_at.event" is "output"
    And the response "data.stopped_at.moe_phase" is "router_post_topk"

  @deferred
  @deferred
  Scenario: Inspect at router_post_topk shows selected expert indices and weights
    Given the client has stepped forward 1 tick at "router_post_topk" granularity
    When the client sends "rocket/inspect" with:
      """json
      {
        "target": "mixtral:0:0:router.decision:output",
        "detail": "summary"
      }
      """
    Then the response status is "stopped"
    And the response "data.routing_decision" is not null
    And the response "data.routing_decision.selected_experts" is an array
    And each entry in "data.routing_decision.selected_experts" is an integer in range [0, 7]
    And the response "data.routing_decision.selected_experts" has exactly 2 entries
    And the response "data.routing_decision.expert_weights" is an array with exactly 2 entries
    And each entry in "data.routing_decision.expert_weights" is a number > 0

  # ── expert granularity ───────────────────────────────────────────

  @deferred
  @deferred
  Scenario: Step with granularity=expert stops at individual expert execution
    When the client sends "rocket/step" with:
      | direction   | forward |
      | count       | 1       |
      | granularity | expert  |
    Then the response status is "stopped"
    And the response "data.ticks_executed" is 1
    And the response "data.stopped_at.component" starts with "experts["
    And the response "data.stopped_at.moe_phase" is "expert"
    And the response "data.stopped_at.expert_index" is an integer in range [0, 7]

  @deferred
  @deferred
  Scenario: Stepping through all selected experts visits each one
    Given the client has stepped forward 1 tick at "router_post_topk" granularity
    And the routing decision selected experts [2, 5]
    When the client sends "rocket/step" with:
      | direction   | forward |
      | count       | 1       |
      | granularity | expert  |
    And the response "data.stopped_at.expert_index" is saved as "first_expert"
    And the client sends "rocket/step" with:
      | direction   | forward |
      | count       | 1       |
      | granularity | expert  |
    And the response "data.stopped_at.expert_index" is saved as "second_expert"
    Then "first_expert" and "second_expert" are distinct
    And the set {"first_expert", "second_expert"} equals {2, 5}

  # ── moe_layer granularity ────────────────────────────────────────

  @deferred
  @deferred
  Scenario: Step with granularity=moe_layer advances past entire MoE block
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | moe_layer |
    Then the response status is "stopped"
    And the response "data.ticks_executed" is 1
    And the response "data.stopped_at.moe_phase" is "moe_layer_complete"
    And the response "data.stopped_at.event" is "output"

  # ── route_override intervention ──────────────────────────────────

  @deferred
  @deferred
  Scenario: route_override intervention changes expert selection
    Given the client has stepped forward 1 tick at "router_pre_topk" granularity
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "iv-route-override-1",
          "type": "route_override",
          "target": "mixtral:0:0:router.decision:output",
          "params": {
            "token": 0,
            "experts": [0, 7]
          }
        }
      }
      """
    Then the response status is "stopped"
    And the response data field "applied" is true
    When the client sends "rocket/step" with:
      | direction   | forward           |
      | count       | 1                 |
      | granularity | router_post_topk  |
    Then the response "data.routing_decision.selected_experts" includes 0
    And the response "data.routing_decision.selected_experts" includes 7

  # ── Capability gating ────────────────────────────────────────────

  @deferred
  @deferred
  Scenario: MoE tick granularities require supports_moe capability
    Given the server capability "supports_moe" is true
    When the client sends "rocket/step" with:
      | direction   | forward          |
      | count       | 1                |
      | granularity | router_pre_topk  |
    Then the response status is "stopped"
    And the response "data.ticks_executed" is 1

  @deferred
  @deferred
  Scenario: Dense model step with MoE granularity returns CAPABILITY_NOT_SUPPORTED
    Given the session is initialized with protocol_version "0.3.0"
    And a model "llama-7b" is attached with model_family "llama"
    And the server capability "supports_moe" is false
    When the client sends "rocket/step" with:
      | direction   | forward          |
      | count       | 1                |
      | granularity | router_pre_topk  |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "CAPABILITY_NOT_SUPPORTED"
    And the error "data.severity" is "recoverable"
    And the error "data.context.required_capability" is "supports_moe"
