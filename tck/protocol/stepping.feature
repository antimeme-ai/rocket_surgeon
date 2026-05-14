@stepping
Feature: Forward-pass stepping — rocket/step verb
  The step verb advances (or reverses) the forward pass by one or more
  ticks. Each step transitions STOPPED -> STEPPING -> STOPPED and returns
  the new tick position plus the number of ticks actually executed.

  Background:
    Given the session is initialized and a model is attached
    And the session is in "stopped" state

  # ── Basic forward stepping ─────────────────────────────────────────

  Scenario: Step forward count=1 at component granularity
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response status is "stopped"
    And the response "data.ticks_executed" is 1
    And the response "data.stopped_at" contains "layer"
    And the response "data.stopped_at" contains "component"
    And the response "data.stopped_at" contains "event"
    And the response "data.stopped_at.direction" is "forward"

  Scenario: Step forward count=5 executes 5 ticks
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 5         |
      | granularity | component |
    Then the response status is "stopped"
    And the response "data.ticks_executed" is 5

  Scenario Outline: Step forward with different granularities
    When the client sends "rocket/step" with:
      | direction   | forward       |
      | count       | <count>       |
      | granularity | <granularity> |
    Then the response status is "stopped"
    And the response "data.ticks_executed" is <count>

    Examples:
      | granularity | count |
      | component   | 1     |
      | component   | 3     |
      | layer       | 1     |
      | layer       | 2     |

  Scenario: Layer granularity produces fewer ticks than component for same layer count
    Given the client steps forward 1 tick at "layer" granularity
    And the resulting tick_id is saved as "layer_tick"
    When the session is reset to stopped at tick 0
    And the client steps forward 1 tick at "component" granularity
    And the resulting tick_id is saved as "component_tick"
    Then "layer_tick" advanced further in layer index than "component_tick"

  # ── First step from initial position ───────────────────────────────

  Scenario: First step starts from layer 0
    Given no steps have been executed in this session
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response status is "stopped"
    And the response "data.stopped_at.layer" is 0
    And the response "data.ticks_executed" is 1

  # ── tick_id monotonicity ───────────────────────────────────────────

  Scenario: tick_id monotonically increases across steps
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    And the response "state.tick_id" is saved as "tick_a"
    And the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    And the response "state.tick_id" is saved as "tick_b"
    And the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    And the response "state.tick_id" is saved as "tick_c"
    Then "tick_a" < "tick_b" < "tick_c"

  Scenario: tick_id is never reused across steps
    When the client executes 10 forward steps at "component" granularity
    Then all observed tick_ids are unique

  # ── Position structure ─────────────────────────────────────────────

  Scenario: Position includes required fields after step
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response "data.stopped_at" has field "tick_id" of type integer
    And the response "data.stopped_at" has field "direction" of type string
    And the response "data.stopped_at" has field "layer" of type integer
    And the response "data.stopped_at" has field "component" of type string
    And the response "data.stopped_at" has field "event" of type string
    And the response "data.stopped_at.event" is one of "input" or "output"

  Scenario: Position layer advances across successive steps
    When the client sends "rocket/step" with:
      | direction   | forward |
      | count       | 1       |
      | granularity | layer   |
    And the response "data.stopped_at.layer" is saved as "layer_a"
    And the client sends "rocket/step" with:
      | direction   | forward |
      | count       | 1       |
      | granularity | layer   |
    And the response "data.stopped_at.layer" is saved as "layer_b"
    Then "layer_b" > "layer_a"

  # ── Response envelope ──────────────────────────────────────────────

  Scenario: Step response includes full SessionState in envelope
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response "state" has field "session_id" of type string
    And the response "state" has field "model_id" of type string
    And the response "state" has field "status" of type string
    And the response "state" has field "position" of type object
    And the response "state" has field "tick_id" of type integer
    And the response "state" has field "active_probes" of type array
    And the response "state" has field "checkpoints" of type array
    And the response "state" has field "available_actions" of type array

  # ── Backward step errors ───────────────────────────────────────────

  Scenario: Step backward without checkpointing returns CAPABILITY_NOT_SUPPORTED
    Given the server capability "supports_reverse_step" is false
    When the client sends "rocket/step" with:
      | direction   | backward  |
      | count       | 1         |
      | granularity | component |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "CAPABILITY_NOT_SUPPORTED"
    And the error "data.severity" is "recoverable"
