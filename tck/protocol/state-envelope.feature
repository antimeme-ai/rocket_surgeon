@envelope
Feature: Response envelope contract — SessionState in every response
  Every protocol response includes a "state" object conforming to the
  SessionState schema. This envelope carries the session_id, model_id,
  current status, tick position, active probes, checkpoints, and the
  list of valid actions for the current state. The TCK verifies that the
  envelope is present, correctly typed, and consistent with the state
  machine at every stage of the session lifecycle.

  # ── Presence and shape ─────────────────────────────────────────────

  Scenario: Every response contains a "state" object with session_id
    Given the session is in "uninitialized" state
    When the client sends "initialize" with:
      | client_name      | tck-harness |
      | protocol_version | 0.1.0       |
    Then the response has a "state" object
    And the response "state" has field "session_id" of type string
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response has a "state" object
    And the response "state" has field "session_id" of type string
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response has a "state" object
    And the response "state" has field "session_id" of type string
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    Then the response has a "state" object
    And the response "state" has field "session_id" of type string
    When the client sends "detach" with no parameters
    Then the response has a "state" object
    And the response "state" has field "session_id" of type string

  # ── session_id ─────────────────────────────────────────────────────

  Scenario: session_id is UUID format
    Given the session is in "uninitialized" state
    When the client sends "initialize" with:
      | client_name      | tck-harness |
      | protocol_version | 0.1.0       |
    Then the response "state.session_id" matches UUID format "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

  Scenario: session_id is stable across responses in a session
    Given the session is in "uninitialized" state
    When the client sends "initialize" with:
      | client_name      | tck-harness |
      | protocol_version | 0.1.0       |
    And the response "state.session_id" is saved as "sid_init"
    And the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    And the response "state.session_id" is saved as "sid_attach"
    And the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    And the response "state.session_id" is saved as "sid_step"
    And the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    And the response "state.session_id" is saved as "sid_inspect"
    And the client sends "detach" with no parameters
    And the response "state.session_id" is saved as "sid_detach"
    Then "sid_init" equals "sid_attach"
    And "sid_attach" equals "sid_step"
    And "sid_step" equals "sid_inspect"
    And "sid_inspect" equals "sid_detach"

  # ── model_id ───────────────────────────────────────────────────────

  Scenario: model_id is null before attach
    Given the session is in "uninitialized" state
    When the client sends "initialize" with:
      | client_name      | tck-harness |
      | protocol_version | 0.1.0       |
    Then the response "state.model_id" is null

  Scenario: model_id is populated after attach
    Given the session is in "initialized" state
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response "state.model_id" is a non-empty string

  Scenario: model_id returns to null after detach
    Given the session is in "stopped" state with model "llama"
    When the client sends "detach" with no parameters
    Then the response "state.model_id" is null

  # ── status ─────────────────────────────────────────────────────────

  Scenario Outline: status matches expected state machine position
    Given the session has been advanced to "<precondition>" state
    Then the most recent response "state.status" is "<expected_status>"

    Examples:
      | precondition            | expected_status |
      | after initialize        | initialized     |
      | after attach            | stopped         |
      | after step              | stopped         |
      | after inspect           | stopped         |
      | after detach            | initialized     |

  # ── position ───────────────────────────────────────────────────────

  Scenario: position is null before first step
    Given the session is in "initialized" state
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response "state.position" is null

  Scenario: position is populated after first step
    Given the session is in "stopped" state with model "llama"
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response "state.position" is not null
    And the response "state.position" has field "tick_id" of type integer
    And the response "state.position" has field "layer" of type integer
    And the response "state.position" has field "component" of type string
    And the response "state.position" has field "event" of type string

  # ── tick_id ────────────────────────────────────────────────────────

  Scenario: tick_id is null before first step
    Given the session is in "stopped" state with model "llama"
    And no steps have been executed in this session
    Then the most recent response "state.tick_id" is null

  Scenario: tick_id is an integer after first step
    Given the session is in "stopped" state with model "llama"
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response "state.tick_id" is of type integer
    And the response "state.tick_id" >= 0

  Scenario: tick_id is monotonically increasing across steps
    Given the session is in "stopped" state with model "llama"
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    And the response "state.tick_id" is saved as "t1"
    And the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    And the response "state.tick_id" is saved as "t2"
    And the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    And the response "state.tick_id" is saved as "t3"
    Then "t1" < "t2" < "t3"

  Scenario: tick_id is unique and never reused
    Given the session is in "stopped" state with model "llama"
    When the client executes 10 forward steps at "component" granularity
    Then all observed "state.tick_id" values are unique

  # ── active_probes ──────────────────────────────────────────────────

  Scenario: active_probes is empty initially
    Given the session is in "stopped" state with model "llama"
    And no probes have been defined in this session
    Then the most recent response "state.active_probes" is an empty array

  # ── available_actions per state ────────────────────────────────────

  Scenario: available_actions for "initialized" state after initialize
    Given the session is in "uninitialized" state
    When the client sends "initialize" with:
      | client_name      | tck-harness |
      | protocol_version | 0.1.0       |
    Then the response "state.status" is "initialized"
    And the response "state.available_actions" is the array ["attach"]

  Scenario: available_actions for "stopped" state includes all domain verbs
    Given the session is in "stopped" state with model "llama"
    Then the most recent response "state.available_actions" includes "step"
    And the most recent response "state.available_actions" includes "inspect"
    And the most recent response "state.available_actions" includes "intervene"
    And the most recent response "state.available_actions" includes "probe"
    And the most recent response "state.available_actions" includes "detach"
    And the most recent response "state.available_actions" includes "status"
    And the most recent response "state.available_actions" includes "subscribe"

  Scenario Outline: available_actions matches state machine for each state
    Given the session has been advanced to "<precondition>" state
    Then the most recent response "state.available_actions" equals <expected_actions>

    Examples:
      | precondition     | expected_actions                                                              |
      | after initialize | ["attach"]                                                                    |
      | after attach     | ["step", "inspect", "intervene", "probe", "checkpoint", "replay", "detach", "status", "subscribe"] |
