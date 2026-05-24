@tick-position @phase
Feature: TickPosition phase and token_position fields
  TickPosition must carry a phase (prefill | decode | prefill_chunked)
  and an optional token_position so the token axis, KV cache views,
  and worldline branching have a coordinate to attach to.

  Protocol version 0.2.0 introduces these fields.

  Background:
    Given the session is initialized and a model is attached
    And the session is in "stopped" state

  # ── Protocol version ──────────────────────────────────────────────

  @deferred
  Scenario: Server advertises protocol version 0.2.0
    When the client sends "initialize" with:
      | client_name      | tck-runner |
      | protocol_version | 0.2.0      |
    Then the response "data.capabilities.protocol_version" is "0.2.0"

  # ── Phase field presence ──────────────────────────────────────────

  @deferred
  Scenario: Step response includes phase field on stopped_at
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response "data.stopped_at" has field "phase" of type object
    And the response "data.stopped_at.phase.type" is one of "prefill", "decode", or "prefill_chunked"

  @deferred
  Scenario: Phase is decode during single-token generation
    Given the model has completed prefill
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response "data.stopped_at.phase.type" is "decode"

  @deferred
  Scenario: Phase is prefill during initial forward pass
    Given a prompt has been submitted but prefill has not completed
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response "data.stopped_at.phase.type" is "prefill"

  # ── token_position field ──────────────────────────────────────────

  @deferred
  Scenario: token_position is present after a step
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response "data.stopped_at" has field "token_position" of type integer

  @deferred
  Scenario: token_position advances during decode
    Given the model has completed prefill for a prompt of length N
    When the client executes 3 forward steps at "layer" granularity
    Then the observed token_position values are non-decreasing

  # ── Phase enum serialization ──────────────────────────────────────

  @deferred
  Scenario: Decode phase serializes as tagged object
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    And the response "data.stopped_at.phase.type" is "decode"
    Then the response "data.stopped_at.phase" has no field "chunk_size"
    And the response "data.stopped_at.phase" has no field "chunk_index"
    And the response "data.stopped_at.phase" has no field "total_chunks"

  @deferred
  Scenario: PrefillChunked phase carries chunk metadata
    Given the model is configured for chunked prefill with chunk_size 512
    And a prompt of length 2048 has been submitted
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | layer     |
    And the response "data.stopped_at.phase.type" is "prefill_chunked"
    Then the response "data.stopped_at.phase" has field "chunk_size" of type integer
    And the response "data.stopped_at.phase" has field "chunk_index" of type integer
    And the response "data.stopped_at.phase" has field "total_chunks" of type integer
    And the response "data.stopped_at.phase.total_chunks" equals ceil(2048 / 512)

  # ── Forward compatibility ─────────────────────────────────────────

  @deferred
  Scenario: TickPosition without phase deserializes with decode default
    Given a JSON TickPosition from protocol 0.3.0 without "phase" or "token_position"
    When the client deserializes the JSON as TickPosition
    Then the phase is "decode"
    And the token_position is null

  # ── Event notifications ───────────────────────────────────────────

  @deferred
  Scenario: tick.stopped event includes phase and token_position
    Given the client has subscribed to events
    When a step completes and a "tick.stopped" event is emitted
    Then the event "position.phase" has field "type"
    And the event "position" has field "token_position"

  @deferred
  Scenario: tick.heartbeat event includes phase on position
    Given the client has subscribed to events
    When a "tick.heartbeat" event is emitted
    Then the event "position.phase" has field "type"
