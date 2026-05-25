@replay @phase3
Feature: Replay execution from checkpoint — with interventions, verification, divergence
  The replay verb re-executes the forward pass from a checkpoint, optionally
  applying interventions and verifying fidelity against the original execution.
  Replay is ULP-close, not bit-exact. Divergences beyond tolerance are reported
  via structured data and a rocket/replay.divergence notification event.

  Background:
    Given the session is in "stopped" state with model "llama"
    And the server capability "supports_checkpointing" is true
    And the session has been stepped to tick 10 at layer 8
    And the session has an activation checkpoint "ckpt-origin" at tick 3 layer 2

  # ── Happy path ─────────────────────────────────────────────────────

  @deferred
  Scenario: Replay from checkpoint returns ticks_replayed and stopped_at
    When the client sends "rocket/replay" with:
      | from_checkpoint | ckpt-origin |
    Then the response status is "stopped"
    And the response "data.ticks_replayed" is greater than 0
    And the response "data.stopped_at" includes:
      | field     | type    |
      | tick_id   | integer |
      | direction | string  |
      | layer     | integer |
      | component | string  |
      | event     | string  |

  @deferred
  Scenario: Replay with interventions applies them during replay
    When the client sends "rocket/replay" with:
      | from_checkpoint | ckpt-origin |
    And the request includes "interventions" array:
      """json
      [
        {
          "id": "ablate-head-7",
          "type": "ablate",
          "target": "llama:0:5:attn.o_proj:output",
          "params": {}
        }
      ]
      """
    Then the response status is "stopped"
    And the response "data.ticks_replayed" is greater than 0
    And the response "data.divergences" is an array

  @deferred
  Scenario: Replay with stop_at layer stops at specified layer
    When the client sends "rocket/replay" with:
      | from_checkpoint | ckpt-origin |
    And the request includes "stop_at" object:
      | layer     | 5           |
      | component | attn.o_proj |
    Then the response status is "stopped"
    And the response "data.stopped_at.layer" is 5
    And the response "data.stopped_at.component" is "attn.o_proj"

  @deferred
  Scenario: Replay with verify=true returns verified boolean in response
    When the client sends "rocket/replay" with:
      | from_checkpoint | ckpt-origin |
      | verify          | true        |
    Then the response status is "stopped"
    And the response "data.verified" is a boolean
    And the response "data.divergences" is an array

  # ── Divergence detection ───────────────────────────────────────────

  @deferred
  Scenario: Replay divergence detected populates divergences array
    Given the session has an activation checkpoint "ckpt-before-mut" at tick 3 layer 2
    When the client sends "rocket/replay" with:
      | from_checkpoint | ckpt-before-mut |
      | verify          | true            |
    And the request includes "interventions" array:
      """json
      [
        {
          "id": "scale-residual",
          "type": "scale",
          "target": "llama:0:4:residual_post:output",
          "params": {"factor": 100.0}
        }
      ]
      """
    Then the response status is "stopped"
    And the response "data.divergences" is a non-empty array
    And each entry in "data.divergences" includes:
      | field              | type   |
      | tick_id            | integer|
      | original_tick_id   | integer|
      | probe_point        | string |
      | cosine_similarity  | number |
      | max_relative_error | number |
      | message            | string |

  @deferred
  Scenario: Replay divergence fires rocket/replay.divergence event
    Given the client has subscribed to "rocket/replay.divergence" events
    And the session has an activation checkpoint "ckpt-before-mut" at tick 3 layer 2
    When the client sends "rocket/replay" with:
      | from_checkpoint | ckpt-before-mut |
      | verify          | true            |
    And the request includes "interventions" array:
      """json
      [
        {
          "id": "scale-residual",
          "type": "scale",
          "target": "llama:0:4:residual_post:output",
          "params": {"factor": 100.0}
        }
      ]
      """
    Then the client receives a "rocket/replay.divergence" notification
    And the notification "params" includes:
      | field              | type    |
      | tick_id            | integer |
      | original_tick_id   | integer |
      | probe_point        | string  |
      | cosine_similarity  | number  |
      | max_relative_error | number  |
      | message            | string  |

  # ── Tick identity ──────────────────────────────────────────────────

  @deferred
  Scenario: Replayed ticks get fresh tick_ids with replay_of referencing original
    When the client sends "rocket/replay" with:
      | from_checkpoint | ckpt-origin |
    Then the response status is "stopped"
    And the response "data.stopped_at.tick_id" is greater than 10
    And the response "data.stopped_at.replay_of" is not null
    And the response "state.tick_id" is greater than 10

  # ── Error paths ────────────────────────────────────────────────────

  @deferred
  Scenario: Replay from nonexistent checkpoint returns CHECKPOINT_NOT_FOUND error
    When the client sends "rocket/replay" with:
      | from_checkpoint | nonexistent |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "CHECKPOINT_NOT_FOUND"
    And the error "data.severity" is "recoverable"
    And the error "data.suggestion" is a non-empty string
