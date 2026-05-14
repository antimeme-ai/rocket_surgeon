@checkpoint @phase3
Feature: Checkpoint management — create, list, restore, delete, bookmark
  The checkpoint verb manages activation checkpoints and full snapshots
  of model state. Checkpoints enable reverse stepping and replay by
  capturing residual streams, RNG state, and input IDs at tick boundaries.
  All checkpoint verbs require an attached model in "stopped" state and
  the supports_checkpointing capability.

  Background:
    Given the session is in "stopped" state with model "llama"
    And the server capability "supports_checkpointing" is true
    And the session has been stepped to tick 5 at layer 3

  # ── Create ─────────────────────────────────────────────────────────

  Scenario: Create activation checkpoint returns checkpoint_id
    When the client sends "rocket/checkpoint" with:
      | action | create     |
      | tier   | activation |
    Then the response status is "stopped"
    And the response "data.checkpoint_id" is a non-empty string
    And the response "data.checkpoints" contains an entry where:
      | checkpoint_id | equals response "data.checkpoint_id" |
      | tier          | activation                           |
      | tick_id       | 5                                    |
      | layer_idx     | 3                                    |

  Scenario: Create full_snapshot checkpoint returns checkpoint_id
    When the client sends "rocket/checkpoint" with:
      | action | create        |
      | tier   | full_snapshot |
    Then the response status is "stopped"
    And the response "data.checkpoint_id" is a non-empty string
    And the response "data.checkpoints" contains an entry where:
      | checkpoint_id | equals response "data.checkpoint_id" |
      | tier          | full_snapshot                        |
      | tick_id       | 5                                    |

  # ── List ───────────────────────────────────────────────────────────

  Scenario: List checkpoints returns all with tier, tick_id, layer_idx
    Given the session has an activation checkpoint "ckpt-a" at tick 3 layer 2
    And the session has a full_snapshot checkpoint "ckpt-b" at tick 5 layer 3
    When the client sends "rocket/checkpoint" with:
      | action | list |
    Then the response status is "stopped"
    And the response "data.checkpoints" has 2 entries
    And each entry in "data.checkpoints" includes:
      | field         | type    |
      | checkpoint_id | string  |
      | tick_id       | integer |
      | layer_idx     | integer |
      | tier          | string  |
      | created_at    | string  |

  # ── Restore ────────────────────────────────────────────────────────

  Scenario: Restore checkpoint by id moves position to checkpointed tick
    Given the session has an activation checkpoint "ckpt-a" at tick 3 layer 2
    And the session has been stepped to tick 5 at layer 3
    When the client sends "rocket/checkpoint" with:
      | action        | restore |
      | checkpoint_id | ckpt-a  |
    Then the response status is "stopped"
    And the response "data.restored_to.tick_id" is 3
    And the response "data.restored_to.layer" is 2
    And the response "state.position.tick_id" is 3

  # ── Delete ─────────────────────────────────────────────────────────

  Scenario: Delete checkpoint by id removes it from list
    Given the session has an activation checkpoint "ckpt-a" at tick 3 layer 2
    And the session has a full_snapshot checkpoint "ckpt-b" at tick 5 layer 3
    When the client sends "rocket/checkpoint" with:
      | action        | delete |
      | checkpoint_id | ckpt-a |
    Then the response status is "stopped"
    And the response "data.checkpoints" has 1 entry
    And the response "data.checkpoints" does not contain checkpoint_id "ckpt-a"

  # ── Bookmark ───────────────────────────────────────────────────────

  Scenario: Bookmark a tick_id with a name appears on checkpoint
    When the client sends "rocket/checkpoint" with:
      | action  | bookmark            |
      | tick_id | 5                   |
      | name    | before-intervention |
    Then the response status is "stopped"
    And the response "data.checkpoints" contains an entry where:
      | tick_id  | 5                   |
      | bookmark | before-intervention |

  # ── Error paths ────────────────────────────────────────────────────

  Scenario: Restore nonexistent checkpoint returns CHECKPOINT_NOT_FOUND error
    When the client sends "rocket/checkpoint" with:
      | action        | restore     |
      | checkpoint_id | nonexistent |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "CHECKPOINT_NOT_FOUND"
    And the error "data.severity" is "recoverable"
    And the error "data.suggestion" is a non-empty string

  Scenario: Delete nonexistent checkpoint returns CHECKPOINT_NOT_FOUND error
    When the client sends "rocket/checkpoint" with:
      | action        | delete      |
      | checkpoint_id | nonexistent |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "CHECKPOINT_NOT_FOUND"
    And the error "data.severity" is "recoverable"
    And the error "data.suggestion" is a non-empty string

  # ── Response envelope ──────────────────────────────────────────────

  Scenario: Checkpoint response includes full SessionState
    When the client sends "rocket/checkpoint" with:
      | action | create     |
      | tier   | activation |
    Then the response "state" includes:
      | field             | type    |
      | session_id        | string  |
      | model_id          | string  |
      | status            | string  |
      | position          | object  |
      | tick_id           | integer |
      | active_probes     | array   |
      | checkpoints       | array   |
      | available_actions | array   |
    And the response "state.session_id" matches UUID format
    And the response "state.status" is "stopped"
    And the response "state.checkpoints" is not empty
