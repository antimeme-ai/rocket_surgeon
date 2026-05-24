@bundle
Feature: Session bundle — export and restore full session state
  A session bundle captures the complete debugger session state into a
  portable JSON file. Bundles include the model identifier, tick position,
  active probes, checkpoints, interventions, and the protocol_version
  used to create them. Restoring a bundle recreates the session state.
  Bundles are versioned: restoring from an incompatible protocol version
  is rejected.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.3.0"
    And a model "llama-7b" is attached
    And the session has been stepped to tick 5 at layer 3

  # ── Export ─────────────────────────────────────────────────────────

  Scenario: Bundle export creates a file containing session state
    When the client sends "rocket/bundle" with:
      """json
      {
        "action": "export",
        "path": "/tmp/tck-bundle-export.json"
      }
      """
    Then the response status is "stopped"
    And the response data field "bundle_path" is "/tmp/tck-bundle-export.json"
    And the response data field "exported" is true
    And the file "/tmp/tck-bundle-export.json" exists

  Scenario: Bundle format is JSON
    When the client sends "rocket/bundle" with:
      """json
      {
        "action": "export",
        "path": "/tmp/tck-bundle-format.json"
      }
      """
    Then the response status is "stopped"
    And the file "/tmp/tck-bundle-format.json" is valid JSON

  Scenario: Bundle includes model_id, position, active_probes, checkpoints, interventions
    Given a defined probe "p-bundle-1" at point "llama:0:12:attn.o_proj:output" with action "capture"
    And an active intervention "iv-bundle-1" of type "ablate" on "llama:0:8:mlp:output"
    When the client sends "rocket/bundle" with:
      """json
      {
        "action": "export",
        "path": "/tmp/tck-bundle-contents.json"
      }
      """
    Then the response status is "stopped"
    And the bundle at "/tmp/tck-bundle-contents.json" contains field "model_id" as a non-empty string
    And the bundle at "/tmp/tck-bundle-contents.json" contains field "position" as an object
    And the bundle "position.tick_id" is 5
    And the bundle "position.layer" is 3
    And the bundle at "/tmp/tck-bundle-contents.json" contains field "active_probes" as an array
    And the bundle "active_probes" contains an entry with id "p-bundle-1"
    And the bundle at "/tmp/tck-bundle-contents.json" contains field "checkpoints" as an array
    And the bundle at "/tmp/tck-bundle-contents.json" contains field "interventions" as an array
    And the bundle "interventions" contains an entry with id "iv-bundle-1"

  Scenario: Bundle includes protocol_version for compatibility checking
    When the client sends "rocket/bundle" with:
      """json
      {
        "action": "export",
        "path": "/tmp/tck-bundle-version.json"
      }
      """
    Then the response status is "stopped"
    And the bundle at "/tmp/tck-bundle-version.json" contains field "protocol_version" equal to "0.3.0"

  # ── Restore ────────────────────────────────────────────────────────

  Scenario: Bundle restore recreates session state
    Given a defined probe "p-restore-1" at point "llama:0:12:attn.o_proj:output" with action "capture"
    And a bundle has been exported to "/tmp/tck-bundle-restore.json"
    And the client sends "detach" with no parameters
    And the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    When the client sends "rocket/bundle" with:
      """json
      {
        "action": "restore",
        "path": "/tmp/tck-bundle-restore.json"
      }
      """
    Then the response status is "stopped"
    And the response data field "restored" is true
    And the response "state.model_id" is a non-empty string
    And the response "state.position.tick_id" is 5
    And the response "state.position.layer" is 3
    And the response "state.active_probes" contains an entry with id "p-restore-1"

  Scenario: Bundle restore with model_id match succeeds
    Given a bundle has been exported to "/tmp/tck-bundle-model-match.json"
    And the client sends "detach" with no parameters
    And the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    When the client sends "rocket/bundle" with:
      """json
      {
        "action": "restore",
        "path": "/tmp/tck-bundle-model-match.json"
      }
      """
    Then the response status is "stopped"
    And the response data field "restored" is true
    And the response "data.model_id" matches the bundle "model_id"

  # ── Error paths ────────────────────────────────────────────────────

  Scenario: Bundle restore with wrong model returns INVALID_PARAMS
    Given a bundle has been exported to "/tmp/tck-bundle-wrong-model.json"
    And the client sends "detach" with no parameters
    And the client sends "attach" with:
      | model_path   | /models/gpt-neox-20b |
      | model_family | gpt-neox             |
    When the client sends "rocket/bundle" with:
      """json
      {
        "action": "restore",
        "path": "/tmp/tck-bundle-wrong-model.json"
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_PARAMS"
    And the error "data.severity" is "recoverable"
    And the error "data.context.reason" is "model_mismatch"
    And the error "data.suggestion" is a non-empty string

  Scenario: Restore bundle from incompatible protocol version returns INVALID_PARAMS
    Given a bundle file "/tmp/tck-bundle-old-version.json" with protocol_version "0.0.1"
    When the client sends "rocket/bundle" with:
      """json
      {
        "action": "restore",
        "path": "/tmp/tck-bundle-old-version.json"
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_PARAMS"
    And the error "data.severity" is "recoverable"
    And the error "data.context.reason" is "version_incompatible"
    And the error "data.context.expected_version" is "0.3.0"
    And the error "data.context.bundle_version" is "0.0.1"
    And the error "data.suggestion" is a non-empty string

  Scenario: Restore bundle from nonexistent path returns INVALID_PARAMS
    When the client sends "rocket/bundle" with:
      """json
      {
        "action": "restore",
        "path": "/tmp/tck-bundle-does-not-exist.json"
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_PARAMS"
    And the error "data.severity" is "recoverable"
    And the error "data.context.reason" is "bundle_not_found"

  # ── Response envelope ──────────────────────────────────────────────

  Scenario: Bundle response includes full SessionState in envelope
    When the client sends "rocket/bundle" with:
      """json
      {
        "action": "export",
        "path": "/tmp/tck-bundle-envelope.json"
      }
      """
    Then the response "state" has field "session_id" of type string
    And the response "state" has field "model_id" of type string
    And the response "state" has field "status" of type string
    And the response "state" has field "position" of type object
    And the response "state" has field "tick_id" of type integer
    And the response "state" has field "active_probes" of type array
    And the response "state" has field "checkpoints" of type array
    And the response "state" has field "available_actions" of type array
    And the response "state.status" is "stopped"
