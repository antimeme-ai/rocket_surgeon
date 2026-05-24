@session-export
Feature: Session bundle export
  The rocket/session.export verb assembles a tar.gz archive containing
  session artifacts: manifest, interventions, protocol trace, and
  optionally captured tensors.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.1.0"
    And a model "llama-7b" is attached
    And the session has been stepped to tick 0 at layer 0

  # ── Happy path ───────────────────────────────────────────────────

  Scenario: Export produces a tar.gz with manifest and interventions
    When the client sends "rocket/session.export" with:
      """json
      {
        "path": "/tmp/test-bundle.tar.gz",
        "include_tensors": false
      }
      """
    Then the response status is "stopped"
    And the response data field "path" is "/tmp/test-bundle.tar.gz"
    And the response data field "size_bytes" is greater than 0
    And the response data field "artifact_count" is at least 2
    And the file "/tmp/test-bundle.tar.gz" is a valid gzip-compressed tar archive
    And the archive contains "manifest.json"
    And the archive contains "interventions.json"

  Scenario: Manifest contains session metadata
    When the client sends "rocket/session.export" with:
      """json
      {
        "path": "/tmp/manifest-check.tar.gz",
        "include_tensors": false
      }
      """
    Then the archive entry "manifest.json" is valid JSON
    And the manifest contains key "session_id"
    And the manifest contains key "bundle_schema_version"

  Scenario: Export includes protocol trace when messages have been exchanged
    When the client sends "rocket/session.export" with:
      """json
      {
        "path": "/tmp/trace-check.tar.gz",
        "include_tensors": false
      }
      """
    Then the archive contains "protocol-trace.jsonl"

  Scenario: Export with include_tensors true includes tensor files
    Given the client has inspected a tensor at "llama:0:0:attn.o_proj:output"
    When the client sends "rocket/session.export" with:
      """json
      {
        "path": "/tmp/tensors-check.tar.gz",
        "include_tensors": true
      }
      """
    Then the archive contains at least one entry matching "tensors/*.bin"

  Scenario: Export with include_tensors defaults to true
    When the client sends "rocket/session.export" with:
      """json
      {
        "path": "/tmp/default-tensors.tar.gz"
      }
      """
    Then the response data field "path" is "/tmp/default-tensors.tar.gz"

  Scenario: Export produces all required artifacts
    When the client sends "rocket/session.export" with:
      """json
      {
        "path": "/tmp/full-bundle.tar.gz",
        "include_tensors": false
      }
      """
    Then the response status is "stopped"
    And the file "/tmp/full-bundle.tar.gz" is a valid gzip-compressed tar archive
    And the archive contains "manifest.json"
    And the archive contains "interventions.json"
    And the archive contains "protocol-trace.jsonl"
    And the archive contains "env.json"
    And the archive contains "model-info.json"
    And the archive contains "prompt.json"
    And the archive contains "bookmarks.json"
    And the response data field "artifact_count" is at least 8

  # ── Interventions in bundle ──────────────────────────────────────

  Scenario: Export after registering interventions includes them in bundle
    Given an active intervention "iv-scale-1" of type "scale" on "llama:0:12:attn.o_proj:output"
    When the client sends "rocket/session.export" with:
      """json
      {
        "path": "/tmp/with-interventions.tar.gz",
        "include_tensors": false
      }
      """
    Then the archive entry "interventions.json" is valid JSON
    And the interventions array contains an entry with id "iv-scale-1"

  # ── Error cases ──────────────────────────────────────────────────

  Scenario: Export while session is not in stopped state returns INVALID_STATE
    Given the session is in "stepping" state
    When the client sends "rocket/session.export" with:
      """json
      {
        "path": "/tmp/bad-state.tar.gz",
        "include_tensors": false
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_STATE"
    And the error "data.severity" is "recoverable"

  Scenario: Export with missing path parameter returns INVALID_PARAMS
    When the client sends "rocket/session.export" with:
      """json
      {
        "include_tensors": false
      }
      """
    Then the response is a JSON-RPC error

  Scenario: Export before model attached returns INVALID_STATE
    Given the session is initialized but no model is attached
    When the client sends "rocket/session.export" with:
      """json
      {
        "path": "/tmp/no-model.tar.gz",
        "include_tensors": false
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_STATE"
