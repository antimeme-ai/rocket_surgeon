@probes
Feature: Probe system for observing and asserting on the forward pass
  The rocket/probe verb allows clients to define, enable, disable,
  remove, and configure probe points that fire at matching positions
  in the forward pass. Probes persist across ticks until removed.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.1.0"
    And a model "llama-7b" is attached
    And the session has been stepped to tick 0 at layer 0

  # ── Define probes ─────────────────────────────────────────────────

  Scenario: Define probe with capture action returns probe_id
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "p-cap-1",
          "point": "llama:0:12:attn.o_proj:0:fwd",
          "action": "capture",
          "config": {
            "summary": true,
            "capture_tensor": false
          },
          "enabled": true,
          "priority": 0
        }
      }
      """
    Then the response status is "stopped"
    And the response data field "probe_id" is "p-cap-1"
    And the response data field "probes" contains an entry with id "p-cap-1"
    And the entry "p-cap-1" has action "capture"
    And the entry "p-cap-1" has enabled true

  Scenario: Define probe with six-level hierarchical point pattern
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "p-hier-1",
          "point": "llama:*:12:attn.o_proj:*:fwd",
          "action": "capture",
          "config": {"summary": true}
        }
      }
      """
    Then the response status is "stopped"
    And the response data field "probe_id" is "p-hier-1"
    And the entry "p-hier-1" has point "llama:*:12:attn.o_proj:*:fwd"

  # ── List probes ───────────────────────────────────────────────────

  Scenario: List probes returns all defined probes
    Given a defined probe "p-alpha" at point "llama:0:12:attn.o_proj:0:fwd" with action "capture"
    And a defined probe "p-beta" at point "llama:0:8:mlp:0:fwd" with action "trace"
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "list"
      }
      """
    Then the response status is "stopped"
    And the response data field "probes" has 2 entries
    And the response data field "probes" contains an entry with id "p-alpha"
    And the response data field "probes" contains an entry with id "p-beta"
    And the response data field "probe_id" is null

  # ── Enable / disable / remove ─────────────────────────────────────

  Scenario: Enable probe by ID
    Given a defined probe "p-dis-1" at point "llama:0:12:attn.o_proj:0:fwd" with action "capture" and enabled false
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "enable",
        "probe_id": "p-dis-1"
      }
      """
    Then the response status is "stopped"
    And the entry "p-dis-1" has enabled true

  Scenario: Disable probe by ID
    Given a defined probe "p-en-1" at point "llama:0:12:attn.o_proj:0:fwd" with action "capture" and enabled true
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "disable",
        "probe_id": "p-en-1"
      }
      """
    Then the response status is "stopped"
    And the entry "p-en-1" has enabled false

  Scenario: Remove probe by ID
    Given a defined probe "p-rm-1" at point "llama:0:12:attn.o_proj:0:fwd" with action "capture"
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "remove",
        "probe_id": "p-rm-1"
      }
      """
    Then the response status is "stopped"
    And the response data field "probes" does not contain an entry with id "p-rm-1"

  # ── Error cases ───────────────────────────────────────────────────

  Scenario: Enable nonexistent probe returns PROBE_NOT_FOUND
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "enable",
        "probe_id": "p-does-not-exist"
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "PROBE_NOT_FOUND"
    And the error "data.severity" is "recoverable"

  # ── Wildcard matching ─────────────────────────────────────────────

  Scenario: Wildcard probe matches all points
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "p-wildcard",
          "point": "*:*:*:*:*:*",
          "action": "capture",
          "config": {"summary": true, "capture_tensor": false}
        }
      }
      """
    Then the response data field "probe_id" is "p-wildcard"
    When the client sends "rocket/step" with direction "forward"
    And the client subscribes to "probe.fired" events
    Then probe "p-wildcard" fires for every component at every layer

  # ── Priority ordering ─────────────────────────────────────────────

  Scenario: Two probes at same point fire in priority order
    Given a defined probe "p-lo" at point "llama:0:12:attn.o_proj:0:fwd" with action "capture" and priority 0
    And a defined probe "p-hi" at point "llama:0:12:attn.o_proj:0:fwd" with action "capture" and priority 10
    When the client sends "rocket/step" with direction "forward"
    Then probe "p-lo" fires before probe "p-hi" at point "llama:0:12:attn.o_proj:0:fwd"

  # ── Assert action ─────────────────────────────────────────────────

  Scenario: Probe with assert action pauses execution on predicate violation
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "define",
        "probe": {
          "id": "p-assert-norm",
          "point": "llama:0:12:attn.o_proj:0:fwd",
          "action": "assert",
          "config": {
            "assertion": "norm < 100.0"
          }
        }
      }
      """
    Then the response data field "probe_id" is "p-assert-norm"
    When the forward pass reaches layer 12 and the norm exceeds 100.0
    Then execution pauses at "llama:0:12:attn.o_proj:0:fwd"
    And the session is in "stopped" state
    And a "probe.fired" event is emitted with probe_id "p-assert-norm"
    And the event includes the assertion violation details

  # ── Granularity control ───────────────────────────────────────────

  Scenario: set_granularity changes tick granularity for matching layers
    When the client sends "rocket/probe" with:
      """json
      {
        "action": "set_granularity",
        "scopes": [
          {"match": "layers[12]", "granularity": "component"},
          {"match": "layers[*]", "granularity": "layer"}
        ]
      }
      """
    Then the response status is "stopped"
    When the client sends "rocket/step" with direction "forward"
    Then layer 12 ticks at "component" granularity
    And all other layers tick at "layer" granularity

  # ── Probe and intervention composition ────────────────────────────

  Scenario: Probe at same point as intervention both execute
    Given an active intervention "iv-scale-comp" of type "scale" on "llama:0:12:attn.o_proj:0:fwd" with params {"factor": 0.5}
    And a defined probe "p-cap-comp" at point "llama:0:12:attn.o_proj:0:fwd" with action "capture" and priority 0
    When the client sends "rocket/step" with direction "forward"
    Then probe "p-cap-comp" captures the tensor at "llama:0:12:attn.o_proj:0:fwd"
    And intervention "iv-scale-comp" is applied at "llama:0:12:attn.o_proj:0:fwd"
    And the non-mutating probe "p-cap-comp" executes before the mutating intervention "iv-scale-comp"
