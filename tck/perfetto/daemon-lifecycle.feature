@perfetto @daemon-lifecycle
Feature: Perfetto trace integration with daemon lifecycle
  The daemon creates a .pftrace file when a model is attached,
  writes trace packets as tick and probe events occur during stepping,
  and flushes/closes the trace on detach. The resulting file must be
  valid Perfetto wire format with non-zero content.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.3.0"

  # ── Trace creation on attach ────────────────────────────────────

  @deferred
  Scenario: Attaching a model creates a .pftrace file
    When a model "gpt2" is attached
    Then the trace file exists at "{session_id}.pftrace" with non-zero size

  # ── Events during stepping ──────────────────────────────────────

  @deferred
  Scenario: Stepping produces trace packets from tick events
    Given a model "gpt2" is attached
    And the client subscribes to "tick.stopped" events
    When the client sends "rocket/step" with direction "forward"
    And the client sends "rocket/step" with direction "forward"
    And the client sends "rocket/step" with direction "forward"
    And the client sends "detach" with:
      """json
      {}
      """
    Then the trace file exists at "{session_id}.pftrace" with non-zero size
    And the .pftrace file is valid field-1 framed protobuf

  # ── Flush on detach ─────────────────────────────────────────────

  @deferred
  Scenario: Detaching flushes and closes the trace file
    Given a model "gpt2" is attached
    When the client sends "rocket/step" with direction "forward"
    And the client sends "detach" with:
      """json
      {}
      """
    Then the trace file exists at "{session_id}.pftrace" with non-zero size
    And the .pftrace file is valid field-1 framed protobuf

  # ── Minimum packet count ────────────────────────────────────────

  @deferred
  Scenario: Trace contains at least a process track and tick events
    Given a model "gpt2" is attached
    When the client sends "rocket/step" with direction "forward"
    And the client sends "rocket/step" with direction "forward"
    And the client sends "rocket/step" with direction "forward"
    And the client sends "detach" with:
      """json
      {}
      """
    Then the trace file exists at "{session_id}.pftrace" with non-zero size
    And the output contains at least 4 TracePackets

  # ── Clean shutdown ──────────────────────────────────────────────

  @deferred
  Scenario: All open slices are terminated on detach
    Given a model "gpt2" is attached
    When the client sends "rocket/step" with direction "forward"
    And the client sends "detach" with:
      """json
      {}
      """
    Then all open slices have been terminated with SLICE_END
