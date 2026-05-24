@shm
Feature: Shared memory data plane for tensor transfer
  The shared memory ring buffer (DoomRing) provides an optimized
  transport for tensor data between the worker and daemon processes.
  It is transparent to the client — the protocol contract does not
  change. When shared memory is unavailable or a tensor exceeds the
  slot size, the system falls back to base64 encoding.

  Background:
    Given the session is initialized and a model is attached
    And the client has stepped forward at least 1 tick at "component" granularity
    And the session is in "stopped" state

  # ── Capability advertisement ───────────────────────────────────────

  @deferred
  Scenario: Capabilities include shared_memory_supported flag
    When the client sends "initialize" with:
      """json
      {
        "client_name": "tck-runner",
        "protocol_version": "0.3.0"
      }
      """
    Then the capabilities field "shared_memory_supported" is true

  # ── Transparent tensor transfer ────────────────────────────────────

  @deferred
  Scenario: Inspect returns valid tensor summary with shm transport
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    Then the response status is "stopped"
    And the response "data.tensors" is an array with at least 1 element
    And the first tensor in "data.tensors" has field "tensor_id" of type string
    And the first tensor "tensor_id" matches the pattern "^[0-9a-f]{64}$"
    And the first tensor in "data.tensors" has field "stats" of type object
    And the first tensor "stats" includes "mean", "std", "min", "max"

  @deferred
  Scenario: Inspect returns valid tensor slice with shm transport
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | slice                        |
      | slices | [[0, 10]]                    |
    Then the response status is "stopped"
    And the response "data.slice_data" is a non-null base64-encoded string

  # ── Fallback behavior ──────────────────────────────────────────────

  @deferred
  Scenario: Inspect succeeds when shared memory is unavailable
    Given the shared memory data plane is disabled
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    Then the response status is "stopped"
    And the response "data.tensors" is an array with at least 1 element
    And the first tensor in "data.tensors" has field "tensor_id" of type string
