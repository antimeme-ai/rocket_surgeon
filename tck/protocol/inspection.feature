@inspection
Feature: Tensor inspection — rocket/inspect verb
  The inspect verb reads tensor data at a probe point without modifying
  model state. It returns TensorSummary objects (always), optional slice
  data, and optional built-in view results. Transitions STOPPED ->
  INSPECTING -> STOPPED (transient).

  Background:
    Given the session is initialized and a model is attached
    And the client has stepped forward at least 1 tick at "component" granularity
    And the session is in "stopped" state

  # ── Summary inspection ─────────────────────────────────────────────
  Scenario: Inspect with detail=summary returns TensorSummary
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    Then the response status is "stopped"
    And the response "data.tensors" is an array with at least 1 element
    And the first tensor in "data.tensors" has field "shape" of type array
    And the first tensor in "data.tensors" has field "dtype" of type string
    And the first tensor in "data.tensors" has field "stats" of type object
    And the first tensor "stats" includes "mean", "std", "min", "max"
    And the first tensor "stats" includes "abs_max", "sparsity", "l2_norm", "histogram"

  Scenario: Inspect defaults to summary when detail is omitted
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
    Then the response status is "stopped"
    And the response "data.tensors" is an array with at least 1 element
    And the response "data.slice_data" is null
    And the response "data.view_result" is null

  # ── Target matching ────────────────────────────────────────────────
  Scenario: Inspect with target matching attn.o_proj returns tensor for that component
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    Then the response "data.tensors" has exactly 1 element

  Scenario: Inspect with wildcard target returns multiple tensors
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:*:output |
      | detail | summary            |
    Then the response "data.tensors" is an array with at least 2 elements

  Scenario: Inspect nonexistent target returns INVALID_TARGET error
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:nonexistent_component:output |
      | detail | summary                                |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_TARGET"
    And the error "data.severity" is "recoverable"

  # ── Slice inspection ───────────────────────────────────────────────
  Scenario: Inspect with detail=slice and valid slices returns slice_data
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | slice                        |
      | slices | [[0, 10]]                    |
    Then the response status is "stopped"
    And the response "data.slice_data" is a non-null base64-encoded string
    And the response "data.tensors" is an array with at least 1 element

  Scenario: Inspect with slice out of bounds returns SLICE_OUT_OF_BOUNDS error
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | slice                        |
      | slices | [[0, 999999999]]             |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "SLICE_OUT_OF_BOUNDS"
    And the error "data.severity" is "recoverable"

  # ── Built-in views ─────────────────────────────────────────────────
  Scenario: Inspect with built-in view "residual_stream_norm" returns view_result
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | view   | residual_stream_norm         |
    Then the response status is "stopped"
    And the response "data.view_result" is not null
    And the response "data.view_result" is an object

  # ── tensor_id contract ─────────────────────────────────────────────
  Scenario: TensorSummary includes tensor_id as BLAKE3 hash (64 hex chars)
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    Then the first tensor in "data.tensors" has field "tensor_id" of type string
    And the first tensor "tensor_id" matches the pattern "^[0-9a-f]{64}$"

  Scenario: Same tensor content at two probe points yields same tensor_id
    Given the model has two probe points observing the same tensor content
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    And the first tensor "tensor_id" is saved as "id_a"
    And the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj_alias:output |
      | detail | summary                            |
    And the first tensor "tensor_id" is saved as "id_b"
    Then "id_a" equals "id_b"

  # ── Response envelope ──────────────────────────────────────────────
  Scenario: Inspect response includes full SessionState in envelope
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    Then the response "state" has field "session_id" of type string
    And the response "state" has field "model_id" of type string
    And the response "state" has field "status" of type string
    And the response "state" has field "position" of type object
    And the response "state" has field "tick_id" of type integer
    And the response "state" has field "active_probes" of type array
    And the response "state" has field "checkpoints" of type array
    And the response "state" has field "available_actions" of type array
    And the response "state.status" is "stopped"
