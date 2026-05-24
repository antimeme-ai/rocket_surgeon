@tensors
Feature: Tensor handle lifecycle — addressing, summaries, slicing, eviction
  Captured tensors are assigned content-addressable tensor_ids (BLAKE3 hash,
  64 lowercase hex characters). Identical tensor content always yields the
  same tensor_id regardless of probe point. Tensor summaries include shape,
  dtype, and 8 stat fields (min, max, mean, std, abs_max, sparsity, l2_norm,
  histogram). Slice access returns base64-encoded raw data. The tensor store
  has a finite capacity; when exceeded, the oldest tensors are evicted and
  subsequent access returns TENSOR_NOT_FOUND.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.3.0"
    And a model "llama-7b" is attached
    And the client has stepped forward at least 1 tick at "component" granularity
    And the session is in "stopped" state

  # ── Content-addressable tensor_id ─────────────────────────────────

  @deferred
  @deferred
  Scenario: Captured tensor gets a content-addressable tensor_id (BLAKE3 hash)
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    Then the response status is "stopped"
    And the response "data.tensors" is an array with at least 1 element
    And the first tensor in "data.tensors" has field "tensor_id" of type string
    And the first tensor "tensor_id" matches the pattern "^[0-9a-f]{64}$"

  @deferred
  @deferred
  Scenario: Same tensor content at different probe points yields same tensor_id
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

  @deferred
  @deferred
  Scenario: Different tensor content yields different tensor_ids
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    And the first tensor "tensor_id" is saved as "id_attn"
    And the client sends "rocket/inspect" with:
      | target | llama:0:0:mlp.down_proj:output |
      | detail | summary                        |
    And the first tensor "tensor_id" is saved as "id_mlp"
    Then "id_attn" does not equal "id_mlp"

  # ── Tensor summary ────────────────────────────────────────────────

  @deferred
  @deferred
  Scenario: Tensor summary includes shape, dtype, and all 8 stat fields
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    Then the response "data.tensors" is an array with at least 1 element
    And the first tensor in "data.tensors" has field "shape" of type array
    And the first tensor in "data.tensors" has field "dtype" of type string
    And the first tensor in "data.tensors" has field "stats" of type object
    And the first tensor "stats" includes all 8 required fields:
      | field     |
      | min       |
      | max       |
      | mean      |
      | std       |
      | abs_max   |
      | sparsity  |
      | l2_norm   |
      | histogram |
    And the first tensor "stats.min" is of type number
    And the first tensor "stats.max" is of type number
    And the first tensor "stats.mean" is of type number
    And the first tensor "stats.std" is of type number
    And the first tensor "stats.abs_max" is of type number
    And the first tensor "stats.sparsity" is of type number
    And the first tensor "stats.l2_norm" is of type number
    And the first tensor "stats.histogram" is of type object

  @deferred
  @deferred
  Scenario: Tensor stats satisfy basic invariants
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    Then the first tensor "stats.min" <= the first tensor "stats.mean"
    And the first tensor "stats.mean" <= the first tensor "stats.max"
    And the first tensor "stats.abs_max" >= 0
    And the first tensor "stats.std" >= 0
    And the first tensor "stats.sparsity" >= 0
    And the first tensor "stats.sparsity" <= 1.0
    And the first tensor "stats.l2_norm" >= 0
    And the first tensor "stats.histogram.bins" >= 1

  # ── Tensor slice ──────────────────────────────────────────────────

  @deferred
  @deferred
  Scenario: Tensor slice returns base64-encoded raw data
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | slice                        |
      | slices | [[0, 10]]                    |
    Then the response status is "stopped"
    And the response "data.slice_data" is a non-null base64-encoded string
    And the response "data.slice_shape" is an array
    And the response "data.slice_dtype" is a non-empty string

  @deferred
  @deferred
  Scenario: Tensor slice out of bounds returns SLICE_OUT_OF_BOUNDS
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | slice                        |
      | slices | [[0, 999999999]]             |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "SLICE_OUT_OF_BOUNDS"
    And the error "data.severity" is "recoverable"

  @deferred
  @deferred
  Scenario: Tensor slice with negative indices returns SLICE_OUT_OF_BOUNDS
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | slice                        |
      | slices | [[-1, 10]]                   |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "SLICE_OUT_OF_BOUNDS"
    And the error "data.severity" is "recoverable"

  # ── Tensor eviction ───────────────────────────────────────────────

  @deferred
  @deferred
  Scenario: After exceeding store capacity oldest tensors are evicted
    Given the tensor store capacity is configured to hold at most 4 tensors
    When the client captures 6 tensors by inspecting 6 distinct components
    And the first captured tensor_id is saved as "oldest_id"
    Then the tensor store contains at most 4 tensors
    And the tensor "oldest_id" has been evicted

  @deferred
  @deferred
  Scenario: Evicted tensor_id returns TENSOR_NOT_FOUND
    Given the tensor store capacity is configured to hold at most 4 tensors
    When the client captures 6 tensors by inspecting 6 distinct components
    And the first captured tensor_id is saved as "evicted_id"
    When the client sends "rocket/inspect" with:
      """json
      {
        "tensor_id": "evicted_id",
        "detail": "slice",
        "slices": [[0, 10]]
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "TENSOR_NOT_FOUND"
    And the error "data.severity" is "recoverable"

  # ── dtype preservation ────────────────────────────────────────────

  @deferred
  @deferred
  Scenario Outline: Tensor dtype preservation across supported dtypes
    Given a model "llama-7b" is attached with dtype "<dtype>"
    And the client has stepped forward at least 1 tick at "component" granularity
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | summary                      |
    Then the response "data.tensors" is an array with at least 1 element
    And the first tensor in "data.tensors" has field "dtype" equal to "<dtype>"

    Examples:
      | dtype    |
      | float16  |
      | float32  |
      | bfloat16 |

  @deferred
  @deferred
  Scenario: Tensor slice data preserves dtype encoding
    Given a model "llama-7b" is attached with dtype "float16"
    And the client has stepped forward at least 1 tick at "component" granularity
    When the client sends "rocket/inspect" with:
      | target | llama:0:0:attn.o_proj:output |
      | detail | slice                        |
      | slices | [[0, 4]]                     |
    Then the response "data.slice_data" is a non-null base64-encoded string
    And the response "data.slice_dtype" is "float16"
    And the decoded slice data length in bytes equals 4 * 2
