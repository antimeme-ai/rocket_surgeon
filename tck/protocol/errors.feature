@errors
Feature: Error contract — every error code has a trigger, every response has structure
  All 18 error codes in the protocol registry (errors.json) must be triggerable.
  Every error response conforms to the ErrorData schema: error_code (string),
  numeric_code (integer), severity ("fatal" or "recoverable"), and suggestion
  (non-empty string). INVALID_STATE errors additionally include current_state
  and valid_states fields.

  # ── Error code triggers ──────────────────────────────────────────

  Scenario: INVALID_STATE — attach without initializing
    Given the session is in "uninitialized" state
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_STATE"
    And the error "data.severity" is "recoverable"
    And the error "data.current_state" is "uninitialized"

  Scenario: INVALID_TARGET — intervene with nonexistent target
    Given the session is in "stopped" state with model "llama"
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "bad-target",
          "type": "ablate",
          "target": "llama:0:999:nonexistent.component:output",
          "params": {}
        }
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_TARGET"
    And the error "data.severity" is "recoverable"

  Scenario: INVALID_RECIPE — intervene with missing params
    Given the session is in "stopped" state with model "llama"
    When the client sends "rocket/intervene" with:
      """json
      {
        "action": "set",
        "recipe": {
          "id": "bad-recipe",
          "type": "scale",
          "target": "llama:0:0:attn.o_proj:output",
          "params": {}
        }
      }
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_RECIPE"
    And the error "data.severity" is "recoverable"

  Scenario: MODEL_NOT_ATTACHED — step before calling attach
    Given the session is in "initialized" state
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "MODEL_NOT_ATTACHED"
    And the error "data.severity" is "recoverable"

  @deferred
  Scenario: TENSOR_NOT_FOUND — inspect slice with nonexistent tensor_id
    Given the session is in "stopped" state with model "llama"
    When the client sends "rocket/inspect" with:
      | tensor_id | 0000000000000000000000000000000000000000000000000000000000000000 |
      | detail    | slice                                                          |
      | slice     | [0, 10]                                                        |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "TENSOR_NOT_FOUND"
    And the error "data.severity" is "recoverable"

  Scenario: CHECKPOINT_NOT_FOUND — restore checkpoint "nonexistent"
    Given the session is in "stopped" state with model "llama"
    When the client sends "rocket/checkpoint" with:
      | action        | restore     |
      | checkpoint_id | nonexistent |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "CHECKPOINT_NOT_FOUND"
    And the error "data.severity" is "recoverable"

  Scenario: PROBE_NOT_FOUND — enable probe "nonexistent"
    Given the session is in "stopped" state with model "llama"
    When the client sends "rocket/probe" with:
      | action   | enable      |
      | probe_id | nonexistent |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "PROBE_NOT_FOUND"
    And the error "data.severity" is "recoverable"

  @deferred
  Scenario: CAPABILITY_NOT_SUPPORTED — call rocket/checkpoint when supports_checkpointing=false
    Given the session is in "stopped" state with model "llama"
    And the server capability "supports_checkpointing" is false
    When the client sends "rocket/checkpoint" with:
      | action | create     |
      | tier   | activation |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "CAPABILITY_NOT_SUPPORTED"
    And the error "data.severity" is "recoverable"

  @deferred
  Scenario: SLICE_OUT_OF_BOUNDS — inspect slice [0, 999999] on a small tensor
    Given the session is in "stopped" state with model "llama"
    And the session has a tensor "t-small" with shape [1, 32, 128]
    When the client sends "rocket/inspect" with:
      | tensor_id | t-small       |
      | detail    | slice         |
      | slice     | [0, 999999]   |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "SLICE_OUT_OF_BOUNDS"
    And the error "data.severity" is "recoverable"

  @deferred
  Scenario: RESPONSE_TOO_LARGE — inspect detail=full on large tensor
    Given the session is in "stopped" state with model "llama"
    And the session has a tensor "t-huge" with shape [1, 32, 4096, 128]
    When the client sends "rocket/inspect" with:
      | tensor_id | t-huge |
      | detail    | full   |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "RESPONSE_TOO_LARGE"
    And the error "data.severity" is "recoverable"

  @deferred @integration
  Scenario: HOST_ERROR — host process crash (simulated)
    Given the session is in "stopped" state with model "llama"
    And the host process is configured to simulate a crash on next step
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "HOST_ERROR"
    And the error "data.severity" is "fatal"

  @deferred @integration
  Scenario: GPU_OOM — GPU out of memory (simulated)
    Given the session is in "stopped" state with model "llama"
    And the GPU memory is configured to simulate OOM on next step
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "GPU_OOM"
    And the error "data.severity" is "fatal"

  @deferred @phase5 @integration
  Scenario: NCCL_TIMEOUT — NCCL timeout (simulated)
    Given the session is in "stopped" state with model "llama" on 2 ranks
    And the NCCL backend is configured to simulate a timeout
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "NCCL_TIMEOUT"
    And the error "data.severity" is "fatal"

  @deferred @phase3
  Scenario: REPLAY_DIVERGENCE — replay with mutation causing divergence
    Given the session is in "stopped" state with model "llama"
    And the server capability "supports_checkpointing" is true
    And the session has been stepped to tick 10 at layer 8
    And the session has an activation checkpoint "ckpt-pre" at tick 3 layer 2
    When the client sends "rocket/replay" with:
      | from_checkpoint | ckpt-pre |
      | verify          | true     |
    And the request includes "interventions" array:
      """json
      [
        {
          "id": "nuke-residual",
          "type": "scale",
          "target": "llama:0:4:residual_post:output",
          "params": {"factor": 1000.0}
        }
      ]
      """
    Then the response status is "stopped"
    And the response "data.divergences" is a non-empty array
    And the response "data.divergences[0].cosine_similarity" is less than 0.99995

  Scenario: UNSUPPORTED_MODEL — attach with model_family "unknown_arch"
    Given the session is in "initialized" state
    When the client sends "attach" with:
      | model_path   | /models/mystery-model |
      | model_family | unknown_arch          |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "UNSUPPORTED_MODEL"
    And the error "data.severity" is "recoverable"

  @deferred
  Scenario: COMPILED_MODEL — attach a torch.compile model
    Given the session is in "initialized" state
    When the client sends "attach" with:
      | model_path      | /models/llama-7b-compiled |
      | model_family    | llama                     |
      | execution_mode  | compiled                  |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "COMPILED_MODEL"
    And the error "data.severity" is "recoverable"

  Scenario: MODEL_ALREADY_ATTACHED — call attach twice without detach
    Given the session is in "stopped" state with model "llama"
    When the client sends "attach" with:
      | model_path   | /models/gpt-neox-20b |
      | model_family | gpt-neox             |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "MODEL_ALREADY_ATTACHED"
    And the error "data.severity" is "recoverable"

  Scenario: INVALID_PARAMS — send malformed JSON-RPC params (missing required field)
    Given the session is in "initialized" state
    When the client sends "attach" with:
      """json
      {}
      """
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_PARAMS"
    And the error "data.severity" is "recoverable"

  # ── Error response structure contract ──────────────────────────────

  Scenario: Every error response includes error_code string
    Given the session is in "initialized" state
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is a non-empty string

  Scenario: Every error response includes numeric_code integer
    Given the session is in "initialized" state
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response is a JSON-RPC error
    And the error "data.numeric_code" is an integer
    And the error "code" is an integer
    And the error "code" equals the error "data.numeric_code"

  Scenario: Every error response includes severity
    Given the session is in "initialized" state
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response is a JSON-RPC error
    And the error "data.severity" is one of "fatal", "recoverable"

  Scenario: Every error response includes suggestion string
    Given the session is in "initialized" state
    When the client sends "rocket/step" with:
      | direction   | forward   |
      | count       | 1         |
      | granularity | component |
    Then the response is a JSON-RPC error
    And the error "data.suggestion" is a non-empty string

  Scenario: INVALID_STATE error includes current_state and valid_states
    Given the session is in "uninitialized" state
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_STATE"
    And the error "data.current_state" is "uninitialized"
    And the error "data.valid_states" is a non-empty array
    And each entry in error "data.valid_states" is a valid session status
