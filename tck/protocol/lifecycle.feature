@lifecycle
Feature: Session lifecycle — initialize, attach, detach
  The protocol state machine begins at UNINITIALIZED and moves through
  INITIALIZED, ATTACHING, STOPPED, DETACHING, and back to INITIALIZED.
  Every lifecycle verb enforces valid transitions and rejects invalid ones
  with structured error responses.

  # ── Happy path ──────────────────────────────────────────────────────

  Scenario: Initialize with valid client_name and protocol_version
    Given the session is in "uninitialized" state
    When the client sends "initialize" with:
      | client_name      | rocket-tui |
      | protocol_version | 0.1.0      |
    Then the response status is "initialized"
    And the response contains a "data.capabilities" object
    And the response "data.capabilities.protocol_version" is "0.1.0"

  Scenario: Initialize response contains negotiated capabilities
    Given the session is in "uninitialized" state
    When the client sends "initialize" with:
      | client_name      | claude-agent |
      | protocol_version | 0.1.0        |
    Then the response "data.capabilities" includes at least:
      | field                  | type    |
      | protocol_version       | string  |
      | supports_reverse_step  | boolean |
      | supports_checkpointing | boolean |
      | supports_moe           | boolean |
      | supports_backward      | boolean |
      | supports_sae           | boolean |
      | execution_mode         | string  |
      | parallelism            | string  |
      | tick_granularities     | array   |
      | intervention_types     | array   |
      | built_in_views         | array   |
      | head_granularity       | string  |
      | transports             | array   |
      | wire_formats           | array   |
      | max_response_bytes     | integer |
    And the response "data.capabilities.protocol_version" is "0.1.0"

  Scenario: Attach with model_path and model_family reaches stopped state
    Given the session is in "initialized" state
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response status is "stopped"
    And the response "state.model_id" is not null
    And the response "data.model_id" is a non-empty string
    And the response "data.model_family" is "llama"
    And the response "data.num_layers" is a positive integer
    And the response "data.num_heads" is a positive integer
    And the response "data.hidden_dim" is a positive integer
    And the response "data.num_ranks" is a positive integer
    And the response "data.capabilities" includes "model_family"

  Scenario: Detach returns to initialized state
    Given the session is in "stopped" state with model "llama"
    When the client sends "detach" with no parameters
    Then the response status is "initialized"
    And the response "state.model_id" is null
    And the response "data.detached_model_id" is a non-empty string

  Scenario: Re-attach after detach succeeds
    Given the session is in "initialized" state after a previous detach
    When the client sends "attach" with:
      | model_path   | /models/mixtral-8x7b |
      | model_family | mixtral              |
    Then the response status is "stopped"
    And the response "data.model_family" is "mixtral"

  Scenario: Full lifecycle round-trip
    Given the session is in "uninitialized" state
    When the client sends "initialize" with:
      | client_name      | tck-harness |
      | protocol_version | 0.1.0       |
    Then the response status is "initialized"
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response status is "stopped"
    When the client sends "detach" with no parameters
    Then the response status is "initialized"
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response status is "stopped"

  # ── Error paths ─────────────────────────────────────────────────────

  Scenario: Double initialize returns INVALID_STATE error
    Given the session is in "initialized" state
    When the client sends "initialize" with:
      | client_name      | rocket-tui |
      | protocol_version | 0.1.0      |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_STATE"
    And the error "data.severity" is "recoverable"
    And the error "data.current_state" is "initialized"

  Scenario: Attach without initialize returns INVALID_STATE error
    Given the session is in "uninitialized" state
    When the client sends "attach" with:
      | model_path   | /models/llama-7b |
      | model_family | llama            |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "INVALID_STATE"
    And the error "data.severity" is "recoverable"
    And the error "data.current_state" is "uninitialized"
    And the error "data.valid_states" includes "initialized"

  Scenario: Double attach returns MODEL_ALREADY_ATTACHED error
    Given the session is in "stopped" state with model "llama"
    When the client sends "attach" with:
      | model_path   | /models/gpt-neox-20b |
      | model_family | gpt-neox             |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "MODEL_ALREADY_ATTACHED"
    And the error "data.severity" is "recoverable"

  Scenario: Detach without attach returns MODEL_NOT_ATTACHED error
    Given the session is in "initialized" state
    When the client sends "detach" with no parameters
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "MODEL_NOT_ATTACHED"
    And the error "data.severity" is "recoverable"

  # ── Backend integration (BEAD-0008) ─────────────────────────────────

  Scenario: Attach response carries real backend model metadata
    Given the session is in "initialized" state
    And the backend worker reports a model with 2 layers and 4 heads
    When the client sends "attach" with:
      | model_path   | hf-internal-testing/tiny-random-LlamaForCausalLM |
      | model_family | llama                                            |
    Then the response status is "stopped"
    And the response "data.num_layers" is 2
    And the response "data.num_heads" is 4
    And the response "data.hidden_dim" matches the backend report

  Scenario: Attach with broken backend returns BACKEND_ATTACH_FAILED error
    Given the session is in "initialized" state
    And the backend worker cannot load the requested model
    When the client sends "attach" with:
      | model_path   | /models/does-not-exist |
      | model_family | llama                  |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "BACKEND_ATTACH_FAILED"
    And the error "data.severity" is "recoverable"
    And the error "data.context" includes the backend error message
    And the session remains in "initialized" state

  Scenario: Attach response model_family reflects worker, not client claim
    Given the session is in "initialized" state
    And the backend worker reports model_type "mixtral"
    When the client sends "attach" with:
      | model_path   | hf-internal-testing/tiny-random-LlamaForCausalLM |
      | model_family | llama                                            |
    Then the response status is "stopped"
    And the response "data.model_family" is "mixtral"

  Scenario: Worker returning zero-valued metadata is rejected
    Given the session is in "initialized" state
    And the backend worker reports num_layers=0
    When the client sends "attach" with:
      | model_path   | /models/buggy-worker |
      | model_family | llama                |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "BACKEND_ATTACH_FAILED"
    And the error "data.context.backend_error" mentions "invalid metadata"

  Scenario: Duplicate attach rejected without spawning a new worker
    Given the session is in "stopped" state with model "llama"
    When the client sends "attach" with:
      | model_path   | /models/some-other-model |
      | model_family | llama                    |
    Then the response is a JSON-RPC error
    And the error "data.error_code" is "MODEL_ALREADY_ATTACHED"
    And no orchestrator subprocess was spawned for the duplicate request
