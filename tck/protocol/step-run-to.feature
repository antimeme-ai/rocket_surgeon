Feature: Step with run_to destination
  LLM clients can name a destination instead of counting ticks.

  Scenario: Step to a specific component
    Given an attached session at layer 0
    When the client sends step with run_to "llama:*:12:attn.o_proj:output"
    Then the session stops at layer 12 component "attn.o_proj"

  Scenario: Step to completion
    Given an attached session at layer 0
    When the client sends step with run_to "completion"
    Then the session stops at the final component of the final layer

  Scenario: run_to with invalid target
    Given an attached session
    When the client sends step with run_to "llama:*:99:nonexistent:output"
    Then the response is an error with code "INVALID_TARGET"
    And error details include "nearest_matches"
