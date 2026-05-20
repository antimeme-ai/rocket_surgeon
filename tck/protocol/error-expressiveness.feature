Feature: Error expressiveness
  Every error carries what happened, why, and what to do about it.

  Scenario: ErrorData includes recovery_hint
    Given any error response
    Then the error data has a "recovery_hint" field (string or null)

  Scenario: INVALID_TARGET includes nearest matches
    Given an attached session
    When the client inspects target "llama:*:12:attn.out_proj:output"
    Then the error code is "INVALID_TARGET"
    And error context includes "attempted" = "attn.out_proj"
    And error context includes "nearest_matches" as a non-empty array
    And error context includes "valid_components_at_layer" as an array

  Scenario: E_VRAM_EXHAUSTED includes memory accounting
    Given a session near VRAM capacity
    When an operation would exceed the VRAM headroom
    Then the error code is "VRAM_EXHAUSTED"
    And error context includes "used_mb", "total_mb", "headroom_mb"
    And error context includes "per_branch" array with id and size_mb per branch
    And error context includes "recommendation" string
