Feature: Probe-point discovery

  @deferred
  Scenario: Discover with wildcard returns matching points
    Given an attached session with model_family "llama"
    When the client sends discover with pattern "llama:*:12:*:output"
    Then the response includes all layer 12 output components
    And each entry has canonical name, tensor_shape, and aliases

  @deferred
  Scenario: Discover with partial match suggests corrections
    Given an attached session
    When the client sends discover with pattern "llama:*:12:attn.out_proj:output"
    Then the response includes 0 exact matches
    And includes "suggestions" with nearest valid patterns
