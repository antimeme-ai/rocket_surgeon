Feature: Model discovery via attach response
  The attach response provides everything an LLM or TUI needs
  to construct valid probe points without trial and error.

  @deferred
  Scenario: Attach response includes component vocabulary
    Given an initialized session
    When the client sends attach with model_family "llama"
    Then the response includes "component_vocabulary" as an array
    And each entry has "canonical" (string), "event" (string), "tensor_shape" (array)

  @deferred
  Scenario: Attach response includes module tree
    Given an attached session with model_family "llama"
    Then the response includes "module_tree" as an array of strings
    And the tree contains at least one entry per layer

  @deferred
  Scenario: Attach response includes alias table
    Given an attached session
    Then the response includes "alias_table" as an array
    And each entry has "canonical" and "aliases" fields
    And "blocks.0.attn.hook_q" appears as an alias for the layer 0 attn.q component

  @deferred
  Scenario: Attach response includes tick map
    Given an attached session
    Then the response includes "tick_map" as an object
    And tick_map contains an entry for granularity "component"
    And each granularity entry lists ticks per layer with ordering
