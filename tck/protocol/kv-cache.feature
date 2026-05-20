Feature: KV cache protocol surface

  Scenario: kv.read returns cache slice
    Given an attached session with KV cache populated
    When the client sends kv.read with layers [0, 1], positions [0, 1, 2]
    Then the response includes cache entries with norms per layer and position

  Scenario: kv.read with evicted position
    Given a session where position 5 was evicted
    When the client sends kv.read for position 5
    Then the error code is "KV_EVICTED"
    And error context includes evicted_at_tick and nearest_checkpoint
