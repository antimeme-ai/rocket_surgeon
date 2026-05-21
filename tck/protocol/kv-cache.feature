Feature: KV cache protocol surface
  The kv.read and kv.intervene verbs expose the attention key/value cache.
  kv.read returns per-(layer, position, head) cache norms without mutating
  model state. kv.intervene performs surgery on cache slots (zero, scale,
  evict, pin) between ticks. Reading an evicted position is a recoverable
  KV_EVICTED error carrying the eviction tick and nearest checkpoint.

  Scenario: kv.read returns cache slice
    Given an attached session with KV cache populated
    When the client sends kv.read with layers [0, 1], positions [0, 1, 2]
    Then the response includes cache entries with norms per layer and position

  Scenario: kv.read with evicted position
    Given a session where position 5 was evicted
    When the client sends kv.read for position 5
    Then the error code is "KV_EVICTED"
    And error context includes evicted_at_tick and nearest_checkpoint

  Scenario: kv.intervene evicts a position
    Given an attached session with KV cache populated
    When the client sends kv.intervene with op "evict" on position 5
    Then the response reports the applied op "evict"
    And the response reports a positive slots_modified count

  Scenario: kv.intervene then kv.read sees the eviction
    Given an attached session with KV cache populated
    When the client sends kv.intervene with op "evict" on position 5
    And the client sends kv.read for position 5
    Then the error code is "KV_EVICTED"

  Scenario: kv.intervene pin protects a position from eviction
    Given an attached session with KV cache populated
    When the client sends kv.intervene with op "pin" on position 5
    Then the response reports the applied op "pin"

  Scenario: kv.intervene with empty layers is rejected
    Given an attached session with KV cache populated
    When the client sends kv.intervene with op "zero" and empty layers
    Then the error code is "INVALID_PARAMS"
