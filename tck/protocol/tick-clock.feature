Feature: Three-clock tick model
  The tick model carries three incommensurable clocks:
  token (sequence position), operator (within-token traversal),
  and wall (nanosecond real time).

  Scenario: TickPosition carries all three clocks
    Given a session in Stopped state at layer 5 component "attn.q"
    Then the tick position has a "clock" field
    And clock.token is the current token position
    And clock.operator is the within-token traversal index
    And clock.wall_ns is a non-zero nanosecond timestamp

  Scenario: tick_id is alias for clock.operator
    Given a tick position with clock.operator = 42
    Then tick_id equals 42

  Scenario: clock.operator resets each token
    Given a session stepping through token 0
    When the session advances to token 1
    Then clock.token increments by 1
    And clock.operator resets to 0

  Scenario: Backward compatibility — tick_id still present
    Given a response from protocol version 0.3.0
    Then the tick position JSON contains both "tick_id" and "clock" fields
    And tick_id equals clock.operator
