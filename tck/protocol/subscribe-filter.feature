Feature: Subscribe with event filtering
  Clients can filter which events they receive to reduce
  notification volume.

  Scenario: Filter by event type
    Given an attached session
    When the client subscribes with filter events ["tick.stopped"]
    Then the client receives tick.stopped events
    And the client does not receive probe.fired events

  Scenario: Filter by layer range
    Given an attached session with probes on layers 0-31
    When the client subscribes with filter layers [10, 11, 12]
    Then probe.fired events only arrive for layers 10, 11, 12

  Scenario: Filter by component pattern
    Given an attached session
    When the client subscribes with filter components ["attn.*"]
    Then tick.stopped events only arrive for attention components
