Feature: Response envelope compactness
  Clients can negotiate response envelope verbosity to manage
  context window pressure.

  Scenario: Default envelope is full
    Given an attached session
    When the client sends step with no envelope field
    Then the response includes the complete SessionState

  Scenario: Position-only envelope
    Given an attached session
    When the client sends step with envelope "position"
    Then the response includes status and tick position
    And the response does not include active_probes or checkpoints

  Scenario: No envelope
    Given an attached session
    When the client sends step with envelope "none"
    Then the response includes only the data payload
    And no SessionState fields are present
