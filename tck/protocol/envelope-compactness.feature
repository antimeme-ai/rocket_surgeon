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

  Scenario: Inspect default envelope is full
    Given a stopped session with a captured tensor
    When the client sends inspect with no envelope field
    Then the response includes the complete SessionState
    And the response includes the inspected tensors

  Scenario: Inspect position-only envelope
    Given a stopped session with a captured tensor
    When the client sends inspect with envelope "position"
    Then the response includes status and tick position
    And the response does not include active_probes or checkpoints
    And the response includes the inspected tensors

  Scenario: Inspect no envelope
    Given a stopped session with a captured tensor
    When the client sends inspect with envelope "none"
    Then the response includes only the data payload
    And no SessionState fields are present

  Scenario: View default envelope is full
    Given a stopped session with view data available
    When the client sends view with no envelope field
    Then the response includes the complete SessionState
    And the response includes the view data

  Scenario: View position-only envelope
    Given a stopped session with view data available
    When the client sends view with envelope "position"
    Then the response includes status and tick position
    And the response does not include active_probes or checkpoints
    And the response includes the view data

  Scenario: View no envelope
    Given a stopped session with view data available
    When the client sends view with envelope "none"
    Then the response includes only the data payload
    And no SessionState fields are present
