Feature: Batch experiment sweep

  @deferred
  Scenario: Sweep runs multiple trials from a checkpoint
    Given a session with checkpoint "ckpt-clean"
    When the client sends sweep with 3 trial specs
    Then the response includes results keyed by trial index
    And each result includes collected tensor summaries

  @deferred
  Scenario: Sweep streams trial_complete events
    Given a subscribed session running a sweep
    Then a sweep.trial_complete event fires after each trial
