@mailbox
Feature: Lock-based mailbox barrier for tick-by-tick stepping
  The mailbox is a single-slot synchronization primitive using
  _thread.allocate_lock(). It enables the barrier pattern where a
  capture hook puts a value and blocks until the controller resumes it.
  Two mailboxes (result + resume) form the barrier pair.

  # ── Single-slot semantics ────────────────────────────────────────

  Scenario: Put then wait delivers the value
    Given a fresh mailbox
    When a value is put into the mailbox
    Then wait returns that value

  Scenario: Restore makes the mailbox reusable
    Given a fresh mailbox
    When a value is put into the mailbox
    And the value is consumed via wait
    And the mailbox is restored
    Then the mailbox can accept a new value

  Scenario: Get retrieves value without blocking
    Given a fresh mailbox
    When a value is put into the mailbox
    Then get returns that value without blocking

  # ── Barrier pattern ──────────────────────────────────────────────

  Scenario: Two-mailbox barrier cycle completes
    Given a result mailbox and a resume mailbox
    When a producer puts a value on the result mailbox and waits on the resume mailbox
    And the consumer waits on the result mailbox
    Then the consumer receives the produced value
    When the consumer restores the result mailbox and puts a signal on the resume mailbox
    Then the producer is unblocked

  Scenario: Multiple barrier rounds succeed
    Given a result mailbox and a resume mailbox
    When 3 barrier rounds are executed
    Then all 3 rounds complete successfully
