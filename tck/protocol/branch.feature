Feature: Worldline branching

  @deferred
  Scenario: branch.fork creates a new branch
    Given a session with checkpoint "ckpt-1"
    When the client sends branch.fork from "ckpt-1"
    Then the response includes a branch_id
    And a branch.created event is emitted

  @deferred
  Scenario: branch.compare returns divergence metrics
    Given two branches "branch-a" and "branch-b" from the same checkpoint
    When the client sends branch.compare for "branch-a" and "branch-b"
    Then the response includes cosine_similarity, max_relative_error, kl_divergence
    And per_layer_norm_delta is an array with one entry per layer

  @deferred
  Scenario: branch.drop releases resources
    Given a live branch "branch-x"
    When the client sends branch.drop for "branch-x"
    Then a branch.tier_changed event is emitted with tier "dropped"
