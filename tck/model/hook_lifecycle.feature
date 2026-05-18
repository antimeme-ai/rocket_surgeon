@hook_lifecycle
Feature: Hook installation, capture barrier, and forward pass lifecycle
  Sentinel hooks defeat PyTorch's fast-path optimization. Capture hooks
  use the mailbox barrier to pause execution at each tick, deliver the
  output tensor, and wait for the controller to resume. The forward pass
  runs in a background thread so the controller can drive the tick loop.

  Background:
    Given a tiny llama model is loaded on CPU

  # ── Sentinel hooks ───────────────────────────────────────────────

  Scenario: Sentinel hooks installed on all modules return handles
    Given the model's module paths are discovered
    When sentinel hooks are installed on all module paths
    Then a handle is returned for each module path
    And the handles can be removed without error

  # ── Capture hooks ────────────────────────────────────────────────

  Scenario: Capture hook delivers path, call_index, and tensor on forward pass
    Given a result mailbox and a resume mailbox
    And sentinel hooks are installed on all module paths
    And a capture hook is installed on "model.layers.0.self_attn.q_proj"
    When a forward pass is started in a background thread
    And the result mailbox is waited on
    Then the captured value contains the module path "model.layers.0.self_attn.q_proj"
    And the captured value contains a non-negative call_index
    And the captured value contains a torch.Tensor
    When the result mailbox is restored and the resume mailbox signals continue
    Then the forward pass completes without error

  Scenario: Capture hook with no active probes does not block
    Given a result mailbox and a resume mailbox
    And sentinel hooks are installed on all module paths
    And a capture hook is installed on "model.layers.0.self_attn.q_proj" with no active probes
    When a forward pass runs to completion
    Then the forward pass completes without error
    And the result mailbox was never written to

  # ── Hook removal ─────────────────────────────────────────────────

  Scenario: Removing hooks allows clean forward pass
    Given sentinel hooks are installed on all module paths
    When all hooks are removed
    And a forward pass runs to completion
    Then the forward pass completes without error

  # ── run_forward lifecycle ────────────────────────────────────────

  Scenario: run_forward calls done_callback with None on success
    When run_forward is called with valid input
    Then the done callback receives None (no error)

  Scenario: run_forward calls done_callback with exception on bad input
    When run_forward is called with invalid input
    Then the done callback receives an exception
