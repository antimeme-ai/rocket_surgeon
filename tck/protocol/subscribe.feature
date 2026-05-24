@subscribe
Feature: Event subscription and notification delivery
  The rocket/subscribe verb enables push-everything event delivery.
  A boolean gate — no subscription state, no per-event filtering.
  Events are JSON-RPC Notifications with monotonic seq in params.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.3.0"
    And a model "llama-7b" is attached
    And the session has been stepped to tick 0 at layer 0

  # ── Subscribe enables events ─────────────────────────────────────

  Scenario: Subscribe enables event delivery and returns available event types
    When the client sends "rocket/subscribe" with:
      """json
      {}
      """
    Then the response status is "stopped"
    And the response data field "available_events" contains "tick.stopped"
    And the response data field "available_events" contains "tick.heartbeat"
    And the response data field "available_events" contains "probe.fired"
    When the client sends "rocket/step" with direction "forward"
    Then the client receives a "tick.stopped" notification
    And the notification includes a "seq" field with a non-negative integer

  # ── Events not sent before subscribe ─────────────────────────────

  Scenario: No notifications before subscribing
    When the client sends "rocket/step" with direction "forward"
    Then the client does not receive any notifications

  # ── Unsubscribe stops events ─────────────────────────────────────

  Scenario: Unsubscribe disables event delivery
    When the client sends "rocket/subscribe" with:
      """json
      {}
      """
    And the client sends "rocket/step" with direction "forward"
    Then the client receives a "tick.stopped" notification
    When the client sends "rocket/unsubscribe" with:
      """json
      {}
      """
    Then the response status is "stopped"
    When the client sends "rocket/step" with direction "forward"
    Then the client does not receive any notifications

  # ── Heartbeat while stopped ──────────────────────────────────────

  Scenario: Heartbeat notifications sent approximately every 1 second while stopped
    When the client sends "rocket/subscribe" with:
      """json
      {}
      """
    And the session remains in "stopped" state for 3 seconds
    Then the client receives at least 2 "tick.heartbeat" notifications
    And each "tick.heartbeat" notification includes "position" and "uptime_seconds"

  # ── Probe.fired delivered after step ─────────────────────────────

  Scenario: Probe.fired notifications delivered after step completes
    Given a defined probe "p-sub-fire" at point "*:*:*:*:*:*" with action "capture"
    When the client sends "rocket/subscribe" with:
      """json
      {}
      """
    And the client sends "rocket/step" with direction "forward"
    Then the client receives a "tick.stopped" notification
    And the client receives at least one "probe.fired" notification
    And each "probe.fired" notification includes "probe_id" and "seq"

  # ── Monotonic seq ────────────────────────────────────────────────

  Scenario: Notifications have strictly increasing seq values
    When the client sends "rocket/subscribe" with:
      """json
      {}
      """
    And the client sends "rocket/step" with direction "forward"
    And the client sends "rocket/step" with direction "forward"
    Then all received notifications have strictly increasing "seq" values

  # ── Subscribe is idempotent ──────────────────────────────────────

  Scenario: Subscribing twice does not error
    When the client sends "rocket/subscribe" with:
      """json
      {}
      """
    And the client sends "rocket/subscribe" with:
      """json
      {}
      """
    Then the response status is "stopped"
    And the response data field "available_events" contains "tick.stopped"

  # ── Unsubscribe is idempotent ────────────────────────────────────

  Scenario: Unsubscribing without prior subscribe does not error
    When the client sends "rocket/unsubscribe" with:
      """json
      {}
      """
    Then the response status is "stopped"

  # ── Subscribe requires stopped state ─────────────────────────────
  # Background attaches+steps, so detach first to reach Initialized state.

  Scenario: Subscribe without attached model returns model-not-attached error
    When the client sends "detach" with:
      """json
      {}
      """
    And the client sends "rocket/subscribe" with:
      """json
      {}
      """
    Then the response contains an error with code "MODEL_NOT_ATTACHED"

  # ── Notification wire format ─────────────────────────────────────

  Scenario: Notifications are JSON-RPC Notifications with no id field
    When the client sends "rocket/subscribe" with:
      """json
      {}
      """
    And the client sends "rocket/step" with direction "forward"
    Then the client receives a notification message
    And the notification has field "jsonrpc" equal to "2.0"
    And the notification has field "method" as a string
    And the notification has field "params" as an object
    And the notification has no "id" field
    And the notification "params" contains "seq" as a non-negative integer
