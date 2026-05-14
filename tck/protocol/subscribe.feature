@subscribe
Feature: Event subscription for real-time notifications
  The rocket/subscribe verb allows clients to register for streaming
  event notifications with optional layer and rank filtering.
  Subscriptions persist until the session ends or the client
  unsubscribes.

  Background:
    Given a rocket_surgeon server is running
    And the session is initialized with protocol_version "0.1.0"
    And a model "llama-7b" is attached
    And the session has been stepped to tick 0 at layer 0

  # ── Basic subscriptions ───────────────────────────────────────────

  Scenario: Subscribe to tick.stopped event and receive notification after step
    When the client sends "rocket/subscribe" with:
      """json
      {
        "events": ["tick.stopped"]
      }
      """
    Then the response status is "stopped"
    And the response data field "subscription_id" is a non-empty string
    And the response data field "subscribed_events" contains "tick.stopped"
    When the client sends "rocket/step" with direction "forward"
    Then the client receives a "tick.stopped" notification
    And the notification includes the current tick position

  Scenario: Subscribe to probe.fired event and receive notification when probe fires
    Given a defined probe "p-sub-fire" at point "llama:0:12:attn.o_proj:output" with action "capture"
    When the client sends "rocket/subscribe" with:
      """json
      {
        "events": ["probe.fired"]
      }
      """
    Then the response data field "subscription_id" is a non-empty string
    And the response data field "subscribed_events" contains "probe.fired"
    When the client sends "rocket/step" with direction "forward"
    And the forward pass reaches layer 12
    Then the client receives a "probe.fired" notification
    And the notification includes probe_id "p-sub-fire"
    And the notification includes a tensor summary

  # ── Filtered subscriptions ────────────────────────────────────────

  Scenario: Subscribe with layer filter only receives events for specified layers
    When the client sends "rocket/subscribe" with:
      """json
      {
        "events": ["tick.stopped"],
        "filter": {
          "layer": [12, 13]
        }
      }
      """
    Then the response data field "subscription_id" is a non-empty string
    When the client sends "rocket/step" with direction "forward"
    Then the client receives "tick.stopped" notifications only for layers 12 and 13
    And the client does not receive "tick.stopped" notifications for layer 0

  Scenario: Subscribe with rank filter only receives events for specified ranks
    When the client sends "rocket/subscribe" with:
      """json
      {
        "events": ["tick.stopped"],
        "filter": {
          "rank": [0]
        }
      }
      """
    Then the response data field "subscription_id" is a non-empty string
    When the client sends "rocket/step" with direction "forward"
    Then the client receives "tick.stopped" notifications only for rank 0

  # ── Multiple subscribers ──────────────────────────────────────────

  Scenario: Multiple subscribers all receive events
    When client "A" sends "rocket/subscribe" with:
      """json
      {
        "events": ["tick.stopped"]
      }
      """
    And client "B" sends "rocket/subscribe" with:
      """json
      {
        "events": ["tick.stopped"]
      }
      """
    Then client "A" receives a subscription_id
    And client "B" receives a subscription_id
    And client "A" subscription_id differs from client "B" subscription_id
    When the client sends "rocket/step" with direction "forward"
    Then client "A" receives a "tick.stopped" notification
    And client "B" receives a "tick.stopped" notification

  # ── Heartbeat ─────────────────────────────────────────────────────

  Scenario: tick.heartbeat sent approximately every 1 second while stopped
    When the client sends "rocket/subscribe" with:
      """json
      {
        "events": ["tick.heartbeat"]
      }
      """
    Then the response data field "subscribed_events" contains "tick.heartbeat"
    When the session remains in "stopped" state for 3 seconds
    Then the client receives at least 2 "tick.heartbeat" notifications
    And each "tick.heartbeat" notification includes per-rank status

  # ── Unsubscribed events ───────────────────────────────────────────

  Scenario: Unsubscribed events are not received
    When the client sends "rocket/subscribe" with:
      """json
      {
        "events": ["tick.stopped"]
      }
      """
    And the client sends "rocket/step" with direction "forward"
    Then the client receives a "tick.stopped" notification
    And the client does not receive a "probe.fired" notification
    And the client does not receive a "tick.heartbeat" notification

  # ── Response shape ────────────────────────────────────────────────

  Scenario: Subscribe response includes subscription_id
    When the client sends "rocket/subscribe" with:
      """json
      {
        "events": ["tick.stopped", "probe.fired"]
      }
      """
    Then the response data field "subscription_id" is a non-empty string
    And the response data field "subscribed_events" has 2 entries
    And the response data field "subscribed_events" contains "tick.stopped"
    And the response data field "subscribed_events" contains "probe.fired"
