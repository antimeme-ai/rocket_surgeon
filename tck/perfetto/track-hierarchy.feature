@perfetto @track-hierarchy
Feature: Perfetto track hierarchy and event mapping
  PerfettoSink maps the rocket_surgeon model topology to Perfetto's
  track tree: Session→Process, Rank→Thread, Layer→Track, Component→Track.
  Tick events emit SLICE_BEGIN/END duration pairs on component tracks.
  Probe events emit TYPE_INSTANT with DebugAnnotations for tensor stats.

  Background:
    Given a PerfettoSink has been created for session "test-session" with model "gpt2"

  # ── Process track ───────────────────────────────────────────────

  Scenario: Session creates a process track with uuid 1
    Then a TrackDescriptor packet exists with uuid 1 and name "test-session"
    And the process track has a ProcessDescriptor

  Scenario: Process track has EXPLICIT child ordering
    Then a TrackDescriptor packet exists with uuid 1 and name "test-session"
    And every child track has child_ordering set to EXPLICIT

  # ── Rank tracks ─────────────────────────────────────────────────

  Scenario: Declaring rank 0 creates a thread track under the process
    Given rank 0 has been declared
    Then a TrackDescriptor packet exists with uuid 100 and name "rank:0"
    And a TrackDescriptor packet exists with uuid 100 and parent_uuid 1
    And the rank track has a ThreadDescriptor

  Scenario: Declaring rank 1 creates a distinct thread track
    Given rank 0 has been declared
    And rank 1 has been declared
    Then a TrackDescriptor packet exists with uuid 100 and name "rank:0"
    And a TrackDescriptor packet exists with uuid 101 and name "rank:1"
    And a TrackDescriptor packet exists with uuid 101 and parent_uuid 1

  # ── Layer tracks ────────────────────────────────────────────────

  Scenario: Declaring layer 0 under rank 0 creates a child track
    Given rank 0 has been declared
    And layer 0 under rank 0 has been declared
    Then a TrackDescriptor packet exists with uuid 1000 and name "L0"
    And a TrackDescriptor packet exists with uuid 1000 and parent_uuid 100

  Scenario: Layer tracks under different ranks have distinct uuids
    Given rank 0 has been declared
    And rank 1 has been declared
    And layer 0 under rank 0 has been declared
    And layer 0 under rank 1 has been declared
    Then a TrackDescriptor packet exists with uuid 1000 and name "L0"
    And a TrackDescriptor packet exists with uuid 2000 and name "L0"

  # ── Component tracks ────────────────────────────────────────────

  Scenario: Declaring a component creates a track under its layer
    Given rank 0 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" at index 0 under layer 0 rank 0 has been declared
    Then a TrackDescriptor packet exists with uuid 10000 and name "L0::attn::q_proj"
    And a TrackDescriptor packet exists with uuid 10000 and parent_uuid 1000

  Scenario: Multiple components under same layer have distinct uuids
    Given rank 0 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" at index 0 under layer 0 rank 0 has been declared
    And component "attn::k_proj" at index 1 under layer 0 rank 0 has been declared
    Then a TrackDescriptor packet exists with uuid 10000 and name "L0::attn::q_proj"
    And a TrackDescriptor packet exists with uuid 10001 and name "L0::attn::k_proj"

  # ── Tick events → duration slices ───────────────────────────────

  Scenario: First tick_stopped emits SLICE_BEGIN on the component track
    Given rank 0 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" at index 0 under layer 0 rank 0 has been declared
    When on_tick_stopped is called with layer 0 component "attn::q_proj"
    Then a SLICE_BEGIN TrackEvent exists on the component track

  Scenario: Second tick_stopped on same component emits SLICE_END then SLICE_BEGIN
    Given rank 0 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" at index 0 under layer 0 rank 0 has been declared
    When on_tick_stopped is called with layer 0 component "attn::q_proj"
    And on_tick_stopped is called with layer 0 component "attn::q_proj"
    Then a SLICE_END TrackEvent exists on the component track
    And a SLICE_BEGIN TrackEvent exists on the component track

  # ── Probe events → instants ─────────────────────────────────────

  Scenario: Probe fired emits TYPE_INSTANT with probe name
    Given rank 0 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" at index 0 under layer 0 rank 0 has been declared
    When on_tick_stopped is called with layer 0 component "attn::q_proj"
    And on_probe_fired is called with probe_id "p1" and tensor summary
    Then a TYPE_INSTANT TrackEvent exists with name "probe:p1"

  Scenario: Probe instant carries DebugAnnotations for tensor stats
    Given rank 0 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" at index 0 under layer 0 rank 0 has been declared
    When on_tick_stopped is called with layer 0 component "attn::q_proj"
    And on_probe_fired is called with probe_id "p1" and tensor summary
    Then a TYPE_INSTANT TrackEvent exists with name "probe:p1"
    And the instant event has DebugAnnotations for "shape,mean,std,l2_norm"

  # ── Close terminates open slices ────────────────────────────────

  Scenario: Close emits SLICE_END for every open slice
    Given rank 0 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" at index 0 under layer 0 rank 0 has been declared
    When on_tick_stopped is called with layer 0 component "attn::q_proj"
    And close is called on the PerfettoSink
    Then all open slices have been terminated with SLICE_END

  # ── Auto-vivification ──────────────────────────────────────────

  Scenario: tick_stopped on undeclared component auto-creates the track
    When on_tick_stopped is called with layer 0 component "attn::q_proj"
    Then a SLICE_BEGIN TrackEvent exists on the component track
