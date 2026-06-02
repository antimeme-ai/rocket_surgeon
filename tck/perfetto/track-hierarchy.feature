@perfetto @track-hierarchy
Feature: Perfetto track hierarchy and event mapping (multi-rank)
  PerfettoSink maps the rocket_surgeon process topology to Perfetto's track
  tree per ADR-0010:

    [Process] daemon:{session}                       uuid=process(DAEMON)
    [Process] rank:N                                 uuid=process(rank=N)
      [Track] L{l}                                   uuid=layer(rank=N, layer=l)
        [Track] L{l}::{component}                    uuid=component(rank=N, l, c)

  UUIDs are bit-packed as (kind:4 | rank:12 | layer:16 | component:32).
  Each Perfetto sequence (sequence_id = 1000 + rank, or 999 for the daemon)
  owns an independent InternTable. Tick events emit SLICE_BEGIN/END duration
  pairs on the component track of their originating rank's sequence; probe
  events emit TYPE_INSTANT carrying their rank via ProbeFiredEvent.rank.

  Background:
    Given a PerfettoSink has been created for session "test-session" with model "gpt2" and daemon pid 1

  # ── Daemon process ───────────────────────────────────────────────

  Scenario: Session creates a daemon process track
    Then a TrackDescriptor packet exists with uuid 0x1FFF000000000000 and name "daemon:test-session"
    And the daemon track has a ProcessDescriptor

  # ── Worker ranks as ProcessDescriptors ──────────────────────────

  Scenario: Declaring rank 0 creates a worker process track under the daemon
    Given worker process rank 0 with pid 100 has been declared
    Then a TrackDescriptor packet exists with uuid 0x1000000000000000 and name "rank:0"
    And the rank track has a ProcessDescriptor

  Scenario: Declaring multiple worker ranks emits distinct process tracks
    Given worker process rank 0 with pid 100 has been declared
    And worker process rank 1 with pid 101 has been declared
    Then a TrackDescriptor packet exists with uuid 0x1000000000000000 and name "rank:0"
    And a TrackDescriptor packet exists with uuid 0x1001000000000000 and name "rank:1"

  # ── Layer tracks ────────────────────────────────────────────────

  Scenario: Declaring layer 0 under rank 0 creates a child track
    Given worker process rank 0 with pid 100 has been declared
    And layer 0 under rank 0 has been declared
    Then a TrackDescriptor packet exists with uuid 0x2000000000000000 and name "L0"
    And a TrackDescriptor packet exists with uuid 0x2000000000000000 and parent_uuid 0x1000000000000000

  Scenario: Layer tracks under different ranks have distinct uuids
    Given worker process rank 0 with pid 100 has been declared
    And worker process rank 1 with pid 101 has been declared
    And layer 0 under rank 0 has been declared
    And layer 0 under rank 1 has been declared
    Then a TrackDescriptor packet exists with uuid 0x2000000000000000 and name "L0"
    And a TrackDescriptor packet exists with uuid 0x2001000000000000 and name "L0"

  # ── Component tracks ────────────────────────────────────────────

  Scenario: Declaring a component creates a track under its layer
    Given worker process rank 0 with pid 100 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" under layer 0 rank 0 has been declared
    Then a TrackDescriptor packet exists with uuid 0x3000000000000000 and name "L0::attn::q_proj"
    And a TrackDescriptor packet exists with uuid 0x3000000000000000 and parent_uuid 0x2000000000000000

  Scenario: Multiple components under same layer have distinct uuids
    Given worker process rank 0 with pid 100 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" under layer 0 rank 0 has been declared
    And component "attn::k_proj" under layer 0 rank 0 has been declared
    Then a TrackDescriptor packet exists with uuid 0x3000000000000000 and name "L0::attn::q_proj"
    And a TrackDescriptor packet exists with uuid 0x3000000000000001 and name "L0::attn::k_proj"

  Scenario: Same component name on different ranks emits distinct tracks (BEAD-0010 M-4)
    Given worker process rank 0 with pid 100 has been declared
    And worker process rank 1 with pid 101 has been declared
    And layer 0 under rank 0 has been declared
    And layer 0 under rank 1 has been declared
    And component "attn::q_proj" under layer 0 rank 0 has been declared
    And component "attn::q_proj" under layer 0 rank 1 has been declared
    Then a TrackDescriptor packet exists with uuid 0x3000000000000000
    And a TrackDescriptor packet exists with uuid 0x3001000000000000
    And the two component tracks have parent_uuid 0x2000000000000000 and 0x2001000000000000 respectively

  # ── Per-sequence interning (BEAD-0010 M-2) ──────────────────────

  Scenario: Each rank's InternedData packet is scoped to its own sequence
    Given worker process rank 0 with pid 100 has been declared
    And worker process rank 1 with pid 101 has been declared
    And layer 0 under rank 0 has been declared
    And layer 0 under rank 1 has been declared
    And component "attn::q_proj" under layer 0 rank 0 has been declared
    And component "mlp::gate" under layer 0 rank 1 has been declared
    When emit_interned_names is called for rank 0
    And emit_interned_names is called for rank 1
    Then an InternedData packet exists with trusted_packet_sequence_id 1000 containing name "L0::attn::q_proj"
    And an InternedData packet exists with trusted_packet_sequence_id 1001 containing name "L0::mlp::gate"
    And no InternedData packet on sequence 1001 contains "L0::attn::q_proj"
    And no InternedData packet on sequence 1000 contains "L0::mlp::gate"

  Scenario: Each new sequence emits SEQ_INCREMENTAL_STATE_CLEARED on its first packet
    Given worker process rank 0 with pid 100 has been declared
    And worker process rank 1 with pid 101 has been declared
    And layer 0 under rank 0 has been declared
    And layer 0 under rank 1 has been declared
    And component "attn::q_proj" under layer 0 rank 0 has been declared
    And component "attn::q_proj" under layer 0 rank 1 has been declared
    When emit_interned_names is called for rank 0
    And emit_interned_names is called for rank 1
    Then the first InternedData packet on sequence 1000 has SEQ_INCREMENTAL_STATE_CLEARED set
    And the first InternedData packet on sequence 1001 has SEQ_INCREMENTAL_STATE_CLEARED set
    And first_packet_on_sequence is true for both initial packets

  # ── Tick events → duration slices on the originating rank ──────

  Scenario: tick_stopped routes SLICE_BEGIN to the rank's sequence
    Given worker process rank 0 with pid 100 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" under layer 0 rank 0 has been declared
    When on_tick_stopped is called with rank 0 layer 0 component "attn::q_proj"
    Then a SLICE_BEGIN TrackEvent exists on track 0x3000000000000000
    And the SLICE_BEGIN packet has trusted_packet_sequence_id 1000

  Scenario: Two ranks ticking the same component name emit on their own sequences
    Given worker process rank 0 with pid 100 has been declared
    And worker process rank 1 with pid 101 has been declared
    And layer 0 under rank 0 has been declared
    And layer 0 under rank 1 has been declared
    And component "attn::q_proj" under layer 0 rank 0 has been declared
    And component "attn::q_proj" under layer 0 rank 1 has been declared
    When on_tick_stopped is called with rank 0 layer 0 component "attn::q_proj"
    And on_tick_stopped is called with rank 1 layer 0 component "attn::q_proj"
    Then a SLICE_BEGIN packet exists on track 0x3000000000000000 with trusted_packet_sequence_id 1000
    And a SLICE_BEGIN packet exists on track 0x3001000000000000 with trusted_packet_sequence_id 1001

  Scenario: Second tick_stopped on same component emits SLICE_END then SLICE_BEGIN on same sequence
    Given worker process rank 0 with pid 100 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" under layer 0 rank 0 has been declared
    When on_tick_stopped is called with rank 0 layer 0 component "attn::q_proj"
    And on_tick_stopped is called with rank 0 layer 0 component "attn::q_proj"
    Then a SLICE_END TrackEvent exists on track 0x3000000000000000 with trusted_packet_sequence_id 1000
    And a SLICE_BEGIN TrackEvent exists on track 0x3000000000000000 with trusted_packet_sequence_id 1000

  # ── Probe events carry rank → instants on that rank's sequence ─

  Scenario: on_probe_fired emits TYPE_INSTANT on the event's rank sequence
    Given worker process rank 0 with pid 100 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" under layer 0 rank 0 has been declared
    When on_probe_fired is called with rank 0 probe_id "p1" point "L0::attn::q_proj"
    Then a TYPE_INSTANT TrackEvent exists with name "probe:p1"
    And the INSTANT packet has trusted_packet_sequence_id 1000

  Scenario: Probe firing on rank 2 lands on rank 2's sequence (BEAD-0010 C-2)
    Given worker process rank 2 with pid 102 has been declared
    When on_probe_fired is called with rank 2 probe_id "p1" point "L0::attn::q_proj"
    Then a TYPE_INSTANT TrackEvent exists with name "probe:p1"
    And the INSTANT packet has trusted_packet_sequence_id 1002

  Scenario: Probe instant carries DebugAnnotations for tensor stats
    Given worker process rank 0 with pid 100 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" under layer 0 rank 0 has been declared
    When on_probe_fired is called with rank 0 probe_id "p1" point "L0::attn::q_proj" and tensor summary
    Then a TYPE_INSTANT TrackEvent exists with name "probe:p1"
    And the instant event has DebugAnnotations for "shape,mean,std,l2_norm"

  # ── Close terminates open slices on their originating sequence ─

  Scenario: Close emits SLICE_END on the same sequence the SLICE_BEGIN was emitted on
    Given worker process rank 0 with pid 100 has been declared
    And worker process rank 1 with pid 101 has been declared
    And layer 0 under rank 0 has been declared
    And layer 0 under rank 1 has been declared
    And component "attn::q_proj" under layer 0 rank 0 has been declared
    And component "attn::q_proj" under layer 0 rank 1 has been declared
    When on_tick_stopped is called with rank 0 layer 0 component "attn::q_proj"
    And on_tick_stopped is called with rank 1 layer 0 component "attn::q_proj"
    And close is called on the PerfettoSink
    Then all open slices have been terminated with SLICE_END
    And the rank-0 SLICE_END has trusted_packet_sequence_id 1000
    And the rank-1 SLICE_END has trusted_packet_sequence_id 1001

  # ── Auto-vivification ──────────────────────────────────────────

  Scenario: tick_stopped on undeclared (rank, component) auto-creates the track
    When on_tick_stopped is called with rank 0 layer 0 component "attn::q_proj"
    Then a SLICE_BEGIN TrackEvent exists on track 0x3000000000000000

  Scenario: Probe firing on undeclared rank lazily materializes that rank's intern table
    When on_probe_fired is called with rank 3 probe_id "p1" point "L0::attn::q_proj"
    Then a TYPE_INSTANT TrackEvent exists with name "probe:p1"
    And the INSTANT packet has trusted_packet_sequence_id 1003
