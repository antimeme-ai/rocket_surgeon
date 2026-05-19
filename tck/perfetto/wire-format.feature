@perfetto @wire-format
Feature: Perfetto trace wire format correctness
  The perfetto-writer crate produces .pftrace output using the Perfetto
  native trace format: a stream of `repeated TracePacket packet = 1`
  records, each field-1 length-delimited framed as [0x0A][varint(len)][bytes].
  Every packet must round-trip through protobuf encode/decode unchanged.

  Background:
    Given a TraceSink is opened for writing

  # ── Field-1 framing ──────────────────────────────────────────────

  Scenario: Single packet output begins with field-1 tag byte
    When a TracePacket is written with timestamp 42
    Then the output begins with byte 0x0A

  Scenario: Single packet output is valid field-1 framed protobuf
    When a TracePacket is written with timestamp 42
    Then the output is valid field-1 framed protobuf
    And the output contains exactly 1 TracePacket

  Scenario: Multiple packets are each individually field-1 framed
    When a TracePacket is written with timestamp 100
    And a TracePacket is written with timestamp 200
    And a TracePacket is written with timestamp 300
    Then the output is valid field-1 framed protobuf
    And the output contains exactly 3 TracePackets
    And each packet in the output decodes as a valid TracePacket

  # ── Protobuf roundtrip ──────────────────────────────────────────

  Scenario: Every TracePacket survives encode-decode roundtrip
    When a TracePacket is written with timestamp 1000000
    Then each packet in the output decodes as a valid TracePacket
    And the re-encoded packet equals the original bytes

  Scenario: Process track descriptor roundtrips through protobuf
    When a process track is written with uuid 1 and name "test-session"
    Then each packet in the output decodes as a valid TracePacket
    And the re-encoded packet equals the original bytes

  Scenario: Thread track descriptor roundtrips through protobuf
    When a process track is written with uuid 1 and name "test-session"
    And a thread track is written with uuid 100 parent 1
    Then the output contains exactly 2 TracePackets
    And the re-encoded packet equals the original bytes

  # ── Interning wire format ───────────────────────────────────────

  Scenario: Interned names packet carries SEQ_INCREMENTAL_STATE_CLEARED
    Given rank 0 has been declared
    And component "attn::q_proj" at index 0 under layer 0 rank 0 has been declared
    And interned names have been emitted for rank 0
    Then the InternedData packet has sequence_flags SEQ_INCREMENTAL_STATE_CLEARED

  Scenario: Interned names have unique iids starting from 1
    Given rank 0 has been declared
    And component "attn::q_proj" at index 0 under layer 0 rank 0 has been declared
    And component "attn::k_proj" at index 1 under layer 0 rank 0 has been declared
    And interned names have been emitted for rank 0
    Then each interned name has a unique iid starting from 1

  # ── Full trace validity ─────────────────────────────────────────

  Scenario: Complete trace with hierarchy and events is valid wire format
    Given a PerfettoSink has been created for session "test-session" with model "gpt2"
    And rank 0 has been declared
    And layer 0 under rank 0 has been declared
    And component "attn::q_proj" at index 0 under layer 0 rank 0 has been declared
    And interned names have been emitted for rank 0
    When on_tick_stopped is called with layer 0 component "attn::q_proj"
    And on_tick_stopped is called with layer 0 component "attn::q_proj"
    And close is called on the PerfettoSink
    Then the output is valid field-1 framed protobuf
    And the output contains at least 7 TracePackets
    And the re-encoded packet equals the original bytes
