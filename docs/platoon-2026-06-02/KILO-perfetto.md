# B004 — KILO findings (perfetto-writer)

**Lane:** `crates/perfetto-writer` — `varint.rs`, `proto.rs`, `writer.rs`,
`intern.rs`, plus track-hierarchy invariants.
**Branch:** `platoon2/perfetto`. Dev-dep added: `proptest.workspace = true`.

## Summary

Brought the crate from example-only (0 property tests) to MATERIA oracle tiers
4–7. **18 new tests** across 4 integration-test files, all green; the existing
30 unit tests are untouched and still pass. `cargo clippy --workspace
--all-targets -- -D warnings` is clean.

| File | New tests | Highest oracle |
| --- | --- | --- |
| `tests/varint_props.rs` | 6 | tier 6 differential (prost) + tier 7 boundary table |
| `tests/intern_props.rs` | 5 | tier 6 stateful model (dict + counter, lockstep) |
| `tests/proto_props.rs` | 4 | tier 4/6 roundtrip + bit-level NaN metamorphic |
| `tests/writer_model.rs` | 3 | tier 6 stateful stream model + forest invariants + exception-raising |

## Techniques applied

- **Differential / model oracle (varint).** `encode_varint` is checked
  byte-for-byte against prost's reference `encode_varint`, and prost's
  `decode_varint` must recover the original and consume all bytes. A fully
  independent local decoder gives a second (non-prost) roundtrip oracle.
- **Specification oracle (varint boundaries).** Hand-derived table over every
  7-bit group transition `2^(7k)−1 / 2^(7k) / +1` for k=1..9 plus
  `0, 1, u64::MAX, u64::MAX−1`, asserting exact encoded length AND roundtrip.
- **Structural property (varint).** Continuation-bit invariant (every
  non-terminal byte has 0x80 set, terminal byte clear), `len ≤ 10`, length ==
  `ceil(significant_bits/7)`, and minimal-encoding (no overlong trailing zero
  group).
- **Stateful model-based testing (intern).** Lockstep against a `HashMap +
  counter` reference over generated `Intern`/`Get` op sequences; after *every*
  op we assert real==model plus the bijection/density invariants (iids are
  exactly the contiguous range `1..=len`, `entries()` is the inverse of `get`).
- **Metamorphic (intern).** Idempotence (re-interning never grows the table or
  changes an iid) and order-independence of the interned *name-set* under input
  permutation (the id *assignment* is first-occurrence rank, so only the set /
  size are permutation-invariant — encoded explicitly, not conflated).
- **Stateful model-based testing (writer).** Generate sequences of the 9
  high-level writer calls; decode the produced byte stream through a framing
  reader; assert (a) exactly one record per call (stream tiles with no leftover
  bytes), (b) each record's salient fields re-derived from Perfetto semantics,
  and (c) the `first_packet_on_sequence` flag follows a parallel seen-set model.
- **Forest invariants (track hierarchy).** Generate a valid track tree
  (acyclic by construction: node i's parent ∈ 0..i), emit via the writer, decode,
  and assert: distinct uuids, exactly one root, no dangling parent references,
  and the parent relation survives transit unchanged.
- **Exception-raising property (writer).** A sink whose `write_all` fails after
  N successes must surface `WriteError::Io` — never panic, never silently
  succeed. (MATERIA: 113x oracle; almost nobody writes these.)
- **Metamorphic bit-level (proto).** NaN doubles break struct `PartialEq`
  (NaN≠NaN) but their *bits* must round-trip; checked via `to_bits()` for quiet
  / signaling / negative NaN.

## Generator-distribution evidence (measured, not assumed)

- **varint values** — bucketed by encoded byte-length over 20 000 samples:
  `[2383, 2117, 2077, 2170, 2074, 2223, 2148, 2153, 2013, 642]` for lengths
  1..10. All ten byte-lengths exercised (asserted in
  `generator_covers_all_byte_lengths`). Note: bare `any::<u64>()` would dump
  >99% into the 9–10-byte buckets; the right-shift generator spreads mass
  evenly, and the 10-byte bucket is deliberately the thinnest (only values
  ≥ 2^63 land there).
- **intern op stream** — over 2 000 sequences: `novel=56352 reintern=93685
  gets=50472` (total interns 150 037). Both the novel-insert and the
  idempotent re-intern paths are heavily exercised — the small name pool is
  what makes re-interning common; an unconstrained name generator would make
  every intern unique and silently skip the idempotence path.
- **writer ops** — sequence ids drawn from `1..=3` so the
  `first_packet_on_sequence` state machine sees real repeats rather than
  all-unique ids (otherwise the `false` branch would never fire).

## Bugs / weak oracles found

**None — the production code is correct under every property tried.** This is a
genuine (if quieter) result: `encode_varint` matches prost bit-for-bit across
the full u64 range incl. all boundaries; the intern table is a faithful
first-occurrence map with dense contiguous ids; the writer faithfully frames and
maps every field; IO errors propagate as `WriteError::Io`. The Wave-1 modules
surfaced two defects; this crate did not, and the property suite is the evidence
that the absence is real rather than untested.

## Gaps left (for the next agent / NOVEMBER's aim)

- **No production varint decoder exists.** The roundtrip is closed only through
  test-side decoders (local + prost). If a decode path is ever added to the
  crate, it needs its own exception-raising properties: truncated varint
  (continuation bit set on the last byte), overlong (>10 groups), and
  non-minimal encodings must all error rather than silently mis-decode. The
  test-side `decode_ref` already models the correct rejection behavior and can
  seed those tests.
- **No structural validation in the writer.** `write_track` et al. accept any
  `parent_uuid`, including dangling or self-referential ones — the writer is a
  faithful emitter, not a validator. The forest invariant is tested over
  *valid* generated trees; whether the writer *should* reject a child whose
  parent was never declared is a design question, not a bug, and is left for the
  protocol owner.
- **`InternTable` iid space is `u64` and never recycles.** Overflow at 2^64
  interns is untested (and untestable in finite time); not a practical concern.
- Mutation testing (NOVEMBER) is the right next lens to confirm these oracles
  are sensitive where it matters — the differential varint and stateful writer
  models should kill the obvious arithmetic/tag mutants; the survivors will
  point at any remaining oracle gaps.
