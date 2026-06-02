# ALPHA — protocol wire-format properties — findings

**Lane:** `crates/rocket-surgeon-protocol`. **Branch:** `platoon/test-protocol`.
Authored by worker ALPHA; findings doc + commit finalized by the commander after
ALPHA's session ended with the suite green but uncommitted.

## What was built

`crates/rocket-surgeon-protocol/tests/proptest_wire_format.rs` (1187 lines, new),
`proptest` added as a dev-dependency. 46 property tests pass, 1 ignored
(documents a real defect, below). Production code left untouched — all `Strategy`s
are hand-written in the test crate, not `#[derive(Arbitrary)]` on the wire types.

Oracle tiers climbed over the existing example-based `serde_roundtrip.rs` surface:
- **Tier 4 roundtrip** — `from_str(to_string(v)) == v` over *generated* values for
  every wire type (request/response/notification, TickPosition, ProbeDefinition,
  InterventionRecipe, WorldlineState, Capabilities, TensorSummary, …).
- **Tier 6 model** — `WorldlineState::is_empty()` checked against an independent
  reference predicate AND its real `skip_serializing_if` wire contract inside
  `SessionState` (`is_empty_iff_field_skipped_in_session_state`).
- **Tier 5 exception-raising** — unknown enum variants (`dtype`, `status`) and
  malformed JSON must error, never panic / never silently coerce.
- **Tier 4 metamorphic** — the parser is idempotent on accepted inputs.
- Generators are **measured** (`generator_distribution_*` tests) so we know the
  non-trivial variants are exercised.

## Bugs / weak oracles found

1. **`InterventionParams` silently coerces malformed params to `Ablate`** (real
   robustness defect). `InterventionParams` is `#[serde(untagged)]` with an
   all-optional `Ablate` variant, so a typo'd or incomplete params object does not
   error — it falls through to `Ablate` with defaults. Test
   `untagged_params_typo_should_be_rejected` encodes the *desired* exception-raising
   behaviour and is `#[ignore]`d with a pointer here, alongside two tests that pin
   the *current* (wrong) coercion so the change is detected when it's fixed.
   **Recommendation:** make the params representation reject unknown shapes —
   internally-tagged enum, or a deny-unknown-fields wrapper, so a bad recipe is a
   loud error at the wire boundary, not a silent ablate.

2. **Non-finite / extreme `f64` do not roundtrip through `serde_json`.** Generated
   values like `Inline([-5.87e298])` and non-finite floats shrink to roundtrip
   failures (NaN/Inf are not representable in JSON; extreme magnitudes lose
   identity). Saved in `proptest_wire_format.proptest-regressions`. Generators are
   constrained to roundtrip-safe finite floats so the suite is green, and the
   limitation is recorded here. **Recommendation:** decide the contract explicitly
   — reject non-finite tensor values at the wire boundary (preferred for a
   debugger that must not lie), or document that NaN/Inf are out of protocol scope.

## Gate status

`cargo test -p rocket-surgeon-protocol --test proptest_wire_format` → 46 passed,
1 ignored. `cargo clippy -p rocket-surgeon-protocol --all-targets -- -D warnings`
→ clean.

## Left for follow-up

Finding 1 is a production-code change (intervention params representation) — out
of scope for a test-only lane; filed here for the daemon/protocol owner. The
ignored test will go green once it lands.
