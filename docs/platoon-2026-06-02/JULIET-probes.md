# PLATOON-FINDINGS — JULIET (B004, probe grammar lane)

**Lane:** `crates/rocket-surgeon-probes` — `grammar.rs` (winnow parser),
`assertion.rs`, `registry.rs`.
**Branch:** `platoon2/probes`. **Base:** `origin/master` @ ab78f99.
**Dev-dep added:** `proptest = "1"` (via `proptest.workspace = true`; the crate
had none).

## Techniques applied (MATERIA oracle tiers 4–7)

| File | Test file | Tier | Oracle |
| --- | --- | --- | --- |
| `grammar.rs` | `tests/grammar_props.rs` | 4 roundtrip | `parse ∘ render == id` over generated `ProbePoint` ASTs |
| `grammar.rs` | `tests/grammar_props.rs` | 6 model | `matches` vs an `Option`-per-field reference (`model_matches`) |
| `grammar.rs` | `tests/grammar_props.rs` | 4 metamorphic | `matches` symmetric, reflexive, wildcard-monotone, full-`*` matches all |
| `grammar.rs` | `tests/grammar_props.rs` | exception-raising | 8 malformation strategies → specific `ParseError`, never panic |
| `registry.rs` | `tests/registry_model.rs` | 6 stateful-model | op-sequence vs `BTreeMap`+`next_seq` abstract model, asserted after every step, shrinks failing sequences |
| `assertion.rs` | `tests/assertion_props.rs` | 6 model | `evaluate` vs direct `f64` compare; float parse vs std `str::parse` (bit-exact) |
| `assertion.rs` | `tests/assertion_props.rs` | 4 metamorphic | operator complementarity (`<`/`>=`, `>`/`<=`, `==`/`!=`); whitespace inert |
| `assertion.rs` | `tests/assertion_props.rs` | exception-raising | unknown field / missing op / non-numeric / trailing junk / bare field → error |

**New test count:** 29 property/model `#[test]` functions across 3 integration
test files (11 assertion, 17 grammar, 1 stateful registry running 0..40-op
sequences × 400 cases), each exercising hundreds–thousands of generated cases.
All green; existing 66 unit tests untouched and still passing.
`cargo clippy --workspace --all-targets -- -D warnings` clean; `cargo fmt` clean.

## Generator-distribution evidence

`grammar_props::generator_distribution` samples `arb_probe_point()` 4000× and
asserts each interesting form is meaningfully represented (not theatre). Measured
(deterministic seed):

```
model wildcard:        19.9%
any numeric wildcard:  49.3%   (rank | layer | call_index)
component wildcard:    20.5%
has indexed seg (MoE): 53.1%   (experts[3] / head[k] form)
multi-segment path:    59.5%   (dotted paths attn.o_proj …)
all-concrete point:    26.6%
full wildcard point:    0.3%
```

No category is trivial-dominated; wildcards, MoE expert/head indexing, and
multi-segment paths are all heavily exercised, so the roundtrip and matching
oracles run against the forms that matter, not just `a:0:0:b:0:c`.

## Findings

### F1 (latent semantic, documented — NOT a code bug): NaN breaks assertion operator complementarity

`Assertion::evaluate` compares a `TensorStats` field against a threshold. When the
field is `NaN` (e.g. an assertion-`Assert` probe firing on an all-NaN / overflowed
activation — a realistic debugging scenario), IEEE-754 semantics make **both**
sides of every complementary operator pair false:

- `mean < t` is `false` **and** `mean >= t` is `false`
- `mean == t` is `false` (`(NaN - t).abs() < EPSILON`) **and** `mean != t` is
  `false` (`(NaN - t).abs() >= EPSILON`)

Minimal case (pinned in `assertion_props::nan_breaks_operator_complementarity`):
field = `NaN`, any threshold → every comparison returns `false`.

**Why it matters:** an assertion probe whose job is "alarm if `abs_max > 1e4`"
will silently **not** fire when the tensor has gone to NaN — exactly the
pathological case the user most wants flagged. `evaluate` returning `false` reads
as "assertion passed".

**Root cause:** standard IEEE-754 NaN ordering; `evaluate` (`assertion.rs:50–60`)
does the comparisons directly with no NaN guard. This is *correct* float behavior,
so I did **not** "fix" it (no spec says otherwise, and silently flipping a
comparison would be worse). Flagged for the protocol/probe-semantics owner to
decide whether `Assert` probes should treat a NaN field as a forced failure
(`is_nan() → assertion violated`). The metamorphic complementarity property is
therefore stated over **finite** inputs only, with the NaN behavior pinned by a
dedicated regression test so any future change is intentional and visible.

### Weak-oracle note: `define` checks duplicate-id *before* parsing the point

`ProbeRegistry::define` (`registry.rs:35–52`) returns `DuplicateId` for an existing
id **regardless of whether the supplied point is valid** — i.e. redefining an
existing id with a malformed point yields `DuplicateId`, not `InvalidPoint`. This
is a reasonable precedence but was previously unspecified and untested. The
stateful model now mirrors and pins this ordering, so a future refactor that
parses first (and would surface `InvalidPoint` for a duplicate id) is caught.

### No production defects surfaced in the parser or registry

Under the roundtrip, model, and stateful-model oracles the winnow parser, the
`matches` glob logic, and the registry state machine are correct for every
generated case (thousands of roundtrips, 400 op-sequences). `render` is injective
on the generated AST space; `matches` is exactly the `Option`-per-field reference;
the registry tracks the abstract model after every operation including dedup
precedence, seq-counter non-advancement on rejected inserts, `(priority, seq)`
ordering, and enable/disable/remove on absent ids. Recording this as evidence, not
as an absence of effort: the oracles are tier 4–6 and the suite shrinks
counterexamples — they would have caught a real defect.

## Gaps left (out of scope for this lane / this wave)

- **MC/DC on `matches`/`evaluate` boolean decisions** — the AND-fold in `matches`
  and the op dispatch in `evaluate` would benefit from MC/DC coverage
  (`-Z coverage-options=mcdc`) to prove each condition independently affects the
  outcome. Not run here; property + model oracles already exercise the conditions
  but MC/DC would formalize it.
- **Grammar fuzzing (cargo-fuzz)** — `arbitrary_input_never_panics` is a proptest
  `.*` sweep (tier-2 implicit oracle for robustness). A coverage-guided
  `cargo-fuzz` target on `ProbePoint::parse` / `Assertion::parse` would push
  deeper into the winnow state space; deferred (would add a fuzz harness, larger
  than this wave's bounded scope).
- **Protocol-level `ProbeConfig.assertion` wiring** — these tests cover the
  `assertion.rs` parser/evaluator in isolation; the path where a probe's
  `config.assertion` string is parsed and evaluated against live tensor stats is
  another crate's lane.
