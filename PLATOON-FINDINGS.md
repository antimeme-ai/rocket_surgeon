# B004 — MIKE lane findings: tick model

**Callsign:** MIKE · **Lane:** `crates/rocket-surgeon-worker` — `tick.rs` (tick_id
invariant across forward/reverse step sequences) + the uncovered `step_driver.rs`.
**Branch:** `platoon2/worker-tick`. **Base:** master @ a40e9aa (Wave-1, PR #49).

## Scope delivered

`tick.rs` already carried BRAVO's (B002) stateful model-based suite for the
*forward* three-clock semantics. This wave adds the MIKE mandate on top:

1. The **tick_id identity contract** across forward *and* reverse navigation,
   pinning what the impl actually guarantees and witnessing the ADR contradiction.
2. Property/metamorphic coverage for **`step_driver.rs`**, which had zero.
3. A real **overflow defect** surfaced by an exception-raising property, recorded
   below, then fixed (the property is the win).

I did **not** touch other lanes, did not refactor for style, did not push/PR.

## Techniques applied (oracle tiers)

| Test | Technique | Tier |
| --- | --- | --- |
| `tick_id_aliases_operator_and_direction_is_always_forward` | property over fwd/rev nav seq | 5 |
| `tick_id_is_not_a_unique_session_key_but_clock_and_step_count_are` | model (identity-as-set) | 6 |
| `step_count_is_the_monotonic_never_reset_clock` | model-based | 6 |
| `seek_token_preserves_operator` | metamorphic (reverse motion) | 4 |
| `advance_token_never_overflows_or_regresses_token` | exception-raising / robustness | 5 |
| `advance_token_saturates_at_ceiling` | deterministic boundary regression | 3→spec |
| `cross_token_tick_id_collision_witness` | regression witness for the ADR finding | 3 |
| `tick_id_is_many_to_one_across_tokens` | model (many-to-one map) | 6 |
| `generator_audit_nav_distribution` | generator distribution audit (cover) | — |
| `plan_step_is_identity_on_count_and_defaults_component` | model-based | 6 |
| `plan_step_count_independent_of_granularity` | metamorphic | 4 |
| `is_layer_boundary_matches_reference` | model-based | 6 |
| `is_layer_boundary_reflexive_and_strict` | property (both Some arms) | 5 |
| `is_layer_boundary_none_is_never_boundary` | property | 5 |
| `generator_audit_layer_boundary_distribution` | generator distribution audit (cover) | — |

**New tests:** 9 in `tick.rs` (`mod tick_id_contract_tests`), 6 in `step_driver.rs`
(`mod prop_tests`). All green. `cargo clippy --workspace --all-targets -D warnings`
clean; `cargo fmt --check` clean; full `rocket-surgeon-worker` suite: 147 passed.

## Generator-distribution evidence

- **`tick.rs` nav generator** (`generator_audit_nav_distribution`, N=2000 sampled
  sequences): >50% of sequences contain a token-reset (`NextToken`), >50% contain
  an operator advance, and high-half `SeekToken` values (≥ u64::MAX/2) are
  present — so we genuinely reach the region near the overflow boundary that
  BRAVO's generator (SetTokenPosition capped at `0..1000`, `tick.rs:374`) could
  never touch.
- **`step_driver` boundary generator** (`generator_audit_layer_boundary_distribution`,
  N=4000): the natural `(Option<layer>, layer)` generator hits the
  `Some(p) if p == new` (same-layer, no-boundary) branch only **~0.6%** of the
  time — recorded in-test as the reason `is_layer_boundary_reflexive_and_strict`
  *constructs* `Some(prev), prev` instead of waiting for a random collision.

## BUG FOUND & FIXED — `advance_token` integer overflow

- **Minimal failing input:** `set_token_position(u64::MAX); advance_token()`.
- **Symptom:** panic `attempt to add with overflow` (debug) / silent wrap to `0`
  (release) — the release wrap silently violates token-clock monotonicity.
- **Root cause:** `tick.rs` `advance_token` did `self.token += 1`. `set_token_position`
  (`tick.rs:90`) accepts any `u64`, so `token` can already sit at the ceiling.
- **Why it was latent:** BRAVO's `SetTokenPosition` generator caps at `0..1000`
  (`tick.rs:374`), and `any::<u64>()` essentially never samples *exactly* `u64::MAX`
  in 256 cases — the boundary must be injected explicitly. The MIKE generator
  does `prop_oneof![Just(u64::MAX), Just(MAX-1), Just(MAX-2), any::<u64>()]` plus a
  deterministic ceiling witness. (Lesson: boundary values are not "free" from a
  uniform generator.)
- **Reachability:** `set_token_position` currently has **zero non-test callers**
  (whole-repo grep), so this is a public-API-surface defect, not a live crash
  path today — but it is reachable the moment a reverse/seek path is wired up.
- **Fix:** `self.token = self.token.saturating_add(1)` — matches the existing
  `wall_ns` saturating idiom in the same file (`tick.rs:142`). Saturating, not
  wrapping, preserves monotonicity at the ceiling. `operator`/`step_count` keep
  bare `+= 1`: they'd need 2^64 advances to overflow (physically unreachable),
  whereas `token` is reachable in a single call. Minimal, targeted, idiom-matching.

## ⚠️ ADR / IMPL CONTRADICTION — for the protocol owner (NOT fixed here)

Per the brief, this is **pinned, not silently resolved.** The impl and the ADR
disagree about what `tick_id` *is*:

- **ADR-0005-tick-model.md:83** — "`tick_id` is a monotonic `u64`, **never reused,
  never reset within a session**. It is **the primary key** for checkpoints, probe
  firings, intervention attachment, and session bundle references."
- **ADR-0005-tick-model.md:84** — replayed ticks get *fresh* `tick_id` with
  `replay_of: Option<u64>`.
- **`tick.rs:96-98`** — `tick_id()` returns `self.operator`.
- **`tick.rs:67-80`** — `advance_token()` resets `self.operator = 0` every token.
- **`tick.rs:13-15`** (struct doc) — explicitly states `tick_id` "is NOT a global
  monotonic counter. It resets to 0 at each new token."

**Consequence (proven by test):** the impl's `tick_id` **collides across tokens** —
it is not unique, not monotonic, not a primary key. `tick_id_is_not_a_unique_session_key_but_clock_and_step_count_are`
shows distinct ticks share a `tick_id`, while the full `(token, operator)` clock
pair and `step_count` are unique. `cross_token_tick_id_collision_witness` is the
two-line concrete counterexample (token 0 op 1 and token 1 op 1 both report
`tick_id == 1`).

**This is observable on the wire:** `dispatch.rs:461` and `dispatch.rs:1641` surface
this `tick_id` as the protocol-level identity (e.g. `pre_replay_tick`), so any
client using `tick_id` as a checkpoint/probe/intervention key per the ADR will get
cross-token collisions.

**The field that *does* satisfy the ADR contract** is `step_count` (`tick.rs:21-23`,
"Total operators traversed across the whole session"), currently marked
"diagnostics only, never surfaced as tick_id". `step_count_is_the_monotonic_never_reset_clock`
pins that it is non-decreasing, +1 per advance, and untouched by token motion.

**Also unimplemented vs. ADR-0005:87-102:** the backward-tick schema. `to_tick_position`
hardcodes `direction: StepDirection::Forward` (`tick.rs:127`) and `replay_of: None`
(`tick.rs:132`); `TickState` never emits `Backward`. This matches the ADR's "backward
support deferred to Phase 8+" note, so it is consistent *as deferred* — but it means
"reverse step sequences" in this lane are modeled as `set_token_position` seeks (the
only backward motion the cursor offers), not true reverse traversal. Pinned by
`tick_id_aliases_operator_and_direction_is_always_forward` (direction always Forward)
and `seek_token_preserves_operator` (a backward seek moves only the token clock).

**Resolution is the protocol owner's call** — either (a) reconcile the ADR to the
implemented operator-alias semantics and name `(token, operator)`/`step_count` as
the identity, or (b) change the impl to surface a true monotonic `tick_id`
(candidate: `step_count`) and keep `operator` separate. The contract tests are
written to **break loudly** under either reconciliation, flagging it as deliberate.

## Gaps left / not in scope

- `bridge.rs` (PyO3 FFI, 0 tests) and `main.rs` (entrypoint, 0 tests) remain
  uncovered. `bridge.rs` is runtime-PyTorch-dependent (the dragons live there);
  pure-Rust property tests can't reach it without a torch process — out of scope
  for this lane and better left where the runtime dependency is honest.
- True reverse-traversal tick semantics are unspecified pending the ADR
  reconciliation above; testing them now would mean inventing the contract.
- The ADR contradiction itself is a finding for the protocol owner, deliberately
  left unresolved here.
</content>
</invoke>
