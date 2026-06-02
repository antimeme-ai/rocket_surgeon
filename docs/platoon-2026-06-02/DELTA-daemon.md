# PLATOON-FINDINGS — DELTA (daemon session FSM)

**Brief:** B002 — raise the test suite to MATERIA oracle standards.
**Lane:** `crates/rocket-surgeon` — stateful model-based testing of the
`Session` `Status` state machine and dispatch; model the legal transitions,
fuzz the illegal ones.
**Branch:** `platoon/test-daemon`.

## What I did

Added one test module, `crates/rocket-surgeon/src/session_fsm_proptest.rs`
(wired into `main.rs` under `#[cfg(test)]`), plus `proptest` as a dev-dependency
(workspace + crate). The suite for this crate had **0** property/stateful tests
(289 example-based fns at oracle tiers 2–3). This module climbs to tiers 5–6.

### Techniques applied

| Technique | Oracle tier | Where |
| --- | --- | --- |
| **Stateful model-based testing** (abstract model in lockstep, full-equality + projection oracles, integrated shrinking) | 6 (model) | `fsm_matches_model_under_random_action_sequences` |
| **Exception-raising properties** (invalid input → the *right* `ErrorCode`, never panic/silent no-op) | 5 (property) | 8 focused `proptest!` cases |
| **Worldline tree structural invariant** (metamorphic/property on the segment forest) | 4–5 | `check_worldline_shape` |
| **Generator-distribution measurement** (classify the corpus, assert non-triviality) | — | `generator_distribution_is_non_trivial` |

The stateful harness generates random sequences (1–40) of FSM actions —
**legal and illegal** — over the command set `{Initialize, Attach, Detach,
Status, Step, CheckpointCreate/Delete/Restore, AdvanceWorldline}`. It drives the
real `Session` and an independently-written abstract `Model` in lockstep and,
after **every** action, asserts:

1. **Outcome oracle:** real `Ok`/`Err` matches the model's prediction down to
   the `ErrorCode` (this is the model-based property — it checks the prediction
   on *every* step, not one sampled postcondition).
2. `status` agreement, and that the FSM only ever occupies the three synchronous
   states `Uninitialized/Initialized/Stopped` (never an intermediate state).
3. `available_actions` is exactly the pure function of `status`.
4. `model_id.is_some()` iff `Stopped`.
5. **`session_id` stability** — empty iff `Uninitialized`, and once minted it
   never changes across attach/detach/re-attach cycles.
6. **Checkpoint-list projection** equals the model's `(id, tick, layer)` list
   (Hughes' abstraction-function-to-a-simpler-type), and is empty whenever not
   `Stopped`.
7. **Worldline full structural equality** vs the model, plus an independent
   tree-shape invariant.

### New test count

**10** new test functions (2 stateful/distribution + 8 exception-raising
properties), each running 256–500 generated cases → ~3,000 generated
scenarios/run. Total crate tests: 289 → **298**, all green.
`cargo clippy --workspace --all-targets -- -D warnings` clean (with
`PYO3_PYTHON` pointed at the main-checkout `.venv`; see Environment note).

### Generator distribution evidence

Measured over 500 sequences / 10,294 commands (`generator_distribution_is_non_trivial`,
deterministic seed):

```
reached Stopped:                 47.8%   (52% stay pre-attach → illegal-path coverage)
rejected (illegal) commands:     66.5%
sequences w/ checkpoint create:  30.0%
sequences w/ worldline advance:  87.4%
sequences w/ successful restore:  8.2%
sequences w/ successful attach:  47.8%   detach: 19.2%
```

The corpus is genuinely rich and exercises **both** regions (post-attach
`Stopped` verbs *and* pre-attach rejection paths). The test asserts a band
(reached-Stopped ∈ 35–95%, rejection ∈ 10–85%, etc.) ~25% below observed, so it
fails loudly if the generator ever degenerates to trivial inputs.

## Bugs found

### BUG-1 (fixed) — self-parented worldline root

`Session::advance_worldline_segment` (`session.rs:1109`) never seeded a root
segment, so the **first** call minted segment `id=0` with
`parent_segment: Some(0)` — a node that is its own parent. The root of a tree
must be parentless. This function had **zero** prior test coverage.

- **Minimal failing case** (shrunk by proptest): a single
  `advance_worldline_segment(0)` on a fresh session →
  `WorldlineSegment { id: 0, parent_segment: Some(0), branch_tick: Some(0), .. }`
  where the correct value is `parent_segment: None, branch_tick: None`.
- **Root cause:** the method always set `parent_segment: Some(current_segment)`
  and `current_segment` is `0` before the first push, so the root pointed at
  itself. The worldline `WorldlineState::default()` starts empty and nothing
  (attach included) seeds a root, so `advance` was conflating "create the root"
  with "branch from the cursor."
- **Fix (minimal, my lane):** when the tree is empty the new segment is the
  root — `parent_segment` and `branch_tick` are `None`; later segments still
  branch from the cursor. 4-line change in `session.rs`; no production refactor.
  Caught by the stateful model (which encodes the corrected shape) and pinned by
  `check_worldline_shape` + a committed `proptest-regressions` seed.
- **Blast radius:** `dispatch.rs` `session.export` and `main.rs:230/257`
  serialize `parent_segment`; the self-parent would surface in exported
  worldline JSON and any future tree-walk (a `parent==id` cycle). Replay
  divergence (BRAVO's lane) walks segments — worth a cross-check there.

## Weak oracles / gaps left for follow-up

- **GAP-1 — checkpoint state-guard lives only in dispatch.** `Session::
  checkpoint_create/_with_id/_delete/_restore/_bookmark` and
  `set_intervention/clear_intervention` have **no internal precondition**; the
  `require_stopped` guard is enforced only in `dispatch::handle_checkpoint` /
  `handle_intervene`. Calling the `Session` method directly while `Initialized`
  silently succeeds. My harness reproduces the dispatch composition
  (`require_stopped` then the method) so it tests the real system contract, but
  the model layer lacks defense-in-depth. Not changed (would widen scope). A
  belt-and-suspenders guard inside the methods would let a future caller skip
  dispatch safely.
- **GAP-2 — replay excluded from the stateful command set.** `Session::replay`'s
  worker-fallback path computes `stopped_at.tick_id = current_tick + ticks` with
  branchy `saturating_sub(...).max(1)` arithmetic; a faithful parallel model is
  fragile. Replay's FSM edges (`MODEL_NOT_ATTACHED` when not stopped,
  `CHECKPOINT_NOT_FOUND` for unknown ids) are covered by the exception
  properties, but its tick arithmetic and worldline side-effect are **not**
  model-checked here. Candidate for a dedicated metamorphic test (overlaps
  BRAVO's replay-divergence lane).
- **GAP-3 — intermediate `Status` states are unreachable.** `Attaching`,
  `Stepping`, `Inspecting`, `Modifying`, `Replaying`, `Detaching` are defined in
  the protocol enum but never entered by the synchronous daemon FSM (an
  asserted invariant here). If async/streaming transitions land later, the model
  needs new edges. Flagged so it isn't mistaken for dead code.
- **GAP-4 — `detach` does not reset the worldline.** Faithfully modeled (the
  worldline persists across attach/detach), but it means a re-attached model
  inherits the prior model's worldline tree. May be intentional (worldline =
  connection-scoped) or a latent leak; left as an observation for the owner.

## Stop condition

Met: new property/stateful/metamorphic tests green (`cargo test -p
rocket-surgeon` → 298 passed), workspace clippy `-D warnings` clean, this note
committed on `platoon/test-daemon`. BUG-1 recorded *and* fixed (4-line, in-lane);
no PR opened, no other lane touched.
