# PLATOON-FINDINGS — BRAVO (worker replay/checkpoint, stateful model-based)

Lane: `crates/rocket-surgeon-worker`. Brief: B002. Branch: `platoon/test-worker`.

Baseline before this work: 107 worker unit tests (all example-based, tiers 2-3),
**0** property/stateful tests. After: **132** tests; **25 new** property /
stateful-model / metamorphic / exception-raising tests.

Build note: PyO3 (`auto-initialize`) needs an interpreter. This worktree has no
`.venv`, so the crate is built/tested with
`PYO3_PYTHON=/Users/patrickbeam/projects/rocket_surgeon/.venv/bin/python`.
The crate is **binary-only** (no `lib.rs`), so property tests live inside the
modules' `#[cfg(test)] mod prop_tests` (an integration `tests/` dir cannot reach
the crate's private items / modules).

## How to run

```
PYO3_PYTHON=/Users/patrickbeam/projects/rocket_surgeon/.venv/bin/python \
  cargo test -p rocket-surgeon-worker --bin rs-worker prop_tests
PYO3_PYTHON=…/.venv/bin/python cargo clippy --workspace --all-targets -- -D warnings
```

Gate status: 132/132 pass; `cargo clippy --workspace --all-targets -- -D warnings`
clean; `cargo fmt --check` clean.

## Techniques applied (by oracle tier)

| Test | Tier | Technique |
| --- | --- | --- |
| `checkpoint::arena_matches_model` | 6 | **Stateful model-based** — random alloc/free op sequences vs. an independent count/set/order model; asserted after every op; proptest shrinks |
| `checkpoint::arena_exhaustion_is_exact` | 4/exc | **Exception-raising + metamorphic** — alloc fails iff full; freed slots reusable |
| `checkpoint::oldest_evictable_matches_model` | 6 | **Model-based** — eviction parser vs. independent reimplementation |
| `checkpoint::step_and_checkpoint_are_independent` | 6 | **Stateful** — interleaved step/checkpoint ops prove the two subsystems never perturb each other |
| `checkpoint::spill_load_roundtrip` | 4 | **Metamorphic round-trip** — spill∘load preserves every payload byte + dtype/ndim/shape |
| `checkpoint::spill_corruption_always_detected` | exc | **Exception-raising** — flipping any byte in the payload region is always caught by CRC32 (never silent wrong-data, never panic) |
| `checkpoint::spill_truncation_rejected` | exc | **Exception-raising** — truncation at any offset never panics |
| `checkpoint::spill_bad_magic_rejected` | exc | bad-magic rejection |
| `checkpoint::slot_header_write_read_roundtrip` | 4 | **Roundtrip** — binary header serializer is its own inverse |
| `checkpoint::slot_header_read_total` | 2/exc | **Totality** — `read_from` never panics on arbitrary bytes; `Some` only when magic+dtype valid |
| `checkpoint::spill_index_entry_roundtrip` | 4 | **Roundtrip** — NVMe index-entry serializer |
| `checkpoint::dtype_from_u8_total` / `dtype_torch_str_roundtrip` | 4/exc | **Roundtrip + exception** for both dtype codecs |
| `checkpoint::align_up_properties` / `align_up_monotonic` | 5 | **Property** — full algebraic contract (≥, aligned, <align overshoot, idempotent, monotonic) |
| `tick::tick_state_matches_model` | 6 | **Stateful model-based** — the step-driver three-clock cursor vs. an independent model after every advance/advance_token/set_phase/set_token_position |
| `tick::advance_does_not_touch_token` | 4 | **Metamorphic** — operator motion never moves the token clock |
| `tick::step_count_survives_token_resets` | 4 | **Metamorphic** — step_count is invariant across token boundaries |
| `tick::advance_token_phase_rule` | 5 | **Property** — Prefill→Decode transition; fixed point elsewhere |
| `replay::from_request_propagates_fields` | 6 | **Model-based** — `from_request` copies every control field verbatim (float fields compared bit-exact) |
| `replay::should_stop_matches_predicate` | 6 | **Model-based** — `should_stop` true for exactly the configured coordinate |
| `replay::no_stop_point_never_stops` | 4 | **Metamorphic** — no stop point ⇒ never stops, any coordinate |
| `replay::stop_point_fires_only_at_target` | 4 | **Metamorphic** — fires at target, false everywhere else |

The two crown jewels per the brief — the step-driver FSM (`tick.rs`) and the
checkpoint allocator (`checkpoint.rs`) — are both covered by **stateful
model-based** tests that maintain an abstract model in parallel and assert
`real == model` after every operation, with automatic shrinking of failing
sequences. Each model is an independent reimplementation (counts/sets vs.
free-list+mmap; a mirror struct vs. the real cursor), not shared code.

## Generator distribution evidence

`checkpoint::arena_generator_distribution` (a `#[test]` that samples 400 op
sequences and asserts coverage floors — fails if the generator goes trivial):

```
arena generator over 400 cases: total_ops=11778 alloc_ok=3462
  alloc_exhausted=2461 free_present=1987 free_absent=3868
  cases_hitting_exhaustion=303 cases_with_present_free=354
```

- **75.8%** of cases hit allocator exhaustion (alloc-on-full) — the exception
  path is genuinely exercised, not incidental.
- **88.5%** of cases free a *live* checkpoint (not a no-op free).
- Small alphabets (4 ckpt ids × 4 layers, capacity 1–5) deliberately force
  collisions, duplicate `(ckpt, layer)` allocations, and re-use of freed slots.

Asserted floors: ≥20% exhaustion, ≥20% live-free, and all of {alloc_ok,
alloc_exhausted, free_absent} > 0.

## Bugs / weak oracles found

**None in this lane.** All properties hold; the allocator, the binary
serializers, the spill/load path, the step-driver cursor, and the replay-control
predicates conform to their models across the generated input space. This is
honest conformance evidence, not a null result from weak oracles — the same
model-based tests *would* have caught: off-by-one free accounting, insertion-order
drift in `oldest_checkpoint`/`oldest_evictable`, a `tick_id`/operator alias break,
a Prefill→Decode transition bug, silent CRC bypass on corruption, and field-drop
bugs in `from_request` (the class of wire-format bug ALPHA's lane just hit).

Two design facts worth recording (verified, not bugs):
- Duplicate `alloc_slot(ckpt, layer)` consumes a second physical slot and
  overwrites the index entry; both slots are still owned by the checkpoint and
  returned to the free list on `free_checkpoint`. The model encodes this and the
  property confirms it.
- The step-driver cursor API (`tick.rs`) is **total** — no operation can fail or
  panic on any input, so there are no exception-raising properties to write
  there. Stated explicitly rather than left as an apparent gap.

## Gaps left for follow-up

1. **Replay-divergence metamorphic relations** (brief bullets: "no
   interventions ⇒ zero divergence", "replay twice ⇒ match", "intervention at
   tick T doesn't perturb divergence before T"). The divergence math is
   `bridge::compare_activations_from_ptr` (PyO3 → torch) driven by
   `dispatch::run_replay_loop` over real forward-pass mailboxes — it cannot be
   unit-tested in pure Rust without a live model host. The pure numerical
   relations (cosine/MRE) live in Python `replay.py` and are **ECHO's lane**
   (B002 ECHO targets exactly these metamorphic + exception properties). The
   Rust-side `ReplayContext` *control* surface is fully covered here.
2. **`cargo-mutants` confirmation.** These oracles claim sensitivity; FOXTROT's
   mutation run over `rocket-surgeon-worker` should confirm the new model-based
   tests actually kill the relevant mutants in `checkpoint.rs` / `tick.rs`. If
   surviving mutants remain in the allocator or cursor, point them back here.
3. **Concurrency.** `CheckpointArena` is `unsafe impl Send` with a `RefCell` and
   a documented single-thread assumption; if the worker ever goes multi-threaded
   a `loom` interleaving test (or the `RefCell`→`Mutex` swap noted in the source)
   would be required. Single-threaded model only, by design, today.
