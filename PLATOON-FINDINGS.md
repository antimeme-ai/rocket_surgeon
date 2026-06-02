# PLATOON-FINDINGS — BRAVO (worker replay/checkpoint, stateful model-based)

Lane: `crates/rocket-surgeon-worker`. Brief: B002. Branch: `platoon/test-worker`.
Baseline before this work: 107 worker unit tests, **0** property/stateful tests.

Build note: PyO3 (auto-initialize) needs an interpreter. This worktree has no
`.venv`; tests are run with
`PYO3_PYTHON=/Users/patrickbeam/projects/rocket_surgeon/.venv/bin/python`.

## Sub-plan (JSMNTL)

Pure-Rust testable surface in this crate (PyO3 divergence math lives in
`dispatch.rs`/Python `replay.py` — ECHO's lane; see Gaps):

1. **CheckpointArena — stateful model-based** (crown jewel). Abstract model =
   {free count, ckpt→slot-count, present (ckpt,layer) key set, insertion order}.
   Drive random alloc/free op sequences; assert real == model after every op;
   proptest shrinks. + exhaustion exception property.
2. **TickState (step-driver FSM) — stateful model-based.** Model the three-clock
   cursor (token/operator/step_count/phase/layer/component); drive
   advance/advance_token/set_phase/set_token_position; assert == model each step.
3. **Combined step+checkpoint op sequences** — interleave, prove independence.
4. **spill/load — metamorphic roundtrip** (checkpoint round-trips) +
   exception-raising (truncation, bad magic, bad version, single-byte corruption).
5. **Binary serializers — roundtrip + exception:** SlotHeader, SpillIndexEntry,
   DtypeTag, align_up properties.
6. **ReplayContext::should_stop — model-based + metamorphic** (stop_at semantics);
   oldest_evictable parser model-based.

(Findings, counts, distributions, bugs filled in below as work lands.)
