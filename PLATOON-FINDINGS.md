# PLATOON-FINDINGS — NOVEMBER (Adversary): Mutation Re-Audit (Wave 2)

**Brief:** B004 / NOVEMBER. Re-run `cargo-mutants` against **current master**
(carrying Wave-1's +104 tests, PR #49 / `a40e9aa`). Two jobs:
1. **Confirm Wave-1 oracles** — re-measure FOXTROT's exact scopes and report the
   per-crate mutation-score delta vs his Wave-1 baseline
   (`docs/platoon-2026-06-02/FOXTROT-mutation.md`).
2. **Aim Wave-2 oracles** — a refreshed, prioritized surviving-mutant report on the
   modules **INDIA / JULIET / KILO / MIKE** own this wave.

This lane does **not** fix production code and (by FOXTROT's scope discipline, see
below) does **not** add tests inside the four lanes' actively-owned crates.

**Tool:** `cargo-mutants 27.0.0`, copy-mode, on this isolated worktree.
**Mutation score = caught / (caught + missed)**, excluding unviable (won't-compile).
A *survivor* (MISSED) = we changed behavior and **every existing test still passed**
— no oracle constrains it. Counts are apples-to-apples with FOXTROT: identical tool,
identical scopes, identical mutant populations (verified — see protocol/transport).

## Harness notes (for the commander)

- **pyo3 interpreter.** Exported
  `PYO3_PYTHON=/Users/patrickbeam/projects/rocket_surgeon/.venv/bin/python`
  (absolute → survives cargo-mutants' source copy) for every run. The `.venv`
  symlink in this worktree is enough for the pre-commit hook but cargo-mutants
  copies the tree, so the absolute path is required.
- **Parallelism.** Wave-2 job at `-j6`; Wave-1 re-confirmation job at `-j4`, run
  concurrently. On a less-loaded machine `-j8` per FOXTROT is fine and faster.
- **Arid-node policy (Google / Petrovic 2021).** Survivors that are pure noise —
  `Debug::fmt`, `Drop::drop`, `len`/`is_empty` accessor stubs that mirror a
  tested `len`, version/string getters returning `""`/`"xyzzy"`, and **config
  constants** (e.g. `DEFAULT_MAX_BYTES = 2*1024*1024*1024`) — are *counted in raw
  scores* but excluded from the high-signal target lists. They are flagged inline.
- **Equivalent mutants** are called out explicitly where found (varint `| → ^`,
  several float min/max boundary flips). These are NOT test gaps; no oracle can
  kill them and they should not be chased.

---

## Part 1 — Wave-1 confirmation: did the +104 tests move the needle?

| Crate | Scope | FOXTROT baseline | **Wave-2 re-audit** | Δ score | Verdict |
| --- | --- | --- | --- | ---: | --- |
| `rocket-surgeon-transport` | full | 9 / 4 → **69%** | 9 / 4 → **69%** | **0** | **PBT landed, 0 kills** |
| `rocket-surgeon-protocol` | full | 31 / 27 → **53%** | 31 / 27 → **53%** | **0** | **PBT landed, 0 kills** |
| `rocket-surgeon-shm` | full | 91 / 84 → **52%** | 125 / 50 → **71%** | **+19pp** | strong |
| `rocket-surgeon-worker` | scoped\* | 172 / 54 → **76%** | 196 / 30 → **87%** | **+11pp** | strong |
| `rocket-surgeon` | `session.rs` | 58 / 116 → **33%** | 67 / 109 → **38%** | **+5pp** | partial |

\* worker scope = `checkpoint.rs, step_driver.rs, replay.rs, tick.rs, kv.rs`
(identical to FOXTROT). Mutant populations match baseline exactly (transport 17,
protocol 66, shm 185, worker 244) — the deltas are pure oracle gains, not
generation drift.

### Finding W-1 (HIGH): `protocol`/`transport` got property tests in Wave 1 that
**killed zero mutants** — the new oracles miss exactly the surviving behaviors.
PR #49 *did* land ALPHA's `protocol/tests/proptest_wire_format.rs` (46 tests,
incl. a `.proptest-regressions` file — PBT found a failing case during dev) and
CHARLIE's `transport/tests/prop_framing.rs` (285 lines, with a
`body_generator_distribution` classifier, FIFO roundtrips, `message_too_large`,
`truncated_body_errors`). I verified cargo-mutants **runs** these (baseline
"build + 2s test"; `cargo test --test proptest_wire_format` → 46 passed). Yet the
mutant population *and* the caught/missed split are **byte-identical** to FOXTROT
(protocol 31/27/8, transport 9/4/4). The properties added robustness but moved the
**mutation score not at all**, because every survivor is structurally off the path
the new generators exercise:
- **`checkpoint_layers`** (protocol `lib.rs:9-15`) — a pure function **not on the
  serde wire-format path**. All 8 mutations survive (body → `vec![]`/`vec![0]`/
  `vec![1]`, `<=`→`>`, `/`→`%`/`*`, `*`→`+`/`/`). FOXTROT *predicted* this exactly:
  "the roundtrip property will NOT catch A1-A3 — they aren't on the serde path."
- **`ErrorCode::numeric_code`** (`errors.rs`, 14) + **JSON-RPC constants**
  (`jsonrpc.rs:129-133`, 5) — spec-value pinning, which a roundtrip generator
  never asserts (it round-trips whatever value is there).
- **transport** (4): `framing.rs:3` frame-size const `*`→`+` (arid-ish);
  `framing.rs:50` max-length `>`→`>=` — the `message_too_large` property tests
  `len > max` (reject) but never `len == max` (accept), so the boundary flip is
  invisible; `stdio.rs:44` `send_response → Ok(())` — no test asserts bytes reach
  the writer (a side-effect oracle, not a roundtrip).

**This is the central MATERIA lesson of the re-audit, not a scolding:** tier-4
roundtrip properties are necessary but do **not** subsume tier-6 model/value-pinning
(`checkpoint_layers`, error codes) or tier-2/boundary exception oracles (the `==max`
acceptance case, the `send_response` side effect). FOXTROT's A1/A2/A3 targets
remain the fix and are cheap (a `checkpoint_layers` reference model + an error-code
table + one exact-boundary framing case). No active Wave-2 lane owns these crates,
so they stay at 53% / 69% unless the commander assigns them.

### Finding W-2 (MED): worker improved, but `kv_metric` (FOXTROT B5) never landed.
Worker rose 76→87%. `replay.rs::should_stop` (FOXTROT's #1 worker target, B1) is
now **fully killed**, and `tick.rs` / `step_driver.rs` / `replay.rs` have **zero**
surviving non-arid mutants (BRAVO's stateful model + replay properties worked).
But the 30 remaining survivors are concentrated in **`kv.rs` (15) and
`checkpoint.rs` (15)**. **`kv_metric` is still 100% unconstrained** — body →
`0.0`/`1.0`/`-1.0` and every operator (`+ - * / %` at lines 94/97/98) survives.
FOXTROT's B5 (a `kv_metric` range/monotonicity/reference-value property) was not
written. Checkpoint spill/offset arithmetic (B4) is the other remaining cluster.

### Finding W-3 (MED): daemon `now_rfc3339` (FOXTROT D4) and `find_checkpoint_before`
(D2) never landed. Of the 109 daemon survivors, **~74 are still inside the
hand-rolled `now_rfc3339` civil-date routine** (`session.rs:60-75`) — the cheap,
high-count win FOXTROT flagged (pin a handful of epoch→RFC-3339 pairs). The
FSM-proper improved modestly (some `validate_attach`/`step` predicates now caught),
but `find_checkpoint_before` (8 survivors, the replay start-checkpoint selector —
correctness-critical) and `update_arena_utilization` (8) remain unconstrained.

**Bottom line:** the two lanes that did **stateful model-based** work (CHARLIE-shm,
BRAVO-worker) delivered the largest gains — the oracle hierarchy paid off exactly as
MATERIA predicts. The lanes that wrote **roundtrip-only** properties (ALPHA-protocol,
CHARLIE-transport) added passing tests but **zero mutation-score movement** — a
textbook demonstration that the *style* of property determines the power. And the
partially-landed work (daemon-`now_rfc3339`/`find_checkpoint_before`,
worker-`kv_metric`) is still wide open.

---

## Part 2 — Wave-2 target survivor report (INDIA / JULIET / KILO / MIKE)

| Lane | Crate / module | Mutants | Caught | **Missed** | Score | Runtime (-j6) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **INDIA** | `rocket-surgeon` `tensor_stats.rs`+`tensor_store.rs` | 304 | 193 | **95** | **67%** | 4m |
| **JULIET** | `rocket-surgeon-probes` `grammar.rs` | 67 | 28 | **0** | **100%** | 30s |
| **KILO** | `perfetto-writer` (full) | 94 | 73 | **19** | **79%** | 31s |
| **MIKE** | `rocket-surgeon-worker` `tick.rs` | 26 | 25 | **0** | **100%** | 67s |

### INDIA — `tensor_stats.rs` (62 survivors) + `tensor_store.rs` (33) *(highest signal this wave)*

This is where the survivors live. The example suite is genuinely good (numpy
reference, Welford-merge cases, NaN/Inf handling) but it pins *fixed* inputs and
leaves the operator algebra and half the dtype matrix unconstrained.

**I1. Five dtype decoders have ZERO test (21 survivors) — top priority, trivial.**
`compute_summary`'s example tests cover only `Float32/Float16/Bfloat16/Int32/Bool`.
**`read_f64`, `read_i8`, `read_i16`, `read_i64`, `read_u8` are never exercised** —
each whole-body mutation survives (`-> vec![]`/`vec![0.0]`/`vec![1.0]`/`vec![-1.0]`).
→ INDIA's **model-based decode property**: for every one of the 10 `DType`s,
generate values, encode to LE bytes, `decode_values`, and assert equality to a
naive reference decoder — *including* NaN/Inf/denormal/empty per the brief. Kills
all 21 at once and is exactly the brief's mandate.

**I2. `read_bool_values` inversion is invisible to the symmetric example (1).**
`tensor_stats.rs:351 != → ==` survives because `bool_dtype_stats` uses
`[1,1,0,1,0,0]` → mean 0.5 / sparsity 0.5, which is *invariant under inversion*.
→ Use an **asymmetric** bool case (e.g. `[1,1,1,0]` → mean 0.75) or a property
`mean(bool bytes) == (count of non-zero)/n`.

**I3. Chan–Golub–LeVeque `merge_pass1` arithmetic (~25 survivors) — the brief's
named metamorphic target.** Lines 225-264: the finite-count deltas (`-`→`+`), the
Welford merge terms (`+`→`-`/`*`), and especially the **L2 rescale branches**
(252-264: `>=`→`<`, `&&`→`||`, `>`→`==`/`<`/`>=`, `/`→`%`/`*`, `*`→`+`/`/`) nearly
all survive. The two example merge tests use fixed 50/50 and 500/500 splits with
loose tolerances. → INDIA's **metamorphic property**
`merge(stats(A), stats(B)) ≈ stats(A ∪ B)` over **generated** A,B with varied
sizes *and varied magnitude scales* (the L2 rescale branch only activates when
`a.l2_scale` and `b.l2_scale` straddle), plus **order-invariance** of mean/var.
Classify the generator so the cross-scale and empty-partition cases are actually
hit — that is what kills the rescale cluster.

**I4. Histogram binning + pass1 boundaries (~13 survivors).** `compute_pass2:168`
`(x - range_min) / range` — `-`→`+`, `/`→`%`/`*` survive; `:183`/`:193` top-k and
edge `>`→`>=`. `compute_pass1` min/max/sparsity/L2 comparison flips (103,106,110,
115,120,121,127). → A model-based histogram check (expected bin counts for a known
distribution) kills 168/183/193. **CAUTION — likely equivalent mutants:** the
min/max idempotent flips `if x < min` → `<=` (103) and `if x > max` → `>=` (106),
and abs_max `>`→`>=` (110), produce identical results (assigning `min = x` when
`x == min` is a no-op). Don't burn oracle effort on these; a sparsity-epsilon
boundary test (`ax == 1e-8` exactly) is the only boundary flip here worth a case.

**I5. `tensor_store` LRU is a stateful-model gap (~20 survivors).** The
`access_generation += 1` counter (lines 79,137,157,179,188,214,232,259) mutates to
`-=`/`*=` and survives — the example LRU tests assert *which* id is evicted but
never pin the **generation ordering** across a long mixed access pattern.
Critically, **`insert_with_id`'s eviction loop is entirely untested** (150/151/149:
`&&`→`||`, `delete !`, `>=`→`<`, `>`→`==`/`<`/`>=`, `+`→`-`/`*` all survive) — the
eviction tests only drive `insert`. → INDIA's **dict + reference-LRU model** (track
a `HashMap` + an ordered access log; assert real eviction id == model after each op)
over generated `insert`/`insert_with_id`/`get`/`raw_data` sequences kills this
cluster. Also untested: `ids()`, `raw_data()`, `is_empty()` (one-liner kills);
`DEFAULT_MAX_BYTES` const mutation (9:36/43/50) is **arid** — ignore.

### JULIET — `grammar.rs`: already 100% on cargo-mutants' fault model (0 survivors).

Honest result: the existing example suite (roundtrip `parse∘render`, the `reject_*`
exception cases, wildcard-matching matrix) **already kills every viable mutant**
(28 caught / 0 missed; 39 unviable are winnow-combinator artifacts). cargo-mutants
finds *nothing* for JULIET to aim at.

This does **not** make JULIET's property work pointless — but it reframes it. The
value is **beyond cargo-mutants' single-point fault model**, not in killing
survivors:
- **Roundtrip as a property** (generate `ProbePoint` → `render` → `parse` → assert
  `id`) hardens against *combinations* the 9 hand-picked examples miss (deep
  component paths, multi-indexed segments, mixed wildcard/MoE forms). cargo-mutants
  can't mutate "the example you forgot to write."
- **Exception-raising properties** (113× leverage per MATERIA) — generate malformed
  strings and assert the *specific* `ParseError` + that `offset` points at the
  right column. The `offset` field is currently never asserted by any test — a
  future regression there is invisible to both the suite and cargo-mutants today.
- **Classify the generator** (per the brief) to *prove* wildcards / `experts[i]` /
  multi-segment paths are actually exercised, not just assumed.

Recommend JULIET still add `proptest = "1"` and the property + exception suite, and
**state in their findings that the mutation score was already 100%** so the
commander knows the property tests are defense-in-depth, not gap-closing.

### KILO — `perfetto-writer` (19 survivors): a prost-decode roundtrip is incomplete.

The writer tests *already* prost-`decode` written packets — but they assert only a
subset of fields, so "delete field X" survives. **17 of 19 survivors are
field-level decode gaps; 2 are accessor/equivalent.**

**K1. Track/event field assertions are partial (15 "delete field" survivors).**
`write_thread_track` (`name`), `write_counter_track` (whole fn → `Ok(())`, plus
`track_descriptor`/`uuid`/`parent_uuid`/`name`/`counter`/`unit_name`), `slice_end`
(`trusted_packet_sequence_id`), `instant` (`timestamp`/`seq_id`/`track_uuid`),
`counter_double` (`timestamp`/`seq_id`/`track_uuid`), and `flush → Ok(())`.
`write_process_track` and `write_track` are *fully* asserted (no survivors) — they
are the template. → KILO's **model-based roundtrip**: build each packet type from
generated params, write, prost-decode, and assert **every** field equals the input
(the brief's "field values survive"). A single well-built generator over the writer
API kills all 15. `flush → Ok(())` needs a test that asserts bytes actually reached
the inner writer.

**K2. `InternTable::is_empty` untested (2, one-liner).** `intern.rs:45` → `true`/
`false` survive. Trivial: `assert!(t.is_empty())` then `t.intern("x");
assert!(!t.is_empty())`. The brief's interning-as-a-function property (same string →
same id, distinct → distinct, table consistent with use) is already example-covered;
generalize it to a property and fold in `is_empty`.

**K3. EQUIVALENT MUTANT — do not chase.** `varint.rs:9 | → ^` survives and is
**provably equivalent**: `byte = (value & 0x7F)` guarantees bit 7 is clear, so
`byte | 0x80` and `byte ^ 0x80` produce identical bytes for every input. No oracle
can distinguish them. KILO's encode∘decode roundtrip over the full u64 range
(0, 2^7±1, …, u64::MAX) is still worth writing for the *other* varint logic, but
this specific survivor should be annotated equivalent, not "fixed."

### MIKE — `tick.rs`: already 100% (0 survivors) + the ADR contradiction stands.

`tick.rs` is at **25 caught / 0 missed** — BRAVO's Wave-1 stateful model
(`tick_state_matches_model` + the three metamorphic properties) already kills every
viable mutant. MIKE has no survivors to aim at; their job is verification + the
contradiction note.

**M1. (FINDING, not a fix) ADR ↔ impl tick_id contradiction, pinned to the code.**
`crates/rocket-surgeon-worker/src/tick.rs:14-15` and `:89-92` document and implement
`tick_id` as an **alias for the operator clock that RESETS to 0 each token**
(`advance_token` sets `operator = 0`, `tick.rs:69`). The existing tests
(`advance_token_increments_token_and_resets_operator:223-224`, and the model in
`prop_tests`) **pin the reset behavior** — i.e. they encode the *impl* side. Per the
B004 brief and FOXTROT's codebase scout, the ADR says `tick_id` is
*monotonic-never-reset*. The suite is internally consistent and correctly tests
what the code does; it does **not** silently bless the ADR. This is a
**protocol-owner decision**, not a test bug — surfaced here with file:line so it
can be resolved upstream. MIKE should restate this in their findings; do not
"fix" it in a test lane.

---

## Equivalent mutants found (documented, not chased)

| Location | Mutation | Why equivalent |
| --- | --- | --- |
| `perfetto-writer/varint.rs:9:23` | `\| → ^` | `byte` masked to `0x7F`; bit 7 always clear, so `\|0x80 ≡ ^0x80`. |
| `tensor_stats.rs:103` | `if x < min` → `<=` | assigning `min = x` when `x == min` is a no-op → identical min. |
| `tensor_stats.rs:106` | `if x > max` → `>=` | same idempotence on max. |
| `tensor_stats.rs:110` | abs_max `>` → `>=` | same idempotence on abs_max. |

(The min/max trio are the classic example-vs-property trap: a *property* "result
equals the true min/max" also can't kill them, because they don't change the
result. Correctly identified as equivalent rather than counted against INDIA.)

---

## Runtimes (this machine, copy-mode)

Wave-2 job (`-j6`, sequential): perfetto 31s · grammar 30s · tick 67s · tensor 4m0s
→ **≈ 5.5 min** total. Wave-1 job (`-j4`, sequential, run concurrently):
transport 32s · protocol 2m0s · shm 4m0s · worker-scoped 7m0s · daemon 3m0s
→ **≈ 16 min**. Both jobs in parallel finished in **≈ 16 min** wall. Re-running a
single lane after a builder lands is ≤4 min each (tensor is the slow one).

## Scope discipline (why NOVEMBER added no patches)

Every Wave-2 survivor lands inside a crate **actively owned by another lane this
wave** (INDIA=`tensor_*`, JULIET=`grammar`, KILO=`perfetto-writer`,
MIKE=`tick`). Per FOXTROT's precedent, adding a test inside a builder-owned file
would collide with their branch at integration. NOVEMBER therefore delivers
**targets, not patches** — each builder kills their own survivors with the
oracle named above. The one-line kills are flagged (I1, I2, K2, store `ids`/
`raw_data`/`is_empty`) so they're cheap to land in the owning branch. The
protocol/transport survivors (Finding W-1) are *not* owned by any active lane and
are out of NOVEMBER's focus scope — flagged for the commander to assign, not
fixed here.

## Gaps left for a follow-up

- **protocol / transport — PBT landed but 0 kills** (Finding W-1): roundtrip
  properties shipped; the 31 survivors (`checkpoint_layers`, error-code table,
  JSON-RPC constants, transport framing `==max` boundary / `send_response` side
  effect) are off the roundtrip path and need model/value-pinning + boundary +
  side-effect oracles. All cheap, no current owner.
- **worker `kv_metric` (W-2)** and **daemon `now_rfc3339` + `find_checkpoint_before`
  (W-3)** — Wave-1 lanes that didn't fully land; ~90 survivors between them.
- **Unmeasured code** (unchanged from FOXTROT): worker `dispatch.rs`/`adapter.rs`/
  `bridge.rs`; daemon `dispatch.rs`/`notifications.rs`/`perfetto_sink.rs`; the
  secondary crates `orchestrator`, `tui`, `python`; and **mutmut on Python** (the
  three new `python/tests/*_properties.py` from PR #49 are unmeasured by any Rust
  pass). A dedicated Python mutation pass remains owed.
- **perfetto-writer `proto.rs`** generated-code mutants were included in the full
  run; none surfaced as high-signal beyond the writer field gaps above.
