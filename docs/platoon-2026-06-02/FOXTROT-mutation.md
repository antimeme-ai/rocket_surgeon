# PLATOON-FINDINGS — FOXTROT (Adversary): Mutation Audit

**Brief:** B002 / FOXTROT. Measure where the *current* test suite's oracles are
blind, using `cargo-mutants`. Produce a prioritized surviving-mutant gap report
and rank where ALPHA / BRAVO / CHARLIE / DELTA should aim. This lane does **not**
fix production code, and (by design — see "Scope discipline" below) does **not**
add tests inside the builder-owned crates.

**Tool:** `cargo-mutants 27.0.0`. Runs in this isolated worktree.
**Mutation score = caught / (caught + missed)**, excluding unviable (won't-compile)
mutants. A *survived* (MISSED) mutant means: we changed the code's behavior and
**every existing test still passed** — i.e. no oracle constrains that behavior.

## Harness notes (for the commander)

- **pyo3 interpreter.** This worktree has no `.venv`; `crates/rocket-surgeon-worker`
  and `crates/rocket-surgeon-python` need a Python interpreter at build time
  (`.cargo/config.toml` pins `PYO3_PYTHON = .venv/bin/python`, relative). All runs
  here exported `PYO3_PYTHON=/Users/patrickbeam/projects/rocket_surgeon/.venv/bin/python`
  (absolute → survives cargo-mutants' source copy). Re-runs need the same.
- **Parallelism:** `-j 8` on a 16-core machine. Copy-mode (not `--in-place`) so
  jobs don't collide.
- **Scoping the big crates:** `rocket-surgeon-worker` (583 mutants) and
  `rocket-surgeon` (876) are too large to fully mutate cheaply. They were scoped
  with `-f <file>` to the builder-relevant modules (worker: checkpoint/step_driver/
  replay/tick/kv; daemon: session.rs). The *unscoped* remainder (dispatch.rs,
  adapter.rs, perfetto_sink.rs, etc.) is **not yet measured** — see "Gaps left".
- **Arid nodes:** per Google's mutation-testing practice, ignore survivors that are
  `Debug::fmt -> Ok(Default::default())`, `Drop::drop -> ()`, and version/string
  accessors (`name -> ""`/`"xyzzy"`). They are noise, not missing oracles. They are
  excluded from the "high-signal" lists below but counted in raw scores.

## Scoreboard

| Crate | Scope | Caught | Missed | Unviable | **Score** | Wall time (-j8) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `rocket-surgeon-transport` | full | 9 | 4 | 4 | **69%** | 25s |
| `rocket-surgeon-protocol` | full | 31 | 27 | 8 | **53%** | 59s |
| `rocket-surgeon-shm` | full | 91 | 84 | 10 | **52%** | 62s |
| `rocket-surgeon-worker` | scoped* | 172 | 54 | 18 | **76%** | 2m |
| `rocket-surgeon` | session.rs | 58 | 116 | 68 | **33%** | 2m |

\* worker scope = checkpoint.rs, step_driver.rs, replay.rs, tick.rs, kv.rs.

Headline: the measured crates sit at **33–76%**. Roughly a third to a half of the
behavior in the wire format, the shared-memory ring, and the session FSM is
unconstrained by any oracle. This is exactly the example-based-only suite MATERIA
warns about: tests pin a few happy-path examples and leave the arithmetic,
boundaries, and roundtrips wide open. Total audit wall time ≈ **8 minutes** (all
five runs, `-j8`). To re-run after a builder lands, re-mutate only that builder's
crate (each scoped run above is ≤2m); a full unscoped worker/daemon pass would be
~3–4× longer and is only worth it for a final sign-off.

---

## ALPHA — `rocket-surgeon-protocol` (53%, 27 survivors)

### A1. `checkpoint_layers()` has NO oracle at all *(highest signal)*
`src/lib.rs:8-17`. A pure function (sqrt-spaced checkpoint layer selection).
**8 distinct mutations survive**, including replacing the entire body with
`vec![]`, `vec![0]`, `vec![1]`, and corrupting every operator:

```
lib.rs:9:5   checkpoint_layers -> vec![]   /  vec![0]  /  vec![1]
lib.rs:9:19  replace <= with >             (the num_layers<=1 guard)
lib.rs:13:42 replace / with %  /  / with *  (interval = L / sqrt_l)
lib.rs:15:32 replace * with +  /  * with /  (i * interval)
```

No test calls this function. This is the textbook ALPHA model-based target. A
reference oracle (`checkpoint_layers(n)` for small n, plus properties: result is
strictly increasing; every element `< n`; length `== ceil(sqrt(n)) - 1`;
empty iff `n <= 1`; deterministic) kills all 8 at once.

### A2. `ErrorCode::numeric_code()` — spec codes unpinned (14 survivors)
`src/errors.rs:69-...`. Every arm is `delete -` (e.g. `-32001` → `32001`) and
survives. The JSON-RPC error codes are spec-mandated negatives; no test asserts a
single concrete value. **Exception-raising / wire-format oracle:** a table test
mapping each `ErrorCode` variant to its exact code, plus a property that all codes
are negative and pairwise-unique. Kills 14.

### A3. JSON-RPC standard constants unpinned (5 survivors)
`src/jsonrpc.rs:129-133`. `PARSE_ERROR`/`INVALID_REQUEST`/`METHOD_NOT_FOUND`/
`INVALID_PARAMS`/`INTERNAL_ERROR` — all `delete -` survive. One assertion per
constant (or assert against the spec values `-32700..-32603`).

**ALPHA verdict:** the roundtrip property in the brief is necessary but will NOT
catch A1–A3 (they aren't on the serde path). ALPHA must *also* add a model-based
test for `checkpoint_layers` and value-pinning tests for the error codes. These
are the survivors a pure roundtrip generator leaves behind.

---

## CHARLIE — `rocket-surgeon-shm` (52%, 84 survivors)

The ring buffer (`ring.rs`) and frame serialization (`lib.rs`) are CHARLIE's
crown jewels and they are the weakest-oracle code in the audit.

### C1. Ring read path has no roundtrip oracle *(highest signal)*
`src/ring.rs`. Read functions can be replaced by constant garbage and tests pass:

```
ring.rs:228  read_slot_bytes -> Ok(vec![])  / Ok(vec![0]) / Ok(vec![1])
ring.rs:236  read_absolute   -> Ok(vec![])  / Ok(vec![0]) / Ok(vec![1])
ring.rs:231  read_slot_bytes: replace + with - / + with *  (slot_offset + offset_in_slot)
```
No test does **publish(payload) → consume → assert bytes == payload**. This is the
brief's exact "bytes written then read back == identity" roundtrip, currently absent.

### C2. Ring full/empty boundary unconstrained
```
ring.rs:54   publish:      replace > with >=   (data.len() > slot_data_capacity guard)
ring.rs:62/83 (maketic - nettics) >= backuptics  — full check
ring.rs:78   publish:      replace - with + / - with /   (Ok(self.maketic - 1))
ring.rs:83   is_full:      replace - with +
ring.rs:176  try_consume:  replace > with == / > with >=  (size > slot_data_cap guard)
```
No test fills the ring to *exactly* capacity, nor checks the returned tick is
`maketic-1`. CHARLIE's **stateful FIFO model** (track a `VecDeque` + expected
ticks; assert RingFull fires at exactly `backuptics` and never before) kills these.

### C3. `serialize_probe_frame()` header-offset math (≈23 survivors)
`src/lib.rs:157-174`. Nearly every `+`/`*` in the header field-offset computation
can be mutated and tests pass. The frame serializer is not exercised against a
**decode roundtrip** that checks individual fields land at the right offsets — only
(at most) total length. CHARLIE's "framing encode/decode roundtrip for arbitrary
payloads incl. boundary sizes" with field-level assertions kills this cluster.

### C4. `RingConfig::mask()` wraparound (`lib.rs:138`)
`replace - with + / - with /` survive. `mask = size - 1` (power-of-two wrap); no
test pins it. Property: `mask(n)` for power-of-two `n` equals `n-1`, and
`x & mask == x % n`.

### C5. `region.rs` bounds/validation off-by-one
```
region.rs:254  bounds_check: replace > with >=   (boundary off-by-one — safety!)
region.rs:283  validate_name: replace > with >=
region.rs:109/130 ShmRegion::open: replace < with <= / ==  (size validation)
region.rs:186  is_empty: -> true / -> false; == with !=
```
`bounds_check`'s `>`→`>=` surviving is a latent **safety** gap: a test must hit the
exact-boundary access (offset+len == capacity) and assert it's accepted while
+1 is rejected.

### C6. `cleanup.rs` crash-recovery (stale-region GC) — mostly untested
`parse_pid_from_region_name -> Some(0)/Some(-1)/None`, `discover_stale_region_names
-> vec![]`, `is_pid_alive -> true/false`, `register_region_name -> ()` all survive.
`parse_pid_from_region_name` is pure and trivially model-testable; `is_pid_alive`
is OS-dependent (note as harder). Lower priority than C1–C3 but a real gap.

**CHARLIE verdict:** loom is not required to kill any of the above — they are all
single-threaded oracle gaps (roundtrip + stateful FIFO model + boundary tests).
Recommend CHARLIE do the stateful model first (C1+C2), then framing roundtrip (C3),
then boundary tests (C5). Reserve loom for a *separate* interleaving question.

---

## CHARLIE (cont.) — `rocket-surgeon-transport` (69%, 4 survivors)

Smallest crate, best score, but 4 real survivors:
```
framing.rs:3:41 / 3:48  replace * with +   (a frame-size constant expression)
framing.rs:50:23        read_message: replace > with >=   (length-limit boundary)
stdio.rs:44:9           StdioTransport::send_response -> Ok(())   (no oracle on send)
```
`framing.rs:50` `>`→`>=` is a max-message-length off-by-one with no boundary test.
`send_response -> Ok(())` survives because no test asserts bytes actually reach the
writer. Both are one-assertion kills and belong in CHARLIE's transport lane.

---

## Ranked target list for the builders

1. **CHARLIE — shm ring roundtrip + stateful FIFO (C1, C2).** Highest survivor
   density on the most safety-critical code (uninitialized-slot / torn-read paths).
2. **ALPHA — `checkpoint_layers` model test (A1).** An entire pure function with
   zero oracle; 8 kills from one reference model.
3. **CHARLIE — `serialize_probe_frame` field-level decode roundtrip (C3).** ~23
   survivors from one well-built roundtrip generator.
4. **ALPHA — error-code value pinning (A2, A3).** 19 survivors, trivial table test.
5. **DELTA — worldline + checkpoint-before stateful model (D1, D2).** Two
   correctness-critical invariants (worldline `tick_range`, replay start-checkpoint
   selection) with *zero* oracle — whole functions replaceable by no-ops/constants.
6. **BRAVO — `should_stop` + eviction/dtype models (B1–B3).** B1 (replay stop
   condition) is five kills from one test and underpins the divergence relations.
7. **CHARLIE — shm/transport boundary off-by-ones (C5, transport framing:50).**
   Safety-relevant `>`/`>=` and `<`/`<=` boundaries; need exact-boundary cases.
8. **DELTA — `now_rfc3339` date-math cluster (D4).** Lowest semantic priority but
   ~52 survivors from one small property test; the cheapest score win in the audit.

### Cross-cutting pattern (for all builders)
The dominant survivor class everywhere is **pure functions with no oracle**
(`checkpoint_layers`, `kv_metric`, `now_rfc3339`, `serialize_probe_frame`,
`find_checkpoint_before`) and **off-by-one boundaries** (`>`↔`>=`, `<`↔`<=`).
Example-based tests pin one happy-path value and leave the operator algebra and the
exact boundary unconstrained. Property + boundary-value generators are the highest-
leverage fix; classify generated inputs to confirm boundaries are actually hit.

---

## BRAVO — `rocket-surgeon-worker` (scoped, 76%, 54 survivors)

Scope = checkpoint.rs (34), kv.rs (15), replay.rs (5). step_driver.rs and tick.rs
had no surviving non-arid mutants in scope. Best score of the audit, but the
survivors sit on exactly BRAVO's named targets.

### B1. `ReplayContext::should_stop()` — the replay stop-condition is unconstrained *(highest signal)*
`src/replay.rs:31-37`. The predicate is
`layer == stop.layer && component == stop.component`. **All five mutations survive:**
```
replay.rs:32  should_stop -> true   /  -> false
replay.rs:33  replace == with !=    (layer == stop.layer)
replay.rs:33  replace && with ||
replay.rs:33  replace == with !=    (component == stop.component)
```
No test sets a `stop_at` and checks replay halts at the right (layer, component) and
*nowhere else*. This is the foundation of BRAVO's metamorphic relation "an
intervention at tick T must not perturb divergence before T" — if `should_stop` is
this loose, that relation can't be trusted. **Stateful model:** drive a replay with
a known `stop_at`, assert the driver stops at exactly that point.

### B2. `DtypeTag` torch-string mapping has no roundtrip oracle
`src/checkpoint.rs:29-39`. `from_torch_str` arms can be deleted (`float16`,
`bfloat16`, `float32`, `float64` → fall through), `from_torch_str -> None`,
`to_torch_str -> ""`/`"xyzzy"` all survive. **Model-based:** `from_torch_str(to_torch_str(t)) == Some(t)`
for every `DtypeTag`, plus a table pinning each torch string. Kills ~7.

### B3. `CheckpointArena::oldest_evictable()` — eviction policy unconstrained
`src/checkpoint.rs:309-316`. `-> None`, `-> Some("")`, `-> Some("xyzzy")`, and the
selection predicate `!=`→`==` (line 316) all survive. The checkpoint *eviction
order* (which checkpoint is dropped under memory pressure) is untested. This is
precisely BRAVO's "checkpoint set" abstract-model territory: build a model of the
arena, evict, and assert the real arena evicts the same id.

### B4. Spill-to-disk roundtrip + offset math
`src/checkpoint.rs`. `SpillIndexEntry::write_to`/`read_from` (396/411) `*`→`+`/`/`,
`spill_checkpoint` offset arithmetic (444-452) and size-compare boundaries
(492 `>`→`>=`, 499 `>`→`==`/`<`/`>=`), `load_spilled_checkpoint` (527/544) all
survive. No **spill→load roundtrip** asserting bytes and the size-boundary
(spill triggers at exactly the threshold). `alloc_slot` (217) `-`→`+`/`/` likewise.

### B5. `kv_metric()` numeric function — no oracle
`src/kv.rs:89-98`. Body → `0.0`/`1.0`/`-1.0` and every operator (`+ - * / %`)
survive — a metric with zero constraint. Also `intervene` (220) `*`→`/`. BRAVO:
property test (range, monotonicity, known reference values).

**BRAVO verdict:** the brief's stateful step-driver model is the right tool, but the
*single highest-value* test is B1 (`should_stop`) — five kills and it underpins
the divergence metamorphic relations. B2/B3 are model-based (dtype roundtrip,
eviction order) and B4 is a spill roundtrip.

## DELTA — `rocket-surgeon` session.rs (33%, 116 survivors)

Lowest score in the audit — but read it carefully. **52 of the 116 survivors are
in one function, `now_rfc3339` (D4 below).** Excluding that pure date utility, the
session-FSM-proper is **58 caught / 64 missed ≈ 48%** — still the weakest FSM
coverage measured, and the survivors land squarely on the invariants the brief
names (worldline `tick_range` consistency, checkpoint list).

### D1. `advance_worldline_segment()` — worldline invariant has NO oracle *(highest signal)*
`src/session.rs:1109-1122`. The entire function can be replaced with `()` (no-op)
and tests pass:
```
1110  advance_worldline_segment with ()        (the whole branch op is a no-op)
1111  replace < with == / <= / >               (current_idx < segments.len() guard)
```
This is the brief's "worldline `tick_range` consistency" invariant, and nothing
constrains it. **Stateful model:** track an abstract worldline (segment list + cur);
after each `advance_worldline_segment(t)` assert a new segment exists, the prior
segment's `tick_range.1 == t`, the new segment's range is `(t, 0)`, and
`current_segment` advanced. Kills D1.

### D2. `find_checkpoint_before()` — checkpoint-selection-for-replay unconstrained
`src/session.rs:1124-1130`. "Find the latest checkpoint strictly before tick T."
**Every operator and the whole body survive:**
```
1125  -> None / Some("") / Some("xyzzy")
1127  replace < with == / > / <=     (pos.tick_id < target_tick)
1127  replace > with == / < / >=     (pos.tick_id > best.1  — the "latest" comparison)
1127  replace && with || , || with &&
```
This decides which checkpoint a replay starts from — a correctness-critical
selection with zero oracle. **Model-based:** a reference scan over a `BTreeMap` of
(tick → id); assert real == model for random checkpoint sets and target ticks,
incl. ties, empty set, and target before/after all checkpoints.

### D3. FSM validation predicates (`||`/`&&`/`==` flips survive)
```
441:49  validate_attach: replace || with &&     (attach precondition logic)
562:53  detach:          replace || with &&     (detach precondition logic)
865:20  Session::step:   replace < with >        (a step boundary)
986-987 suggest_patterns: && with ||, delete !
376:48  invalid_state_error: == with !=
1194    replay: + with *  ;  1222 replay: < with >
```
These are the **exception-raising / legal-transition** gaps the brief asks DELTA to
target: drive each action in each `Status`, assert the *right* rejection fires. The
`validate_attach`/`detach` `||`→`&&` survivors mean some illegal attach/detach is
currently accepted-or-rejected identically under the mutation — no test pins the
precondition.

### D4. `now_rfc3339()` — hand-rolled civil-date algorithm, zero tests (52 survivors)
`src/session.rs:60-75`. A from-scratch `civil_from_days` (Hinnant) date formatter
(no chrono dependency — consistent with RS's no-deps stance). Essentially every
arithmetic operator in it can be mutated and tests pass; the body can be replaced
with `"xyzzy".into()`. It's pure and trivially oracle-able: pin a handful of known
`Unix epoch → RFC-3339` pairs (epoch 0 = `1970-01-01T00:00:00Z`, a known leap-year
date, a recent timestamp) and add a monotonicity property. This single test cluster
moves the daemon score from 33% → ~48%. Low FSM-importance but very high
survivor-count-per-test; an obvious one-shot win. *(It produces `created_at`
timestamps on checkpoints, so it is user/LLM-visible, not dead code.)*

### D5. Arena utilization accounting (minor)
`src/session.rs:1098-1106`. `arena_utilization -> 0.0/1.0/-1.0`,
`update_arena_utilization` body→`()`, and its arithmetic all survive. No test checks
utilization tracks captured bytes. Lower priority than D1/D2.

**DELTA verdict:** prioritize D1 (worldline stateful model) and D2 (checkpoint-
before model) — they are the named invariants and they have *no* oracle. D3 is the
illegal-transition exception-raising suite. D4 is a cheap high-count cleanup the
commander could even hand to FOXTROT as a fast-follow (see Scope discipline).

---

## Scope discipline (why FOXTROT added no tests)

Every high-signal survivor lands inside a crate already owned by another lane
(ALPHA=protocol, BRAVO=worker, CHARLIE=shm/transport, DELTA=daemon). The brief
permits FOXTROT to add a test where a survivor is a trivial one-assertion kill,
but doing so inside builder-owned files would collide with their branches at
integration. FOXTROT therefore delivers **targets, not patches** — each builder
kills their own survivors. If the commander wants FOXTROT to also land the trivial
kills (A2/A3 error-code tables, transport framing boundary), say so and they're a
fast follow.

## Gaps left for a follow-up

- **Unscoped large-crate code is unmeasured:** worker `dispatch.rs`/`adapter.rs`/
  `bridge.rs`, daemon `dispatch.rs`/`tensor_stats.rs`/`tensor_store.rs`/
  `notifications.rs`/`perfetto_sink.rs`. These hold the bulk of the mutants
  (worker 583 total vs 244 scoped; daemon 876 total vs 242 in session.rs).
- **Secondary crates not run:** `perfetto-writer`, `rocket-surgeon-orchestrator`,
  `rocket-surgeon-probes`, `rocket-surgeon-tui`, `rocket-surgeon-python`.
- **mutmut on Python** (ECHO's lane) was not run — out of FOXTROT's tractable scope
  this pass; flag for a dedicated Python mutation pass.
