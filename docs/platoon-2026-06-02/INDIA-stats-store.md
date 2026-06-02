# B004 — Wave 2 findings: INDIA (stats + tensor store)

**Lane.** `crates/rocket-surgeon`: `tensor_stats.rs` (Welford mean/var, scaled
`dnrm2` L2, min/max/abs_max/sparsity, 64-bin histogram, top-k heap, parallel
merge) and `tensor_store.rs` (BLAKE3 content-addressing + dual-budget LRU).

**Branch.** `platoon2/stats-store`. proptest was already a dev-dep; no new deps.

## Techniques applied (climbing the MATERIA oracle hierarchy)

The existing suite for both modules was example-based (oracle tiers 2-3). Added
**13 property/model/metamorphic tests** that reach tiers 4-6:

### `tensor_stats.rs` — in-file `mod proptests` (needs private `compute_pass1` / `Pass1Result` / `merge_pass1`)

| Test | Tier | Oracle |
| --- | --- | --- |
| `pass1_matches_naive_reference` | 6 model-based | independent two-pass f64 reference (`naive_ref`); counts/min/max/abs_max/sparsity **exact**, mean/M2/L2 within rel 1e-9 / 1e-6 |
| `merge_equals_concatenation` | 4 metamorphic | Chan–Golub–LeVeque: `merge(pass1(A),pass1(B)) == pass1(A++B)` |
| `statistics_are_order_invariant` | 4 metamorphic | `pass1(A) == pass1(perm(A))` on every statistic |
| `summary_is_always_well_formed` | 5 exception/invariant | over arbitrary bytes × all 10 dtypes: no panic, histogram shape constant, sparsity∈[0,1], top-k sorted by \|value\| desc, len≤10 |
| `bounded_dtype_summary_is_finite` | 5 invariant | over arbitrary bytes × the 9 bounded dtypes: every emitted stat + histogram edge is finite (JSON-serialisability contract) |
| `documents_f64_huge_value_overflow` | regression-pin | pins the **bug below** |
| `value_generator_distribution_is_non_trivial` | — | `cover` discipline (see distribution) |

### `tensor_store.rs` — `tensor_store_proptest.rs` (public API suffices)

| Test | Tier | Oracle |
| --- | --- | --- |
| `store_matches_lru_model` | 6 stateful model-based | abstract `HashMap`+recency-`Vec` model driven in lockstep; after **every** op assert handle/presence/bytes, id-set, `len()`, `bytes_used()`, bounds |
| `identical_bytes_dedup` | 6 model-based | repeated insert of equal bytes ⇒ 1 entry, BLAKE3 id, no byte growth |
| `content_determines_identity` | 6 model-based | `a==b ⇔ same id ⇔ collapsed`; distinct ⇒ 2 entries |
| `slice_bounds_oracle` | 5 exception | `slice` total over bounds: in-range ⇒ exact bytes; OOB / `offset+len` overflow ⇒ `SliceOutOfBounds` |
| `missing_id_reports_absence` | 5 exception | every op on an absent id ⇒ `NotFound`/`None`, never a panic |
| `op_generator_distribution_is_non_trivial` | — | `cover` discipline |

The stateful model encodes the store's *exact* LRU access contract, which is
where the subtle bugs would be: every hit (`get`/`raw_data`/`summarize`/in-range
or out-of-bounds `slice` on a **present** id) touches recency; a miss never
does; eviction takes the least-recently-touched victim; a dedup insert touches
but never evicts; an oversized single payload evicts everything then inserts
anyway (store may exceed `max_bytes` only at `len()==1`). The real store matched
the model on every generated sequence — **no LRU divergence found.**

## Generator-distribution evidence (measured, not assumed)

`tensor_stats` value generator over 500 streams (avg len 124.5):
```
empty: 0.8%  has-NaN: 49.8%  has-Inf: 52.0%  has-zero: 97.6%  non-empty all-finite: 47.2%
```
Both regimes well represented: ~47% clean numeric streams exercise the
accumulators; ~50% carry NaN/Inf exercising the non-finite filter/count paths.

`tensor_store` op generator over 400 scenarios:
```
scenarios w/ eviction: 55.2%   w/ dedup: 70.2%
totals — inserts: 2511  dedup-hits: 1047  evictions: 864  oversized: 451
get hit/miss: 935/1515   slice ok/oob/not-found: 125/899/1515
```
Eviction, dedup, oversized inserts, get hit+miss, and all three slice outcomes
are all exercised (the distribution test asserts these thresholds, so a future
generator regression that stops covering a path fails loudly).

## Bug found (the win)

**`compute_summary` emits non-finite stats for finite-but-huge `Float64`
inputs — violates the JSON-serialisability contract.**

- **Minimal failing input:** a `Float64` tensor of `[f64::MAX, -f64::MAX]`
  (16 bytes LE). Both values are *finite*, so the existing per-value non-finite
  guard (which handles NaN/±Inf inputs) does not fire.
- **Observed output:** `stats.mean = -inf`, `stats.std = NaN`, and the histogram
  edges are `inf` (`range = max - min = f64::MAX - (-f64::MAX)` overflows to
  `+inf`).
- **Root cause (two independent overflow sites):**
  1. *Welford* (`tensor_stats.rs:97-100`): on the second value, the running
     delta `x - mean = -f64::MAX - f64::MAX` overflows to `-inf`, driving
     `mean → -inf` and then `m2 += delta*delta2 → -inf`; `std = sqrt(m2/n) →
     NaN`.
  2. *Histogram* (`tensor_stats.rs:154,168,194`): `range = range_max -
     range_min` overflows to `+inf`; the `range.mul_add(...)` edge construction
     then yields `inf` edges, and per-value `frac = (x-min)/range` is `inf/inf =
     NaN` (binned as bin 0 via `NaN as usize`).
- **Why it matters:** `TensorStats` is serialized to JSON over the LLM/TUI
  protocol; `serde_json` cannot represent `inf`/`NaN` (serializes as `null` or
  errors), so an extreme-magnitude f64 tensor produces a malformed/lossy
  summary. The contract demonstrably *holds* for all 9 bounded dtypes
  (`bounded_dtype_summary_is_finite` passes), because f16/bf16/f32/int values
  cast to f64 stay far inside f64 range and the intermediates never overflow.
- **Severity:** low in practice (transformer activations are never near
  1e308, and f64 activation tensors are rare), but it is a genuine correctness
  gap against the finite-output contract.
- **Status: recorded, NOT fixed** (per brief: recording the failing case is the
  win; fixing is out of this lane's scope and would change production
  numeric semantics). `documents_f64_huge_value_overflow` pins the current
  behaviour so the regression is visible; it carries a comment telling whoever
  fixes it to update this file.
- **Suggested fix for the protocol owner:** after `compute_summary` builds the
  result, sanitise non-finite outputs (e.g. clamp `mean`/`std`/`l2_norm` and
  histogram edges to finite, or compute the histogram range with a saturating
  subtraction). The cleanest place is a single output-sanitisation pass, since
  the engine already promises finite, JSON-safe stats.

No other divergences were found: the streaming engine matches the naive
reference, the parallel merge satisfies the CGL identity, statistics are
order-invariant, and the LRU store matches its abstract model on every
generated op sequence.

## Gaps left / not in scope

- The f64-overflow bug is documented, not fixed (deliberate).
- `tensor_stats` numeric tolerances assume finite magnitudes ≤ ~1e6 in the
  model-based generators; that is intentional (tight tolerance > wide coverage
  of pathological floats, which the finiteness/well-formedness properties cover
  separately across the full byte space).
- `insert_with_id` (precomputed-hash trust path) is exercised by the existing
  example tests; the stateful model uses the hashing `insert` path, which is
  the production ingestion path. Trust-the-caller id/data mismatch is by design
  and not modelled.
- Mutation-score confirmation for these new oracles is NOVEMBER's lane
  (`cargo-mutants`), not re-run here.
