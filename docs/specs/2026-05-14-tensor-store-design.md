# WU 1.4 — Content-Addressable Tensor Store

## Purpose

The tensor store is the daemon-side cache for captured activation tensors. It provides:
- Content-addressable storage keyed by BLAKE3 hash of raw tensor bytes
- High-performance, numerically stable summary statistics computation
- Slice access for returning raw tensor subsets to clients
- LRU eviction under memory pressure

This is on the hot path of timestop accounting: every tensor captured during a forward-pass tick flows through this store.

## Architecture

Two modules in `crates/rocket-surgeon/src/`:

| Module | Responsibility |
|--------|---------------|
| `tensor_store.rs` | Content-addressable cache: insert, get, summarize, slice, evict |
| `tensor_stats.rs` | Two-pass fused statistics engine: `(bytes, dtype, shape) → TensorSummary` |

The stats engine is a pure function with no side effects. Its inner loops are designed to be replaceable with `std::arch` intrinsics or inline assembly without touching the store API.

## Data Flow

```
Python host captures activation tensor
    ↓
Writes raw bytes to shared memory (WU 1.8)
    ↓
Daemon reads bytes + metadata (shape, dtype, device)
    ↓
tensor_store.insert(bytes, shape, dtype, device)
    ↓
BLAKE3(bytes) → tensor_id (64 hex chars)
    ↓
StoredTensor cached in HashMap
    ↓
TensorHandle { tensor_id, shape, dtype } returned
    ↓
On inspect → tensor_store.summarize(tensor_id) → TensorSummary
```

## Store Structure

```rust
pub struct StoredTensor {
    pub tensor_id: String,           // BLAKE3 hex digest
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub device: String,
    data: Vec<u8>,                   // raw bytes, little-endian
    summary: Option<TensorSummary>,  // lazily computed, cached
    inserted_at: Instant,
    last_access: Instant,
}

pub struct TensorStore {
    entries: HashMap<String, StoredTensor>,
    access_order: VecDeque<String>,  // LRU tracking (front = oldest)
    max_entries: usize,              // default 1024
    max_bytes: usize,                // default 2 GiB
    current_bytes: usize,
}
```

### Public API

```rust
impl TensorStore {
    pub fn new() -> Self;
    pub fn with_limits(max_entries: usize, max_bytes: usize) -> Self;

    /// Insert raw tensor bytes. Computes BLAKE3, deduplicates.
    /// Returns TensorHandle. Evicts LRU entries if over capacity.
    pub fn insert(
        &mut self, data: Vec<u8>, shape: Vec<u64>, dtype: DType, device: String,
    ) -> TensorHandle;

    /// Get stored tensor metadata by ID.
    pub fn get(&mut self, tensor_id: &str) -> Option<&StoredTensor>;

    /// Compute or return cached TensorSummary.
    pub fn summarize(&mut self, tensor_id: &str) -> Option<TensorSummary>;

    /// Extract a contiguous byte range from the flattened tensor.
    /// ranges is a list of [start, end) pairs along the flattened (1D) axis.
    /// Multi-dimensional slicing is the caller's responsibility (inspect handler
    /// computes flat ranges from shape + requested dim slices).
    pub fn slice(&mut self, tensor_id: &str, offset: u64, len: u64) -> Result<Vec<u8>, StoreError>;

    /// Check if a tensor_id exists.
    pub fn contains(&self, tensor_id: &str) -> bool;

    /// Current number of stored tensors.
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;

    /// Current total bytes stored.
    pub fn bytes_used(&self) -> usize;
}
```

### Eviction Policy

LRU by `last_access` timestamp. On insert, if `entries.len() >= max_entries` or `current_bytes + new_size > max_bytes`, evict from the front of `access_order` until both constraints are satisfied. Eviction is O(1) per entry.

Content-addressable deduplication: if a tensor with the same BLAKE3 hash is already present, `insert` updates `last_access` and returns the existing handle without storing duplicate data.

## Statistics Engine

### Literature Basis

| Statistic | Algorithm | Reference |
|-----------|-----------|-----------|
| Mean | Welford's online algorithm | Welford 1962, Knuth TAOCP Vol 2 §4.2.2 |
| Std / Variance | Welford M2 accumulator | Welford 1962 |
| Min, Max, Abs-max | Running comparison | — |
| Sparsity | Zero-count with epsilon threshold | — |
| L2 norm | Blue's scaled accumulation | Blue 1978 (LAPACK `dnrm2`) |
| Histogram | Fixed-bin linear binning | NumPy approach |
| Top-k | Min-heap of size k | Lemire 2017 |
| Parallel merge | Chan/Golub/LeVeque formula | Chan et al. 1979 |

### Two-Pass Architecture

**Pass 1 — Fused streaming scan.** For each element x (converted to accumulator type):
1. Welford update: `delta = x - mean; mean += delta / n; delta2 = x - mean; M2 += delta * delta2`
2. `min = min(min, x); max = max(max, x); abs_max = max(abs_max, |x|)`
3. Sparsity: `if |x| < epsilon { sparse_count += 1 }`
4. L2 norm (Blue's method): if `|x| > l2_scale`, rescale accumulator: `l2_accum *= (l2_scale / |x|)²; l2_scale = |x|`; then `l2_accum += (x / l2_scale)²`

Pass 1 state fits in registers: `(n: u64, mean: f64, m2: f64, min: f64, max: f64, abs_max: f64, sparse_count: u64, l2_accum: f64, l2_scale: f64)`.

**Pass 2 — Fused histogram + top-k.** Requires min/max from pass 1.
1. Histogram: `bin = floor((x - min) / (max - min) * n_bins)`, clamped to `[0, n_bins-1]`. Increment `counts[bin]`.
2. Top-k: maintain min-heap of size k on `|x|`. If `|x| > heap.peek()`, replace and sift down. Store `(abs_value, original_value, flat_index)`.

Pass 2 state: `counts: [u64; 64]` + heap of k entries. Fits in L1 cache.

**Why two passes:** Histogram requires the data range. Adaptive single-pass histograms produce inconsistent bin edges across tensors, making cross-tensor comparison meaningless. The tensor data stays cache-hot between passes; the cost is ~2x memory bandwidth, which is acceptable.

### Dtype Dispatch

| Input dtype | Accumulator | Conversion |
|-------------|-------------|-----------|
| f16 | f32 | `half::f16::to_f32()` (maps to `vcvtph2ps` on x86) |
| bf16 | f32 | `half::bf16::to_f32()` (16-bit left shift) |
| f32 | f32 | None |
| f64 | f64 | None |
| i8, i16, i32 | f64 | Widen to f64 for mean/std |
| i64 | f64 | Widen (note: f64 cannot represent all i64 exactly) |
| u8 | f64 | Widen |
| bool | u64 | Popcount for sum; sparsity = 1 - mean |

f16/bf16 accumulate in f32 following PyTorch and JAX convention (`aten/src/ATen/AccumulateType.h`). f64 accumulation is unnecessary for these types.

### SIMD / Assembly Escape Hatch

The stats engine is structured as a trait with a generic scalar implementation and room for specialized implementations:

```rust
pub trait StatsAccumulator {
    fn process_chunk(&mut self, data: &[u8], dtype: DType);
    fn finalize_pass1(&self) -> Pass1Result;
    fn process_chunk_pass2(&mut self, data: &[u8], dtype: DType, range: (f64, f64));
    fn finalize_pass2(&self) -> Pass2Result;
}
```

The scalar implementation covers all dtypes and is the correctness reference. SIMD-optimized implementations can be added per-dtype behind `#[cfg(target_arch)]` without touching the store or any consumer. If profiling shows the inner loop is the bottleneck, inline assembly or `std::arch` intrinsics slot in at this trait boundary.

### Parallel Merge

For multi-threaded stats (future work, not WU 1.4 scope):

```rust
pub fn merge_pass1(a: &Pass1Result, b: &Pass1Result) -> Pass1Result {
    // Chan/Golub/LeVeque parallel Welford merge
    let n = a.n + b.n;
    let delta = b.mean - a.mean;
    let mean = a.mean + delta * (b.n as f64 / n as f64);
    let m2 = a.m2 + b.m2 + delta * delta * (a.n as f64 * b.n as f64 / n as f64);
    // min/max/abs_max: element-wise min/max
    // sparsity: sum counts
    // L2 norm: rescale and merge accumulators
    ...
}
```

This is implemented in WU 1.4 even though single-threaded, because it's needed for multi-GPU merge in later phases and the tests validate numerical correctness of the merge.

## Protocol Crate Addition

Add to `DType` in `crates/rocket-surgeon-protocol/src/types.rs`:

```rust
impl DType {
    pub fn byte_size(self) -> usize {
        match self {
            Self::Float16 | Self::Bfloat16 | Self::Int16 => 2,
            Self::Float32 | Self::Int32 => 4,
            Self::Float64 | Self::Int64 => 8,
            Self::Int8 | Self::Uint8 | Self::Bool => 1,
        }
    }
}
```

## Dependencies

- `blake3` — already in workspace (used by WU 1.9 python bridge)
- `half` — new workspace dependency for f16/bf16 types with SIMD conversion

No other new dependencies. Stats engine is implemented from scratch per project principle.

## Test Plan

### tensor_store.rs (~14 tests)

1. `insert_returns_tensor_handle` — shape, dtype, tensor_id populated
2. `insert_computes_blake3_id` — tensor_id is 64 hex chars matching BLAKE3
3. `duplicate_content_deduplicates` — same bytes → same tensor_id, no duplicate storage
4. `different_content_different_id` — different bytes → different tensor_id
5. `get_returns_stored_tensor` — metadata matches what was inserted
6. `get_nonexistent_returns_none`
7. `get_updates_last_access` — LRU tracking
8. `contains_works`
9. `eviction_by_entry_count` — oldest evicted when max_entries reached
10. `eviction_by_byte_limit` — oldest evicted when max_bytes reached
11. `eviction_preserves_recently_accessed` — LRU order respected
12. `slice_returns_correct_bytes` — flat byte range extraction
13. `slice_out_of_bounds_returns_error` — offset + len exceeds data size
14. `len_and_bytes_used_accurate`

### tensor_stats.rs (~18 tests)

15. `mean_known_values` — hand-computed mean for small arrays
16. `std_known_values` — hand-computed std
17. `min_max_abs_max` — including negative values
18. `sparsity_all_zeros` — sparsity = 1.0
19. `sparsity_no_zeros` — sparsity = 0.0
20. `l2_norm_known_values` — verified against numpy
21. `l2_norm_does_not_overflow_f16` — large f16 values don't produce inf
22. `histogram_uniform_distribution` — roughly equal bin counts
23. `histogram_single_value` — all in one bin
24. `histogram_edges_correct` — edges span [min, max]
25. `top_k_returns_largest` — correct values and indices
26. `top_k_with_k_larger_than_n` — returns all elements
27. `welford_merge_two_halves` — split array, merge, compare to whole
28. `welford_merge_numerical_stability` — large offset values (e.g., 1e8 + small noise)
29. `f16_accumulates_in_f32` — no precision loss vs naive f16 accumulation
30. `bf16_accumulates_in_f32` — same
31. `integer_dtype_stats` — i32 tensor produces correct mean/std
32. `bool_dtype_stats` — mean = fraction of true, sparsity = fraction of false

### Cross-validation tests

33. `summary_matches_numpy` — for a known tensor, compare all stats to numpy reference values (hardcoded expected values computed offline)
34. `blake3_id_matches_python_bridge` — same bytes produce same BLAKE3 hash as the Python `blake3_hash()` function

## Execution Order

1. Add `half` to workspace deps
2. Add `DType::byte_size()` to protocol crate + test
3. Implement `tensor_stats.rs` (the research-backed engine) + tests
4. Implement `tensor_store.rs` (store wrapping the engine) + tests
5. Wire module declarations in `main.rs`
6. Clippy, fmt, CI
7. Code review → fix ALL findings
8. Commit and push
