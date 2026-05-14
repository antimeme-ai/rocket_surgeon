---
topic: Tensor Summary Statistics — numerically stable, high-performance algorithms for computing summary stats on dense activation tensors (float16/bf16/f32/f64/int/bool) in the hot path
status: draft
created: 2026-05-14
sources: Welford 1962, Chan/Golub/LeVeque 1979, Kahan 1965, Higham 1993 (pairwise summation), NumPy/PyTorch/JAX internals, RadiK (Li et al. 2024), Lemire 2017, half-rs crate
---

# Tensor Summary Statistics: Lit Review

Algorithms for computing mean, std, min, max, abs-max, sparsity, L2 norm, histogram, and top-k on captured activation tensors. Correctness and hot-path performance are co-equal requirements.

## 1. Mean

**Canonical algorithm:** Welford's online algorithm (Welford 1962, Knuth TAOCP Vol 2 Sec 4.2.2).

Update: `mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n`. Single-pass, O(1) state. Numerically stable because it only accumulates differences from the running mean, avoiding catastrophic cancellation inherent in the naive `sum/n` approach.

**Accumulator dtype:** PyTorch's TensorIterator unconditionally promotes float16/bfloat16 reductions to float32 accumulators (see `aten/src/ATen/AccumulateType.h`). Google TPUs do the same: bf16 inputs, f32 accumulation. **Recommendation: accumulate in f32 for f16/bf16 inputs. f64 accumulation is unnecessary** -- f32 has 24 bits of mantissa, more than sufficient for mean computation over tensors up to ~16M elements without Kahan compensation.

**Alternative -- pairwise summation:** NumPy and Julia use pairwise (cascade) summation for `sum()` rather than Kahan. Error grows as O(epsilon * log n) vs Kahan's O(epsilon), but pairwise requires the same FLOP count as naive summation and parallelizes trivially (Higham 1993). For SIMD: pairwise summation maps directly to tree reductions in vector lanes. Kahan's 4x arithmetic overhead and serial dependency chain make it hostile to SIMD.

**Recommendation:** Welford for streaming single-pass. If we ever need standalone sum, pairwise summation in f32 accumulators.

## 2. Standard Deviation

**Canonical algorithm:** Welford's method computes variance alongside mean with zero additional cost. The M2 accumulator tracks `sum((x_i - mean)^2)` incrementally: `M2_n = M2_{n-1} + (x_n - mean_{n-1}) * (x_n - mean_n)`. Variance = `M2 / n`, std = `sqrt(M2 / n)`.

**Parallel merge (Chan/Golub/LeVeque 1979):** Two partial results (n_A, mean_A, M2_A) and (n_B, mean_B, M2_B) merge as:
```
delta = mean_B - mean_A
mean_AB = mean_A + delta * (n_B / n_AB)
M2_AB = M2_A + M2_B + delta^2 * (n_A * n_B / n_AB)
```
This is critical for multi-GPU reduce: each rank computes local Welford state, then merges via allreduce. Also maps to SIMD lane reduction.

**Recommendation:** Welford with Chan et al. parallel merge. f32 accumulator for f16/bf16.

## 3. Min, Max, Abs-Max

**Algorithm:** Trivial -- running min/max with single comparisons per element. Abs-max is `max(|x|)`.

**Cache considerations:** These are pure streaming operations, limited entirely by memory bandwidth. The key optimization is fusing them into the same pass as mean/std to avoid re-reading the tensor from memory.

**SIMD:** Min/max map directly to `_mm256_min_ps` / `_mm256_max_ps` intrinsics (x86 AVX2) or equivalent ARM NEON. Process 8 f32s or 16 f16s per instruction. For f16/bf16: convert to f32 in SIMD registers, compare, no need to convert back.

**Recommendation:** Fuse into the Welford pass. Zero additional cost beyond comparisons.

## 4. Sparsity

**Algorithm:** Count elements where `|x| < epsilon` (or exactly zero for integer types). Single counter, incremented per element.

**For bool tensors:** Sparsity = fraction of `false` values. Bit-counting: Polars uses branchless SIMD with popcount on packed bitmasks, processing 64+ bools per cycle.

**SIMD:** Compare-and-mask: `_mm256_cmp_ps(abs_x, epsilon, _CMP_LT_OS)` produces a bitmask, `_mm256_movemask_ps` extracts it, `popcnt` counts set bits.

**Recommendation:** Fuse into the Welford pass. One compare + popcount per SIMD lane.

## 5. L2 Norm

**The problem:** For float16 (max ~65504), squaring values above ~256 overflows. For bfloat16 (max ~3.4e38, same as f32) overflow is less acute but precision loss during accumulation is severe.

**Canonical algorithm:** Scaled accumulation (Blue 1978, LAPACK `dnrm2`). Compute `max_val = max(|x|)`, then `norm = max_val * sqrt(sum((x / max_val)^2))`. This is inherently two-pass (first for max, then for scaled sum).

**Single-pass variant:** Track running max and rescale the accumulator on the fly. When a new max is encountered: `accum = accum * (old_max / new_max)^2`, then continue. This is what modern BLAS implementations do.

**PyTorch/NumPy behavior:** NumPy's `linalg.norm` overflows on float16 (known bug, issue #8775). Their suggested fix is the scale-and-correct method. PyTorch upcasts to f32 before computing norm.

**Recommendation:** Since we are already tracking max in the fused pass, use the single-pass scaled accumulation. Accumulate `sum((x / running_max)^2)` in f32. When running_max updates, rescale the accumulator. Final result: `running_max * sqrt(accum / n_nonzero)`. Wait -- that's L2 norm, not RMS. L2 norm = `running_max * sqrt(accum)`.

## 6. Histogram

**The problem:** Fixed-bin histogram requires knowing the data range. Unknown-range data requires either two passes or adaptive binning.

**NumPy's approach:** Two-pass. First `min(a), max(a)`, then linear binning. This is the standard for batch data.

**Recommendation:** Since our fused pass already computes min and max, the histogram is naturally a second pass: `bin_index = floor((x - min) / (max - min) * n_bins)`. This two-pass approach is correct and simple. The first pass (Welford + min/max + sparsity + L2 norm) is memory-bound; the second pass (histogram + top-k) is also memory-bound. Two passes over a hot-in-cache tensor is acceptable. Trying to do adaptive single-pass histogramming adds complexity for negligible gain -- the data is already cached from pass 1.

**SIMD for binning:** Compute bin indices with SIMD float ops, then scatter-add to bin counters. The scatter step is inherently serial (histogram bins alias), but for reasonable bin counts (64-256), the counters fit in L1 and the serial scatter is fast.

**Bin count recommendation:** 64 bins default. Sturges' rule (log2(n)+1) gives ~20 bins for 1M elements; 64 provides finer granularity without memory pressure.

## 7. Top-k

**Use case:** k smallest, typically k=10-100, n up to hundreds of millions.

**Min-heap of size k:** O(n log k). Constant O(k) memory. Cache-friendly for small k. Consistent performance, no adversarial worst-case. Branch-heavy (heap sift operations).

**Partial quickselect (Hoare):** O(n) average, O(n^2) worst case. O(n) memory (in-place on mutable copy). For small k and large n, the constant factors dominate and heap wins on cache behavior (Lemire 2017: heap ~37K ops/s vs quickselect ~45K ops/s at n=1408, k=128 -- only 20% difference).

**GPU state of the art:** RadiK (Li et al. 2024) uses radix-based selection: find the k-th element via radix sort prefix scanning, then filter. 2.5-4.8x faster than prior GPU top-k. But this is a GPU algorithm; on CPU for our case (small k, large n), a heap is simpler.

**Recommendation:** Min-heap of size k, operating on absolute values, storing (abs_value, original_value, flat_index) tuples. For k <= 128 and n in millions, the heap fits in L1 cache and the O(n log k) cost is dominated by the memory-bandwidth-bound scan. Can be fused into pass 2 (same pass as histogram) since both require reading every element.

**Index tracking:** Store flat indices; convert to multi-dimensional indices lazily on demand.

## Fused Pass Architecture

**Pass 1 (single scan, fused):** For each element x:
- Welford update (mean, M2) -> mean, variance, std
- running min, max, abs_max
- sparsity counter (compare |x| < eps)
- L2 norm scaled accumulator (rescale when max updates)

State: `(n, mean, M2, min, max, abs_max, sparse_count, l2_accum, l2_scale)` -- fits in registers.

**Pass 2 (second scan, fused):** Requires min/max from pass 1.
- Histogram binning (using min/max as range)
- Top-k via min-heap on |x|

State: `bin_counts[64]` + heap of size k -- fits in L1.

**Why not single-pass for everything?** Histogram requires knowing the range. Adaptive histograms add complexity and produce inconsistent bin edges across tensors, making cross-tensor comparison meaningless. Two passes over a tensor that's hot in LLC (or even L2 for smaller activations) costs roughly 2x memory bandwidth but keeps the code simple and the histogram semantics clean.

## Dtype Dispatch Strategy

| Input dtype | Accumulator | Convert strategy |
|-------------|-------------|-----------------|
| f16         | f32         | SIMD f16->f32 (vcvtph2ps on x86, fcvt on ARM) |
| bf16        | f32         | SIMD bf16->f32 (shift left 16 bits -- it's a truncated f32) |
| f32         | f32         | No conversion |
| f64         | f64         | No conversion |
| i8/i16/i32  | i64 or f64  | Widen to i64 for exact sums, or f64 for mean/std |
| i64         | i128 or f64 | i128 for exact count; f64 for mean/std (note: f64 cannot represent all i64 exactly) |
| u8          | u64 or f32  | Widen |
| bool        | u64         | Popcount for sum; sparsity = 1 - mean |

## Rust Implementation Notes

**Crates:**
- `half` crate (half-rs) for f16/bf16 types with SIMD conversion support
- No dependency on ndarray or polars -- we reimplement (per project principle: no deps as dependencies, only as reference)

**SIMD strategy:** Use `std::simd` (portable_simd, nightly) or `std::arch` intrinsics for stable Rust. Key operations: f16->f32 conversion, f32 min/max/compare, horizontal reductions. Polars demonstrates that branchless SIMD with bitmask operations gives constant good performance.

**Parallelism:** Chunk the tensor across threads. Each thread runs the full fused pass on its chunk, producing local Welford state. Merge via Chan et al. parallel formula. This is embarrassingly parallel and maps directly to multi-core and to multi-GPU (each rank computes local stats, allreduce merges).

## Key References

- Welford, B.P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3).
- Chan, T.F., Golub, G.H., LeVeque, R.J. (1979). "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances."
- Kahan, W. (1965). "Pracniques: Further Remarks on Reducing Truncation Errors." CACM 8(1).
- Higham, N.J. (1993). "The Accuracy of Floating Point Summation." SIAM J. Sci. Comput.
- Blue, J.L. (1978). "A Portable Fortran Program to Find the Euclidean Norm of a Vector." ACM TOMS.
- Li et al. (2024). "RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection." ACM ICS.
- Lemire, D. (2017). "QuickSelect versus binary heap for top-k queries." Blog post.
