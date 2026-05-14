use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[allow(dead_code)]
const NUM_HISTOGRAM_BINS: usize = 64;
#[allow(dead_code)]
const SPARSITY_EPSILON: f64 = 1e-8;
#[allow(dead_code)]
const DEFAULT_TOP_K: usize = 10;

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Pass1Result {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    abs_max: f64,
    sparse_count: u64,
    nan_count: u64,
    l2_accum: f64,
    l2_scale: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
struct TopKHeapEntry {
    abs_value: f64,
    original_value: f64,
    flat_index: u64,
}

impl Eq for TopKHeapEntry {}

impl PartialOrd for TopKHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TopKHeapEntry {
    #[allow(dead_code)]
    fn new(abs_value: f64, original_value: f64, flat_index: u64) -> Self {
        debug_assert!(
            !abs_value.is_nan(),
            "TopKHeapEntry abs_value must not be NaN"
        );
        Self {
            abs_value,
            original_value,
            flat_index,
        }
    }
}

impl Ord for TopKHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.abs_value
            .partial_cmp(&other.abs_value)
            .unwrap_or(Ordering::Equal)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Pass2Result {
    counts: [u64; NUM_HISTOGRAM_BINS],
    edges: Vec<f64>,
    top_k: Vec<TopKHeapEntry>,
}

#[allow(dead_code)]
fn compute_pass1(values: &[f64]) -> Pass1Result {
    let mut n: u64 = 0;
    let mut mean: f64 = 0.0;
    let mut m2: f64 = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut abs_max: f64 = 0.0;
    let mut sparse_count: u64 = 0;
    let mut nan_count: u64 = 0;
    let mut l2_accum: f64 = 0.0;
    let mut l2_scale: f64 = 0.0;

    for &x in values {
        n += 1;

        // NaN guard — count but skip all accumulators so they stay clean
        if x.is_nan() {
            nan_count += 1;
            continue;
        }

        // Welford update
        let non_nan = n - nan_count;
        let delta = x - mean;
        mean += delta / non_nan as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;

        // Min / max / abs_max
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        let ax = x.abs();
        if ax > abs_max {
            abs_max = ax;
        }

        // Sparsity
        if ax < SPARSITY_EPSILON {
            sparse_count += 1;
        }

        // LAPACK-style scaled L2 accumulation (running-max, cf. dnrm2)
        if ax > l2_scale {
            if l2_scale > 0.0 {
                let ratio = l2_scale / ax;
                l2_accum = l2_accum * ratio * ratio;
            }
            l2_scale = ax;
        }
        if l2_scale > 0.0 {
            let scaled = x / l2_scale;
            l2_accum += scaled * scaled;
        }
    }

    Pass1Result {
        n,
        mean,
        m2,
        min,
        max,
        abs_max,
        sparse_count,
        nan_count,
        l2_accum,
        l2_scale,
    }
}

#[allow(dead_code)]
fn compute_pass2(values: &[f64], range_min: f64, range_max: f64, k: usize) -> Pass2Result {
    debug_assert!(
        range_min <= range_max || range_min.is_nan(),
        "compute_pass2: range_min must be <= range_max"
    );
    let mut counts = [0u64; NUM_HISTOGRAM_BINS];
    let range = range_max - range_min;

    // Min-heap: std::collections::BinaryHeap is a max-heap,
    // so we use std::cmp::Reverse for min-heap behavior.
    let mut heap: BinaryHeap<std::cmp::Reverse<TopKHeapEntry>> = BinaryHeap::with_capacity(k + 1);

    for (i, &x) in values.iter().enumerate() {
        // Skip NaNs — they were counted in pass 1
        if x.is_nan() {
            continue;
        }

        // Histogram binning
        if range > 0.0 {
            let frac = (x - range_min) / range;
            let bin = (frac * NUM_HISTOGRAM_BINS as f64).floor() as usize;
            let bin = bin.min(NUM_HISTOGRAM_BINS - 1);
            counts[bin] += 1;
        } else {
            // All values identical — put everything in the middle bin
            counts[NUM_HISTOGRAM_BINS / 2] += 1;
        }

        // Top-k min-heap on |x|
        let ax = x.abs();
        let entry = TopKHeapEntry::new(ax, x, i as u64);
        if heap.len() < k {
            heap.push(std::cmp::Reverse(entry));
        } else if let Some(std::cmp::Reverse(min_entry)) = heap.peek() {
            if ax > min_entry.abs_value {
                heap.pop();
                heap.push(std::cmp::Reverse(entry));
            }
        }
    }

    // Build histogram edges: n_bins + 1 edges from min to max
    let edges: Vec<f64> = (0..=NUM_HISTOGRAM_BINS)
        .map(|i| {
            if range > 0.0 {
                range.mul_add(i as f64 / NUM_HISTOGRAM_BINS as f64, range_min)
            } else {
                range_min
            }
        })
        .collect();

    // Extract top-k sorted by descending abs_value
    let mut top_k: Vec<TopKHeapEntry> = heap.into_iter().map(|r| r.0).collect();
    top_k.sort_by(|a, b| {
        b.abs_value
            .partial_cmp(&a.abs_value)
            .unwrap_or(Ordering::Equal)
    });

    Pass2Result {
        counts,
        edges,
        top_k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn mean_known_values() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = compute_pass1(&values);
        assert!(approx_eq(r.mean, 3.0, 1e-10));
    }

    #[test]
    fn std_known_values() {
        // Population std of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let r = compute_pass1(&values);
        let variance = r.m2 / r.n as f64;
        let std = variance.sqrt();
        assert!(approx_eq(std, 2.0, 1e-10));
    }

    #[test]
    fn min_max_abs_max() {
        let values = vec![-10.0, -3.0, 0.0, 2.0, 7.0];
        let r = compute_pass1(&values);
        assert!(approx_eq(r.min, -10.0, 1e-10));
        assert!(approx_eq(r.max, 7.0, 1e-10));
        assert!(approx_eq(r.abs_max, 10.0, 1e-10));
    }

    #[test]
    fn sparsity_all_zeros() {
        let values = vec![0.0, 0.0, 0.0, 0.0];
        let r = compute_pass1(&values);
        let sparsity = r.sparse_count as f64 / r.n as f64;
        assert!(approx_eq(sparsity, 1.0, 1e-10));
    }

    #[test]
    fn sparsity_no_zeros() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let r = compute_pass1(&values);
        let sparsity = r.sparse_count as f64 / r.n as f64;
        assert!(approx_eq(sparsity, 0.0, 1e-10));
    }

    #[test]
    fn l2_norm_known_values() {
        // [3, 4] -> L2 = 5.0
        let values = vec![3.0, 4.0];
        let r = compute_pass1(&values);
        let l2 = r.l2_scale * r.l2_accum.sqrt();
        assert!(approx_eq(l2, 5.0, 1e-10));
    }

    #[test]
    fn l2_norm_single_element() {
        let values = vec![42.0];
        let r = compute_pass1(&values);
        let l2 = r.l2_scale * r.l2_accum.sqrt();
        assert!(approx_eq(l2, 42.0, 1e-10));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn pass1_empty_slice() {
        let r = compute_pass1(&[]);
        assert_eq!(r.n, 0);
        assert!(approx_eq(r.mean, 0.0, 1e-10));
        assert_eq!(r.min, f64::INFINITY);
        assert_eq!(r.max, f64::NEG_INFINITY);
    }

    #[test]
    fn sparsity_mixed() {
        let values = vec![0.0, 5e-9, 1.0, 2.0];
        let r = compute_pass1(&values);
        assert_eq!(r.sparse_count, 2);
        let sparsity = r.sparse_count as f64 / r.n as f64;
        assert!(approx_eq(sparsity, 0.5, 1e-10));
    }

    #[test]
    fn l2_norm_negative_values() {
        // [-3, 4] -> L2 = 5.0, signs must not affect norm
        let values = vec![-3.0, 4.0];
        let r = compute_pass1(&values);
        let l2 = r.l2_scale * r.l2_accum.sqrt();
        assert!(approx_eq(l2, 5.0, 1e-10));
    }

    #[test]
    fn l2_norm_large_values_no_overflow() {
        // Values near sqrt(f64::MAX) (~1.34e154). Naive squaring overflows to
        // infinity; LAPACK-style scaling keeps this finite.
        let values = vec![1e154, 1e154, 1e154];
        let r = compute_pass1(&values);
        let l2 = r.l2_scale * r.l2_accum.sqrt();
        let expected = (3.0_f64).sqrt() * 1e154;
        assert!(approx_eq(l2, expected, expected * 1e-10));
    }

    #[test]
    fn histogram_uniform_distribution() {
        // 640 values spread uniformly => ~10 per bin
        let values: Vec<f64> = (0..640).map(|i| f64::from(i) / 640.0).collect();
        let min = 0.0;
        let max = 639.0 / 640.0;
        let r = compute_pass2(&values, min, max, 0);
        let total: u64 = r.counts.iter().sum();
        assert_eq!(total, 640);
        for &c in &r.counts {
            assert!((8..=12).contains(&c), "bin count {c} not near 10");
        }
    }

    #[test]
    fn histogram_single_value() {
        let values = vec![5.0, 5.0, 5.0];
        let r = compute_pass2(&values, 5.0, 5.0, 0);
        let total: u64 = r.counts.iter().sum();
        assert_eq!(total, 3);
        assert_eq!(r.counts[NUM_HISTOGRAM_BINS / 2], 3);
    }

    #[test]
    fn histogram_edges_correct() {
        let r = compute_pass2(&[0.0, 10.0], 0.0, 10.0, 0);
        assert_eq!(r.edges.len(), NUM_HISTOGRAM_BINS + 1);
        assert!(approx_eq(r.edges[0], 0.0, 1e-10));
        assert!(approx_eq(*r.edges.last().unwrap(), 10.0, 1e-10));
    }

    #[test]
    fn top_k_returns_largest() {
        let values = vec![1.0, -5.0, 3.0, -8.0, 2.0, 7.0, -1.0];
        let r = compute_pass2(&values, -8.0, 7.0, 3);
        assert_eq!(r.top_k.len(), 3);
        // By abs_value descending: 8, 7, 5
        assert!(approx_eq(r.top_k[0].abs_value, 8.0, 1e-10));
        assert!(approx_eq(r.top_k[0].original_value, -8.0, 1e-10));
        assert_eq!(r.top_k[0].flat_index, 3);
        assert!(approx_eq(r.top_k[1].abs_value, 7.0, 1e-10));
        assert!(approx_eq(r.top_k[2].abs_value, 5.0, 1e-10));
    }

    #[test]
    fn top_k_with_k_larger_than_n() {
        let values = vec![1.0, 2.0, 3.0];
        let r = compute_pass2(&values, 1.0, 3.0, 100);
        assert_eq!(r.top_k.len(), 3);
    }

    #[test]
    fn histogram_and_top_k_skip_nan() {
        let values = vec![f64::NAN, 3.0, f64::NAN, -5.0];
        let r = compute_pass2(&values, -5.0, 3.0, 2);
        let total: u64 = r.counts.iter().sum();
        assert_eq!(total, 2); // only 2 non-NaN values counted
        assert_eq!(r.top_k.len(), 2); // NaN not in top-k
    }
}
