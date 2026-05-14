// All items are used by `compute_summary` (pub API for tensor_store), but
// the binary crate has no caller yet — suppress dead-code until Task 7-8 wires
// the tensor store.
#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rocket_surgeon_protocol::types::{DType, Histogram, TensorStats, TopKEntry};

const NUM_HISTOGRAM_BINS: usize = 64;
const SPARSITY_EPSILON: f64 = 1e-8;
const DEFAULT_TOP_K: usize = 10;

#[derive(Debug, Clone)]
pub struct Pass1Result {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    abs_max: f64,
    sparse_count: u64,
    nan_count: u64,
    inf_count: u64,
    l2_accum: f64,
    l2_scale: f64,
}

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

#[derive(Debug, Clone)]
struct Pass2Result {
    counts: [u64; NUM_HISTOGRAM_BINS],
    edges: Vec<f64>,
    top_k: Vec<TopKHeapEntry>,
}

fn compute_pass1(values: &[f64]) -> Pass1Result {
    let mut n: u64 = 0;
    let mut mean: f64 = 0.0;
    let mut m2: f64 = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut abs_max: f64 = 0.0;
    let mut sparse_count: u64 = 0;
    let mut nan_count: u64 = 0;
    let mut inf_count: u64 = 0;
    let mut l2_accum: f64 = 0.0;
    let mut l2_scale: f64 = 0.0;

    for &x in values {
        n += 1;

        // Non-finite guard — count but skip all accumulators so they stay clean
        if !x.is_finite() {
            if x.is_nan() {
                nan_count += 1;
            } else {
                inf_count += 1;
            }
            continue;
        }

        // Welford update
        let finite = n - nan_count - inf_count;
        let delta = x - mean;
        mean += delta / finite as f64;
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
        inf_count,
        l2_accum,
        l2_scale,
    }
}

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
        // Skip non-finite values — they were counted in pass 1
        if !x.is_finite() {
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

pub fn merge_pass1(a: &Pass1Result, b: &Pass1Result) -> Pass1Result {
    if a.n == 0 {
        return b.clone();
    }
    if b.n == 0 {
        return a.clone();
    }

    let a_finite = a.n - a.nan_count - a.inf_count;
    let b_finite = b.n - b.nan_count - b.inf_count;
    let total_finite = a_finite + b_finite;

    let n = a.n + b.n;
    let nan_count = a.nan_count + b.nan_count;
    let inf_count = a.inf_count + b.inf_count;

    // Chan/Golub/LeVeque parallel Welford merge
    let (mean, m2) = if total_finite == 0 {
        (0.0, 0.0)
    } else {
        let delta = b.mean - a.mean;
        let mean = delta.mul_add(b_finite as f64 / total_finite as f64, a.mean);
        let m2 = (delta * delta).mul_add(
            a_finite as f64 * b_finite as f64 / total_finite as f64,
            a.m2 + b.m2,
        );
        (mean, m2)
    };

    let min = a.min.min(b.min);
    let max = a.max.max(b.max);
    let abs_max = a.abs_max.max(b.abs_max);
    let sparse_count = a.sparse_count + b.sparse_count;

    // Merge LAPACK-style L2 accumulators: rescale to the larger scale
    let (l2_scale, l2_accum) = if a.l2_scale >= b.l2_scale {
        if a.l2_scale > 0.0 && b.l2_scale > 0.0 {
            let ratio = b.l2_scale / a.l2_scale;
            (a.l2_scale, (b.l2_accum * ratio).mul_add(ratio, a.l2_accum))
        } else {
            (a.l2_scale, a.l2_accum)
        }
    } else if b.l2_scale > 0.0 && a.l2_scale > 0.0 {
        let ratio = a.l2_scale / b.l2_scale;
        (b.l2_scale, (a.l2_accum * ratio).mul_add(ratio, b.l2_accum))
    } else {
        (b.l2_scale, b.l2_accum)
    };

    Pass1Result {
        n,
        mean,
        m2,
        min,
        max,
        abs_max,
        sparse_count,
        nan_count,
        inf_count,
        l2_accum,
        l2_scale,
    }
}

// ---------------------------------------------------------------------------
// Byte-to-value conversion functions
// ---------------------------------------------------------------------------

fn read_f16_values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f64::from(half::f16::from_bits(bits))
        })
        .collect()
}

fn read_bf16_values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f64::from(half::bf16::from_bits(bits))
        })
        .collect()
}

fn read_f32_values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(4)
        .map(|chunk| f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect()
}

fn read_f64_values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| {
            f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect()
}

fn read_i8_values(data: &[u8]) -> Vec<f64> {
    data.iter().map(|&b| f64::from(b as i8)).collect()
}

fn read_i16_values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(2)
        .map(|chunk| f64::from(i16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

fn read_i32_values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(4)
        .map(|chunk| f64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect()
}

fn read_i64_values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f64
        })
        .collect()
}

fn read_u8_values(data: &[u8]) -> Vec<f64> {
    data.iter().map(|&b| f64::from(b)).collect()
}

fn read_bool_values(data: &[u8]) -> Vec<f64> {
    data.iter()
        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
        .collect()
}

fn decode_values(data: &[u8], dtype: DType) -> Vec<f64> {
    match dtype {
        DType::Float16 => read_f16_values(data),
        DType::Bfloat16 => read_bf16_values(data),
        DType::Float32 => read_f32_values(data),
        DType::Float64 => read_f64_values(data),
        DType::Int8 => read_i8_values(data),
        DType::Int16 => read_i16_values(data),
        DType::Int32 => read_i32_values(data),
        DType::Int64 => read_i64_values(data),
        DType::Uint8 => read_u8_values(data),
        DType::Bool => read_bool_values(data),
    }
}

// ---------------------------------------------------------------------------
// Flat-to-multi-dim index conversion
// ---------------------------------------------------------------------------

fn flat_index_to_multi(flat: u64, shape: &[u64]) -> Vec<u64> {
    if shape.is_empty() {
        return vec![];
    }
    let mut indices = vec![0u64; shape.len()];
    let mut remaining = flat;
    for i in (0..shape.len()).rev() {
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    indices
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn compute_summary(data: &[u8], dtype: DType, shape: &[u64]) -> (TensorStats, Vec<TopKEntry>) {
    let values = decode_values(data, dtype);

    if values.is_empty() {
        let empty_stats = TensorStats {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            abs_max: 0.0,
            sparsity: 1.0,
            l2_norm: 0.0,
            histogram: Histogram {
                bins: NUM_HISTOGRAM_BINS as u32,
                edges: vec![0.0; NUM_HISTOGRAM_BINS + 1],
                counts: vec![0; NUM_HISTOGRAM_BINS],
            },
        };
        return (empty_stats, vec![]);
    }

    let p1 = compute_pass1(&values);
    let finite_count = p1.n - p1.nan_count - p1.inf_count;

    if finite_count == 0 {
        // All values are non-finite (or empty after decode) — return zero stats
        let empty_stats = TensorStats {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            abs_max: 0.0,
            sparsity: 1.0,
            l2_norm: 0.0,
            histogram: Histogram {
                bins: NUM_HISTOGRAM_BINS as u32,
                edges: vec![0.0; NUM_HISTOGRAM_BINS + 1],
                counts: vec![0; NUM_HISTOGRAM_BINS],
            },
        };
        return (empty_stats, vec![]);
    }

    let variance = if finite_count > 1 {
        p1.m2 / finite_count as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();
    let sparsity = if finite_count > 0 {
        p1.sparse_count as f64 / finite_count as f64
    } else {
        1.0
    };
    let l2_norm = if p1.l2_scale > 0.0 {
        p1.l2_scale * p1.l2_accum.sqrt()
    } else {
        0.0
    };

    let k = DEFAULT_TOP_K.min(finite_count as usize);
    let p2 = compute_pass2(&values, p1.min, p1.max, k);

    let histogram = Histogram {
        bins: NUM_HISTOGRAM_BINS as u32,
        edges: p2.edges,
        counts: p2.counts.to_vec(),
    };

    let top_k: Vec<TopKEntry> = p2
        .top_k
        .iter()
        .map(|entry| TopKEntry {
            index: flat_index_to_multi(entry.flat_index, shape),
            value: entry.original_value,
        })
        .collect();

    let stats = TensorStats {
        mean: p1.mean,
        std: std_dev,
        min: p1.min,
        max: p1.max,
        abs_max: p1.abs_max,
        sparsity,
        l2_norm,
        histogram,
    };

    (stats, top_k)
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

    // --- dtype dispatch tests ---

    #[test]
    fn f32_dtype_stats() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (stats, _top_k) = compute_summary(&data, DType::Float32, &[5]);
        assert!(approx_eq(stats.mean, 3.0, 1e-6));
        assert!(approx_eq(stats.min, 1.0, 1e-6));
        assert!(approx_eq(stats.max, 5.0, 1e-6));
    }

    #[test]
    fn f16_decodes_correctly() {
        let f16_vals: Vec<half::f16> = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
        ];
        let data: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (stats, _) = compute_summary(&data, DType::Float16, &[3]);
        assert!(approx_eq(stats.mean, 2.0, 1e-3));
    }

    #[test]
    fn bf16_accumulates_in_f32() {
        let bf16_vals: Vec<half::bf16> = vec![
            half::bf16::from_f32(10.0),
            half::bf16::from_f32(20.0),
            half::bf16::from_f32(30.0),
        ];
        let data: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (stats, _) = compute_summary(&data, DType::Bfloat16, &[3]);
        assert!(approx_eq(stats.mean, 20.0, 0.5));
    }

    #[test]
    fn integer_dtype_stats() {
        let values: Vec<i32> = vec![10, 20, 30, 40, 50];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (stats, _) = compute_summary(&data, DType::Int32, &[5]);
        assert!(approx_eq(stats.mean, 30.0, 1e-10));
        assert!(approx_eq(stats.min, 10.0, 1e-10));
        assert!(approx_eq(stats.max, 50.0, 1e-10));
    }

    #[test]
    fn bool_dtype_stats() {
        let data: Vec<u8> = vec![1, 1, 0, 1, 0, 0];
        let (stats, _) = compute_summary(&data, DType::Bool, &[6]);
        assert!(approx_eq(stats.mean, 0.5, 1e-10));
        assert!(approx_eq(stats.sparsity, 0.5, 1e-10));
    }

    #[test]
    fn all_nan_tensor_returns_zero_stats() {
        let values: Vec<f32> = vec![f32::NAN, f32::NAN, f32::NAN];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (stats, top_k) = compute_summary(&data, DType::Float32, &[3]);
        assert!(approx_eq(stats.mean, 0.0, 1e-10));
        assert!(approx_eq(stats.std, 0.0, 1e-10));
        assert!(approx_eq(stats.min, 0.0, 1e-10));
        assert!(approx_eq(stats.max, 0.0, 1e-10));
        assert!(approx_eq(stats.sparsity, 1.0, 1e-10));
        assert!(top_k.is_empty());
    }

    #[test]
    fn flat_index_to_multi_index() {
        assert_eq!(flat_index_to_multi(17, &[2, 3, 4]), vec![1, 1, 1]);
        assert_eq!(flat_index_to_multi(0, &[2, 3, 4]), vec![0, 0, 0]);
        assert_eq!(flat_index_to_multi(23, &[2, 3, 4]), vec![1, 2, 3]);
    }

    #[test]
    fn welford_merge_two_halves() {
        let full: Vec<f64> = (0..100).map(|i| f64::from(i).mul_add(0.7, -20.0)).collect();
        let (left, right) = full.split_at(50);

        let full_result = compute_pass1(&full);
        let left_result = compute_pass1(left);
        let right_result = compute_pass1(right);
        let merged = merge_pass1(&left_result, &right_result);

        assert!(approx_eq(merged.mean, full_result.mean, 1e-10));
        assert!(approx_eq(merged.m2, full_result.m2, 1e-6));
        assert!(approx_eq(merged.min, full_result.min, 1e-10));
        assert!(approx_eq(merged.max, full_result.max, 1e-10));
        assert!(approx_eq(merged.abs_max, full_result.abs_max, 1e-10));
        assert_eq!(merged.sparse_count, full_result.sparse_count);

        let merged_l2 = merged.l2_scale * merged.l2_accum.sqrt();
        let full_l2 = full_result.l2_scale * full_result.l2_accum.sqrt();
        assert!(approx_eq(merged_l2, full_l2, 1e-6));
    }

    #[test]
    fn welford_merge_numerical_stability() {
        // Values clustered around 1e8 with small noise.
        // Welford avoids naive cancellation; however, the merge step's
        // floating-point non-associativity at high base/std ratio (~3.5e8)
        // causes ~1.5e-6 relative precision loss, hence 1e-5 tolerance headroom.
        let base = 1e8;
        let full: Vec<f64> = (0..1000)
            .map(|i| f64::from(i).mul_add(0.001, base))
            .collect();
        let (left, right) = full.split_at(500);

        let full_result = compute_pass1(&full);
        let merged = merge_pass1(&compute_pass1(left), &compute_pass1(right));

        let full_std = (full_result.m2 / full_result.n as f64).sqrt();
        let merged_std = (merged.m2 / merged.n as f64).sqrt();
        assert!(
            approx_eq(merged_std, full_std, full_std * 1e-5),
            "merged_std={merged_std}, full_std={full_std}"
        );
    }

    #[test]
    fn welford_merge_with_nan() {
        // Left half has a NaN, right half is clean
        let left = vec![1.0, f64::NAN, 3.0];
        let right = vec![4.0, 5.0, 6.0];
        let full_clean = vec![1.0, 3.0, 4.0, 5.0, 6.0]; // 5 non-NaN values

        let left_result = compute_pass1(&left);
        let right_result = compute_pass1(&right);
        let merged = merge_pass1(&left_result, &right_result);
        let full_result = compute_pass1(&full_clean);

        assert_eq!(merged.nan_count, 1);
        assert_eq!(merged.n - merged.nan_count, 5);
        assert!(approx_eq(merged.mean, full_result.mean, 1e-10));
        assert!(approx_eq(merged.m2, full_result.m2, 1e-6));
    }

    #[test]
    fn welford_merge_empty_partition() {
        let empty = compute_pass1(&[]);
        let values = vec![1.0, 2.0, 3.0];
        let real = compute_pass1(&values);

        // merge(empty, real) == real
        let merged_lr = merge_pass1(&empty, &real);
        assert!(approx_eq(merged_lr.mean, real.mean, 1e-10));
        assert_eq!(merged_lr.n, real.n);

        // merge(real, empty) == real
        let merged_rl = merge_pass1(&real, &empty);
        assert!(approx_eq(merged_rl.mean, real.mean, 1e-10));
        assert_eq!(merged_rl.n, real.n);
    }

    #[test]
    fn summary_matches_numpy_reference() {
        // 10 f32 values: [0.1, -0.5, 1.2, 0.0, -3.3, 2.1, 0.7, -0.9, 0.0, 1.5]
        // Pre-computed with NumPy (accounting for f32 precision loss vs f64):
        //   np.mean  =  0.089999996... (0.09)
        //   np.std   =  1.430699110... (population std)
        //   np.min   = -3.299999952...
        //   np.max   =  2.099999904...
        //   abs_max  =  3.299999952...
        //   sparsity =  0.2 (two zeros with eps=1e-8)
        //   l2_norm  =  np.linalg.norm = 4.533210754...
        let raw: Vec<f32> = vec![0.1, -0.5, 1.2, 0.0, -3.3, 2.1, 0.7, -0.9, 0.0, 1.5];
        let data: Vec<u8> = raw.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (stats, top_k) = compute_summary(&data, DType::Float32, &[10]);

        assert!(approx_eq(stats.mean, 0.09, 1e-5), "mean={}", stats.mean);
        assert!(approx_eq(stats.std, 1.430_699, 1e-5), "std={}", stats.std);
        assert!(approx_eq(stats.min, -3.3, 1e-5), "min={}", stats.min);
        assert!(approx_eq(stats.max, 2.1, 1e-5), "max={}", stats.max);
        assert!(
            approx_eq(stats.abs_max, 3.3, 1e-5),
            "abs_max={}",
            stats.abs_max
        );
        assert!(
            approx_eq(stats.sparsity, 0.2, 1e-10),
            "sparsity={}",
            stats.sparsity
        );
        assert!(
            approx_eq(stats.l2_norm, 4.533_210, 1e-4),
            "l2_norm={}",
            stats.l2_norm
        );

        // top-k: largest by abs are -3.3 (idx 4), 2.1 (idx 5), 1.5 (idx 9)
        assert!(!top_k.is_empty());
        assert!(approx_eq(top_k[0].value, -3.3, 1e-4));
        assert_eq!(top_k[0].index, vec![4]);
    }

    #[test]
    fn empty_tensor_returns_zero_stats() {
        let (stats, top_k) = compute_summary(&[], DType::Float32, &[0]);
        assert!(approx_eq(stats.mean, 0.0, 1e-10));
        assert!(approx_eq(stats.std, 0.0, 1e-10));
        assert!(approx_eq(stats.sparsity, 1.0, 1e-10));
        assert!(top_k.is_empty());
    }

    #[test]
    fn inf_values_do_not_corrupt_stats() {
        let values: Vec<f32> = vec![1.0, f32::INFINITY, 2.0, f32::NEG_INFINITY, 3.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (stats, top_k) = compute_summary(&data, DType::Float32, &[5]);
        // Only 3 finite values: [1, 2, 3]
        assert!(approx_eq(stats.mean, 2.0, 1e-6), "mean={}", stats.mean);
        assert!(approx_eq(stats.min, 1.0, 1e-6), "min={}", stats.min);
        assert!(approx_eq(stats.max, 3.0, 1e-6), "max={}", stats.max);
        assert!(
            approx_eq(stats.l2_norm, (14.0_f64).sqrt(), 1e-6),
            "l2={}",
            stats.l2_norm
        );
        assert_eq!(top_k.len(), 3);
        // Verify stats are JSON-serializable (no NaN/Inf)
        assert!(stats.mean.is_finite());
        assert!(stats.std.is_finite());
        assert!(stats.min.is_finite());
        assert!(stats.max.is_finite());
        assert!(stats.l2_norm.is_finite());
    }

    #[test]
    fn pass1_handles_infinity() {
        let values = vec![1.0, f64::INFINITY, 2.0, f64::NEG_INFINITY, 3.0];
        let r = compute_pass1(&values);
        assert_eq!(r.inf_count, 2);
        assert_eq!(r.nan_count, 0);
        let finite = r.n - r.nan_count - r.inf_count;
        assert_eq!(finite, 3);
        assert!(approx_eq(r.mean, 2.0, 1e-10));
        assert!(approx_eq(r.min, 1.0, 1e-10));
        assert!(approx_eq(r.max, 3.0, 1e-10));
    }
}
