use std::cmp::Ordering;

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
    let mut l2_accum: f64 = 0.0;
    let mut l2_scale: f64 = 0.0;

    for &x in values {
        n += 1;

        // Welford update
        let delta = x - mean;
        mean += delta / n as f64;
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

        // Blue's scaled L2 norm accumulation
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
        l2_accum,
        l2_scale,
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
    fn l2_norm_large_values_no_overflow() {
        // Values near f32 max (~3.4e38). Blue's method should not overflow.
        let values = vec![1e30, 1e30, 1e30];
        let r = compute_pass1(&values);
        let l2 = r.l2_scale * r.l2_accum.sqrt();
        let expected = (3.0_f64).sqrt() * 1e30;
        assert!(approx_eq(l2, expected, expected * 1e-10));
    }
}
