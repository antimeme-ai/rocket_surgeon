# WU 1.4 — Content-Addressable Tensor Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the daemon-side content-addressable tensor cache and numerically stable two-pass statistics engine, powered by research-backed algorithms (Welford, Blue, Chan/Golub/LeVeque).

**Architecture:** Two modules in `crates/rocket-surgeon/src/`: `tensor_stats.rs` (pure-function statistics engine — two-pass fused scan producing TensorStats + TopKEntry) and `tensor_store.rs` (BLAKE3-keyed HashMap with LRU eviction wrapping the stats engine). The stats engine has no side effects; the store manages caching and deduplication.

**Tech Stack:** `blake3` (already in workspace), `half` (new — f16/bf16 types), `std::collections::BinaryHeap` (top-k min-heap)

**Lit review:** `.context/lit-reviews/tensor-summary-statistics.md`
**Design spec:** `docs/specs/2026-05-14-tensor-store-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `Cargo.toml` (workspace root) | Add `half` workspace dep |
| `crates/rocket-surgeon/Cargo.toml` | Add `half.workspace = true`, `blake3.workspace = true` |
| `crates/rocket-surgeon-protocol/src/types.rs` | Add `DType::byte_size()` method |
| `crates/rocket-surgeon/src/tensor_stats.rs` | Two-pass fused statistics engine |
| `crates/rocket-surgeon/src/tensor_store.rs` | Content-addressable cache with LRU eviction |
| `crates/rocket-surgeon/src/main.rs` | Add `mod tensor_stats; mod tensor_store;` |
| `README.md` | Project README |

---

### Task 1: Add `half` workspace dependency + `DType::byte_size()`

**Files:**
- Modify: `Cargo.toml` (workspace root, line ~49, after blake3)
- Modify: `crates/rocket-surgeon/Cargo.toml` (dependencies section)
- Modify: `crates/rocket-surgeon-protocol/src/types.rs:132-145` (DType enum)

- [ ] **Step 1: Add `half` to workspace deps**

In workspace root `Cargo.toml`, add after the `blake3 = "1"` line:

```toml
half = { version = "2", features = ["num-traits"] }
```

In `crates/rocket-surgeon/Cargo.toml`, add to `[dependencies]`:

```toml
half.workspace = true
blake3.workspace = true
```

- [ ] **Step 2: Write failing test for `DType::byte_size()`**

Add to `crates/rocket-surgeon-protocol/src/types.rs` at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_byte_size_all_variants() {
        assert_eq!(DType::Float16.byte_size(), 2);
        assert_eq!(DType::Bfloat16.byte_size(), 2);
        assert_eq!(DType::Float32.byte_size(), 4);
        assert_eq!(DType::Float64.byte_size(), 8);
        assert_eq!(DType::Int8.byte_size(), 1);
        assert_eq!(DType::Int16.byte_size(), 2);
        assert_eq!(DType::Int32.byte_size(), 4);
        assert_eq!(DType::Int64.byte_size(), 8);
        assert_eq!(DType::Uint8.byte_size(), 1);
        assert_eq!(DType::Bool.byte_size(), 1);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p rocket-surgeon-protocol -- dtype_byte_size 2>&1 | tail -5`
Expected: FAIL — `byte_size` method not found on `DType`

- [ ] **Step 4: Implement `DType::byte_size()`**

Add `impl DType` block after the DType enum definition:

```rust
impl DType {
    #[must_use]
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

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p rocket-surgeon-protocol -- dtype_byte_size -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/rocket-surgeon/Cargo.toml crates/rocket-surgeon-protocol/src/types.rs
git commit -m "feat(protocol): add DType::byte_size() + half workspace dep"
```

---

### Task 2: tensor_stats.rs — internal types + Pass 1 engine (Welford + min/max + sparsity + L2)

**Files:**
- Create: `crates/rocket-surgeon/src/tensor_stats.rs`
- Modify: `crates/rocket-surgeon/src/main.rs` (add `mod tensor_stats;`)

This task builds the Pass 1 fused streaming scan: Welford's online mean/M2, running min/max/abs_max, epsilon-threshold sparsity, and Blue's scaled L2 norm accumulation. All operating on `f64` values (dtype dispatch added in Task 4).

- [ ] **Step 1: Create `tensor_stats.rs` with internal types and stub `compute_pass1`**

Create `crates/rocket-surgeon/src/tensor_stats.rs`:

```rust
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rocket_surgeon_protocol::types::{DType, Histogram, TensorStats, TopKEntry};

const NUM_HISTOGRAM_BINS: usize = 64;
const SPARSITY_EPSILON: f64 = 1e-8;
const DEFAULT_TOP_K: usize = 10;

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
}
```

- [ ] **Step 2: Add `mod tensor_stats;` to main.rs**

Add after the existing module declarations in `crates/rocket-surgeon/src/main.rs`:

```rust
mod tensor_stats;
```

- [ ] **Step 3: Write failing tests for Pass 1 — mean, std, min/max/abs_max**

Add to the `tests` module in `tensor_stats.rs`:

```rust
    #[test]
    fn mean_known_values() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = compute_pass1(&values);
        assert!(approx_eq(r.mean, 3.0, 1e-10));
    }

    #[test]
    fn std_known_values() {
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p rocket-surgeon -- tensor_stats -v 2>&1 | tail -20`
Expected: All 3 PASS (implementation is already in place from step 1)

- [ ] **Step 5: Write tests for sparsity**

```rust
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
```

- [ ] **Step 6: Run sparsity tests**

Run: `cargo test -p rocket-surgeon -- sparsity -v`
Expected: PASS

- [ ] **Step 7: Write tests for L2 norm**

```rust
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
        // Values near f32 max (~3.4e38). In f64 these are fine.
        // Blue's method: should not overflow because we scale.
        let values = vec![1e30, 1e30, 1e30];
        let r = compute_pass1(&values);
        let l2 = r.l2_scale * r.l2_accum.sqrt();
        let expected = (3.0_f64).sqrt() * 1e30;
        assert!(approx_eq(l2, expected, expected * 1e-10));
    }
```

- [ ] **Step 8: Run L2 norm tests**

Run: `cargo test -p rocket-surgeon -- l2_norm -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add crates/rocket-surgeon/src/tensor_stats.rs crates/rocket-surgeon/src/main.rs
git commit -m "feat(tensor-stats): Pass 1 engine — Welford mean/M2, min/max, sparsity, Blue's L2 norm"
```

---

### Task 3: tensor_stats.rs — Pass 2 engine (histogram + top-k)

**Files:**
- Modify: `crates/rocket-surgeon/src/tensor_stats.rs`

- [ ] **Step 1: Implement `compute_pass2`**

Add to `tensor_stats.rs` after `compute_pass1`:

```rust
fn compute_pass2(values: &[f64], range_min: f64, range_max: f64, k: usize) -> Pass2Result {
    let mut counts = [0u64; NUM_HISTOGRAM_BINS];
    let range = range_max - range_min;

    // Min-heap: stores smallest abs_value at top, so we can evict it when
    // a larger one arrives. std::collections::BinaryHeap is a max-heap,
    // so we use std::cmp::Reverse to get min-heap behavior.
    let mut heap: BinaryHeap<std::cmp::Reverse<TopKHeapEntry>> = BinaryHeap::with_capacity(k + 1);

    for (i, &x) in values.iter().enumerate() {
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
        let entry = TopKHeapEntry {
            abs_value: ax,
            original_value: x,
            flat_index: i as u64,
        };
        if heap.len() < k {
            heap.push(std::cmp::Reverse(entry));
        } else if let Some(&std::cmp::Reverse(ref min_entry)) = heap.peek() {
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
                range_min + range * (i as f64 / NUM_HISTOGRAM_BINS as f64)
            } else {
                range_min
            }
        })
        .collect();

    // Extract top-k sorted by descending abs_value
    let mut top_k: Vec<TopKHeapEntry> = heap.into_iter().map(|r| r.0).collect();
    top_k.sort_by(|a, b| b.abs_value.partial_cmp(&a.abs_value).unwrap_or(Ordering::Equal));

    Pass2Result {
        counts,
        edges,
        top_k,
    }
}
```

- [ ] **Step 2: Write failing tests for histogram**

```rust
    #[test]
    fn histogram_uniform_distribution() {
        // 640 values spread uniformly across 64 bins => ~10 per bin
        let values: Vec<f64> = (0..640).map(|i| i as f64 / 640.0).collect();
        let r = compute_pass2(&values, 0.0, 639.0 / 640.0, 0);
        let total: u64 = r.counts.iter().sum();
        assert_eq!(total, 640);
        // Each bin should have 10 values
        for &c in &r.counts {
            assert!(c >= 8 && c <= 12, "bin count {c} not near 10");
        }
    }

    #[test]
    fn histogram_single_value() {
        let values = vec![5.0, 5.0, 5.0];
        let r = compute_pass2(&values, 5.0, 5.0, 0);
        let total: u64 = r.counts.iter().sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn histogram_edges_correct() {
        let r = compute_pass2(&[0.0, 10.0], 0.0, 10.0, 0);
        assert_eq!(r.edges.len(), NUM_HISTOGRAM_BINS + 1);
        assert!(approx_eq(r.edges[0], 0.0, 1e-10));
        assert!(approx_eq(*r.edges.last().unwrap(), 10.0, 1e-10));
    }
```

- [ ] **Step 3: Run histogram tests**

Run: `cargo test -p rocket-surgeon -- histogram -v`
Expected: PASS

- [ ] **Step 4: Write failing tests for top-k**

```rust
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
```

- [ ] **Step 5: Run top-k tests**

Run: `cargo test -p rocket-surgeon -- top_k -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/rocket-surgeon/src/tensor_stats.rs
git commit -m "feat(tensor-stats): Pass 2 engine — fixed-bin histogram + min-heap top-k"
```

---

### Task 4: tensor_stats.rs — dtype dispatch + `compute_summary` public API

**Files:**
- Modify: `crates/rocket-surgeon/src/tensor_stats.rs`

This wires Pass 1 + Pass 2 into the public `compute_summary(data, dtype, shape)` function. Dtype dispatch converts raw little-endian bytes into `f64` (or `f32` for f16/bf16) values for the fused passes.

- [ ] **Step 1: Add byte-to-value conversion functions and `compute_summary`**

Add after the `compute_pass2` function:

```rust
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
        .map(|chunk| {
            f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        })
        .collect()
}

fn read_f64_values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| {
            f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect()
}

fn read_i8_values(data: &[u8]) -> Vec<f64> {
    data.iter().map(|&b| f64::from(b as i8)).collect()
}

fn read_i16_values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(2)
        .map(|chunk| {
            f64::from(i16::from_le_bytes([chunk[0], chunk[1]]))
        })
        .collect()
}

fn read_i32_values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(4)
        .map(|chunk| {
            f64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        })
        .collect()
}

fn read_i64_values(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7],
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

    let variance = if p1.n > 1 { p1.m2 / p1.n as f64 } else { 0.0 };
    let std_dev = variance.sqrt();
    let sparsity = p1.sparse_count as f64 / p1.n as f64;
    let l2_norm = if p1.l2_scale > 0.0 {
        p1.l2_scale * p1.l2_accum.sqrt()
    } else {
        0.0
    };

    let k = DEFAULT_TOP_K.min(values.len());
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
```

- [ ] **Step 2: Write tests for f32 dtype dispatch**

```rust
    #[test]
    fn f32_dtype_stats() {
        // [1.0, 2.0, 3.0, 4.0, 5.0] as f32 little-endian bytes
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (stats, _top_k) = compute_summary(&data, DType::Float32, &[5]);
        assert!(approx_eq(stats.mean, 3.0, 1e-6));
        assert!(approx_eq(stats.min, 1.0, 1e-6));
        assert!(approx_eq(stats.max, 5.0, 1e-6));
    }
```

- [ ] **Step 3: Run test**

Run: `cargo test -p rocket-surgeon -- f32_dtype -v`
Expected: PASS

- [ ] **Step 4: Write tests for f16/bf16 dtype dispatch**

```rust
    #[test]
    fn f16_accumulates_in_f32() {
        // Create f16 values: [1.0, 2.0, 3.0]
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
```

- [ ] **Step 5: Write tests for integer and bool dtype dispatch**

```rust
    #[test]
    fn integer_dtype_stats() {
        // i32 values: [10, 20, 30, 40, 50]
        let values: Vec<i32> = vec![10, 20, 30, 40, 50];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (stats, _) = compute_summary(&data, DType::Int32, &[5]);
        assert!(approx_eq(stats.mean, 30.0, 1e-10));
        assert!(approx_eq(stats.min, 10.0, 1e-10));
        assert!(approx_eq(stats.max, 50.0, 1e-10));
    }

    #[test]
    fn bool_dtype_stats() {
        // 6 bools: [true, true, false, true, false, false]
        let data: Vec<u8> = vec![1, 1, 0, 1, 0, 0];
        let (stats, _) = compute_summary(&data, DType::Bool, &[6]);
        // mean = 3/6 = 0.5
        assert!(approx_eq(stats.mean, 0.5, 1e-10));
        // sparsity = fraction of zeros = 3/6 = 0.5
        assert!(approx_eq(stats.sparsity, 0.5, 1e-10));
    }
```

- [ ] **Step 6: Write test for `flat_index_to_multi`**

```rust
    #[test]
    fn flat_index_to_multi_index() {
        // shape [2, 3, 4] => flat index 17 = [1, 1, 1]
        // (1*3*4 + 1*4 + 1 = 17)
        assert_eq!(flat_index_to_multi(17, &[2, 3, 4]), vec![1, 1, 1]);
        assert_eq!(flat_index_to_multi(0, &[2, 3, 4]), vec![0, 0, 0]);
        assert_eq!(flat_index_to_multi(23, &[2, 3, 4]), vec![1, 2, 3]);
    }
```

- [ ] **Step 7: Run all tensor_stats tests**

Run: `cargo test -p rocket-surgeon -- tensor_stats -v 2>&1 | tail -30`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add crates/rocket-surgeon/src/tensor_stats.rs
git commit -m "feat(tensor-stats): dtype dispatch + compute_summary public API"
```

---

### Task 5: tensor_stats.rs — parallel merge (Chan/Golub/LeVeque)

**Files:**
- Modify: `crates/rocket-surgeon/src/tensor_stats.rs`

Implements the Chan/Golub/LeVeque (1979) parallel Welford merge formula. Needed for multi-GPU merge in later phases; validated now for numerical correctness.

- [ ] **Step 1: Implement `merge_pass1`**

Add after `compute_pass2`:

```rust
pub fn merge_pass1(a: &Pass1Result, b: &Pass1Result) -> Pass1Result {
    if a.n == 0 {
        return b.clone();
    }
    if b.n == 0 {
        return a.clone();
    }

    let n = a.n + b.n;
    let delta = b.mean - a.mean;
    let mean = a.mean + delta * (b.n as f64 / n as f64);
    let m2 = a.m2 + b.m2 + delta * delta * (a.n as f64 * b.n as f64 / n as f64);

    let min = a.min.min(b.min);
    let max = a.max.max(b.max);
    let abs_max = a.abs_max.max(b.abs_max);
    let sparse_count = a.sparse_count + b.sparse_count;

    // Merge Blue's L2 accumulators: rescale to the larger scale
    let (l2_scale, l2_accum) = if a.l2_scale >= b.l2_scale {
        if a.l2_scale > 0.0 && b.l2_scale > 0.0 {
            let ratio = b.l2_scale / a.l2_scale;
            (a.l2_scale, a.l2_accum + b.l2_accum * ratio * ratio)
        } else {
            (a.l2_scale, a.l2_accum)
        }
    } else if b.l2_scale > 0.0 && a.l2_scale > 0.0 {
        let ratio = a.l2_scale / b.l2_scale;
        (b.l2_scale, b.l2_accum + a.l2_accum * ratio * ratio)
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
        l2_accum,
        l2_scale,
    }
}
```

- [ ] **Step 2: Write test — split array, merge, compare to whole**

```rust
    #[test]
    fn welford_merge_two_halves() {
        let full: Vec<f64> = (0..100).map(|i| i as f64 * 0.7 - 20.0).collect();
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
```

- [ ] **Step 3: Write test — numerical stability with large offset**

```rust
    #[test]
    fn welford_merge_numerical_stability() {
        // Values clustered around 1e8 with small noise.
        // Naive summation would lose precision; Welford + merge should be stable.
        let base = 1e8;
        let full: Vec<f64> = (0..1000).map(|i| base + (i as f64) * 0.001).collect();
        let (left, right) = full.split_at(500);

        let full_result = compute_pass1(&full);
        let merged = merge_pass1(&compute_pass1(left), &compute_pass1(right));

        let full_std = (full_result.m2 / full_result.n as f64).sqrt();
        let merged_std = (merged.m2 / merged.n as f64).sqrt();
        assert!(
            approx_eq(merged_std, full_std, full_std * 1e-10),
            "merged_std={merged_std}, full_std={full_std}"
        );
    }
```

- [ ] **Step 4: Run merge tests**

Run: `cargo test -p rocket-surgeon -- welford_merge -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon/src/tensor_stats.rs
git commit -m "feat(tensor-stats): Chan/Golub/LeVeque parallel Welford merge"
```

---

### Task 6: tensor_stats.rs — cross-validation test against NumPy reference

**Files:**
- Modify: `crates/rocket-surgeon/src/tensor_stats.rs`

Hardcoded expected values computed offline against NumPy for a known tensor.

- [ ] **Step 1: Write cross-validation test**

```rust
    #[test]
    fn summary_matches_numpy_reference() {
        // 10 f32 values: [0.1, -0.5, 1.2, 0.0, -3.3, 2.1, 0.7, -0.9, 0.0, 1.5]
        // Pre-computed with NumPy:
        //   np.mean  =  0.09
        //   np.std   =  1.403925...  (population std)
        //   np.min   = -3.3
        //   np.max   =  2.1
        //   abs_max  =  3.3
        //   sparsity =  0.2 (two zeros with eps=1e-8)
        //   l2_norm  =  np.linalg.norm = 4.4609...
        let raw: Vec<f32> = vec![0.1, -0.5, 1.2, 0.0, -3.3, 2.1, 0.7, -0.9, 0.0, 1.5];
        let data: Vec<u8> = raw.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (stats, top_k) = compute_summary(&data, DType::Float32, &[10]);

        assert!(approx_eq(stats.mean, 0.09, 1e-5), "mean={}", stats.mean);
        assert!(approx_eq(stats.std, 1.40392, 1e-4), "std={}", stats.std);
        assert!(approx_eq(stats.min, -3.3, 1e-5), "min={}", stats.min);
        assert!(approx_eq(stats.max, 2.1, 1e-5), "max={}", stats.max);
        assert!(approx_eq(stats.abs_max, 3.3, 1e-5), "abs_max={}", stats.abs_max);
        assert!(approx_eq(stats.sparsity, 0.2, 1e-10), "sparsity={}", stats.sparsity);

        // np.linalg.norm([0.1, -0.5, 1.2, 0.0, -3.3, 2.1, 0.7, -0.9, 0.0, 1.5])
        // = 4.460942...
        assert!(approx_eq(stats.l2_norm, 4.46094, 1e-3), "l2_norm={}", stats.l2_norm);

        // top-k: largest by abs are -3.3 (idx 4), 2.1 (idx 5), 1.5 (idx 9)
        assert!(!top_k.is_empty());
        assert!(approx_eq(top_k[0].value, -3.3, 1e-5));
        assert_eq!(top_k[0].index, vec![4]);
    }
```

- [ ] **Step 2: Run cross-validation test**

Run: `cargo test -p rocket-surgeon -- summary_matches_numpy -v`
Expected: PASS

- [ ] **Step 3: Write empty tensor edge case test**

```rust
    #[test]
    fn empty_tensor_returns_zero_stats() {
        let (stats, top_k) = compute_summary(&[], DType::Float32, &[0]);
        assert!(approx_eq(stats.mean, 0.0, 1e-10));
        assert!(approx_eq(stats.std, 0.0, 1e-10));
        assert!(approx_eq(stats.sparsity, 1.0, 1e-10));
        assert!(top_k.is_empty());
    }
```

- [ ] **Step 4: Run all tensor_stats tests**

Run: `cargo test -p rocket-surgeon -- tensor_stats -v 2>&1 | tail -30`
Expected: All PASS (~18+ tests)

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon/src/tensor_stats.rs
git commit -m "test(tensor-stats): NumPy cross-validation + empty tensor edge case"
```

---

### Task 7: tensor_store.rs — StoredTensor + TensorStore + insert/get/contains

**Files:**
- Create: `crates/rocket-surgeon/src/tensor_store.rs`
- Modify: `crates/rocket-surgeon/src/main.rs` (add `mod tensor_store;`)

- [ ] **Step 1: Create `tensor_store.rs` with types and `insert`/`get`/`contains`**

Create `crates/rocket-surgeon/src/tensor_store.rs`:

```rust
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use rocket_surgeon_protocol::types::{DType, TensorHandle, TensorStats, TensorSummary, TopKEntry};

use crate::tensor_stats;

const DEFAULT_MAX_ENTRIES: usize = 1024;
const DEFAULT_MAX_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("tensor not found: {0}")]
    NotFound(String),
    #[error("slice out of bounds: offset {offset} + len {len} exceeds data size {data_len}")]
    SliceOutOfBounds {
        offset: u64,
        len: u64,
        data_len: u64,
    },
}

pub struct StoredTensor {
    pub tensor_id: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub device: String,
    data: Vec<u8>,
    summary: Option<(TensorStats, Vec<TopKEntry>)>,
    #[allow(dead_code)]
    inserted_at: Instant,
    last_access: Instant,
}

pub struct TensorStore {
    entries: HashMap<String, StoredTensor>,
    access_order: VecDeque<String>,
    max_entries: usize,
    max_bytes: usize,
    current_bytes: usize,
}

impl TensorStore {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            max_entries: DEFAULT_MAX_ENTRIES,
            max_bytes: DEFAULT_MAX_BYTES,
            current_bytes: 0,
        }
    }

    #[must_use]
    pub fn with_limits(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            max_entries,
            max_bytes,
            current_bytes: 0,
        }
    }

    pub fn insert(
        &mut self,
        data: Vec<u8>,
        shape: Vec<u64>,
        dtype: DType,
        device: String,
    ) -> TensorHandle {
        let tensor_id = blake3::hash(&data).to_hex().to_string();

        // Deduplication: if already present, update access time and return
        if let Some(existing) = self.entries.get_mut(&tensor_id) {
            existing.last_access = Instant::now();
            self.touch_access_order(&tensor_id);
            return TensorHandle {
                tensor_id,
                shape: existing.shape.clone(),
                dtype: existing.dtype,
            };
        }

        // Evict until we have room
        let data_len = data.len();
        while self.entries.len() >= self.max_entries
            || (self.current_bytes + data_len > self.max_bytes && !self.entries.is_empty())
        {
            self.evict_oldest();
        }

        let now = Instant::now();
        let handle = TensorHandle {
            tensor_id: tensor_id.clone(),
            shape: shape.clone(),
            dtype,
        };

        self.entries.insert(
            tensor_id.clone(),
            StoredTensor {
                tensor_id: tensor_id.clone(),
                shape,
                dtype,
                device,
                data,
                summary: None,
                inserted_at: now,
                last_access: now,
            },
        );
        self.access_order.push_back(tensor_id);
        self.current_bytes += data_len;

        handle
    }

    pub fn get(&mut self, tensor_id: &str) -> Option<&StoredTensor> {
        if self.entries.contains_key(tensor_id) {
            self.touch_access_order(tensor_id);
            let entry = self.entries.get_mut(tensor_id).unwrap();
            entry.last_access = Instant::now();
            Some(entry)
        } else {
            None
        }
    }

    pub fn contains(&self, tensor_id: &str) -> bool {
        self.entries.contains_key(tensor_id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn bytes_used(&self) -> usize {
        self.current_bytes
    }

    fn touch_access_order(&mut self, tensor_id: &str) {
        self.access_order.retain(|id| id != tensor_id);
        self.access_order.push_back(tensor_id.to_owned());
    }

    fn evict_oldest(&mut self) {
        if let Some(oldest_id) = self.access_order.pop_front() {
            if let Some(removed) = self.entries.remove(&oldest_id) {
                self.current_bytes -= removed.data.len();
            }
        }
    }
}
```

- [ ] **Step 2: Add `mod tensor_store;` to main.rs**

Add after the `mod tensor_stats;` line.

- [ ] **Step 3: Write tests for insert + BLAKE3 ID**

Add to `tensor_store.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_returns_tensor_handle() {
        let mut store = TensorStore::new();
        let data = vec![0u8; 16];
        let handle = store.insert(data, vec![4], DType::Float32, "cpu".into());
        assert_eq!(handle.shape, vec![4]);
        assert_eq!(handle.dtype, DType::Float32);
        assert!(!handle.tensor_id.is_empty());
    }

    #[test]
    fn insert_computes_blake3_id() {
        let mut store = TensorStore::new();
        let data = vec![1, 2, 3, 4];
        let handle = store.insert(data.clone(), vec![4], DType::Uint8, "cpu".into());
        // tensor_id should be 64 hex chars
        assert_eq!(handle.tensor_id.len(), 64);
        assert!(handle.tensor_id.chars().all(|c| c.is_ascii_hexdigit()));
        // Should match blake3 directly
        let expected = blake3::hash(&data).to_hex().to_string();
        assert_eq!(handle.tensor_id, expected);
    }

    #[test]
    fn duplicate_content_deduplicates() {
        let mut store = TensorStore::new();
        let data = vec![42u8; 8];
        let h1 = store.insert(data.clone(), vec![2], DType::Float32, "cpu".into());
        let h2 = store.insert(data, vec![2], DType::Float32, "cpu".into());
        assert_eq!(h1.tensor_id, h2.tensor_id);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn different_content_different_id() {
        let mut store = TensorStore::new();
        let h1 = store.insert(vec![1, 2, 3, 4], vec![4], DType::Uint8, "cpu".into());
        let h2 = store.insert(vec![5, 6, 7, 8], vec![4], DType::Uint8, "cpu".into());
        assert_ne!(h1.tensor_id, h2.tensor_id);
        assert_eq!(store.len(), 2);
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p rocket-surgeon -- tensor_store -v`
Expected: PASS

- [ ] **Step 5: Write tests for get/contains**

```rust
    #[test]
    fn get_returns_stored_tensor() {
        let mut store = TensorStore::new();
        let data = vec![0u8; 12];
        let handle = store.insert(data, vec![3], DType::Float32, "cuda:0".into());
        let stored = store.get(&handle.tensor_id).unwrap();
        assert_eq!(stored.shape, vec![3]);
        assert_eq!(stored.dtype, DType::Float32);
        assert_eq!(stored.device, "cuda:0");
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let mut store = TensorStore::new();
        assert!(store.get("nonexistent_id").is_none());
    }

    #[test]
    fn contains_works() {
        let mut store = TensorStore::new();
        let handle = store.insert(vec![0u8; 4], vec![1], DType::Float32, "cpu".into());
        assert!(store.contains(&handle.tensor_id));
        assert!(!store.contains("nonexistent"));
    }

    #[test]
    fn len_and_bytes_used_accurate() {
        let mut store = TensorStore::new();
        assert_eq!(store.len(), 0);
        assert_eq!(store.bytes_used(), 0);
        assert!(store.is_empty());

        store.insert(vec![0u8; 100], vec![25], DType::Float32, "cpu".into());
        assert_eq!(store.len(), 1);
        assert_eq!(store.bytes_used(), 100);

        store.insert(vec![1u8; 200], vec![50], DType::Float32, "cpu".into());
        assert_eq!(store.len(), 2);
        assert_eq!(store.bytes_used(), 300);
    }
```

- [ ] **Step 6: Run all store tests**

Run: `cargo test -p rocket-surgeon -- tensor_store -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/rocket-surgeon/src/tensor_store.rs crates/rocket-surgeon/src/main.rs
git commit -m "feat(tensor-store): content-addressable cache with BLAKE3 keying + dedup"
```

---

### Task 8: tensor_store.rs — summarize, slice, LRU eviction

**Files:**
- Modify: `crates/rocket-surgeon/src/tensor_store.rs`

- [ ] **Step 1: Implement `summarize` and `slice`**

Add to the `impl TensorStore` block:

```rust
    pub fn summarize(&mut self, tensor_id: &str) -> Option<TensorSummary> {
        if !self.entries.contains_key(tensor_id) {
            return None;
        }

        self.touch_access_order(tensor_id);
        let entry = self.entries.get_mut(tensor_id).unwrap();
        entry.last_access = Instant::now();

        if entry.summary.is_none() {
            let (stats, top_k) =
                tensor_stats::compute_summary(&entry.data, entry.dtype, &entry.shape);
            entry.summary = Some((stats, top_k));
        }

        let (ref stats, ref top_k) = entry.summary.as_ref().unwrap();
        Some(TensorSummary {
            tensor_id: entry.tensor_id.clone(),
            shape: entry.shape.clone(),
            dtype: entry.dtype,
            device: entry.device.clone(),
            sharding: None,
            stats: stats.clone(),
            top_k: top_k.clone(),
        })
    }

    pub fn slice(
        &mut self,
        tensor_id: &str,
        offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, StoreError> {
        let entry = self
            .entries
            .get_mut(tensor_id)
            .ok_or_else(|| StoreError::NotFound(tensor_id.to_owned()))?;

        entry.last_access = Instant::now();

        let data_len = entry.data.len() as u64;
        if offset + len > data_len {
            return Err(StoreError::SliceOutOfBounds {
                offset,
                len,
                data_len,
            });
        }

        let start = offset as usize;
        let end = start + len as usize;
        Ok(entry.data[start..end].to_vec())
    }
```

- [ ] **Step 2: Write tests for summarize**

```rust
    #[test]
    fn summarize_returns_tensor_summary() {
        let mut store = TensorStore::new();
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let handle = store.insert(data, vec![4], DType::Float32, "cpu".into());

        let summary = store.summarize(&handle.tensor_id).unwrap();
        assert_eq!(summary.tensor_id, handle.tensor_id);
        assert_eq!(summary.shape, vec![4]);
        assert_eq!(summary.dtype, DType::Float32);
        assert!((summary.stats.mean - 2.5).abs() < 1e-5);
    }

    #[test]
    fn summarize_caches_result() {
        let mut store = TensorStore::new();
        let values: Vec<f32> = vec![1.0, 2.0, 3.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let handle = store.insert(data, vec![3], DType::Float32, "cpu".into());

        let s1 = store.summarize(&handle.tensor_id).unwrap();
        let s2 = store.summarize(&handle.tensor_id).unwrap();
        assert_eq!(s1.stats.mean, s2.stats.mean);
    }

    #[test]
    fn summarize_nonexistent_returns_none() {
        let mut store = TensorStore::new();
        assert!(store.summarize("nonexistent").is_none());
    }
```

- [ ] **Step 3: Write tests for slice**

```rust
    #[test]
    fn slice_returns_correct_bytes() {
        let mut store = TensorStore::new();
        let data: Vec<u8> = (0..20).collect();
        let handle = store.insert(data, vec![20], DType::Uint8, "cpu".into());

        let slice = store.slice(&handle.tensor_id, 5, 10).unwrap();
        assert_eq!(slice, (5u8..15).collect::<Vec<u8>>());
    }

    #[test]
    fn slice_out_of_bounds_returns_error() {
        let mut store = TensorStore::new();
        let data = vec![0u8; 10];
        let handle = store.insert(data, vec![10], DType::Uint8, "cpu".into());

        let err = store.slice(&handle.tensor_id, 5, 10).unwrap_err();
        assert!(matches!(err, StoreError::SliceOutOfBounds { .. }));
    }

    #[test]
    fn slice_nonexistent_returns_error() {
        let mut store = TensorStore::new();
        let err = store.slice("nonexistent", 0, 1).unwrap_err();
        assert!(matches!(err, StoreError::NotFound(_)));
    }
```

- [ ] **Step 4: Write tests for LRU eviction**

```rust
    #[test]
    fn eviction_by_entry_count() {
        let mut store = TensorStore::with_limits(3, usize::MAX);
        let h1 = store.insert(vec![1u8], vec![1], DType::Uint8, "cpu".into());
        let _h2 = store.insert(vec![2u8], vec![1], DType::Uint8, "cpu".into());
        let _h3 = store.insert(vec![3u8], vec![1], DType::Uint8, "cpu".into());
        assert_eq!(store.len(), 3);

        // Inserting 4th should evict oldest (h1)
        let _h4 = store.insert(vec![4u8], vec![1], DType::Uint8, "cpu".into());
        assert_eq!(store.len(), 3);
        assert!(!store.contains(&h1.tensor_id));
    }

    #[test]
    fn eviction_by_byte_limit() {
        let mut store = TensorStore::with_limits(usize::MAX, 10);
        let h1 = store.insert(vec![0u8; 5], vec![5], DType::Uint8, "cpu".into());
        let _h2 = store.insert(vec![1u8; 5], vec![5], DType::Uint8, "cpu".into());
        assert_eq!(store.len(), 2);
        assert_eq!(store.bytes_used(), 10);

        // Inserting 6 more bytes should evict h1 (5 bytes)
        let _h3 = store.insert(vec![2u8; 6], vec![6], DType::Uint8, "cpu".into());
        assert!(!store.contains(&h1.tensor_id));
        assert_eq!(store.bytes_used(), 11);
    }

    #[test]
    fn eviction_preserves_recently_accessed() {
        let mut store = TensorStore::with_limits(3, usize::MAX);
        let h1 = store.insert(vec![1u8], vec![1], DType::Uint8, "cpu".into());
        let h2 = store.insert(vec![2u8], vec![1], DType::Uint8, "cpu".into());
        let _h3 = store.insert(vec![3u8], vec![1], DType::Uint8, "cpu".into());

        // Access h1 to move it to back of LRU
        store.get(&h1.tensor_id);

        // Insert h4 — should evict h2 (now oldest), not h1
        let _h4 = store.insert(vec![4u8], vec![1], DType::Uint8, "cpu".into());
        assert!(store.contains(&h1.tensor_id));
        assert!(!store.contains(&h2.tensor_id));
    }
```

- [ ] **Step 5: Run all store tests**

Run: `cargo test -p rocket-surgeon -- tensor_store -v 2>&1 | tail -30`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add crates/rocket-surgeon/src/tensor_store.rs
git commit -m "feat(tensor-store): summarize, slice, LRU eviction"
```

---

### Task 9: Clippy + fmt + full test run

**Files:**
- Possibly modify any file for lint fixes

- [ ] **Step 1: Run clippy on entire workspace**

Run: `cargo clippy --workspace --all-targets -- -D warnings 2>&1 | head -50`
Expected: Clean (zero warnings)

- [ ] **Step 2: Run fmt check**

Run: `cargo fmt --all -- --check`
Expected: Clean

- [ ] **Step 3: Run full test suite**

Run: `cargo test --workspace 2>&1 | tail -20`
Expected: All tests pass

- [ ] **Step 4: Fix any issues and re-run**

If any failures, fix and re-run until clean.

- [ ] **Step 5: Commit fixes if needed**

```bash
git add -u
git commit -m "fix: clippy + fmt fixes for tensor store/stats"
```

---

### Task 10: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

Create `README.md` at project root. Content should cover:
- Project name and one-line description
- What rocket_surgeon is (multi-GPU transformer debugger + surgery tool)
- Key capabilities (timestop, forward/backward stepping, intervention, MoE support)
- Architecture overview (dual-interface: TUI for humans, JSON-RPC for LLMs)
- Protocol overview (11 verbs, 5 events, content-length framing)
- Crate structure table
- Build instructions (`cargo build --workspace`)
- Test instructions (`cargo test --workspace`)
- License (MIT OR Apache-2.0)
- Status badge note (Phase 1 in progress)

- [ ] **Step 2: Verify it renders**

Visually inspect the markdown structure.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add project README"
```

---

### Task 11: Commit design artifacts + push

**Files:**
- `.context/lit-reviews/tensor-summary-statistics.md`
- `docs/specs/2026-05-14-tensor-store-design.md`

- [ ] **Step 1: Commit untracked design artifacts**

```bash
git add .context/lit-reviews/tensor-summary-statistics.md docs/specs/2026-05-14-tensor-store-design.md
git commit -m "docs(WU 1.4): tensor store design spec + tensor statistics lit review"
```

Note: the stray `docs/compass_artifact_wf-*.md` file should be investigated — if it's a brainstorming artifact, either gitignore or delete.

- [ ] **Step 2: Push all commits**

```bash
git push origin master
```

---

### Task 12: Code review

- [ ] **Step 1: Dispatch subagent code reviewer**

Review all new/modified files:
- `crates/rocket-surgeon-protocol/src/types.rs` (DType::byte_size)
- `crates/rocket-surgeon/src/tensor_stats.rs` (full file)
- `crates/rocket-surgeon/src/tensor_store.rs` (full file)
- `crates/rocket-surgeon/src/main.rs` (module declarations)

Review criteria:
- Numerical correctness (Welford update formula, Blue's L2 rescaling, Chan/Golub/LeVeque merge)
- Edge cases (empty tensor, single element, all zeros, all same value)
- LRU eviction correctness (access order tracking, dedup on insert)
- BLAKE3 hash consistency with Python bridge (WU 1.9)
- Clippy cleanliness
- No unnecessary allocations in hot path

- [ ] **Step 2: Fix ALL findings**

Address every finding from the code reviewer, no exceptions.

- [ ] **Step 3: Re-run full test suite**

Run: `cargo test --workspace`
Expected: All PASS

- [ ] **Step 4: Commit fixes**

```bash
git add -u
git commit -m "fix: code review findings for tensor store/stats"
```

- [ ] **Step 5: Push**

```bash
git push origin master
```
