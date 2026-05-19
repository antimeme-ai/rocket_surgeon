use std::collections::HashMap;
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
    last_access_gen: u64,
}

pub struct TensorStore {
    entries: HashMap<String, StoredTensor>,
    access_generation: u64,
    max_entries: usize,
    max_bytes: usize,
    current_bytes: usize,
}

impl TensorStore {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            access_generation: 0,
            max_entries: DEFAULT_MAX_ENTRIES,
            max_bytes: DEFAULT_MAX_BYTES,
            current_bytes: 0,
        }
    }

    #[must_use]
    #[allow(dead_code)]
    pub fn with_limits(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_generation: 0,
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

        if let Some(existing) = self.entries.get_mut(&tensor_id) {
            existing.last_access_gen = self.access_generation;
            self.access_generation += 1;
            existing.last_access = Instant::now();
            return TensorHandle {
                tensor_id,
                shape: existing.shape.clone(),
                dtype: existing.dtype,
            };
        }

        let data_len = data.len();
        // If data_len > max_bytes, we evict everything and insert anyway.
        // This is intentional — refusing the insert would be worse.
        while !self.entries.is_empty()
            && (self.entries.len() >= self.max_entries
                || self.current_bytes + data_len > self.max_bytes)
        {
            self.evict_oldest();
        }

        let generation = self.access_generation;
        self.access_generation += 1;
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
                last_access_gen: generation,
            },
        );
        self.current_bytes += data_len;

        handle
    }

    #[allow(dead_code)]
    pub fn insert_with_id(
        &mut self,
        tensor_id: String,
        data: Vec<u8>,
        shape: Vec<u64>,
        dtype: DType,
        device: String,
    ) -> TensorHandle {
        if let Some(existing) = self.entries.get_mut(&tensor_id) {
            existing.last_access_gen = self.access_generation;
            self.access_generation += 1;
            existing.last_access = Instant::now();
            return TensorHandle {
                tensor_id,
                shape: existing.shape.clone(),
                dtype: existing.dtype,
            };
        }

        let data_len = data.len();
        // If data_len > max_bytes, we evict everything and insert anyway.
        // This is intentional — refusing the insert would be worse.
        while !self.entries.is_empty()
            && (self.entries.len() >= self.max_entries
                || self.current_bytes + data_len > self.max_bytes)
        {
            self.evict_oldest();
        }

        let generation = self.access_generation;
        self.access_generation += 1;
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
                last_access_gen: generation,
            },
        );
        self.current_bytes += data_len;

        handle
    }

    #[allow(dead_code)]
    pub fn get(&mut self, tensor_id: &str) -> Option<&StoredTensor> {
        if let Some(entry) = self.entries.get_mut(tensor_id) {
            entry.last_access_gen = self.access_generation;
            self.access_generation += 1;
            entry.last_access = Instant::now();
            Some(entry)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub fn contains(&self, tensor_id: &str) -> bool {
        self.entries.contains_key(tensor_id)
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[allow(dead_code)]
    pub fn bytes_used(&self) -> usize {
        self.current_bytes
    }

    pub fn summarize(&mut self, tensor_id: &str) -> Option<TensorSummary> {
        let entry = self.entries.get_mut(tensor_id)?;
        entry.last_access_gen = self.access_generation;
        self.access_generation += 1;
        entry.last_access = Instant::now();

        if entry.summary.is_none() {
            let (stats, top_k) =
                tensor_stats::compute_summary(&entry.data, entry.dtype, &entry.shape);
            entry.summary = Some((stats, top_k));
        }

        let (stats, top_k) = entry.summary.as_ref().unwrap();
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

    pub fn slice(&mut self, tensor_id: &str, offset: u64, len: u64) -> Result<Vec<u8>, StoreError> {
        let entry = self
            .entries
            .get_mut(tensor_id)
            .ok_or_else(|| StoreError::NotFound(tensor_id.to_owned()))?;
        entry.last_access_gen = self.access_generation;
        self.access_generation += 1;
        entry.last_access = Instant::now();

        let data_len = entry.data.len() as u64;
        if offset.checked_add(len).is_none_or(|end| end > data_len) {
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

    fn evict_oldest(&mut self) {
        let oldest_id = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access_gen)
            .map(|(id, _)| id.clone());
        if let Some(id) = oldest_id {
            if let Some(removed) = self.entries.remove(&id) {
                self.current_bytes -= removed.data.len();
            }
        }
    }
}

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
        assert_eq!(handle.tensor_id.len(), 64);
        assert!(handle.tensor_id.chars().all(|c| c.is_ascii_hexdigit()));
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

    // --- summarize tests ---

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

        // First call computes and caches
        assert!(
            store
                .entries
                .get(&handle.tensor_id)
                .unwrap()
                .summary
                .is_none()
        );
        let _s1 = store.summarize(&handle.tensor_id).unwrap();
        assert!(
            store
                .entries
                .get(&handle.tensor_id)
                .unwrap()
                .summary
                .is_some()
        );

        // Second call returns cached (summary field still Some)
        let _s2 = store.summarize(&handle.tensor_id).unwrap();
        assert!(
            store
                .entries
                .get(&handle.tensor_id)
                .unwrap()
                .summary
                .is_some()
        );
    }

    #[test]
    fn summarize_nonexistent_returns_none() {
        let mut store = TensorStore::new();
        assert!(store.summarize("nonexistent").is_none());
    }

    // --- slice tests ---

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

    #[test]
    fn slice_exact_bounds() {
        let mut store = TensorStore::new();
        let data = vec![42u8; 10];
        let handle = store.insert(data, vec![10], DType::Uint8, "cpu".into());
        let slice = store.slice(&handle.tensor_id, 0, 10).unwrap();
        assert_eq!(slice, vec![42u8; 10]);
    }

    // --- insert_with_id tests ---

    #[test]
    fn insert_with_id_accepts_precomputed_hash() {
        let mut store = TensorStore::new();
        let data = vec![1, 2, 3, 4];
        let tensor_id = "a".repeat(64);
        let handle =
            store.insert_with_id(tensor_id.clone(), data, vec![4], DType::Uint8, "cpu".into());
        assert_eq!(handle.tensor_id, tensor_id);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn insert_with_id_dedup_returns_existing() {
        let mut store = TensorStore::new();
        let tensor_id = "b".repeat(64);
        let h1 = store.insert_with_id(
            tensor_id.clone(),
            vec![1, 2, 3, 4],
            vec![4],
            DType::Uint8,
            "cpu".into(),
        );
        let h2 = store.insert_with_id(
            tensor_id,
            vec![1, 2, 3, 4],
            vec![4],
            DType::Uint8,
            "cpu".into(),
        );
        assert_eq!(h1.tensor_id, h2.tensor_id);
        assert_eq!(store.len(), 1);
    }

    // --- LRU eviction tests ---

    #[test]
    fn eviction_by_entry_count() {
        let mut store = TensorStore::with_limits(3, usize::MAX);
        let h1 = store.insert(vec![1u8], vec![1], DType::Uint8, "cpu".into());
        let _h2 = store.insert(vec![2u8], vec![1], DType::Uint8, "cpu".into());
        let _h3 = store.insert(vec![3u8], vec![1], DType::Uint8, "cpu".into());
        assert_eq!(store.len(), 3);

        let _h4 = store.insert(vec![4u8], vec![1], DType::Uint8, "cpu".into());
        assert_eq!(store.len(), 3);
        assert!(!store.contains(&h1.tensor_id));
    }

    #[test]
    fn eviction_by_byte_limit() {
        let mut store = TensorStore::with_limits(usize::MAX, 10);
        let h1 = store.insert(vec![0u8; 5], vec![5], DType::Uint8, "cpu".into());
        let h2 = store.insert(vec![1u8; 5], vec![5], DType::Uint8, "cpu".into());
        assert_eq!(store.len(), 2);
        assert_eq!(store.bytes_used(), 10);

        // Inserting 6 bytes when budget is 10 — must evict both h1 and h2
        // (5 + 6 = 11 still exceeds 10, so eviction continues until 0 + 6 <= 10).
        let h3 = store.insert(vec![2u8; 6], vec![6], DType::Uint8, "cpu".into());
        assert!(!store.contains(&h1.tensor_id));
        assert!(!store.contains(&h2.tensor_id));
        assert!(store.contains(&h3.tensor_id));
        assert_eq!(store.len(), 1);
        assert_eq!(store.bytes_used(), 6);
    }

    #[test]
    fn eviction_preserves_recently_accessed() {
        let mut store = TensorStore::with_limits(3, usize::MAX);
        let h1 = store.insert(vec![1u8], vec![1], DType::Uint8, "cpu".into());
        let h2 = store.insert(vec![2u8], vec![1], DType::Uint8, "cpu".into());
        let h3 = store.insert(vec![3u8], vec![1], DType::Uint8, "cpu".into());

        // Access h1 to move it to back of LRU
        store.get(&h1.tensor_id);

        // Insert h4 — should evict h2 (now oldest), not h1
        let h4 = store.insert(vec![4u8], vec![1], DType::Uint8, "cpu".into());
        assert_eq!(store.len(), 3);
        assert!(store.contains(&h1.tensor_id)); // recently accessed, preserved
        assert!(!store.contains(&h2.tensor_id)); // oldest, evicted
        assert!(store.contains(&h3.tensor_id)); // newer than h2, preserved
        assert!(store.contains(&h4.tensor_id)); // just inserted
    }
}
