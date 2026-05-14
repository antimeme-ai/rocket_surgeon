#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use rocket_surgeon_protocol::types::{DType, TensorHandle, TensorStats, TopKEntry};

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
        if self.entries.contains_key(&tensor_id) {
            self.touch_access_order(&tensor_id);
            let existing = self.entries.get_mut(&tensor_id).unwrap();
            existing.last_access = Instant::now();
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
}
