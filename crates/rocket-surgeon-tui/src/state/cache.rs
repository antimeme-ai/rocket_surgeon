use std::collections::{HashMap, VecDeque};

use rocket_surgeon_protocol::types::TensorSummary;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub tick_id: u64,
    pub probe_point: String,
}

pub struct TensorCache {
    entries: HashMap<CacheKey, TensorSummary>,
    order: VecDeque<CacheKey>,
    max_entries: usize,
}

impl TensorCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_entries: max_entries.max(1),
        }
    }

    pub fn get(&mut self, key: &CacheKey) -> Option<&TensorSummary> {
        if self.entries.contains_key(key) {
            self.promote(key);
            self.entries.get(key)
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: CacheKey, summary: TensorSummary) {
        if self.entries.contains_key(&key) {
            self.promote(&key);
            self.entries.insert(key, summary);
            return;
        }

        while self.entries.len() >= self.max_entries {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
            }
        }

        self.order.push_back(key.clone());
        self.entries.insert(key, summary);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn contains(&self, key: &CacheKey) -> bool {
        self.entries.contains_key(key)
    }

    fn promote(&mut self, key: &CacheKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
            self.order.push_back(key.clone());
        }
    }

    pub fn prefetch_keys(layer: u32, token: u64, component: &str, tick_id: u64) -> Vec<CacheKey> {
        let mut keys = Vec::new();

        if layer > 0 {
            keys.push(CacheKey {
                tick_id,
                probe_point: format!("{}:layer_{}", component, layer - 1),
            });
        }
        keys.push(CacheKey {
            tick_id,
            probe_point: format!("{}:layer_{}", component, layer + 1),
        });
        keys.push(CacheKey {
            tick_id,
            probe_point: format!("{}:token_{}", component, token + 1),
        });

        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocket_surgeon_protocol::types::{DType, Histogram, TensorStats};

    fn make_summary(_label: &str) -> TensorSummary {
        TensorSummary {
            tensor_id: String::new(),
            shape: vec![1, 768],
            dtype: DType::Float32,
            device: "cpu".into(),
            sharding: None,
            stats: TensorStats {
                mean: 0.0,
                std: 1.0,
                min: -3.0,
                max: 3.0,
                abs_max: 3.0,
                sparsity: 0.0,
                l2_norm: 27.7,
                histogram: Histogram {
                    bins: 0,
                    edges: Vec::new(),
                    counts: Vec::new(),
                },
            },
            top_k: Vec::new(),
        }
    }

    fn key(tick: u64, point: &str) -> CacheKey {
        CacheKey {
            tick_id: tick,
            probe_point: point.to_string(),
        }
    }

    #[test]
    fn insert_and_get() {
        let mut cache = TensorCache::new(10);
        cache.insert(key(0, "layer_0"), make_summary("a"));
        assert!(cache.get(&key(0, "layer_0")).is_some());
    }

    #[test]
    fn miss_returns_none() {
        let mut cache = TensorCache::new(10);
        assert!(cache.get(&key(0, "layer_0")).is_none());
    }

    #[test]
    fn evicts_lru_when_full() {
        let mut cache = TensorCache::new(2);
        cache.insert(key(0, "a"), make_summary("a"));
        cache.insert(key(0, "b"), make_summary("b"));
        cache.insert(key(0, "c"), make_summary("c"));

        assert!(!cache.contains(&key(0, "a")));
        assert!(cache.contains(&key(0, "b")));
        assert!(cache.contains(&key(0, "c")));
    }

    #[test]
    fn access_promotes_entry() {
        let mut cache = TensorCache::new(2);
        cache.insert(key(0, "a"), make_summary("a"));
        cache.insert(key(0, "b"), make_summary("b"));

        // Access "a" to promote it
        cache.get(&key(0, "a"));

        // Now insert "c" — should evict "b" (LRU), not "a"
        cache.insert(key(0, "c"), make_summary("c"));

        assert!(cache.contains(&key(0, "a")));
        assert!(!cache.contains(&key(0, "b")));
        assert!(cache.contains(&key(0, "c")));
    }

    #[test]
    fn update_existing_keeps_size() {
        let mut cache = TensorCache::new(2);
        cache.insert(key(0, "a"), make_summary("a"));
        cache.insert(key(0, "a"), make_summary("a2"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn new_with_zero_capacity_uses_minimum() {
        let mut cache = TensorCache::new(0);
        cache.insert(key(0, "a"), make_summary("a"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn prefetch_keys_adjacent() {
        let keys = TensorCache::prefetch_keys(5, 10, "attn.o_proj", 0);
        assert_eq!(keys.len(), 3);
        assert!(keys.iter().any(|k| k.probe_point.contains("layer_4")));
        assert!(keys.iter().any(|k| k.probe_point.contains("layer_6")));
        assert!(keys.iter().any(|k| k.probe_point.contains("token_11")));
    }

    #[test]
    fn prefetch_at_layer_zero_skips_negative() {
        let keys = TensorCache::prefetch_keys(0, 0, "attn.o_proj", 0);
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn prefetch_keys_uses_provided_tick_id() {
        let keys = TensorCache::prefetch_keys(5, 10, "attn.o_proj", 42);
        assert!(keys.iter().all(|k| k.tick_id == 42));
    }
}
