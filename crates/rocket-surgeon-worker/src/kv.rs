//! KV-cache read / intervene support (WU-G).
//!
//! ## Backend limitation
//!
//! The worker does not yet expose real attention key/value tensors from the
//! model adapter — `transformers` past-key-value caches are not threaded
//! through the capture hooks. Until that backend lands, [`kv_metric`] returns
//! a *deterministic stub* norm: a pure function of `(layer, position, head,
//! slot)`. The wire shape ([`KvCacheEntry`] norms per layer/position/head) is
//! the real protocol contract, so a client (human or LLM) sees a coherent,
//! reproducible response and the integrator can swap in real tensor reads
//! without touching the protocol surface.
//!
//! Eviction, by contrast, is *real* worker state: a `kv.intervene` op with
//! `op=evict` records the dropped positions in [`KvCacheState`], and a
//! subsequent `kv.read` of an evicted position is reported with the
//! [`KvOverlay::Evicted`] overlay so the daemon can raise `KV_EVICTED`.

use std::collections::HashMap;

use rocket_surgeon_protocol::messages::{
    HostKvInterveneRequest, HostKvInterveneResponse, HostKvReadRequest, HostKvReadResponse,
    KvCacheEntry, KvEvictionInfo, KvInterveneOp, KvMetric, KvOverlay, KvSlot,
};

/// Default layer count used when no model metadata bounds the request.
const DEFAULT_LAYERS: u32 = 4;
/// Default head count used when a request does not name specific heads.
const DEFAULT_HEADS: u32 = 4;
/// Default position count — the simulated prefill length.
const DEFAULT_POSITIONS: u64 = 8;

/// Per-position eviction bookkeeping for a single (simulated) KV cache.
#[derive(Debug, Default)]
pub struct KvCacheState {
    /// Map of evicted token position -> the tick at which it was evicted.
    evicted: HashMap<u64, u64>,
    /// Positions pinned (exempt from eviction) by a `kv.intervene` `pin` op.
    pinned: std::collections::HashSet<u64>,
}

impl KvCacheState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `position` as evicted at `tick`, unless it is pinned.
    /// Returns `true` if the position was newly evicted.
    pub fn evict(&mut self, position: u64, tick: u64) -> bool {
        if self.pinned.contains(&position) {
            return false;
        }
        self.evicted.insert(position, tick).is_none()
    }

    /// Pin `position` so it is exempt from future eviction. A pinned
    /// position that was already evicted is also un-evicted (re-admitted).
    pub fn pin(&mut self, position: u64) {
        self.pinned.insert(position);
        self.evicted.remove(&position);
    }

    /// Whether `position` is currently evicted.
    #[must_use]
    pub fn is_evicted(&self, position: u64) -> bool {
        self.evicted.contains_key(&position)
    }

    /// The tick at which `position` was evicted, if it is evicted.
    #[must_use]
    pub fn evicted_at(&self, position: u64) -> Option<u64> {
        self.evicted.get(&position).copied()
    }

    /// Clear all eviction/pin bookkeeping (called on detach / re-attach).
    pub fn reset(&mut self) {
        self.evicted.clear();
        self.pinned.clear();
    }
}

/// Deterministic stub norm for one cache slot.
///
/// See the module docs: this is *not* a real tensor norm. It is a stable,
/// reproducible function of the slot coordinates so tests and clients get a
/// coherent shape. The values are kept in a plausible range (roughly 0.1–9.9).
fn kv_metric(metric: KvMetric, layer: u32, position: u64, head: u32, is_key: bool) -> f64 {
    let salt = u64::from(layer)
        .wrapping_mul(31)
        .wrapping_add(position.wrapping_mul(17))
        .wrapping_add(u64::from(head).wrapping_mul(7))
        .wrapping_add(u64::from(is_key));
    let base = f64::from((salt % 99) as u32) / 10.0 + 0.1;
    match metric {
        KvMetric::L2Norm => base,
        KvMetric::Mean => base / 4.0,
        KvMetric::AbsMax => base * 1.5,
    }
}

/// Resolve the requested layer / position / head sets, falling back to the
/// simulated defaults when a request leaves a dimension unspecified.
fn resolve_dims(
    layers: Option<&Vec<u32>>,
    positions: Option<&Vec<u64>>,
    heads: Option<&Vec<u32>>,
) -> (Vec<u32>, Vec<u64>, Vec<u32>) {
    let layers = layers
        .cloned()
        .unwrap_or_else(|| (0..DEFAULT_LAYERS).collect());
    let positions = positions
        .cloned()
        .unwrap_or_else(|| (0..DEFAULT_POSITIONS).collect());
    let heads = heads
        .cloned()
        .unwrap_or_else(|| (0..DEFAULT_HEADS).collect());
    (layers, positions, heads)
}

/// Build a [`HostKvReadResponse`] for a `kv.read` request.
///
/// Produces one [`KvCacheEntry`] per (layer, position, head) tuple. Evicted
/// positions carry `k_metric`/`v_metric == None` and the
/// [`KvOverlay::Evicted`] overlay; live positions carry stub norms (see
/// module docs).
#[must_use]
pub fn read(req: &HostKvReadRequest, cache: &KvCacheState) -> HostKvReadResponse {
    let (layers, positions, heads) = resolve_dims(
        req.layers.as_ref(),
        req.positions.as_ref(),
        req.heads.as_ref(),
    );

    let want_k = matches!(req.slot, KvSlot::K | KvSlot::Both);
    let want_v = matches!(req.slot, KvSlot::V | KvSlot::Both);

    let mut entries = Vec::with_capacity(layers.len() * positions.len() * heads.len());
    let mut evicted_info = Vec::new();
    for &position in &positions {
        if let Some(tick) = cache.evicted_at(position) {
            evicted_info.push(KvEvictionInfo {
                position,
                evicted_at_tick: tick,
            });
        }
    }
    for &layer in &layers {
        for &position in &positions {
            let evicted = cache.is_evicted(position);
            for &head in &heads {
                let (k_metric, v_metric, overlay) = if evicted {
                    (None, None, Some(KvOverlay::Evicted))
                } else {
                    (
                        want_k.then(|| kv_metric(req.metric, layer, position, head, true)),
                        want_v.then(|| kv_metric(req.metric, layer, position, head, false)),
                        None,
                    )
                };
                entries.push(KvCacheEntry {
                    layer,
                    position,
                    head,
                    k_metric,
                    v_metric,
                    overlay,
                });
            }
        }
    }
    HostKvReadResponse {
        entries,
        evicted: evicted_info,
    }
}

/// Apply a `kv.intervene` op against the (simulated) KV cache.
///
/// `current_tick` is the worker's current tick id, recorded as the
/// `evicted_at` tick for `evict` ops. Returns the op tag plus the number of
/// `(layer, position, head)` slots the op nominally touched.
#[must_use]
pub fn intervene(
    req: &HostKvInterveneRequest,
    cache: &mut KvCacheState,
    current_tick: u64,
) -> HostKvInterveneResponse {
    let heads = req
        .heads
        .clone()
        .unwrap_or_else(|| (0..DEFAULT_HEADS).collect());
    let head_count = heads.len().max(1) as u64;
    let slot_factor = match req.slot {
        KvSlot::Both => 2,
        KvSlot::K | KvSlot::V => 1,
    };

    let (applied_op, touched_positions) = match &req.operation {
        KvInterveneOp::Evict => {
            let mut n = 0u64;
            for &pos in &req.positions {
                if cache.evict(pos, current_tick) {
                    n += 1;
                }
            }
            ("evict", n)
        }
        KvInterveneOp::Pin => {
            for &pos in &req.positions {
                cache.pin(pos);
            }
            ("pin", req.positions.len() as u64)
        }
        KvInterveneOp::Zero => ("zero", req.positions.len() as u64),
        KvInterveneOp::Scale { .. } => ("scale", req.positions.len() as u64),
    };

    let slots_modified =
        touched_positions * req.layers.len().max(1) as u64 * head_count * slot_factor;

    HostKvInterveneResponse {
        slots_modified,
        applied_op: applied_op.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_req(layers: Vec<u32>, positions: Vec<u64>) -> HostKvReadRequest {
        HostKvReadRequest {
            model_handle: 1,
            layers: Some(layers),
            positions: Some(positions),
            heads: Some(vec![0]),
            slot: KvSlot::Both,
            metric: KvMetric::L2Norm,
        }
    }

    #[test]
    fn read_returns_entry_per_layer_position_head() {
        let cache = KvCacheState::new();
        let resp = read(&read_req(vec![0, 1], vec![0, 1, 2]), &cache);
        // 2 layers * 3 positions * 1 head
        assert_eq!(resp.entries.len(), 6);
        for e in &resp.entries {
            assert!(e.k_metric.is_some());
            assert!(e.v_metric.is_some());
            assert!(e.overlay.is_none());
        }
    }

    #[test]
    fn read_norms_are_deterministic() {
        let cache = KvCacheState::new();
        let a = read(&read_req(vec![0], vec![3]), &cache);
        let b = read(&read_req(vec![0], vec![3]), &cache);
        assert_eq!(a.entries[0].k_metric, b.entries[0].k_metric);
    }

    #[test]
    fn read_slot_k_only_omits_v_metric() {
        let cache = KvCacheState::new();
        let mut req = read_req(vec![0], vec![0]);
        req.slot = KvSlot::K;
        let resp = read(&req, &cache);
        assert!(resp.entries[0].k_metric.is_some());
        assert!(resp.entries[0].v_metric.is_none());
    }

    #[test]
    fn read_defaults_dims_when_unspecified() {
        let cache = KvCacheState::new();
        let req = HostKvReadRequest {
            model_handle: 1,
            layers: None,
            positions: None,
            heads: None,
            slot: KvSlot::Both,
            metric: KvMetric::L2Norm,
        };
        let resp = read(&req, &cache);
        let expected =
            DEFAULT_LAYERS as usize * DEFAULT_POSITIONS as usize * DEFAULT_HEADS as usize;
        assert_eq!(resp.entries.len(), expected);
    }

    #[test]
    fn evicted_position_reads_with_overlay_and_no_norms() {
        let mut cache = KvCacheState::new();
        cache.evict(5, 42);
        let resp = read(&read_req(vec![0], vec![5]), &cache);
        assert_eq!(resp.entries[0].overlay, Some(KvOverlay::Evicted));
        assert!(resp.entries[0].k_metric.is_none());
        assert!(resp.entries[0].v_metric.is_none());
        // The eviction tick is surfaced for the daemon's KV_EVICTED context.
        assert_eq!(resp.evicted.len(), 1);
        assert_eq!(resp.evicted[0].position, 5);
        assert_eq!(resp.evicted[0].evicted_at_tick, 42);
    }

    #[test]
    fn live_positions_report_no_eviction() {
        let cache = KvCacheState::new();
        let resp = read(&read_req(vec![0], vec![0, 1, 2]), &cache);
        assert!(resp.evicted.is_empty());
    }

    #[test]
    fn intervene_evict_records_position() {
        let mut cache = KvCacheState::new();
        let req = HostKvInterveneRequest {
            model_handle: 1,
            layers: vec![0],
            positions: vec![5],
            heads: Some(vec![0]),
            slot: KvSlot::Both,
            operation: KvInterveneOp::Evict,
        };
        let resp = intervene(&req, &mut cache, 99);
        assert_eq!(resp.applied_op, "evict");
        assert!(cache.is_evicted(5));
        assert_eq!(cache.evicted_at(5), Some(99));
        // 1 position * 1 layer * 1 head * 2 (both slots)
        assert_eq!(resp.slots_modified, 2);
    }

    #[test]
    fn intervene_pin_protects_from_eviction() {
        let mut cache = KvCacheState::new();
        cache.pin(3);
        assert!(!cache.evict(3, 10));
        assert!(!cache.is_evicted(3));
    }

    #[test]
    fn intervene_pin_readmits_evicted_position() {
        let mut cache = KvCacheState::new();
        cache.evict(7, 1);
        assert!(cache.is_evicted(7));
        cache.pin(7);
        assert!(!cache.is_evicted(7));
    }

    #[test]
    fn intervene_scale_does_not_evict() {
        let mut cache = KvCacheState::new();
        let req = HostKvInterveneRequest {
            model_handle: 1,
            layers: vec![0, 1],
            positions: vec![0, 1],
            heads: None,
            slot: KvSlot::Both,
            operation: KvInterveneOp::Scale { factor: 0.5 },
        };
        let resp = intervene(&req, &mut cache, 5);
        assert_eq!(resp.applied_op, "scale");
        assert!(!cache.is_evicted(0));
    }

    #[test]
    fn cache_reset_clears_state() {
        let mut cache = KvCacheState::new();
        cache.evict(1, 1);
        cache.pin(2);
        cache.reset();
        assert!(!cache.is_evicted(1));
        assert!(cache.evict(2, 1));
    }
}
