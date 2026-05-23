use serde::{Deserialize, Serialize};

use crate::errors::ErrorCode;
use crate::types::{
    AliasEntry, BuiltInView, Capabilities, CheckpointRef, CheckpointTier, ComponentEntry, DType,
    EnvelopeMode, GranularityScope, InterventionRecipe, ProbeAction, ProbeDefinition, Status,
    StepDirection, TensorSummary, TickGranularity, TickMapEntry, TickPosition,
};

// ---------------------------------------------------------------------------
// Method names
// ---------------------------------------------------------------------------

pub mod method {
    pub const INITIALIZE: &str = "initialize";
    pub const ATTACH: &str = "attach";
    pub const DETACH: &str = "detach";
    pub const STEP: &str = "rocket/step";
    pub const INSPECT: &str = "rocket/inspect";
    pub const INTERVENE: &str = "rocket/intervene";
    pub const PROBE: &str = "rocket/probe";
    pub const CHECKPOINT: &str = "rocket/checkpoint";
    pub const REPLAY: &str = "rocket/replay";
    pub const STATUS: &str = "rocket/status";
    pub const SUBSCRIBE: &str = "rocket/subscribe";
    pub const UNSUBSCRIBE: &str = "rocket/unsubscribe";
    pub const VIEW: &str = "rocket/view";
    pub const KV_READ: &str = "rocket/kv.read";
    pub const KV_INTERVENE: &str = "rocket/kv.intervene";
    pub const BRANCH_FORK: &str = "rocket/branch.fork";
    pub const BRANCH_DROP: &str = "rocket/branch.drop";
    pub const BRANCH_COMPARE: &str = "rocket/branch.compare";
    pub const DISCOVER: &str = "rocket/discover";
    pub const SWEEP: &str = "rocket/sweep";
    pub const VIEW_FOCUS: &str = "rocket/view.focus";
    pub const VIEW_DEFINE: &str = "rocket/view.define";
    pub const SESSION_EXPORT: &str = "rocket/session.export";
}

pub mod event {
    pub const TICK_STOPPED: &str = "tick.stopped";
    pub const TICK_HEARTBEAT: &str = "tick.heartbeat";
    pub const PROBE_FIRED: &str = "probe.fired";
    pub const REPLAY_DIVERGENCE: &str = "replay.divergence";
    pub const ERROR: &str = "error";
    pub const KV_UPDATE: &str = "kv.update";
    pub const KV_EVICTED: &str = "kv.evicted";
    pub const BRANCH_CREATED: &str = "branch.created";
    pub const BRANCH_TIER_CHANGED: &str = "branch.tier_changed";
    pub const SPEC_STEP: &str = "spec.step";
    pub const SWEEP_TRIAL_COMPLETE: &str = "sweep.trial_complete";
}

pub mod internal {
    pub const HOST_ATTACH: &str = "_host/attach";
    pub const HOST_DETACH: &str = "_host/detach";
    pub const HOST_CONFIGURE_HOOKS: &str = "_host/configure_hooks";
    pub const HOST_STEP: &str = "_host/step";
    pub const HOST_UPDATE_PROBES: &str = "_host/update_probes";
    pub const HOST_INSPECT: &str = "_host/inspect";
    pub const HOST_VIEW: &str = "_host/view";
    pub const HOST_CHECKPOINT: &str = "_host/checkpoint";
    pub const HOST_KV_READ: &str = "_host/kv.read";
    pub const HOST_KV_INTERVENE: &str = "_host/kv.intervene";
    pub const HOST_EXPORT_ENV: &str = "_host/export_env";
}

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitializeRequest {
    pub client_name: String,
    pub protocol_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_capabilities: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitializeResponse {
    pub capabilities: Capabilities,
}

// ---------------------------------------------------------------------------
// attach
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttachRequest {
    pub model_path: String,
    pub model_family: String,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<DType>,
    #[serde(default = "default_one_u32")]
    pub num_ranks: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<serde_json::Value>,
}

fn default_device() -> String {
    "cuda:0".to_owned()
}

fn default_one_u32() -> u32 {
    1
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttachResponse {
    pub model_id: String,
    pub model_family: String,
    pub num_layers: u32,
    pub num_heads: u32,
    pub hidden_dim: u32,
    pub num_ranks: u32,
    pub capabilities: Capabilities,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub component_vocabulary: Vec<ComponentEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub module_tree: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alias_table: Vec<AliasEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tick_map: Vec<TickMapEntry>,
}

// ---------------------------------------------------------------------------
// detach
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetachRequest {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetachResponse {
    pub detached_model_id: String,
}

// ---------------------------------------------------------------------------
// rocket/step
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepRequest {
    pub direction: StepDirection,
    pub count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granularity: Option<TickGranularity>,
    #[serde(default)]
    pub envelope: EnvelopeMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_to: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepResponse {
    pub ticks_executed: u32,
    pub stopped_at: TickPosition,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fired_interventions: Vec<String>,
}

// ---------------------------------------------------------------------------
// rocket/inspect
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectDetail {
    #[default]
    Summary,
    Slice,
    Full,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InspectRequest {
    pub target: String,
    #[serde(default)]
    pub detail: InspectDetail,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slices: Option<Vec<[u64; 2]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<DType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub view: Option<BuiltInView>,
    #[serde(default)]
    pub envelope: EnvelopeMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InspectResponse {
    pub tensors: Vec<TensorSummary>,
    pub view_result: Option<serde_json::Value>,
    pub slice_data: Option<String>,
}

// ---------------------------------------------------------------------------
// rocket/intervene
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum InterveneRequest {
    Set { recipe: InterventionRecipe },
    Clear { intervention_id: String },
    List {},
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterveneResponse {
    pub active_interventions: Vec<InterventionRecipe>,
    pub applied: Option<bool>,
}

// ---------------------------------------------------------------------------
// rocket/probe
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ProbeRequest {
    Define { probe: Box<ProbeDefinition> },
    List {},
    Enable { probe_id: String },
    Disable { probe_id: String },
    Remove { probe_id: String },
    SetGranularity { scopes: Vec<GranularityScope> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeResponse {
    pub probes: Vec<ProbeDefinition>,
    pub probe_id: Option<String>,
}

// ---------------------------------------------------------------------------
// rocket/checkpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreateCheckpointTier {
    Activation,
    FullSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum CheckpointRequest {
    Create {
        #[serde(skip_serializing_if = "Option::is_none")]
        tier: Option<CreateCheckpointTier>,
    },
    Restore {
        checkpoint_id: String,
    },
    List {},
    Delete {
        checkpoint_id: String,
    },
    Bookmark {
        tick_id: u64,
        name: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointResponse {
    pub checkpoints: Vec<CheckpointRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restored_to: Option<TickPosition>,
}

// ---------------------------------------------------------------------------
// rocket/replay
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayStopAt {
    pub layer: u32,
    pub component: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayRequest {
    pub from_checkpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interventions: Option<Vec<InterventionRecipe>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_at: Option<ReplayStopAt>,
    #[serde(default = "crate::types::default_true")]
    pub verify: bool,
    #[serde(default)]
    pub envelope: EnvelopeMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Divergence {
    pub tick_id: u64,
    pub original_tick_id: u64,
    pub probe_point: String,
    pub cosine_similarity: f64,
    pub max_relative_error: f64,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayResponse {
    pub ticks_replayed: u32,
    pub stopped_at: TickPosition,
    pub divergences: Vec<Divergence>,
    pub verified: bool,
}

// ---------------------------------------------------------------------------
// rocket/status
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusRequest {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub gpu_mb: f64,
    pub cpu_mb: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusResponse {
    pub uptime_seconds: f64,
    pub connected_clients: u32,
    pub memory_usage: MemoryUsage,
    pub pending_interventions: u32,
    pub trace_events_recorded: u64,
}

// ---------------------------------------------------------------------------
// rocket/subscribe
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventType {
    #[serde(rename = "tick.stopped")]
    TickStopped,
    #[serde(rename = "tick.heartbeat")]
    TickHeartbeat,
    #[serde(rename = "probe.fired")]
    ProbeFired,
    #[serde(rename = "replay.divergence")]
    ReplayDivergence,
    #[serde(rename = "error")]
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubscribeFilter {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub events: Option<Vec<EventType>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub components: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubscribeRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<SubscribeFilter>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubscribeResponse {
    pub available_events: Vec<EventType>,
    pub status: Status,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnsubscribeRequest {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnsubscribeResponse {
    pub status: Status,
}

// ---------------------------------------------------------------------------
// rocket/view
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewRequest {
    pub view: BuiltInView,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    #[serde(default)]
    pub envelope: EnvelopeMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewResponse {
    pub view: BuiltInView,
    pub data: serde_json::Value,
}

// ---------------------------------------------------------------------------
// rocket/kv.read
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvSlot {
    K,
    V,
    #[default]
    Both,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvMetric {
    #[default]
    L2Norm,
    Mean,
    AbsMax,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvReadRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positions: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heads: Option<Vec<u32>>,
    #[serde(default)]
    pub slot: KvSlot,
    #[serde(default)]
    pub metric: KvMetric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvOverlay {
    Sink,
    HeavyHitter,
    Evicted,
    Quantized,
    PageBoundary,
    SharedPrefix,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCacheEntry {
    pub layer: u32,
    pub position: u64,
    pub head: u32,
    pub k_metric: Option<f64>,
    pub v_metric: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlay: Option<KvOverlay>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvReadResponse {
    pub entries: Vec<KvCacheEntry>,
}

// ---------------------------------------------------------------------------
// rocket/kv.intervene
// ---------------------------------------------------------------------------

/// Surgical operation applied to a KV-cache slot.
///
/// `kv.intervene` lets a client mutate the key/value cache between ticks —
/// the cache-side analogue of `rocket/intervene` on activations. Each variant
/// names a distinct mutation; the daemon validates the target slot exists and
/// is not evicted before forwarding the op to the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum KvInterveneOp {
    /// Zero the selected key/value slot(s).
    Zero,
    /// Scale the selected slot(s) by a multiplicative factor.
    Scale { factor: f64 },
    /// Drop (evict) the selected position(s) from the cache.
    Evict,
    /// Pin the selected position(s) so they are exempt from eviction.
    Pin,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvInterveneRequest {
    pub layers: Vec<u32>,
    pub positions: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heads: Option<Vec<u32>>,
    #[serde(default)]
    pub slot: KvSlot,
    #[serde(flatten)]
    pub operation: KvInterveneOp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvInterveneResponse {
    /// Number of (layer, position, head) cache slots the op touched.
    pub slots_modified: u64,
    /// Echo of the applied operation tag (`zero`, `scale`, `evict`, `pin`).
    pub applied_op: String,
}

// ---------------------------------------------------------------------------
// rocket/branch.*
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BranchTier {
    Live,
    Spilled,
    Dropped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchForkRequest {
    pub from_checkpoint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchForkResponse {
    pub branch_id: String,
    pub tier: BranchTier,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchDropRequest {
    pub branch_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchDropResponse {
    pub branch_id: String,
    pub freed_mb: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchCompareRequest {
    pub branch_a: String,
    pub branch_b: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchCompareResponse {
    pub cosine_similarity: f64,
    pub max_relative_error: f64,
    pub kl_divergence: f64,
    pub per_layer_norm_delta: Vec<f64>,
}

// ---------------------------------------------------------------------------
// rocket/discover
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoverRequest {
    pub pattern: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoverMatch {
    pub canonical: String,
    pub tensor_shape: Vec<u64>,
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoverResponse {
    pub matches: Vec<DiscoverMatch>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
}

// ---------------------------------------------------------------------------
// rocket/view.focus
// ---------------------------------------------------------------------------

// The `By` prefix is intentional and part of the frozen v0.3.0 wire schema
// (serde renames to snake_case `by_id` / `by_position` / ...). Renaming the
// variants to satisfy the lint would change the protocol.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FocusSelector {
    ById { token_id: u64 },
    ByPosition { position: u64 },
    ByRegex { pattern: String },
    ByAnchor { anchor: FocusAnchor },
    ByRange { start: u64, end: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FocusAnchor {
    Bos,
    Eos,
    PadBoundary,
    Sink,
    MaxAttention,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewFocusRequest {
    pub selector: FocusSelector,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewFocusResponse {
    pub position: u64,
    pub token: serde_json::Value,
    pub per_layer_summaries: Vec<TensorSummary>,
}

// ---------------------------------------------------------------------------
// rocket/sweep
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepTrial {
    pub interventions: Vec<InterventionRecipe>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_to: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collect: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SweepMetric {
    #[serde(rename = "type")]
    pub metric_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepRequest {
    pub baseline_checkpoint: String,
    pub trials: Vec<SweepTrial>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<SweepMetric>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepTrialResult {
    pub trial_index: u32,
    pub stopped_at: TickPosition,
    pub collected: Vec<TensorSummary>,
    pub metric_value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepResponse {
    pub results: Vec<SweepTrialResult>,
}

// ---------------------------------------------------------------------------
// rocket/view.define
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewDefineRequest {
    pub name: String,
    pub spec: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewDefineResponse {
    pub name: String,
    pub registered: bool,
}

// ---------------------------------------------------------------------------
// Event notifications
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickStoppedEvent {
    pub position: TickPosition,
    pub state: Status,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankActivity {
    Idle,
    Stopped,
    Waiting,
    Processing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankStatus {
    pub rank: u32,
    pub status: RankActivity,
    pub gpu_memory_used_mb: f64,
    pub gpu_memory_total_mb: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickHeartbeatEvent {
    pub position: TickPosition,
    pub uptime_seconds: f64,
    pub elapsed_stopped_sec: f64,
    pub per_rank_status: Vec<RankStatus>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeFiredEvent {
    pub probe_id: String,
    pub point: String,
    pub tick_id: u64,
    pub tensor_summary: Option<TensorSummary>,
    pub action: ProbeAction,
    pub timestamp: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayDivergenceEvent {
    pub tick_id: u64,
    pub original_tick_id: u64,
    pub probe_point: String,
    pub cosine_similarity: f64,
    pub max_relative_error: f64,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorEvent {
    pub error_code: ErrorCode,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub fatal: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvUpdateEvent {
    pub layer: u32,
    pub new_positions: Vec<u64>,
    pub total_positions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvEvictedEvent {
    pub layer: u32,
    pub evicted_positions: Vec<u64>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchCreatedEvent {
    pub branch_id: String,
    pub from_checkpoint: String,
    pub tier: BranchTier,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchTierChangedEvent {
    pub branch_id: String,
    pub old_tier: BranchTier,
    pub new_tier: BranchTier,
}

// ---------------------------------------------------------------------------
// _host/attach (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostAttachRequest {
    pub model_source: String,
    pub model_family: String,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<DType>,
    pub rank: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostAttachResponse {
    pub model_handle: u64,
    pub num_layers: u32,
    pub num_heads: u32,
    pub hidden_dim: u32,
    pub module_tree: Vec<String>,
    pub model_type: String,
    pub component_vocabulary: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shm_name: Option<String>,
}

// ---------------------------------------------------------------------------
// _host/detach (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostDetachRequest {
    pub model_handle: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostDetachResponse {
    pub released: bool,
}

// ---------------------------------------------------------------------------
// _host/configure_hooks (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostConfigureHooksRequest {
    pub model_handle: u64,
    pub active_probes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostConfigureHooksResponse {
    pub sentinel_count: u32,
    pub capture_count: u32,
}

// ---------------------------------------------------------------------------
// _host/step (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostStepRequest {
    pub model_handle: u64,
    pub count: u32,
    #[serde(default)]
    pub direction: StepDirection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granularity: Option<TickGranularity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_events: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interventions: Vec<InterventionRecipe>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostStepResponse {
    pub position: TickPosition,
    #[serde(default)]
    pub events: Vec<ProbeFiredEvent>,
    pub forward_complete: bool,
    #[serde(default)]
    pub events_truncated: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fired_interventions: Vec<String>,
}

// ---------------------------------------------------------------------------
// _host/update_probes (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostUpdateProbesRequest {
    pub model_handle: u64,
    pub active_probes: Vec<ProbeDefinition>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostUpdateProbesResponse {
    pub probes_active: u32,
}

// ---------------------------------------------------------------------------
// _host/inspect (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostInspectRequest {
    pub model_handle: u64,
    pub target: String,
    #[serde(default)]
    pub detail: InspectDetail,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slices: Option<Vec<[u64; 2]>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostInspectResponse {
    pub tensors: Vec<CapturedTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapturedTensor {
    pub module_path: String,
    pub canonical: String,
    pub layer: u32,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub device: String,
    pub tensor_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shm_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shm_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_base64: Option<String>,
}

// ---------------------------------------------------------------------------
// _host/view (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostViewRequest {
    pub model_handle: u64,
    pub view: BuiltInView,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostViewResponse {
    pub view: BuiltInView,
    pub data: serde_json::Value,
}

// ---------------------------------------------------------------------------
// _host/checkpoint (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

/// Only the two state-affecting checkpoint actions reach the worker.
///
/// `list`, `delete`, and `bookmark` are pure daemon bookkeeping and never
/// round-trip. The daemon mints `checkpoint_id` so the worker can key its
/// snapshot store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum HostCheckpointRequest {
    Create {
        model_handle: u64,
        checkpoint_id: String,
        tier: CreateCheckpointTier,
        tick_id: u64,
        layer_idx: u32,
    },
    Restore {
        model_handle: u64,
        checkpoint_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostCheckpointResponse {
    pub checkpoint_id: String,
    pub tier: CheckpointTier,
    /// Populated on `Restore` with the worker's re-seated tick position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restored_to: Option<TickPosition>,
    /// Resident snapshot size, for VRAM accounting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_captured: Option<u64>,
}

// ---------------------------------------------------------------------------
// _host/kv.read (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostKvReadRequest {
    pub model_handle: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positions: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heads: Option<Vec<u32>>,
    #[serde(default)]
    pub slot: KvSlot,
    #[serde(default)]
    pub metric: KvMetric,
}

/// One evicted (position, tick) pair from a `_host/kv.read` response.
///
/// The worker reports these alongside the read entries; the daemon uses
/// `evicted_at_tick` to populate the `KV_EVICTED` error context when a client
/// reads an evicted position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvEvictionInfo {
    pub position: u64,
    pub evicted_at_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostKvReadResponse {
    pub entries: Vec<KvCacheEntry>,
    /// Eviction metadata for any requested position that is evicted. Empty
    /// when every requested position is live.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evicted: Vec<KvEvictionInfo>,
}

// ---------------------------------------------------------------------------
// _host/kv.intervene (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostKvInterveneRequest {
    pub model_handle: u64,
    pub layers: Vec<u32>,
    pub positions: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heads: Option<Vec<u32>>,
    #[serde(default)]
    pub slot: KvSlot,
    #[serde(flatten)]
    pub operation: KvInterveneOp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostKvInterveneResponse {
    pub slots_modified: u64,
    pub applied_op: String,
}

// ---------------------------------------------------------------------------
// rocket/session.export
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportRequest {
    pub path: String,
    #[serde(default = "crate::types::default_true")]
    pub include_tensors: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportResponse {
    pub path: String,
    pub size_bytes: u64,
    pub artifact_count: u32,
}

// ---------------------------------------------------------------------------
// _host/export_env (internal: daemon → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostExportEnvRequest {
    pub model_handle: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostExportEnvResponse {
    pub env: serde_json::Value,
    pub model_info: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Phase, TickEvent};

    #[test]
    fn host_attach_request_round_trip() {
        let req = HostAttachRequest {
            model_source: "hf-internal-testing/tiny-random-LlamaForCausalLM".to_owned(),
            model_family: "llama".to_owned(),
            device: "cuda:0".to_owned(),
            dtype: None,
            rank: 0,
            config: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HostAttachRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn host_attach_response_round_trip() {
        let resp = HostAttachResponse {
            model_handle: 1,
            num_layers: 32,
            num_heads: 32,
            hidden_dim: 4096,
            module_tree: vec!["model.embed_tokens".to_owned(), "model.layers.0".to_owned()],
            model_type: "llama".to_owned(),
            component_vocabulary: vec!["q_proj".to_owned()],
            shm_name: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HostAttachResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, parsed);
    }

    #[test]
    fn host_detach_request_round_trip() {
        let req = HostDetachRequest { model_handle: 42 };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HostDetachRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn host_detach_response_round_trip() {
        let resp = HostDetachResponse { released: true };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HostDetachResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, parsed);
    }

    #[test]
    fn host_attach_request_default_device() {
        let json = r#"{"model_source":"test","model_family":"llama","rank":0}"#;
        let req: HostAttachRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.device, "cuda:0");
    }

    #[test]
    fn internal_method_constants() {
        assert_eq!(internal::HOST_ATTACH, "_host/attach");
        assert_eq!(internal::HOST_DETACH, "_host/detach");
    }

    #[test]
    fn host_configure_hooks_request_round_trip() {
        let req = HostConfigureHooksRequest {
            model_handle: 1,
            active_probes: vec!["model:0:*:*:0:fwd".to_owned()],
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HostConfigureHooksRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn host_configure_hooks_response_round_trip() {
        let resp = HostConfigureHooksResponse {
            sentinel_count: 50,
            capture_count: 12,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HostConfigureHooksResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, parsed);
    }

    #[test]
    fn host_step_request_round_trip() {
        let req = HostStepRequest {
            model_handle: 1,
            count: 1,
            direction: StepDirection::Forward,
            granularity: None,
            max_events: None,
            interventions: vec![],
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HostStepRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn host_step_response_round_trip() {
        let resp = HostStepResponse {
            position: TickPosition {
                tick_id: 42,
                direction: StepDirection::Forward,
                rank: Some(0),
                layer: 3,
                component: "q_proj".to_owned(),
                event: TickEvent::Output,
                replay_of: None,
                phase: Phase::Decode,
                token_position: None,
                clock: None,
            },
            events: vec![],
            forward_complete: false,
            events_truncated: false,
            fired_interventions: vec![],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HostStepResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.position.tick_id, parsed.position.tick_id);
        assert_eq!(resp.forward_complete, parsed.forward_complete);
        assert!(parsed.events.is_empty());
        assert!(!parsed.events_truncated);
    }

    #[test]
    fn host_update_probes_round_trip() {
        let req = HostUpdateProbesRequest {
            model_handle: 1,
            active_probes: vec![ProbeDefinition {
                id: "p1".to_owned(),
                point: "model:0:*:*:0:fwd".to_owned(),
                action: ProbeAction::Capture,
                config: None,
                enabled: true,
                priority: 0,
            }],
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HostUpdateProbesRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn host_attach_response_includes_component_vocabulary() {
        let resp = HostAttachResponse {
            model_handle: 1,
            num_layers: 4,
            num_heads: 4,
            hidden_dim: 32,
            module_tree: vec!["model.layers.0".to_owned()],
            model_type: "llama".to_owned(),
            component_vocabulary: vec!["q_proj".to_owned(), "k_proj".to_owned()],
            shm_name: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HostAttachResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_type, "llama");
        assert_eq!(parsed.component_vocabulary.len(), 2);
    }

    #[test]
    fn internal_configure_hooks_constant() {
        assert_eq!(internal::HOST_CONFIGURE_HOOKS, "_host/configure_hooks");
    }

    #[test]
    fn internal_step_constant() {
        assert_eq!(internal::HOST_STEP, "_host/step");
    }

    #[test]
    fn internal_update_probes_constant() {
        assert_eq!(internal::HOST_UPDATE_PROBES, "_host/update_probes");
    }

    #[test]
    fn internal_checkpoint_constant() {
        assert_eq!(internal::HOST_CHECKPOINT, "_host/checkpoint");
    }

    #[test]
    fn host_checkpoint_request_create_round_trip() {
        let req = HostCheckpointRequest::Create {
            model_handle: 1,
            checkpoint_id: "ckpt-1".to_owned(),
            tier: CreateCheckpointTier::Activation,
            tick_id: 5,
            layer_idx: 3,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HostCheckpointRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
        assert!(json.contains("\"action\":\"create\""));
    }

    #[test]
    fn host_checkpoint_request_restore_round_trip() {
        let req = HostCheckpointRequest::Restore {
            model_handle: 1,
            checkpoint_id: "ckpt-1".to_owned(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HostCheckpointRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
        assert!(json.contains("\"action\":\"restore\""));
    }

    #[test]
    fn host_checkpoint_response_round_trip() {
        let resp = HostCheckpointResponse {
            checkpoint_id: "ckpt-1".to_owned(),
            tier: CheckpointTier::FullSnapshot,
            restored_to: None,
            bytes_captured: Some(4096),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HostCheckpointResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, parsed);
    }

    #[test]
    fn host_step_request_with_granularity_round_trip() {
        let req = HostStepRequest {
            model_handle: 1,
            count: 3,
            direction: StepDirection::Forward,
            granularity: Some(TickGranularity::Layer),
            max_events: None,
            interventions: vec![],
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HostStepRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
        assert_eq!(parsed.granularity, Some(TickGranularity::Layer));
    }

    #[test]
    fn host_step_request_granularity_defaults_to_none() {
        let json = r#"{"model_handle":1,"count":1}"#;
        let req: HostStepRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.granularity, None);
    }

    #[test]
    fn internal_inspect_constant() {
        assert_eq!(internal::HOST_INSPECT, "_host/inspect");
    }

    #[test]
    fn host_inspect_request_round_trip() {
        let req = HostInspectRequest {
            model_handle: 1,
            target: "model:0:0:*:output".to_owned(),
            detail: InspectDetail::Summary,
            slices: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HostInspectRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn host_inspect_request_with_slices_round_trip() {
        let req = HostInspectRequest {
            model_handle: 1,
            target: "model:0:0:q_proj:output".to_owned(),
            detail: InspectDetail::Slice,
            slices: Some(vec![[0, 10], [20, 30]]),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HostInspectRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
        assert_eq!(parsed.slices.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn host_inspect_response_round_trip() {
        let resp = HostInspectResponse {
            tensors: vec![CapturedTensor {
                module_path: "model.layers.0.self_attn.q_proj".to_owned(),
                canonical: "q_proj".to_owned(),
                layer: 0,
                shape: vec![1, 4, 32],
                dtype: "float32".to_owned(),
                device: "cpu".to_owned(),
                tensor_id: "a".repeat(64),
                shm_name: None,
                shm_offset: None,
                byte_length: None,
                data_base64: Some("AAAA".to_owned()),
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HostInspectResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.tensors.len(), 1);
        assert_eq!(parsed.tensors[0].canonical, "q_proj");
    }

    #[test]
    fn host_inspect_response_empty_tensors_round_trip() {
        let resp = HostInspectResponse { tensors: vec![] };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HostInspectResponse = serde_json::from_str(&json).unwrap();
        assert!(parsed.tensors.is_empty());
    }

    #[test]
    fn captured_tensor_round_trip() {
        let ct = CapturedTensor {
            module_path: "model.layers.3.mlp.gate_proj".to_owned(),
            canonical: "gate_proj".to_owned(),
            layer: 3,
            shape: vec![1, 768],
            dtype: "float16".to_owned(),
            device: "cuda:0".to_owned(),
            tensor_id: "b".repeat(64),
            shm_name: None,
            shm_offset: None,
            byte_length: None,
            data_base64: Some("dGVzdA==".to_owned()),
        };
        let json = serde_json::to_string(&ct).unwrap();
        let parsed: CapturedTensor = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.module_path, ct.module_path);
        assert_eq!(parsed.layer, 3);
        assert_eq!(parsed.shape, vec![1, 768]);
    }

    #[test]
    fn subscribe_request_empty_round_trip() {
        let req = SubscribeRequest { filter: None };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, "{}");
        let parsed: SubscribeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn subscribe_request_backward_compat_no_filter() {
        let json = r"{}";
        let parsed: SubscribeRequest = serde_json::from_str(json).unwrap();
        assert!(parsed.filter.is_none());
    }

    #[test]
    fn subscribe_response_round_trip() {
        let resp = SubscribeResponse {
            available_events: vec![
                EventType::TickStopped,
                EventType::TickHeartbeat,
                EventType::ProbeFired,
            ],
            status: Status::Stopped,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: SubscribeResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, parsed);
        assert!(json.contains("\"available_events\""));
        assert!(json.contains("\"stopped\""));
    }

    #[test]
    fn unsubscribe_request_empty_round_trip() {
        let req = UnsubscribeRequest {};
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, "{}");
        let parsed: UnsubscribeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn unsubscribe_response_round_trip() {
        let resp = UnsubscribeResponse {
            status: Status::Stopped,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: UnsubscribeResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, parsed);
    }

    #[test]
    fn unsubscribe_method_constant() {
        assert_eq!(method::UNSUBSCRIBE, "rocket/unsubscribe");
    }

    // --- WU-G: KV-cache protocol surface -------------------------------

    #[test]
    fn internal_kv_constants() {
        assert_eq!(internal::HOST_KV_READ, "_host/kv.read");
        assert_eq!(internal::HOST_KV_INTERVENE, "_host/kv.intervene");
    }

    #[test]
    fn kv_method_constants() {
        assert_eq!(method::KV_READ, "rocket/kv.read");
        assert_eq!(method::KV_INTERVENE, "rocket/kv.intervene");
    }

    #[test]
    fn kv_intervene_request_zero_round_trip() {
        let req = KvInterveneRequest {
            layers: vec![0, 1],
            positions: vec![3, 4],
            heads: None,
            slot: KvSlot::Both,
            operation: KvInterveneOp::Zero,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"op\":\"zero\""));
        let parsed: KvInterveneRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn kv_intervene_request_scale_round_trip() {
        let req = KvInterveneRequest {
            layers: vec![2],
            positions: vec![0],
            heads: Some(vec![1, 2]),
            slot: KvSlot::K,
            operation: KvInterveneOp::Scale { factor: 0.5 },
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"op\":\"scale\""));
        assert!(json.contains("\"factor\":0.5"));
        let parsed: KvInterveneRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn kv_intervene_response_round_trip() {
        let resp = KvInterveneResponse {
            slots_modified: 12,
            applied_op: "evict".to_owned(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: KvInterveneResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, parsed);
    }

    #[test]
    fn host_kv_read_request_round_trip() {
        let req = HostKvReadRequest {
            model_handle: 7,
            layers: Some(vec![0, 1]),
            positions: Some(vec![0, 1, 2]),
            heads: None,
            slot: KvSlot::Both,
            metric: KvMetric::L2Norm,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HostKvReadRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn host_kv_read_request_defaults() {
        let json = r#"{"model_handle":1}"#;
        let req: HostKvReadRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.slot, KvSlot::Both);
        assert_eq!(req.metric, KvMetric::L2Norm);
        assert!(req.layers.is_none());
    }

    #[test]
    fn host_kv_read_response_round_trip() {
        let resp = HostKvReadResponse {
            entries: vec![KvCacheEntry {
                layer: 0,
                position: 1,
                head: 0,
                k_metric: Some(1.5),
                v_metric: Some(2.5),
                overlay: None,
            }],
            evicted: vec![],
        };
        let json = serde_json::to_string(&resp).unwrap();
        // An empty `evicted` list is omitted from the wire form.
        assert!(!json.contains("evicted"));
        let parsed: HostKvReadResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.entries.len(), parsed.entries.len());
        assert_eq!(parsed.entries[0].k_metric, Some(1.5));
    }

    #[test]
    fn host_kv_read_response_with_eviction_round_trip() {
        let resp = HostKvReadResponse {
            entries: vec![],
            evicted: vec![KvEvictionInfo {
                position: 5,
                evicted_at_tick: 42,
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HostKvReadResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.evicted.len(), 1);
        assert_eq!(parsed.evicted[0].evicted_at_tick, 42);
    }

    #[test]
    fn host_kv_intervene_request_round_trip() {
        let req = HostKvInterveneRequest {
            model_handle: 3,
            layers: vec![0],
            positions: vec![5],
            heads: None,
            slot: KvSlot::Both,
            operation: KvInterveneOp::Pin,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"op\":\"pin\""));
        let parsed: HostKvInterveneRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn host_kv_intervene_response_round_trip() {
        let resp = HostKvInterveneResponse {
            slots_modified: 4,
            applied_op: "zero".to_owned(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HostKvInterveneResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, parsed);
    }
}
