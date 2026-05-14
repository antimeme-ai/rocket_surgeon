use serde::{Deserialize, Serialize};

use crate::errors::ErrorCode;
use crate::types::{
    BuiltInView, Capabilities, CheckpointRef, DType, GranularityScope, InterventionRecipe,
    ProbeAction, ProbeDefinition, SessionState, StepDirection, TensorSummary, TickGranularity,
    TickPosition,
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
}

pub mod event {
    pub const TICK_STOPPED: &str = "tick.stopped";
    pub const TICK_HEARTBEAT: &str = "tick.heartbeat";
    pub const PROBE_FIRED: &str = "probe.fired";
    pub const REPLAY_DIVERGENCE: &str = "replay.divergence";
    pub const ERROR: &str = "error";
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepResponse {
    pub ticks_executed: u32,
    pub stopped_at: TickPosition,
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
    pub checkpoint_id: Option<String>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
pub struct SubscriptionFilter {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe_ids: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubscribeRequest {
    pub events: Vec<EventType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<SubscriptionFilter>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubscribeResponse {
    pub subscription_id: String,
    pub subscribed_events: Vec<EventType>,
}

// ---------------------------------------------------------------------------
// Event notifications
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickStoppedEvent {
    pub position: TickPosition,
    pub state: SessionState,
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
