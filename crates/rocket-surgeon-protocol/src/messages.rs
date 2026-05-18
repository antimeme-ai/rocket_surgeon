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

pub mod internal {
    pub const HOST_ATTACH: &str = "_host/attach";
    pub const HOST_DETACH: &str = "_host/detach";
    pub const HOST_CONFIGURE_HOOKS: &str = "_host/configure_hooks";
    pub const HOST_STEP: &str = "_host/step";
    pub const HOST_UPDATE_PROBES: &str = "_host/update_probes";
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostStepRequest {
    pub model_handle: u64,
    pub count: u32,
    #[serde(default)]
    pub direction: StepDirection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granularity: Option<TickGranularity>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostStepResponse {
    pub position: TickPosition,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture: Option<TensorSummary>,
    pub forward_complete: bool,
}

// ---------------------------------------------------------------------------
// _host/update_probes (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostUpdateProbesRequest {
    pub model_handle: u64,
    pub active_probes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostUpdateProbesResponse {
    pub probes_active: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TickEvent;

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
            },
            capture: None,
            forward_complete: false,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HostStepResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.position.tick_id, parsed.position.tick_id);
        assert_eq!(resp.forward_complete, parsed.forward_complete);
    }

    #[test]
    fn host_update_probes_round_trip() {
        let req = HostUpdateProbesRequest {
            model_handle: 1,
            active_probes: vec!["model:0:3:q_proj:0:fwd".to_owned()],
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
    fn host_step_request_with_granularity_round_trip() {
        let req = HostStepRequest {
            model_handle: 1,
            count: 3,
            direction: StepDirection::Forward,
            granularity: Some(TickGranularity::Layer),
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
}
