use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Session envelope
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: String,
    pub model_id: Option<String>,
    pub status: Status,
    pub position: Option<TickPosition>,
    pub tick_id: Option<u64>,
    pub active_probes: Vec<String>,
    pub checkpoints: Vec<CheckpointRef>,
    pub available_actions: Vec<ActionName>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Uninitialized,
    Initialized,
    Attaching,
    Stopped,
    Stepping,
    Inspecting,
    Modifying,
    Replaying,
    Detaching,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionName {
    Initialize,
    Attach,
    Detach,
    Step,
    Inspect,
    Intervene,
    Probe,
    Checkpoint,
    Replay,
    Status,
    Subscribe,
}

// ---------------------------------------------------------------------------
// Tick model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TickClock {
    pub token: u64,
    pub operator: u64,
    pub wall_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickPosition {
    pub tick_id: u64,
    pub direction: StepDirection,
    pub rank: Option<u32>,
    pub layer: u32,
    pub component: String,
    pub event: TickEvent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_of: Option<u64>,
    #[serde(default)]
    pub phase: Phase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_position: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock: Option<TickClock>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Phase {
    Prefill,
    #[default]
    Decode,
    PrefillChunked {
        chunk_size: u32,
        chunk_index: u32,
        total_chunks: u32,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepDirection {
    #[default]
    Forward,
    Backward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TickEvent {
    Input,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TickGranularity {
    Layer,
    Component,
    Head,
    RouterPreTopk,
    RouterPostTopk,
    Expert,
    MoeLayer,
}

// ---------------------------------------------------------------------------
// Tensor types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorSummary {
    pub tensor_id: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub device: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sharding: Option<ShardingInfo>,
    pub stats: TensorStats,
    pub top_k: Vec<TopKEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorStats {
    pub mean: f64,
    /// Population standard deviation (ddof=0). Matches `NumPy` default, differs from `PyTorch` default (ddof=1).
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub abs_max: f64,
    pub sparsity: f64,
    pub l2_norm: f64,
    pub histogram: Histogram,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Histogram {
    pub bins: u32,
    pub edges: Vec<f64>,
    pub counts: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopKEntry {
    pub index: Vec<u64>,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    Float16,
    Bfloat16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Bool,
}

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardingInfo {
    pub mesh: String,
    pub placements: Vec<Placement>,
    pub local_shape: Vec<u64>,
    pub global_shape: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placement {
    #[serde(rename = "type")]
    pub placement_type: PlacementType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dim: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlacementType {
    Shard,
    Replicate,
    Partial,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorHandle {
    pub tensor_id: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
}

// ---------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeDefinition {
    pub id: String,
    pub point: String,
    pub action: ProbeAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<ProbeConfig>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub priority: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeAction {
    Capture,
    Checkpoint,
    Trace,
    Assert,
    Aggregate,
    Intervene,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeConfig {
    #[serde(default = "default_true")]
    pub summary: bool,
    #[serde(default)]
    pub capture_tensor: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregate_fn: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assertion: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intervention: Option<InterventionRecipe>,
}

pub(crate) fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Interventions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionRecipe {
    pub id: String,
    #[serde(rename = "type")]
    pub intervention_type: InterventionType,
    pub target: String,
    pub params: InterventionParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    #[serde(default)]
    pub priority: i32,
    #[serde(default = "default_additive")]
    pub mode: CompositionMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionType {
    Ablate,
    Scale,
    Add,
    Patch,
    Clamp,
    RouteOverride,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InterventionParams {
    Scale { factor: f64 },
    Add { vector: AddVector },
    Patch { source_tensor_id: String },
    Clamp { min: f64, max: f64 },
    RouteOverride { token: u64, experts: Vec<u64> },
    Ablate {},
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AddVector {
    Inline(Vec<f64>),
    TensorRef(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompositionMode {
    Additive,
    Replace,
}

fn default_additive() -> CompositionMode {
    CompositionMode::Additive
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    pub protocol_version: String,
    pub supports_reverse_step: bool,
    pub supports_checkpointing: bool,
    pub supports_moe: bool,
    pub supports_backward: bool,
    pub supports_sae: bool,
    pub execution_mode: ExecutionMode,
    pub parallelism: Parallelism,
    pub tick_granularities: Vec<TickGranularity>,
    pub intervention_types: Vec<InterventionType>,
    pub built_in_views: Vec<BuiltInView>,
    pub head_granularity: HeadGranularity,
    pub transports: Vec<Transport>,
    pub wire_formats: Vec<WireFormat>,
    pub max_response_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_layers: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_heads: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_dim: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ranks: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_experts: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k_experts: Option<u32>,
    pub shared_memory_supported: bool,
}

impl Capabilities {
    #[must_use]
    pub fn phase1_defaults() -> Self {
        Self {
            protocol_version: "0.2.0".to_owned(),
            supports_reverse_step: false,
            supports_checkpointing: false,
            supports_moe: false,
            supports_backward: false,
            supports_sae: false,
            execution_mode: ExecutionMode::Eager,
            parallelism: Parallelism::SingleGpu,
            tick_granularities: vec![TickGranularity::Layer, TickGranularity::Component],
            intervention_types: vec![
                InterventionType::Ablate,
                InterventionType::Scale,
                InterventionType::Add,
                InterventionType::Patch,
                InterventionType::Clamp,
            ],
            built_in_views: vec![
                BuiltInView::ResidualStreamNorm,
                BuiltInView::AttentionPattern,
            ],
            head_granularity: HeadGranularity::Unavailable,
            transports: vec![Transport::Stdio, Transport::UnixSocket],
            wire_formats: vec![WireFormat::Json],
            max_response_bytes: 65536,
            model_family: None,
            model_id: None,
            num_layers: None,
            num_heads: None,
            hidden_dim: None,
            num_ranks: None,
            num_experts: None,
            top_k_experts: None,
            shared_memory_supported: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Eager,
    Compiled,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Parallelism {
    SingleGpu,
    Ddp,
    Fsdp,
    TensorParallel,
    PipelineParallel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltInView {
    ResidualStreamNorm,
    AttentionPattern,
    HeadOutput,
    LogitLens,
    RoutingDecision,
    RoutingEntropy,
    FeatureAttribution,
    SaeActivation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadGranularity {
    Native,
    RequiresUnfused,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transport {
    Stdio,
    UnixSocket,
    Tcp,
    Websocket,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireFormat {
    Json,
    Protobuf,
}

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointRef {
    pub checkpoint_id: String,
    pub tick_id: u64,
    pub layer_idx: u32,
    pub tier: CheckpointTier,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bookmark: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointTier {
    ProbeLog,
    Activation,
    FullSnapshot,
}

// ---------------------------------------------------------------------------
// Granularity scoping
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GranularityScope {
    #[serde(rename = "match")]
    pub match_pattern: String,
    pub granularity: TickGranularity,
}

// ---------------------------------------------------------------------------
// Response envelope
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseEnvelope<T> {
    pub state: SessionState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
}

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
