use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCode {
    #[serde(rename = "INVALID_STATE")]
    InvalidState,
    #[serde(rename = "INVALID_TARGET")]
    InvalidTarget,
    #[serde(rename = "INVALID_RECIPE")]
    InvalidRecipe,
    #[serde(rename = "MODEL_NOT_ATTACHED")]
    ModelNotAttached,
    #[serde(rename = "TENSOR_NOT_FOUND")]
    TensorNotFound,
    #[serde(rename = "CHECKPOINT_NOT_FOUND")]
    CheckpointNotFound,
    #[serde(rename = "PROBE_NOT_FOUND")]
    ProbeNotFound,
    #[serde(rename = "CAPABILITY_NOT_SUPPORTED")]
    CapabilityNotSupported,
    #[serde(rename = "SLICE_OUT_OF_BOUNDS")]
    SliceOutOfBounds,
    #[serde(rename = "RESPONSE_TOO_LARGE")]
    ResponseTooLarge,
    #[serde(rename = "HOST_ERROR")]
    HostError,
    #[serde(rename = "GPU_OOM")]
    GpuOom,
    #[serde(rename = "NCCL_TIMEOUT")]
    NcclTimeout,
    #[serde(rename = "REPLAY_DIVERGENCE")]
    ReplayDivergence,
    #[serde(rename = "UNSUPPORTED_MODEL")]
    UnsupportedModel,
    #[serde(rename = "COMPILED_MODEL")]
    CompiledModel,
    #[serde(rename = "MODEL_ALREADY_ATTACHED")]
    ModelAlreadyAttached,
    #[serde(rename = "INVALID_PARAMS")]
    InvalidParams,
}

impl ErrorCode {
    #[must_use]
    pub fn numeric_code(self) -> i32 {
        match self {
            Self::InvalidState => -32001,
            Self::InvalidTarget => -32002,
            Self::InvalidRecipe => -32003,
            Self::ModelNotAttached => -32004,
            Self::TensorNotFound => -32005,
            Self::CheckpointNotFound => -32006,
            Self::ProbeNotFound => -32007,
            Self::CapabilityNotSupported => -32008,
            Self::SliceOutOfBounds => -32009,
            Self::ResponseTooLarge => -32010,
            Self::HostError => -32011,
            Self::GpuOom => -32012,
            Self::NcclTimeout => -32013,
            Self::ReplayDivergence => -32014,
            Self::UnsupportedModel => -32015,
            Self::CompiledModel => -32016,
            Self::ModelAlreadyAttached => -32017,
            Self::InvalidParams => -32602,
        }
    }

    #[must_use]
    pub fn severity(self) -> Severity {
        match self {
            Self::HostError | Self::GpuOom | Self::NcclTimeout => Severity::Fatal,
            _ => Severity::Recoverable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Fatal,
    Recoverable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorData {
    pub error_code: ErrorCode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numeric_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_state: Option<super::types::Status>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_states: Option<Vec<super::types::Status>>,
    pub suggestion: String,
    pub severity: Severity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<serde_json::Value>,
}

impl ErrorData {
    #[must_use]
    pub fn new(error_code: ErrorCode, suggestion: impl Into<String>) -> Self {
        Self {
            numeric_code: Some(error_code.numeric_code()),
            severity: error_code.severity(),
            error_code,
            suggestion: suggestion.into(),
            current_state: None,
            valid_states: None,
            context: None,
        }
    }
}
