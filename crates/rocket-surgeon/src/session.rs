use rocket_surgeon_protocol::errors::{ErrorCode, ErrorData, Severity};
use rocket_surgeon_protocol::messages::{
    AttachRequest, AttachResponse, DetachResponse, InitializeRequest, InitializeResponse,
    InspectResponse, MemoryUsage, StatusResponse, StepRequest, StepResponse,
};
use rocket_surgeon_protocol::types::TickPosition;
use rocket_surgeon_protocol::types::{
    ActionName, Capabilities, ResponseEnvelope, SessionState, Status, TensorSummary,
};

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum SessionError {
    #[error("invalid state transition")]
    InvalidState(ErrorData),
    #[error("model already attached")]
    ModelAlreadyAttached(ErrorData),
    #[error("model not attached")]
    ModelNotAttached(ErrorData),
    #[error("unsupported model")]
    UnsupportedModel(ErrorData),
    #[error("compiled model")]
    CompiledModel(ErrorData),
    #[error("capability not supported")]
    CapabilityNotSupported(ErrorData),
    #[error("invalid params")]
    InvalidParams(ErrorData),
}

impl SessionError {
    pub fn error_data(&self) -> &ErrorData {
        match self {
            Self::InvalidState(d)
            | Self::ModelAlreadyAttached(d)
            | Self::ModelNotAttached(d)
            | Self::UnsupportedModel(d)
            | Self::CompiledModel(d)
            | Self::CapabilityNotSupported(d)
            | Self::InvalidParams(d) => d,
        }
    }
}

const SUPPORTED_FAMILIES: &[&str] = &["llama", "mixtral", "gpt-neox", "gpt2"];
const PROTOCOL_VERSION: &str = "0.2.0";

#[derive(Debug)]
pub struct Session {
    state: SessionState,
    start_time: std::time::Instant,
    #[allow(dead_code)]
    detached_model_id: Option<String>,
}

impl Session {
    pub fn new() -> Self {
        Self {
            state: SessionState {
                session_id: String::new(),
                model_id: None,
                status: Status::Uninitialized,
                position: None,
                tick_id: None,
                active_probes: Vec::new(),
                checkpoints: Vec::new(),
                available_actions: Vec::new(),
            },
            start_time: std::time::Instant::now(),
            detached_model_id: None,
        }
    }

    #[allow(dead_code)]
    pub fn state(&self) -> &SessionState {
        &self.state
    }

    pub fn envelope<T>(&self, data: T) -> ResponseEnvelope<T> {
        ResponseEnvelope {
            state: self.state.clone(),
            data: Some(data),
        }
    }

    #[allow(dead_code)]
    pub fn envelope_no_data(&self) -> ResponseEnvelope<()> {
        ResponseEnvelope {
            state: self.state.clone(),
            data: None,
        }
    }

    fn update_available_actions(&mut self) {
        self.state.available_actions = match self.state.status {
            Status::Initialized => vec![ActionName::Attach],
            Status::Stopped => vec![
                ActionName::Step,
                ActionName::Inspect,
                ActionName::Intervene,
                ActionName::Probe,
                ActionName::Checkpoint,
                ActionName::Replay,
                ActionName::Detach,
                ActionName::Status,
                ActionName::Subscribe,
            ],
            Status::Uninitialized
            | Status::Attaching
            | Status::Stepping
            | Status::Inspecting
            | Status::Modifying
            | Status::Replaying
            | Status::Detaching => vec![],
        };
    }

    fn invalid_state_error(&self, method: &str, valid_states: Vec<Status>) -> SessionError {
        let suggestion = if valid_states.len() == 1 {
            format!(
                "The session must be in {:?} state to call {method}",
                valid_states[0]
            )
        } else {
            format!("The session must be in one of {valid_states:?} to call {method}")
        };
        SessionError::InvalidState(ErrorData {
            error_code: ErrorCode::InvalidState,
            numeric_code: Some(ErrorCode::InvalidState.numeric_code()),
            severity: Severity::Recoverable,
            suggestion,
            current_state: Some(self.state.status),
            valid_states: Some(valid_states),
            context: None,
        })
    }

    pub fn initialize(
        &mut self,
        req: &InitializeRequest,
    ) -> Result<ResponseEnvelope<InitializeResponse>, SessionError> {
        if self.state.status != Status::Uninitialized {
            return Err(self.invalid_state_error("initialize", vec![Status::Uninitialized]));
        }

        if req.protocol_version != PROTOCOL_VERSION {
            return Err(SessionError::InvalidParams(ErrorData {
                error_code: ErrorCode::InvalidParams,
                numeric_code: Some(ErrorCode::InvalidParams.numeric_code()),
                severity: Severity::Recoverable,
                suggestion: format!(
                    "Unsupported protocol version '{}', server supports '{PROTOCOL_VERSION}'",
                    req.protocol_version
                ),
                current_state: Some(self.state.status),
                valid_states: None,
                context: None,
            }));
        }

        self.state.session_id = uuid::Uuid::new_v4().to_string();
        self.state.status = Status::Initialized;
        self.update_available_actions();

        let capabilities = Capabilities::phase1_defaults();

        Ok(self.envelope(InitializeResponse { capabilities }))
    }

    /// Cheap state-machine validation for an attach request. Used by the
    /// daemon main loop to reject obviously-bad attaches before paying to
    /// spawn the orchestrator/worker (BEAD-0008 review finding H-1).
    /// Returns `Ok(())` if `commit_attach` would currently succeed.
    pub fn validate_attach(&self, req: &AttachRequest) -> Result<(), SessionError> {
        if self.state.status == Status::Stopped || self.state.model_id.is_some() {
            return Err(SessionError::ModelAlreadyAttached(ErrorData {
                error_code: ErrorCode::ModelAlreadyAttached,
                numeric_code: Some(ErrorCode::ModelAlreadyAttached.numeric_code()),
                severity: Severity::Recoverable,
                suggestion: "Detach the current model before attaching a new one".to_owned(),
                current_state: Some(self.state.status),
                valid_states: None,
                context: None,
            }));
        }

        if self.state.status != Status::Initialized {
            return Err(self.invalid_state_error("attach", vec![Status::Initialized]));
        }

        if let Some(config) = &req.config {
            if let Some(mode) = config.get("execution_mode") {
                if mode.as_str() == Some("compiled") {
                    return Err(SessionError::CompiledModel(ErrorData {
                        error_code: ErrorCode::CompiledModel,
                        numeric_code: Some(ErrorCode::CompiledModel.numeric_code()),
                        severity: Severity::Recoverable,
                        suggestion: "rocket_surgeon requires eager-mode models. Remove torch.compile() wrapper before attaching".to_owned(),
                        current_state: Some(self.state.status),
                        valid_states: None,
                        context: None,
                    }));
                }
            }
        }

        if !SUPPORTED_FAMILIES.contains(&req.model_family.as_str()) {
            return Err(SessionError::UnsupportedModel(ErrorData {
                error_code: ErrorCode::UnsupportedModel,
                numeric_code: Some(ErrorCode::UnsupportedModel.numeric_code()),
                severity: Severity::Recoverable,
                suggestion: format!(
                    "Supported model families: {}",
                    SUPPORTED_FAMILIES.join(", ")
                ),
                current_state: Some(self.state.status),
                valid_states: None,
                context: None,
            }));
        }

        Ok(())
    }

    /// Commit a validated attach with real worker metadata.
    ///
    /// Callers MUST have called `validate_attach` first; this method assumes
    /// validation already passed and only performs state mutation +
    /// response building. `model_family` reflects what the worker actually
    /// loaded (`HostAttachResponse.model_type`), not what the client claimed
    /// — BEAD-0008 review finding H-2.
    pub fn commit_attach(
        &mut self,
        req: &AttachRequest,
        worker_model_type: &str,
        num_layers: u32,
        num_heads: u32,
        hidden_dim: u32,
    ) -> ResponseEnvelope<AttachResponse> {
        let model_id = format!("model-{}", uuid::Uuid::new_v4());
        self.state.model_id = Some(model_id.clone());
        self.state.status = Status::Stopped;
        self.state.position = None;
        self.state.tick_id = None;
        self.update_available_actions();

        self.envelope(AttachResponse {
            model_id,
            model_family: worker_model_type.to_owned(),
            num_layers,
            num_heads,
            hidden_dim,
            num_ranks: req.num_ranks,
            capabilities: Capabilities::phase1_defaults(),
        })
    }

    /// Convenience for tests: validate + commit in one call.
    ///
    /// Production code (the daemon main loop) calls `validate_attach` and
    /// `commit_attach` separately so it can run the validation before
    /// paying to spawn the backend.
    #[cfg(test)]
    pub fn attach(
        &mut self,
        req: &AttachRequest,
        num_layers: u32,
        num_heads: u32,
        hidden_dim: u32,
    ) -> Result<ResponseEnvelope<AttachResponse>, SessionError> {
        self.validate_attach(req)?;
        Ok(self.commit_attach(req, &req.model_family, num_layers, num_heads, hidden_dim))
    }

    pub fn detach(&mut self) -> Result<ResponseEnvelope<DetachResponse>, SessionError> {
        if self.state.status == Status::Initialized || self.state.model_id.is_none() {
            return Err(SessionError::ModelNotAttached(ErrorData {
                error_code: ErrorCode::ModelNotAttached,
                numeric_code: Some(ErrorCode::ModelNotAttached.numeric_code()),
                severity: Severity::Recoverable,
                suggestion: "No model is currently attached".to_owned(),
                current_state: Some(self.state.status),
                valid_states: None,
                context: None,
            }));
        }

        if self.state.status != Status::Stopped {
            return Err(self.invalid_state_error("detach", vec![Status::Stopped]));
        }

        let detached_id = self.state.model_id.take().unwrap_or_default();
        self.detached_model_id = Some(detached_id.clone());
        self.state.status = Status::Initialized;
        self.state.position = None;
        self.state.tick_id = None;
        self.state.active_probes.clear();
        self.state.checkpoints.clear();
        self.update_available_actions();

        Ok(self.envelope(DetachResponse {
            detached_model_id: detached_id,
        }))
    }

    pub fn status(&self) -> Result<ResponseEnvelope<StatusResponse>, SessionError> {
        if self.state.status == Status::Uninitialized {
            return Err(
                self.invalid_state_error("status", vec![Status::Initialized, Status::Stopped])
            );
        }

        Ok(self.envelope(StatusResponse {
            uptime_seconds: self.start_time.elapsed().as_secs_f64(),
            connected_clients: 1,
            memory_usage: MemoryUsage {
                gpu_mb: 0.0,
                cpu_mb: 0.0,
            },
            pending_interventions: 0,
            trace_events_recorded: 0,
        }))
    }

    pub fn require_stopped(&self, method: &str) -> Result<(), SessionError> {
        match self.state.status {
            Status::Stopped => Ok(()),
            Status::Uninitialized | Status::Initialized => {
                Err(SessionError::ModelNotAttached(ErrorData {
                    error_code: ErrorCode::ModelNotAttached,
                    numeric_code: Some(ErrorCode::ModelNotAttached.numeric_code()),
                    severity: Severity::Recoverable,
                    suggestion: "Attach a model before calling this method".to_owned(),
                    current_state: Some(self.state.status),
                    valid_states: Some(vec![Status::Stopped]),
                    context: None,
                }))
            }
            _ => Err(self.invalid_state_error(method, vec![Status::Stopped])),
        }
    }

    #[allow(dead_code)]
    pub fn check_capability(&self, cap: &str) -> Result<(), SessionError> {
        if matches!(
            cap,
            "supports_checkpointing"
                | "supports_reverse_step"
                | "supports_backward"
                | "supports_moe"
                | "supports_sae"
        ) {
            return Err(self.capability_not_supported_error(cap));
        }
        Ok(())
    }

    fn capability_not_supported_error(&self, cap: &str) -> SessionError {
        SessionError::CapabilityNotSupported(ErrorData {
            error_code: ErrorCode::CapabilityNotSupported,
            numeric_code: Some(ErrorCode::CapabilityNotSupported.numeric_code()),
            severity: Severity::Recoverable,
            suggestion: format!("The {cap} capability is not supported in this build"),
            current_state: Some(self.state.status),
            valid_states: None,
            context: None,
        })
    }

    #[allow(dead_code)]
    pub fn step(
        &mut self,
        req: &StepRequest,
        host_position: &TickPosition,
        _forward_complete: bool,
    ) -> Result<ResponseEnvelope<StepResponse>, SessionError> {
        self.require_stopped("rocket/step")?;

        if req.direction == rocket_surgeon_protocol::types::StepDirection::Backward {
            return Err(self.capability_not_supported_error("supports_reverse_step"));
        }

        self.state.tick_id = Some(host_position.tick_id);
        self.state.position = Some(host_position.clone());
        self.update_available_actions();

        Ok(self.envelope(StepResponse {
            ticks_executed: req.count,
            stopped_at: host_position.clone(),
        }))
    }

    #[allow(dead_code)]
    pub fn inspect(
        &self,
        tensors: &[TensorSummary],
        slice_data: Option<String>,
    ) -> Result<ResponseEnvelope<InspectResponse>, SessionError> {
        self.require_stopped("rocket/inspect")?;

        Ok(self.envelope(InspectResponse {
            tensors: tensors.to_vec(),
            view_result: None,
            slice_data,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocket_surgeon_protocol::messages::StepRequest;
    use rocket_surgeon_protocol::types::{
        DType, ExecutionMode, Phase, StepDirection, TensorStats, TensorSummary, TickEvent,
        TickGranularity, TickPosition, TopKEntry,
    };

    fn init_request() -> InitializeRequest {
        InitializeRequest {
            client_name: "test-client".to_owned(),
            protocol_version: "0.2.0".to_owned(),
            client_version: None,
            client_capabilities: None,
        }
    }

    fn attach_request(family: &str) -> AttachRequest {
        AttachRequest {
            model_path: "/models/test-model".to_owned(),
            model_family: family.to_owned(),
            device: "cuda:0".to_owned(),
            dtype: None,
            num_ranks: 1,
            config: None,
        }
    }

    // Stand-in metadata for unit tests. Production code receives these values
    // from the worker's `HostAttachResponse` (BEAD-0008). Distinctive numbers
    // (not the old "llama" stub 32/32/4096) so a future copy-paste can't
    // accidentally re-encode the deleted per-family stub assumption.
    const TEST_NUM_LAYERS: u32 = 7;
    const TEST_NUM_HEADS: u32 = 3;
    const TEST_HIDDEN_DIM: u32 = 256;

    fn test_attach(
        session: &mut Session,
        family: &str,
    ) -> Result<ResponseEnvelope<AttachResponse>, SessionError> {
        session.attach(
            &attach_request(family),
            TEST_NUM_LAYERS,
            TEST_NUM_HEADS,
            TEST_HIDDEN_DIM,
        )
    }

    fn initialized_session() -> Session {
        let mut session = Session::new();
        session.initialize(&init_request()).unwrap();
        session
    }

    fn stopped_session() -> Session {
        let mut session = initialized_session();
        test_attach(&mut session, "llama").unwrap();
        session
    }

    #[test]
    fn new_session_is_uninitialized() {
        let session = Session::new();
        assert_eq!(session.state().status, Status::Uninitialized);
        assert!(session.state().session_id.is_empty());
        assert!(session.state().model_id.is_none());
        assert!(session.state().available_actions.is_empty());
    }

    #[test]
    fn initialize_transitions_to_initialized() {
        let mut session = Session::new();
        let resp = session.initialize(&init_request()).unwrap();
        assert_eq!(resp.state.status, Status::Initialized);
    }

    #[test]
    fn initialize_returns_capabilities() {
        let mut session = Session::new();
        let resp = session.initialize(&init_request()).unwrap();
        let caps = &resp.data.as_ref().unwrap().capabilities;
        assert_eq!(caps.protocol_version, "0.2.0");
        assert_eq!(caps.execution_mode, ExecutionMode::Eager);
    }

    #[test]
    fn unsupported_protocol_version_returns_error() {
        let mut session = Session::new();
        let req = InitializeRequest {
            client_name: "test-client".to_owned(),
            protocol_version: "99.0.0".to_owned(),
            client_version: None,
            client_capabilities: None,
        };
        let err = session.initialize(&req).unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::InvalidParams);
        assert!(data.suggestion.contains("99.0.0"));
        assert_eq!(session.state().status, Status::Uninitialized);
    }

    #[test]
    fn double_initialize_returns_invalid_state() {
        let mut session = initialized_session();
        let err = session.initialize(&init_request()).unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::InvalidState);
        assert_eq!(data.current_state, Some(Status::Initialized));
    }

    #[test]
    fn attach_from_initialized_transitions_to_stopped() {
        let mut session = initialized_session();
        let resp = test_attach(&mut session, "llama").unwrap();
        assert_eq!(resp.state.status, Status::Stopped);
        assert!(resp.state.model_id.is_some());
        assert_eq!(resp.data.as_ref().unwrap().model_family, "llama");
    }

    #[test]
    fn attach_from_uninitialized_returns_invalid_state() {
        let mut session = Session::new();
        let err = test_attach(&mut session, "llama").unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::InvalidState);
        assert_eq!(data.current_state, Some(Status::Uninitialized));
    }

    #[test]
    fn attach_while_stopped_returns_model_already_attached() {
        let mut session = stopped_session();
        let err = test_attach(&mut session, "gpt2").unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::ModelAlreadyAttached);
    }

    #[test]
    fn detach_from_stopped_transitions_to_initialized() {
        let mut session = stopped_session();
        let resp = session.detach().unwrap();
        assert_eq!(resp.state.status, Status::Initialized);
        assert!(resp.state.model_id.is_none());
        assert!(!resp.data.as_ref().unwrap().detached_model_id.is_empty());
    }

    #[test]
    fn detach_from_initialized_returns_model_not_attached() {
        let mut session = initialized_session();
        let err = session.detach().unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::ModelNotAttached);
    }

    #[test]
    fn session_id_is_uuid_and_stable() {
        let mut session = Session::new();
        let resp1 = session.initialize(&init_request()).unwrap();
        let sid = resp1.state.session_id;
        assert_eq!(sid.len(), 36);
        assert!(sid.contains('-'));

        let resp2 = test_attach(&mut session, "llama").unwrap();
        assert_eq!(resp2.state.session_id, sid);

        let resp3 = session.detach().unwrap();
        assert_eq!(resp3.state.session_id, sid);
    }

    #[test]
    fn model_id_null_before_attach_populated_after() {
        let mut session = Session::new();
        let resp = session.initialize(&init_request()).unwrap();
        assert!(resp.state.model_id.is_none());

        let resp = test_attach(&mut session, "llama").unwrap();
        assert!(resp.state.model_id.is_some());
    }

    #[test]
    fn available_actions_initialized_is_attach_only() {
        let session = initialized_session();
        assert_eq!(session.state().available_actions, vec![ActionName::Attach]);
    }

    #[test]
    fn available_actions_stopped_includes_domain_verbs() {
        let session = stopped_session();
        let actions = &session.state().available_actions;
        assert!(actions.contains(&ActionName::Step));
        assert!(actions.contains(&ActionName::Inspect));
        assert!(actions.contains(&ActionName::Intervene));
        assert!(actions.contains(&ActionName::Probe));
        assert!(actions.contains(&ActionName::Detach));
        assert!(actions.contains(&ActionName::Status));
        assert!(actions.contains(&ActionName::Subscribe));
        assert!(actions.contains(&ActionName::Checkpoint));
        assert!(actions.contains(&ActionName::Replay));
    }

    #[test]
    fn unsupported_model_family_returns_error() {
        let mut session = initialized_session();
        let err = test_attach(&mut session, "unknown_arch").unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::UnsupportedModel);
    }

    #[test]
    fn compiled_model_returns_error() {
        let mut session = initialized_session();
        let mut req = attach_request("llama");
        req.config = Some(serde_json::json!({"execution_mode": "compiled"}));
        let err = session
            .attach(&req, TEST_NUM_LAYERS, TEST_NUM_HEADS, TEST_HIDDEN_DIM)
            .unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::CompiledModel);
    }

    #[test]
    fn re_attach_after_detach_succeeds() {
        let mut session = stopped_session();
        session.detach().unwrap();
        let resp = test_attach(&mut session, "mixtral").unwrap();
        assert_eq!(resp.state.status, Status::Stopped);
        assert_eq!(resp.data.as_ref().unwrap().model_family, "mixtral");
    }

    #[test]
    fn require_stopped_when_initialized() {
        let session = initialized_session();
        let err = session.require_stopped("rocket/step").unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::ModelNotAttached);
    }

    #[test]
    fn require_stopped_when_stopped_succeeds() {
        let session = stopped_session();
        assert!(session.require_stopped("rocket/step").is_ok());
    }

    #[test]
    fn check_capability_unsupported() {
        let session = stopped_session();
        let err = session
            .check_capability("supports_checkpointing")
            .unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::CapabilityNotSupported);
    }

    #[test]
    fn full_lifecycle_round_trip() {
        let mut session = Session::new();

        let resp = session.initialize(&init_request()).unwrap();
        assert_eq!(resp.state.status, Status::Initialized);
        let sid = resp.state.session_id;

        let resp = test_attach(&mut session, "llama").unwrap();
        assert_eq!(resp.state.status, Status::Stopped);
        assert_eq!(resp.state.session_id, sid);

        let resp = session.detach().unwrap();
        assert_eq!(resp.state.status, Status::Initialized);
        assert_eq!(resp.state.session_id, sid);

        let resp = test_attach(&mut session, "llama").unwrap();
        assert_eq!(resp.state.status, Status::Stopped);
        assert_eq!(resp.state.session_id, sid);
    }

    #[test]
    fn status_from_stopped() {
        let session = stopped_session();
        let resp = session.status().unwrap();
        assert!(resp.data.as_ref().unwrap().uptime_seconds >= 0.0);
        assert_eq!(resp.data.as_ref().unwrap().connected_clients, 1);
    }

    #[test]
    fn status_from_uninitialized_fails() {
        let session = Session::new();
        let err = session.status().unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::InvalidState);
    }

    #[test]
    fn step_from_stopped_succeeds() {
        let mut session = stopped_session();
        let req = StepRequest {
            direction: StepDirection::Forward,
            count: 1,
            granularity: Some(TickGranularity::Component),
        };
        let host_position = TickPosition {
            tick_id: 1,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer: 0,
            component: "q_proj".to_owned(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        };
        let result = session.step(&req, &host_position, false);
        assert!(result.is_ok());
        let envelope = result.unwrap();
        assert_eq!(envelope.state.status, Status::Stopped);
        let data = envelope.data.as_ref().unwrap();
        assert_eq!(data.ticks_executed, 1);
        assert_eq!(data.stopped_at.component, "q_proj");
    }

    #[test]
    fn step_from_initialized_returns_error() {
        let mut session = initialized_session();
        let req = StepRequest {
            direction: StepDirection::Forward,
            count: 1,
            granularity: None,
        };
        let pos = TickPosition {
            tick_id: 1,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer: 0,
            component: String::new(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        };
        let err = session.step(&req, &pos, false).unwrap_err();
        assert_eq!(err.error_data().error_code, ErrorCode::ModelNotAttached);
    }

    #[test]
    fn step_backward_returns_capability_not_supported() {
        let mut session = stopped_session();
        let req = StepRequest {
            direction: StepDirection::Backward,
            count: 1,
            granularity: None,
        };
        let pos = TickPosition {
            tick_id: 0,
            direction: StepDirection::Backward,
            rank: Some(0),
            layer: 0,
            component: String::new(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        };
        let err = session.step(&req, &pos, false).unwrap_err();
        assert_eq!(
            err.error_data().error_code,
            ErrorCode::CapabilityNotSupported
        );
    }

    #[test]
    fn step_updates_tick_id_in_session_state() {
        let mut session = stopped_session();
        let req = StepRequest {
            direction: StepDirection::Forward,
            count: 1,
            granularity: Some(TickGranularity::Component),
        };
        let pos1 = TickPosition {
            tick_id: 1,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer: 0,
            component: "q_proj".to_owned(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        };
        session.step(&req, &pos1, false).unwrap();
        assert_eq!(session.state().tick_id, Some(1));

        let pos2 = TickPosition {
            tick_id: 2,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer: 0,
            component: "k_proj".to_owned(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        };
        session.step(&req, &pos2, false).unwrap();
        assert_eq!(session.state().tick_id, Some(2));
    }

    #[test]
    fn step_updates_position_in_session_state() {
        let mut session = stopped_session();
        let req = StepRequest {
            direction: StepDirection::Forward,
            count: 1,
            granularity: Some(TickGranularity::Component),
        };
        let pos = TickPosition {
            tick_id: 1,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer: 3,
            component: "gate_proj".to_owned(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        };
        session.step(&req, &pos, false).unwrap();
        let state = session.state();
        assert!(state.position.is_some());
        let session_pos = state.position.as_ref().unwrap();
        assert_eq!(session_pos.layer, 3);
        assert_eq!(session_pos.component, "gate_proj");
    }

    #[test]
    fn step_returns_envelope_with_correct_state() {
        let mut session = stopped_session();
        let req = StepRequest {
            direction: StepDirection::Forward,
            count: 1,
            granularity: Some(TickGranularity::Component),
        };
        let pos = TickPosition {
            tick_id: 1,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer: 0,
            component: "embed".to_owned(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        };
        let envelope = session.step(&req, &pos, false).unwrap();
        assert!(envelope.state.session_id.len() == 36);
        assert!(envelope.state.model_id.is_some());
        assert_eq!(envelope.state.status, Status::Stopped);
        assert!(envelope.state.available_actions.contains(&ActionName::Step));
    }

    fn sample_tensor_summary() -> TensorSummary {
        TensorSummary {
            tensor_id: "a".repeat(64),
            shape: vec![4],
            dtype: DType::Float32,
            device: "cpu".to_owned(),
            sharding: None,
            stats: TensorStats {
                mean: 2.5,
                std: 1.118,
                min: 1.0,
                max: 4.0,
                abs_max: 4.0,
                sparsity: 0.0,
                l2_norm: 5.477,
                histogram: rocket_surgeon_protocol::types::Histogram {
                    bins: 10,
                    edges: vec![1.0, 2.0, 3.0, 4.0],
                    counts: vec![1, 1, 1, 1],
                },
            },
            top_k: vec![TopKEntry {
                index: vec![3],
                value: 4.0,
            }],
        }
    }

    #[test]
    fn inspect_from_stopped_succeeds() {
        let session = stopped_session();
        let tensors = vec![sample_tensor_summary()];
        let result = session.inspect(&tensors, None);
        assert!(result.is_ok());
        let envelope = result.unwrap();
        assert_eq!(envelope.state.status, Status::Stopped);
        let data = envelope.data.as_ref().unwrap();
        assert_eq!(data.tensors.len(), 1);
        assert!(data.slice_data.is_none());
        assert!(data.view_result.is_none());
    }

    #[test]
    fn inspect_from_initialized_returns_error() {
        let session = initialized_session();
        let result = session.inspect(&[], None);
        let err = result.unwrap_err();
        assert_eq!(err.error_data().error_code, ErrorCode::ModelNotAttached);
    }

    #[test]
    fn inspect_with_slice_data() {
        let session = stopped_session();
        let tensors = vec![sample_tensor_summary()];
        let result = session.inspect(&tensors, Some("AQIDBA==".to_owned()));
        assert!(result.is_ok());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data.slice_data.as_deref(), Some("AQIDBA=="));
    }

    #[test]
    fn inspect_does_not_change_session_state() {
        let mut session = stopped_session();
        let req = StepRequest {
            direction: StepDirection::Forward,
            count: 1,
            granularity: Some(TickGranularity::Component),
        };
        let pos = TickPosition {
            tick_id: 5,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer: 2,
            component: "q_proj".to_owned(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        };
        session.step(&req, &pos, false).unwrap();
        let tick_before = session.state().tick_id;
        let pos_before = session.state().position.clone();

        session.inspect(&[sample_tensor_summary()], None).unwrap();

        assert_eq!(session.state().tick_id, tick_before);
        assert_eq!(session.state().position, pos_before);
        assert_eq!(session.state().status, Status::Stopped);
    }
}
