use std::collections::HashMap;

use rocket_surgeon_protocol::errors::{ErrorCode, ErrorData, Severity};
use rocket_surgeon_protocol::messages::{
    AttachRequest, AttachResponse, DetachResponse, DiscoverMatch, DiscoverResponse,
    InitializeRequest, InitializeResponse, InspectResponse, MemoryUsage, StatusResponse,
    StepRequest, StepResponse, SubscribeFilter, ViewDefineResponse,
};
use rocket_surgeon_protocol::types::TickPosition;
use rocket_surgeon_protocol::types::{
    ActionName, Capabilities, EnvelopeMode, PositionEnvelope, ResponseEnvelope, SessionState,
    Status, TensorSummary,
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
const PROTOCOL_VERSION: &str = "0.3.0";

/// A single discoverable probe-point retained in session state after attach.
///
/// `rocket/discover` pattern-matches over a `Vec<ProbePointEntry>` to answer
/// "what can I probe?" without the client guessing probe-point strings. Each
/// entry corresponds to one concrete `family:rank:layer:component:event`
/// coordinate the worker exposed (the daemon retains this from the attach
/// metadata so later verbs do not need a worker round-trip).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbePointEntry {
    /// Model family the point belongs to (e.g. `"llama"`).
    pub family: String,
    /// Layer index the component lives in.
    pub layer: u32,
    /// Canonical component name (e.g. `"attn.q_proj"`).
    pub canonical: String,
    /// Capture event (`"input"` / `"output"` / `"pre_topk"` / ...).
    pub event: String,
    /// Tensor shape produced at this point.
    pub tensor_shape: Vec<u64>,
    /// Alternate names (HookedTransformer-style aliases, etc.).
    pub aliases: Vec<String>,
}

/// One catalog segment of a discover pattern: `*` is a wildcard.
enum PatternSeg<'a> {
    Wild,
    Lit(&'a str),
}

impl<'a> PatternSeg<'a> {
    fn parse(raw: &'a str) -> Self {
        if raw == "*" {
            Self::Wild
        } else {
            Self::Lit(raw)
        }
    }

    fn matches(&self, target: &str) -> bool {
        match self {
            Self::Wild => true,
            Self::Lit(lit) => *lit == target,
        }
    }
}

/// A parsed 5-segment discover pattern: `family:rank:layer:component:event`.
///
/// Note this is intentionally distinct from the 6-segment probe-point grammar
/// (`rocket-surgeon-probes`): discover patterns omit `call_index` per the
/// `tck/protocol/discover.feature` spec (`llama:*:12:*:output`).
struct DiscoverPattern<'a> {
    family: PatternSeg<'a>,
    layer: PatternSeg<'a>,
    component: PatternSeg<'a>,
    event: PatternSeg<'a>,
}

impl<'a> DiscoverPattern<'a> {
    /// Parse a discover pattern. The `rank` segment is accepted but ignored
    /// for matching (probe-points are layer/component coordinates; rank is a
    /// runtime sharding concern, not a discovery axis).
    fn parse(pattern: &'a str) -> Option<Self> {
        let segs: Vec<&str> = pattern.split(':').collect();
        if segs.len() != 5 || segs.iter().any(|s| s.is_empty()) {
            return None;
        }
        Some(Self {
            family: PatternSeg::parse(segs[0]),
            layer: PatternSeg::parse(segs[2]),
            component: PatternSeg::parse(segs[3]),
            event: PatternSeg::parse(segs[4]),
        })
    }

    fn matches(&self, entry: &ProbePointEntry) -> bool {
        let layer_str = entry.layer.to_string();
        self.family.matches(&entry.family)
            && self.layer.matches(&layer_str)
            && self.component.matches(&entry.canonical)
            && self.event.matches(&entry.event)
    }
}

/// Build the default discoverable probe-point catalog for a freshly attached
/// model. The daemon retains this so `rocket/discover` can answer without a
/// worker round-trip. Shapes are derived from the worker-reported model
/// metadata (`hidden_dim`, `num_heads`).
fn default_catalog(
    family: &str,
    num_layers: u32,
    num_heads: u32,
    hidden_dim: u32,
) -> Vec<ProbePointEntry> {
    let hidden = u64::from(hidden_dim);
    let heads = u64::from(num_heads);
    // (canonical, event, shape, alias-suffix)
    // alias-suffix is appended to the HookedTransformer-style `blocks.{l}.`
    // prefix to form a per-layer alias.
    let components: &[(&str, &str, Vec<u64>, &str)] = &[
        ("attn.q_proj", "output", vec![1, hidden], "attn.hook_q"),
        ("attn.k_proj", "output", vec![1, hidden], "attn.hook_k"),
        ("attn.v_proj", "output", vec![1, hidden], "attn.hook_v"),
        ("attn.o_proj", "output", vec![1, hidden], "attn.hook_z"),
        (
            "attn.scores",
            "output",
            vec![1, heads, 1, 1],
            "attn.hook_pattern",
        ),
        ("mlp", "output", vec![1, hidden], "hook_mlp_out"),
        (
            "residual_post",
            "output",
            vec![1, hidden],
            "hook_resid_post",
        ),
    ];

    let mut catalog = Vec::with_capacity(num_layers as usize * components.len());
    for layer in 0..num_layers {
        for (canonical, event, shape, alias_suffix) in components {
            catalog.push(ProbePointEntry {
                family: family.to_owned(),
                layer,
                canonical: (*canonical).to_owned(),
                event: (*event).to_owned(),
                tensor_shape: shape.clone(),
                aliases: vec![format!("blocks.{layer}.{alias_suffix}")],
            });
        }
    }
    catalog
}

#[derive(Debug)]
pub struct Session {
    state: SessionState,
    start_time: std::time::Instant,
    #[allow(dead_code)]
    detached_model_id: Option<String>,
    /// Discoverable probe-points retained from the most recent attach.
    /// Empty until a model is attached; cleared on detach.
    catalog: Vec<ProbePointEntry>,
    /// Named view specifications registered via `rocket/view.define`.
    /// Keyed by view name; cleared on detach.
    defined_views: HashMap<String, serde_json::Value>,
    /// Event-stream filter captured from the most recent `subscribe` request.
    /// `None` means no subscription (or an unfiltered subscription); the
    /// event fan-out in `notifications.rs` consults this to decide which
    /// notifications a subscriber should receive.
    event_filter: Option<SubscribeFilter>,
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
            catalog: Vec::new(),
            defined_views: HashMap::new(),
            event_filter: None,
        }
    }

    #[allow(dead_code)]
    pub fn state(&self) -> &SessionState {
        &self.state
    }

    /// Store the event-stream filter negotiated by a `subscribe` request.
    /// Passing `None` clears any prior filter (unfiltered subscription).
    pub fn set_event_filter(&mut self, filter: Option<SubscribeFilter>) {
        self.event_filter = filter;
    }

    /// The event-stream filter currently in force, if any.
    pub fn event_filter(&self) -> Option<&SubscribeFilter> {
        self.event_filter.as_ref()
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

    /// Build a response body honouring the client-requested [`EnvelopeMode`].
    ///
    /// LLM clients negotiate envelope verbosity to manage context-window
    /// pressure (TCK `envelope-compactness.feature`):
    /// - [`EnvelopeMode::Full`] — the complete [`SessionState`] envelope, the
    ///   historical default. Identical wire shape to [`Self::envelope`].
    /// - [`EnvelopeMode::Position`] — a compact [`PositionEnvelope`] carrying
    ///   only `status` and tick `position`; omits `active_probes`,
    ///   `checkpoints`, and the rest of [`SessionState`].
    /// - [`EnvelopeMode::None`] — the bare `data` payload, with no envelope
    ///   wrapper at all.
    ///
    /// The return type is [`serde_json::Value`] because the three modes have
    /// genuinely different wire shapes; callers hand the result straight to
    /// `serialize_envelope`.
    pub fn envelope_with_mode<T: serde::Serialize>(
        &self,
        mode: EnvelopeMode,
        data: T,
    ) -> serde_json::Value {
        match mode {
            EnvelopeMode::Full => serde_json::to_value(self.envelope(data))
                .unwrap_or_else(|_| serde_json::Value::Null),
            EnvelopeMode::Position => {
                let position = PositionEnvelope {
                    status: self.state.status,
                    position: self.state.position.clone(),
                };
                serde_json::json!({
                    "state": position,
                    "data": data,
                })
            }
            EnvelopeMode::None => {
                serde_json::to_value(data).unwrap_or_else(|_| serde_json::Value::Null)
            }
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
            recovery_hint: Some(
                crate::dispatch::recovery_hint_for(ErrorCode::InvalidState).to_owned(),
            ),
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
                recovery_hint: Some(format!(
                    "Reconnect with protocol_version '{PROTOCOL_VERSION}'."
                )),
                context: Some(serde_json::json!({
                    "requested_version": req.protocol_version,
                    "supported_version": PROTOCOL_VERSION,
                })),
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
                recovery_hint: Some(
                    crate::dispatch::recovery_hint_for(ErrorCode::ModelAlreadyAttached).to_owned(),
                ),
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
                        recovery_hint: Some(
                            crate::dispatch::recovery_hint_for(ErrorCode::CompiledModel)
                                .to_owned(),
                        ),
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
                recovery_hint: Some(
                    crate::dispatch::recovery_hint_for(ErrorCode::UnsupportedModel).to_owned(),
                ),
                context: Some(serde_json::json!({
                    "attempted_family": req.model_family,
                    "supported_families": SUPPORTED_FAMILIES,
                })),
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
        // Retain the discoverable probe-point catalog so `rocket/discover`
        // can answer without a worker round-trip. Built from the worker's
        // reported model metadata (BEAD-0008: metadata reflects the backend).
        self.catalog = default_catalog(worker_model_type, num_layers, num_heads, hidden_dim);
        self.update_available_actions();

        self.envelope(AttachResponse {
            model_id,
            model_family: worker_model_type.to_owned(),
            num_layers,
            num_heads,
            hidden_dim,
            num_ranks: req.num_ranks,
            capabilities: Capabilities::phase1_defaults(),
            component_vocabulary: Vec::new(),
            module_tree: Vec::new(),
            alias_table: Vec::new(),
            tick_map: Vec::new(),
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
                recovery_hint: Some(
                    crate::dispatch::recovery_hint_for(ErrorCode::ModelNotAttached).to_owned(),
                ),
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
        // Discovery metadata and view registrations belong to the attached
        // model; drop them so a stale catalog cannot leak across attaches.
        self.catalog.clear();
        self.defined_views.clear();
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
                    recovery_hint: Some(
                        crate::dispatch::recovery_hint_for(ErrorCode::ModelNotAttached).to_owned(),
                    ),
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
            recovery_hint: Some(
                crate::dispatch::recovery_hint_for(ErrorCode::CapabilityNotSupported).to_owned(),
            ),
            context: Some(serde_json::json!({ "capability": cap })),
        })
    }

    /// Execute a step and build the response honouring `req.envelope`.
    ///
    /// The response wire shape depends on the client-requested
    /// [`EnvelopeMode`] (TCK `envelope-compactness.feature`): `Full` carries
    /// the complete [`SessionState`], `Position` only status + tick position,
    /// `None` only the bare [`StepResponse`] data payload. The return type is
    /// [`serde_json::Value`] so all three shapes flow through one path.
    #[allow(dead_code)]
    pub fn step(
        &mut self,
        req: &StepRequest,
        host_position: &TickPosition,
        _forward_complete: bool,
    ) -> Result<serde_json::Value, SessionError> {
        self.require_stopped("rocket/step")?;

        if req.direction == rocket_surgeon_protocol::types::StepDirection::Backward {
            return Err(self.capability_not_supported_error("supports_reverse_step"));
        }

        self.state.tick_id = Some(host_position.tick_id);
        self.state.position = Some(host_position.clone());
        self.update_available_actions();

        let data = StepResponse {
            ticks_executed: req.count,
            stopped_at: host_position.clone(),
        };
        Ok(self.envelope_with_mode(req.envelope, data))
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

    /// Discover probe-points matching a wildcard `pattern`.
    ///
    /// `pattern` is a 5-segment `family:rank:layer:component:event` string
    /// where any segment may be `*`. Returns every retained probe-point that
    /// matches, sorted by `(layer, canonical)` for deterministic output.
    ///
    /// On zero exact matches, `suggestions` is populated with nearest valid
    /// patterns — distinct canonical component names from the catalog that
    /// share at least the family/layer/event of the requested pattern — so an
    /// LLM client can self-correct a typo'd component name.
    pub fn discover(
        &self,
        pattern: &str,
    ) -> Result<ResponseEnvelope<DiscoverResponse>, SessionError> {
        self.require_stopped("rocket/discover")?;

        let Some(parsed) = DiscoverPattern::parse(pattern) else {
            return Err(SessionError::InvalidParams(ErrorData {
                error_code: ErrorCode::InvalidParams,
                numeric_code: Some(ErrorCode::InvalidParams.numeric_code()),
                severity: Severity::Recoverable,
                suggestion: format!(
                    "Discover pattern '{pattern}' is not a 5-segment \
                     'family:rank:layer:component:event' string"
                ),
                current_state: Some(self.state.status),
                valid_states: None,
                recovery_hint: Some(
                    "Use a pattern like 'llama:*:12:*:output'; segments may be '*'".to_owned(),
                ),
                context: None,
            }));
        };

        let mut matches: Vec<DiscoverMatch> = self
            .catalog
            .iter()
            .filter(|entry| parsed.matches(entry))
            .map(|entry| DiscoverMatch {
                canonical: entry.canonical.clone(),
                tensor_shape: entry.tensor_shape.clone(),
                aliases: entry.aliases.clone(),
            })
            .collect();
        matches.sort_by(|a, b| {
            a.tensor_shape
                .cmp(&b.tensor_shape)
                .then_with(|| a.canonical.cmp(&b.canonical))
        });

        let suggestions = if matches.is_empty() {
            self.suggest_patterns(pattern, &parsed)
        } else {
            Vec::new()
        };

        Ok(self.envelope(DiscoverResponse {
            matches,
            suggestions,
        }))
    }

    /// Build "nearest valid pattern" suggestions for a discover query that
    /// returned no matches. We relax the component segment to each distinct
    /// canonical name available under the requested family/event, rewriting
    /// the requested pattern's component slot.
    ///
    /// The layer constraint is first applied, then dropped as a fallback: a
    /// near-miss on a layer that does not exist (e.g. `llama:*:12:...` on a
    /// 7-layer model) still yields useful component-name corrections.
    fn suggest_patterns(&self, pattern: &str, parsed: &DiscoverPattern<'_>) -> Vec<String> {
        let segs: Vec<&str> = pattern.split(':').collect();
        // `parse` already guaranteed exactly 5 non-empty segments.
        debug_assert_eq!(segs.len(), 5);

        let collect_canonicals = |respect_layer: bool| -> Vec<String> {
            let mut canonicals: Vec<&str> = self
                .catalog
                .iter()
                .filter(|e| {
                    parsed.family.matches(&e.family)
                        && (!respect_layer || parsed.layer.matches(&e.layer.to_string()))
                        && parsed.event.matches(&e.event)
                })
                .map(|e| e.canonical.as_str())
                .collect();
            canonicals.sort_unstable();
            canonicals.dedup();
            canonicals
                .into_iter()
                .map(|canonical| {
                    format!(
                        "{}:{}:{}:{}:{}",
                        segs[0], segs[1], segs[2], canonical, segs[4]
                    )
                })
                .collect()
        };

        let strict = collect_canonicals(true);
        if strict.is_empty() {
            // Layer is out of range — relax it so component typos still get
            // corrected.
            collect_canonicals(false)
        } else {
            strict
        }
    }

    /// Register a named view specification (`rocket/view.define`).
    ///
    /// Stores `spec` under `name` so a later `rocket/view` could resolve a
    /// user-defined view by name. Redefining an existing name overwrites the
    /// prior spec and is reported as `registered: true` (idempotent upsert).
    pub fn define_view(
        &mut self,
        name: &str,
        spec: serde_json::Value,
    ) -> Result<ResponseEnvelope<ViewDefineResponse>, SessionError> {
        self.require_stopped("rocket/view.define")?;

        if name.trim().is_empty() {
            return Err(SessionError::InvalidParams(ErrorData {
                error_code: ErrorCode::InvalidParams,
                numeric_code: Some(ErrorCode::InvalidParams.numeric_code()),
                severity: Severity::Recoverable,
                suggestion: "View name must be a non-empty string".to_owned(),
                current_state: Some(self.state.status),
                valid_states: None,
                recovery_hint: Some(
                    "Provide a 'name' field identifying the view, e.g. \"my_view\"".to_owned(),
                ),
                context: None,
            }));
        }

        self.defined_views.insert(name.to_owned(), spec);

        Ok(self.envelope(ViewDefineResponse {
            name: name.to_owned(),
            registered: true,
        }))
    }

    /// Look up a previously-defined view spec by name. Used by `rocket/view`
    /// to resolve user-defined views.
    #[allow(dead_code)]
    pub fn defined_view(&self, name: &str) -> Option<&serde_json::Value> {
        self.defined_views.get(name)
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
            protocol_version: "0.3.0".to_owned(),
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
        assert_eq!(caps.protocol_version, "0.3.0");
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
            envelope: Default::default(),
            run_to: None,
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
        assert_eq!(envelope["state"]["status"], "stopped");
        let data = &envelope["data"];
        assert_eq!(data["ticks_executed"], 1);
        assert_eq!(data["stopped_at"]["component"], "q_proj");
    }

    #[test]
    fn step_from_initialized_returns_error() {
        let mut session = initialized_session();
        let req = StepRequest {
            direction: StepDirection::Forward,
            count: 1,
            granularity: None,
            envelope: Default::default(),
            run_to: None,
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
            envelope: Default::default(),
            run_to: None,
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
            envelope: Default::default(),
            run_to: None,
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
            envelope: Default::default(),
            run_to: None,
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
            envelope: Default::default(),
            run_to: None,
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
        assert_eq!(envelope["state"]["session_id"].as_str().unwrap().len(), 36);
        assert!(envelope["state"]["model_id"].is_string());
        assert_eq!(envelope["state"]["status"], "stopped");
        let actions = envelope["state"]["available_actions"].as_array().unwrap();
        assert!(actions.iter().any(|a| a == "step"));
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
            envelope: Default::default(),
            run_to: None,
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

    // --- discover tests ---

    #[test]
    fn discover_wildcard_returns_matching_points() {
        let session = stopped_session();
        // model_family in test_attach is "llama"; layer 5 exists (7 layers).
        let envelope = session.discover("llama:*:5:*:output").unwrap();
        let data = envelope.data.as_ref().unwrap();
        assert!(
            !data.matches.is_empty(),
            "wildcard component should match layer-5 points"
        );
        for m in &data.matches {
            assert!(!m.canonical.is_empty());
            assert!(!m.tensor_shape.is_empty());
            assert!(!m.aliases.is_empty(), "every match carries aliases");
        }
        assert!(
            data.suggestions.is_empty(),
            "exact matches => no suggestions"
        );
    }

    #[test]
    fn discover_specific_component_matches_one_per_layer() {
        let session = stopped_session();
        // attn.q_proj exists once per layer => exactly TEST_NUM_LAYERS matches.
        let envelope = session.discover("llama:*:*:attn.q_proj:output").unwrap();
        let data = envelope.data.as_ref().unwrap();
        assert_eq!(data.matches.len(), TEST_NUM_LAYERS as usize);
        for m in &data.matches {
            assert_eq!(m.canonical, "attn.q_proj");
        }
    }

    #[test]
    fn discover_no_match_returns_suggestions() {
        let session = stopped_session();
        // out_proj is not a real canonical name (it is o_proj) => 0 matches.
        let envelope = session.discover("llama:*:5:attn.out_proj:output").unwrap();
        let data = envelope.data.as_ref().unwrap();
        assert_eq!(data.matches.len(), 0);
        assert!(
            !data.suggestions.is_empty(),
            "a near-miss component should produce suggestions"
        );
        // Suggestions rewrite only the component slot, preserving structure.
        for s in &data.suggestions {
            assert_eq!(s.split(':').count(), 5);
            assert!(s.starts_with("llama:*:5:"));
            assert!(s.ends_with(":output"));
        }
        // The real component (attn.o_proj) is among the suggestions.
        assert!(
            data.suggestions
                .iter()
                .any(|s| s == "llama:*:5:attn.o_proj:output")
        );
    }

    #[test]
    fn discover_family_mismatch_returns_no_matches() {
        let session = stopped_session();
        let envelope = session.discover("mixtral:*:5:*:output").unwrap();
        let data = envelope.data.as_ref().unwrap();
        assert_eq!(data.matches.len(), 0);
    }

    #[test]
    fn discover_invalid_pattern_returns_invalid_params_with_recovery_hint() {
        let session = stopped_session();
        // 6 segments — a probe-point string, not a discover pattern.
        let err = session
            .discover("llama:0:5:attn.q_proj:0:output")
            .unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::InvalidParams);
        assert!(
            data.recovery_hint.is_some(),
            "v0.3.0 requires recovery_hint"
        );
    }

    #[test]
    fn discover_when_not_attached_returns_model_not_attached() {
        let session = initialized_session();
        let err = session.discover("llama:*:*:*:output").unwrap_err();
        assert_eq!(err.error_data().error_code, ErrorCode::ModelNotAttached);
    }

    #[test]
    fn discover_catalog_cleared_after_detach() {
        let mut session = stopped_session();
        assert!(!session.catalog.is_empty());
        session.detach().unwrap();
        assert!(session.catalog.is_empty(), "detach drops the catalog");
    }

    #[test]
    fn discover_results_are_deterministically_sorted() {
        let session = stopped_session();
        let first = session.discover("llama:*:3:*:output").unwrap();
        let second = session.discover("llama:*:3:*:output").unwrap();
        assert_eq!(
            first.data.as_ref().unwrap().matches,
            second.data.as_ref().unwrap().matches
        );
    }

    // --- view.define tests ---

    #[test]
    fn define_view_registers_spec() {
        let mut session = stopped_session();
        let spec = serde_json::json!({"reduce": "l2_norm", "over": "residual"});
        let envelope = session.define_view("my_view", spec.clone()).unwrap();
        let data = envelope.data.as_ref().unwrap();
        assert_eq!(data.name, "my_view");
        assert!(data.registered);
        assert_eq!(session.defined_view("my_view"), Some(&spec));
    }

    #[test]
    fn define_view_redefine_overwrites_spec() {
        let mut session = stopped_session();
        session
            .define_view("v", serde_json::json!({"version": 1}))
            .unwrap();
        let envelope = session
            .define_view("v", serde_json::json!({"version": 2}))
            .unwrap();
        assert!(envelope.data.as_ref().unwrap().registered);
        assert_eq!(
            session.defined_view("v"),
            Some(&serde_json::json!({"version": 2}))
        );
    }

    #[test]
    fn define_view_empty_name_returns_invalid_params_with_recovery_hint() {
        let mut session = stopped_session();
        let err = session
            .define_view("   ", serde_json::json!({}))
            .unwrap_err();
        let data = err.error_data();
        assert_eq!(data.error_code, ErrorCode::InvalidParams);
        assert!(
            data.recovery_hint.is_some(),
            "v0.3.0 requires recovery_hint"
        );
    }

    #[test]
    fn define_view_when_not_attached_returns_model_not_attached() {
        let mut session = initialized_session();
        let err = session.define_view("v", serde_json::json!({})).unwrap_err();
        assert_eq!(err.error_data().error_code, ErrorCode::ModelNotAttached);
    }

    #[test]
    fn define_view_registry_cleared_after_detach() {
        let mut session = stopped_session();
        session
            .define_view("v", serde_json::json!({"k": "v"}))
            .unwrap();
        session.detach().unwrap();
        assert!(
            session.defined_views.is_empty(),
            "detach drops view registry"
        );
    }

    #[test]
    fn unknown_defined_view_returns_none() {
        let session = stopped_session();
        assert!(session.defined_view("does_not_exist").is_none());
    }

    // --- EnvelopeMode tests (TCK envelope-compactness.feature) ---

    fn step_req(envelope: EnvelopeMode) -> StepRequest {
        StepRequest {
            direction: StepDirection::Forward,
            count: 1,
            granularity: Some(TickGranularity::Component),
            envelope,
            run_to: None,
        }
    }

    fn step_pos() -> TickPosition {
        TickPosition {
            tick_id: 9,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer: 4,
            component: "attn.q_proj".to_owned(),
            event: TickEvent::Output,
            replay_of: None,
            phase: Phase::Decode,
            token_position: None,
            clock: None,
        }
    }

    #[test]
    fn step_full_envelope_includes_complete_session_state() {
        let mut session = stopped_session();
        let value = session
            .step(&step_req(EnvelopeMode::Full), &step_pos(), false)
            .unwrap();
        let state = &value["state"];
        // The full envelope carries the entire SessionState.
        assert!(state.get("session_id").is_some());
        assert!(state.get("status").is_some());
        assert!(state.get("active_probes").is_some());
        assert!(state.get("checkpoints").is_some());
        assert!(state.get("available_actions").is_some());
        assert!(value.get("data").is_some());
    }

    #[test]
    fn step_default_envelope_is_full() {
        // No explicit envelope field -> EnvelopeMode default -> Full.
        let mut session = stopped_session();
        let value = session
            .step(&step_req(EnvelopeMode::default()), &step_pos(), false)
            .unwrap();
        assert!(value["state"].get("active_probes").is_some());
        assert!(value["state"].get("checkpoints").is_some());
    }

    #[test]
    fn step_position_envelope_has_status_and_position_only() {
        let mut session = stopped_session();
        let value = session
            .step(&step_req(EnvelopeMode::Position), &step_pos(), false)
            .unwrap();
        let state = &value["state"];
        // Position envelope keeps status and tick position...
        assert_eq!(state["status"], "stopped");
        assert_eq!(state["position"]["layer"], 4);
        assert_eq!(state["position"]["component"], "attn.q_proj");
        // ...but omits the heavier SessionState fields.
        assert!(state.get("active_probes").is_none());
        assert!(state.get("checkpoints").is_none());
        assert!(state.get("session_id").is_none());
        assert!(state.get("available_actions").is_none());
        // The data payload is still present.
        assert_eq!(value["data"]["ticks_executed"], 1);
    }

    #[test]
    fn step_none_envelope_is_data_payload_only() {
        let mut session = stopped_session();
        let value = session
            .step(&step_req(EnvelopeMode::None), &step_pos(), false)
            .unwrap();
        // No envelope wrapper at all: the StepResponse fields are top-level.
        assert!(value.get("state").is_none());
        assert!(value.get("data").is_none());
        assert_eq!(value["ticks_executed"], 1);
        assert_eq!(value["stopped_at"]["component"], "attn.q_proj");
    }

    #[test]
    fn envelope_with_mode_position_reflects_no_position_before_step() {
        // Position envelope on a freshly-stopped session: status is set,
        // position is still null (no step taken yet).
        let session = stopped_session();
        let value = session.envelope_with_mode(EnvelopeMode::Position, serde_json::json!({"x": 1}));
        assert_eq!(value["state"]["status"], "stopped");
        assert!(value["state"]["position"].is_null());
        assert_eq!(value["data"]["x"], 1);
    }

    // --- SubscribeFilter session-state tests ---

    #[test]
    fn event_filter_defaults_to_none() {
        let session = stopped_session();
        assert!(session.event_filter().is_none());
    }

    #[test]
    fn set_event_filter_stores_and_clears() {
        let mut session = stopped_session();
        let filter = SubscribeFilter {
            events: Some(vec![
                rocket_surgeon_protocol::messages::EventType::TickStopped,
            ]),
            layers: Some(vec![10, 11, 12]),
            components: None,
        };
        session.set_event_filter(Some(filter.clone()));
        assert_eq!(session.event_filter(), Some(&filter));

        session.set_event_filter(None);
        assert!(session.event_filter().is_none());
    }
}
