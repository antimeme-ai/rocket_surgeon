use rocket_surgeon_protocol::errors::{ErrorCode, ErrorData, Severity};
use rocket_surgeon_protocol::jsonrpc::{
    JSONRPC_VERSION, Notification, RawMessage, Request, RequestId, Response, RpcError,
};
use rocket_surgeon_protocol::messages::{
    AttachRequest, AttachResponse, CheckpointRequest, CheckpointResponse, CreateCheckpointTier,
    DetachRequest, DetachResponse, Divergence, ErrorEvent, EventType, HostViewRequest,
    HostViewResponse, InitializeRequest, InitializeResponse, InspectDetail, InspectRequest,
    InspectResponse, InterveneRequest, MemoryUsage, ProbeFiredEvent, ProbeRequest, ProbeResponse,
    ReplayDivergenceEvent, ReplayRequest, ReplayResponse, ReplayStopAt, StatusRequest,
    StatusResponse, StepRequest, StepResponse, SubscribeRequest, SubscribeResponse,
    TickHeartbeatEvent, TickStoppedEvent, UnsubscribeRequest, UnsubscribeResponse, ViewRequest,
    ViewResponse,
};
use rocket_surgeon_protocol::types::{
    ActionName, BuiltInView, Capabilities, CheckpointRef, CheckpointTier, CompositionMode, DType,
    ExecutionMode, GranularityScope, HeadGranularity, Histogram, InterventionParams,
    InterventionRecipe, InterventionType, Parallelism, Placement, PlacementType, ProbeAction,
    ProbeConfig, ProbeDefinition, ResponseEnvelope, SessionState, ShardingInfo, Status,
    StepDirection, TensorHandle, TensorStats, TensorSummary, TickEvent, TickGranularity,
    TickPosition, TopKEntry, Transport, WireFormat,
};
use serde_json::json;

fn roundtrip<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(
    val: &T,
) {
    let json = serde_json::to_string(val).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(*val, back);
}

fn sample_tick_position() -> TickPosition {
    TickPosition {
        tick_id: 42,
        direction: StepDirection::Forward,
        rank: None,
        layer: 12,
        component: "attn.o_proj".to_owned(),
        event: TickEvent::Output,
        replay_of: None,
    }
}

fn sample_session_state() -> SessionState {
    SessionState {
        session_id: "550e8400-e29b-41d4-a716-446655440000".to_owned(),
        model_id: Some("abc123".to_owned()),
        status: Status::Stopped,
        position: Some(sample_tick_position()),
        tick_id: Some(42),
        active_probes: vec!["probe-1".to_owned()],
        checkpoints: vec![],
        available_actions: vec![ActionName::Step, ActionName::Inspect],
    }
}

fn sample_tensor_summary() -> TensorSummary {
    TensorSummary {
        tensor_id: "a".repeat(64),
        shape: vec![1, 32, 4096],
        dtype: DType::Float16,
        device: "cuda:0".to_owned(),
        sharding: None,
        stats: TensorStats {
            mean: 0.001,
            std: 0.5,
            min: -3.2,
            max: 3.1,
            abs_max: 3.2,
            sparsity: 0.01,
            l2_norm: 12.5,
            histogram: Histogram {
                bins: 10,
                edges: vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                counts: vec![5, 10, 100, 200, 100, 10],
            },
        },
        top_k: vec![TopKEntry {
            index: vec![0, 5, 2048],
            value: 3.1,
        }],
    }
}

// ===== types.rs round-trips =====

#[test]
fn status_enum_serde() {
    let json = serde_json::to_string(&Status::Stopped).unwrap();
    assert_eq!(json, r#""stopped""#);

    let json = serde_json::to_string(&Status::Uninitialized).unwrap();
    assert_eq!(json, r#""uninitialized""#);

    let back: Status = serde_json::from_str(r#""stepping""#).unwrap();
    assert_eq!(back, Status::Stepping);
}

#[test]
fn action_name_serde() {
    let json = serde_json::to_string(&ActionName::Subscribe).unwrap();
    assert_eq!(json, r#""subscribe""#);
}

#[test]
fn tick_position_roundtrip() {
    let pos = sample_tick_position();
    roundtrip(&pos);

    let json = serde_json::to_value(&pos).unwrap();
    assert_eq!(json["direction"], "forward");
    assert_eq!(json["event"], "output");
    assert!(json.get("replay_of").is_none());
}

#[test]
fn tick_position_with_replay() {
    let pos = TickPosition {
        replay_of: Some(10),
        ..sample_tick_position()
    };
    let json = serde_json::to_value(&pos).unwrap();
    assert_eq!(json["replay_of"], 10);
    roundtrip(&pos);
}

#[test]
fn step_direction_serde() {
    assert_eq!(
        serde_json::to_string(&StepDirection::Forward).unwrap(),
        r#""forward""#
    );
    assert_eq!(
        serde_json::to_string(&StepDirection::Backward).unwrap(),
        r#""backward""#
    );
}

#[test]
fn tick_granularity_serde() {
    assert_eq!(
        serde_json::to_string(&TickGranularity::RouterPreTopk).unwrap(),
        r#""router_pre_topk""#
    );
    assert_eq!(
        serde_json::to_string(&TickGranularity::MoeLayer).unwrap(),
        r#""moe_layer""#
    );
}

#[test]
fn dtype_serde() {
    assert_eq!(
        serde_json::to_string(&DType::Bfloat16).unwrap(),
        r#""bfloat16""#
    );
    assert_eq!(
        serde_json::to_string(&DType::Float32).unwrap(),
        r#""float32""#
    );
}

#[test]
fn tensor_summary_roundtrip() {
    roundtrip(&sample_tensor_summary());
}

#[test]
fn sharding_info_roundtrip() {
    let info = ShardingInfo {
        mesh: "tp".to_owned(),
        placements: vec![
            Placement {
                placement_type: PlacementType::Shard,
                dim: Some(0),
            },
            Placement {
                placement_type: PlacementType::Replicate,
                dim: None,
            },
        ],
        local_shape: vec![1, 16, 4096],
        global_shape: vec![1, 32, 4096],
    };
    roundtrip(&info);

    let json = serde_json::to_value(&info).unwrap();
    assert_eq!(json["placements"][0]["type"], "Shard");
    assert_eq!(json["placements"][0]["dim"], 0);
    assert!(json["placements"][1].get("dim").is_none());
}

#[test]
fn tensor_handle_roundtrip() {
    let handle = TensorHandle {
        tensor_id: "b".repeat(64),
        shape: vec![4096],
        dtype: DType::Float16,
    };
    roundtrip(&handle);
}

#[test]
fn probe_definition_roundtrip() {
    let probe = ProbeDefinition {
        id: "p1".to_owned(),
        point: "llama:*:12:attn.o_proj:output".to_owned(),
        action: ProbeAction::Capture,
        config: Some(ProbeConfig {
            summary: true,
            capture_tensor: false,
            filter: Some("norm > 50.0".to_owned()),
            aggregate_fn: None,
            assertion: None,
            intervention: None,
        }),
        enabled: true,
        priority: 0,
    };
    roundtrip(&probe);
}

#[test]
fn probe_action_serde() {
    assert_eq!(
        serde_json::to_string(&ProbeAction::Aggregate).unwrap(),
        r#""aggregate""#
    );
}

#[test]
fn intervention_recipe_roundtrip() {
    let recipe = InterventionRecipe {
        id: "int-1".to_owned(),
        intervention_type: InterventionType::Scale,
        target: "llama:0:12:attn.o_proj:output".to_owned(),
        params: InterventionParams::Scale { factor: 0.5 },
        condition: None,
        priority: 0,
        mode: CompositionMode::Additive,
    };
    roundtrip(&recipe);

    let json = serde_json::to_value(&recipe).unwrap();
    assert_eq!(json["type"], "scale");
    assert_eq!(json["mode"], "additive");
}

#[test]
fn intervention_params_variants() {
    roundtrip(&InterventionParams::Ablate {});
    roundtrip(&InterventionParams::Scale { factor: 2.0 });
    roundtrip(&InterventionParams::Clamp {
        min: -1.0,
        max: 1.0,
    });
    roundtrip(&InterventionParams::Patch {
        source_tensor_id: "x".repeat(64),
    });
    roundtrip(&InterventionParams::RouteOverride {
        token: 5,
        experts: vec![0, 2, 4],
    });
}

#[test]
fn capabilities_phase1_defaults_roundtrip() {
    let caps = Capabilities::phase1_defaults();
    roundtrip(&caps);

    let json = serde_json::to_value(&caps).unwrap();
    assert_eq!(json["protocol_version"], "0.1.0");
    assert_eq!(json["supports_moe"], false);
    assert_eq!(json["execution_mode"], "eager");
    assert_eq!(json["parallelism"], "single_gpu");
    assert_eq!(json["head_granularity"], "unavailable");
    assert!(json.get("model_family").is_none());
    assert!(json.get("num_experts").is_none());
}

#[test]
fn built_in_view_serde() {
    assert_eq!(
        serde_json::to_string(&BuiltInView::ResidualStreamNorm).unwrap(),
        r#""residual_stream_norm""#
    );
    assert_eq!(
        serde_json::to_string(&BuiltInView::SaeActivation).unwrap(),
        r#""sae_activation""#
    );
}

#[test]
fn checkpoint_ref_roundtrip() {
    let cp = CheckpointRef {
        checkpoint_id: "cp-1".to_owned(),
        tick_id: 42,
        layer_idx: 12,
        tier: CheckpointTier::Activation,
        bookmark: Some("before_ablation".to_owned()),
        created_at: "2026-05-14T12:00:00Z".to_owned(),
    };
    roundtrip(&cp);
}

#[test]
fn checkpoint_tier_serde() {
    assert_eq!(
        serde_json::to_string(&CheckpointTier::FullSnapshot).unwrap(),
        r#""full_snapshot""#
    );
}

#[test]
fn granularity_scope_roundtrip() {
    let scope = GranularityScope {
        match_pattern: "layers[12]".to_owned(),
        granularity: TickGranularity::Component,
    };
    let json = serde_json::to_value(&scope).unwrap();
    assert_eq!(json["match"], "layers[12]");
    roundtrip(&scope);
}

#[test]
fn response_envelope_roundtrip() {
    let env = ResponseEnvelope {
        state: sample_session_state(),
        data: Some(StepResponse {
            ticks_executed: 1,
            stopped_at: sample_tick_position(),
        }),
    };
    roundtrip(&env);
}

#[test]
fn session_state_roundtrip() {
    roundtrip(&sample_session_state());
}

// ===== errors.rs =====

#[test]
fn error_code_serde() {
    assert_eq!(
        serde_json::to_string(&ErrorCode::InvalidState).unwrap(),
        r#""INVALID_STATE""#
    );
    assert_eq!(
        serde_json::to_string(&ErrorCode::GpuOom).unwrap(),
        r#""GPU_OOM""#
    );
    assert_eq!(
        serde_json::to_string(&ErrorCode::InvalidParams).unwrap(),
        r#""INVALID_PARAMS""#
    );
}

#[test]
fn error_code_numeric() {
    assert_eq!(ErrorCode::InvalidState.numeric_code(), -32001);
    assert_eq!(ErrorCode::InvalidParams.numeric_code(), -32602);
    assert_eq!(ErrorCode::ModelAlreadyAttached.numeric_code(), -32017);
}

#[test]
fn error_code_severity() {
    assert_eq!(ErrorCode::HostError.severity(), Severity::Fatal);
    assert_eq!(ErrorCode::GpuOom.severity(), Severity::Fatal);
    assert_eq!(ErrorCode::NcclTimeout.severity(), Severity::Fatal);
    assert_eq!(ErrorCode::InvalidState.severity(), Severity::Recoverable);
    assert_eq!(ErrorCode::TensorNotFound.severity(), Severity::Recoverable);
}

#[test]
fn error_data_roundtrip() {
    let err = ErrorData::new(ErrorCode::InvalidState, "Call initialize first");
    roundtrip(&err);

    let json = serde_json::to_value(&err).unwrap();
    assert_eq!(json["error_code"], "INVALID_STATE");
    assert_eq!(json["numeric_code"], -32001);
    assert_eq!(json["severity"], "recoverable");
    assert!(json.get("current_state").is_none());
}

// ===== jsonrpc.rs =====

#[test]
fn request_id_number() {
    let id = RequestId::Number(42);
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, "42");
    roundtrip(&id);
}

#[test]
fn request_id_string() {
    let id = RequestId::String("req-1".to_owned());
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, r#""req-1""#);
    roundtrip(&id);
}

#[test]
fn jsonrpc_request_roundtrip() {
    let req = Request::new(
        RequestId::Number(1),
        "initialize",
        json!({"client_name": "test", "protocol_version": "0.1.0"}),
    );
    assert_eq!(req.jsonrpc, JSONRPC_VERSION);
    roundtrip(&req);
}

#[test]
fn jsonrpc_request_null_params() {
    let req = Request::new(RequestId::Number(1), "rocket/status", json!(null));
    let json = serde_json::to_value(&req).unwrap();
    assert!(json.get("params").is_none());
}

#[test]
fn jsonrpc_response_success() {
    let resp = Response::success(RequestId::Number(1), json!({"status": "ok"}));
    assert!(resp.result.is_some());
    assert!(resp.error.is_none());
    roundtrip(&resp);
}

#[test]
fn jsonrpc_response_error() {
    let err = RpcError::from_error_data(ErrorData::new(
        ErrorCode::ModelNotAttached,
        "Attach a model first",
    ));
    let resp = Response::error(RequestId::Number(1), err);
    assert!(resp.result.is_none());
    assert!(resp.error.is_some());

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["error"]["code"], -32004);
    roundtrip(&resp);
}

#[test]
fn jsonrpc_notification_roundtrip() {
    let notif = Notification::new("tick.stopped", json!({"tick_id": 42}));
    assert_eq!(notif.jsonrpc, JSONRPC_VERSION);
    roundtrip(&notif);
}

#[test]
fn raw_message_request() {
    let raw: RawMessage = serde_json::from_value(json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"client_name": "test", "protocol_version": "0.1.0"}
    }))
    .unwrap();
    assert!(!raw.is_notification());
    assert!(raw.into_request().is_some());
}

#[test]
fn raw_message_notification() {
    let raw: RawMessage = serde_json::from_value(json!({
        "jsonrpc": "2.0",
        "method": "tick.stopped",
        "params": {"tick_id": 42}
    }))
    .unwrap();
    assert!(raw.is_notification());
    assert!(raw.into_notification().is_some());
}

// ===== messages.rs =====

#[test]
fn initialize_request_roundtrip() {
    let req = InitializeRequest {
        client_name: "rocket-tui".to_owned(),
        protocol_version: "0.1.0".to_owned(),
        client_version: Some("1.0.0".to_owned()),
        client_capabilities: None,
    };
    roundtrip(&req);
}

#[test]
fn initialize_response_roundtrip() {
    let resp = InitializeResponse {
        capabilities: Capabilities::phase1_defaults(),
    };
    roundtrip(&resp);
}

#[test]
fn attach_request_defaults() {
    let json = json!({
        "model_path": "/models/llama-3-8b",
        "model_family": "llama"
    });
    let req: AttachRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.device, "cuda:0");
    assert_eq!(req.num_ranks, 1);
    assert!(req.dtype.is_none());
    assert!(req.config.is_none());
}

#[test]
fn attach_response_roundtrip() {
    let resp = AttachResponse {
        model_id: "m".repeat(64),
        model_family: "llama".to_owned(),
        num_layers: 32,
        num_heads: 32,
        hidden_dim: 4096,
        num_ranks: 1,
        capabilities: Capabilities::phase1_defaults(),
    };
    roundtrip(&resp);
}

#[test]
fn detach_roundtrip() {
    roundtrip(&DetachRequest {});
    roundtrip(&DetachResponse {
        detached_model_id: "d".repeat(64),
    });
}

#[test]
fn step_request_roundtrip() {
    let req = StepRequest {
        direction: StepDirection::Forward,
        count: 5,
        granularity: Some(TickGranularity::Component),
    };
    roundtrip(&req);
}

#[test]
fn step_response_roundtrip() {
    let resp = StepResponse {
        ticks_executed: 5,
        stopped_at: sample_tick_position(),
    };
    roundtrip(&resp);
}

#[test]
fn inspect_detail_default() {
    assert_eq!(InspectDetail::default(), InspectDetail::Summary);
}

#[test]
fn inspect_detail_serde() {
    assert_eq!(
        serde_json::to_string(&InspectDetail::Slice).unwrap(),
        r#""slice""#
    );
}

#[test]
fn inspect_request_roundtrip() {
    let req = InspectRequest {
        target: "llama:0:12:attn.o_proj:output".to_owned(),
        detail: InspectDetail::Slice,
        slices: Some(vec![[0, 10], [20, 30]]),
        format: Some(DType::Float32),
        view: None,
    };
    roundtrip(&req);
}

#[test]
fn inspect_response_roundtrip() {
    let resp = InspectResponse {
        tensors: vec![sample_tensor_summary()],
        view_result: Some(json!({"type": "residual_stream_norm", "values": [1.0, 2.0]})),
        slice_data: Some("AAAA".to_owned()),
    };
    roundtrip(&resp);
}

#[test]
fn intervene_request_set_tagged() {
    let req = InterveneRequest::Set {
        recipe: InterventionRecipe {
            id: "int-1".to_owned(),
            intervention_type: InterventionType::Ablate,
            target: "llama:0:12:mlp:output".to_owned(),
            params: InterventionParams::Ablate {},
            condition: None,
            priority: 0,
            mode: CompositionMode::Additive,
        },
    };
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["action"], "set");
    assert!(json.get("recipe").is_some());
    roundtrip(&req);
}

#[test]
fn intervene_request_clear_tagged() {
    let req = InterveneRequest::Clear {
        intervention_id: "int-1".to_owned(),
    };
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["action"], "clear");
    roundtrip(&req);
}

#[test]
fn intervene_request_list_tagged() {
    let req = InterveneRequest::List {};
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["action"], "list");
    roundtrip(&req);
}

#[test]
fn probe_request_define_tagged() {
    let req = ProbeRequest::Define {
        probe: Box::new(ProbeDefinition {
            id: "p1".to_owned(),
            point: "llama:*:*:*:output".to_owned(),
            action: ProbeAction::Capture,
            config: None,
            enabled: true,
            priority: 0,
        }),
    };
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["action"], "define");
    roundtrip(&req);
}

#[test]
fn probe_request_enable_tagged() {
    let req = ProbeRequest::Enable {
        probe_id: "p1".to_owned(),
    };
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["action"], "enable");
    roundtrip(&req);
}

#[test]
fn probe_request_set_granularity_tagged() {
    let req = ProbeRequest::SetGranularity {
        scopes: vec![GranularityScope {
            match_pattern: "layers[12]".to_owned(),
            granularity: TickGranularity::Component,
        }],
    };
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["action"], "set_granularity");
    roundtrip(&req);
}

#[test]
fn probe_response_roundtrip() {
    let resp = ProbeResponse {
        probes: vec![],
        probe_id: Some("p1".to_owned()),
    };
    roundtrip(&resp);
}

#[test]
fn checkpoint_request_create_tagged() {
    let req = CheckpointRequest::Create {
        tier: Some(CreateCheckpointTier::Activation),
    };
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["action"], "create");
    assert_eq!(json["tier"], "activation");
    roundtrip(&req);
}

#[test]
fn checkpoint_request_bookmark_tagged() {
    let req = CheckpointRequest::Bookmark {
        tick_id: 42,
        name: "before_ablation".to_owned(),
    };
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["action"], "bookmark");
    roundtrip(&req);
}

#[test]
fn checkpoint_response_roundtrip() {
    let resp = CheckpointResponse {
        checkpoints: vec![],
        checkpoint_id: Some("cp-1".to_owned()),
        restored_to: None,
    };
    roundtrip(&resp);
}

#[test]
fn replay_request_roundtrip() {
    let req = ReplayRequest {
        from_checkpoint: "cp-1".to_owned(),
        interventions: None,
        stop_at: Some(ReplayStopAt {
            layer: 12,
            component: "attn.o_proj".to_owned(),
        }),
        verify: true,
    };
    roundtrip(&req);
}

#[test]
fn replay_request_verify_default() {
    let json = json!({
        "from_checkpoint": "cp-1"
    });
    let req: ReplayRequest = serde_json::from_value(json).unwrap();
    assert!(req.verify);
}

#[test]
fn replay_response_roundtrip() {
    let resp = ReplayResponse {
        ticks_replayed: 10,
        stopped_at: sample_tick_position(),
        divergences: vec![Divergence {
            tick_id: 50,
            original_tick_id: 42,
            probe_point: "llama:0:12:attn.o_proj:output".to_owned(),
            cosine_similarity: 0.9998,
            max_relative_error: 0.001,
            message: "Minor divergence at layer 12".to_owned(),
        }],
        verified: true,
    };
    roundtrip(&resp);
}

#[test]
fn status_roundtrip() {
    roundtrip(&StatusRequest {});
    let resp = StatusResponse {
        uptime_seconds: 120.5,
        connected_clients: 2,
        memory_usage: MemoryUsage {
            gpu_mb: 8192.0,
            cpu_mb: 4096.0,
        },
        pending_interventions: 1,
        trace_events_recorded: 5000,
    };
    roundtrip(&resp);
}

#[test]
fn event_type_serde() {
    assert_eq!(
        serde_json::to_string(&EventType::TickStopped).unwrap(),
        r#""tick.stopped""#
    );
    assert_eq!(
        serde_json::to_string(&EventType::ProbeFired).unwrap(),
        r#""probe.fired""#
    );
    assert_eq!(
        serde_json::to_string(&EventType::ReplayDivergence).unwrap(),
        r#""replay.divergence""#
    );
}

#[test]
fn subscribe_request_roundtrip() {
    let req = SubscribeRequest {};
    roundtrip(&req);
}

#[test]
fn subscribe_response_roundtrip() {
    let resp = SubscribeResponse {
        available_events: vec![
            EventType::TickStopped,
            EventType::TickHeartbeat,
            EventType::ProbeFired,
        ],
        status: Status::Stopped,
    };
    roundtrip(&resp);
}

#[test]
fn unsubscribe_request_roundtrip() {
    let req = UnsubscribeRequest {};
    roundtrip(&req);
}

#[test]
fn unsubscribe_response_roundtrip() {
    let resp = UnsubscribeResponse {
        status: Status::Stopped,
    };
    roundtrip(&resp);
}

// ===== View =====

#[test]
fn view_request_roundtrip() {
    let req = ViewRequest {
        view: BuiltInView::ResidualStreamNorm,
        params: None,
    };
    roundtrip(&req);
}

#[test]
fn view_request_with_params_roundtrip() {
    let req = ViewRequest {
        view: BuiltInView::AttentionPattern,
        params: Some(json!({"layer": 3, "head": 7})),
    };
    roundtrip(&req);
}

#[test]
fn view_response_roundtrip() {
    let resp = ViewResponse {
        view: BuiltInView::ResidualStreamNorm,
        data: json!({"norms": [0.42, 0.38], "num_layers": 2, "norm_type": "l2"}),
    };
    roundtrip(&resp);
}

#[test]
fn host_view_request_roundtrip() {
    let req = HostViewRequest {
        model_handle: 1,
        view: BuiltInView::AttentionPattern,
        params: Some(json!({"layer": 0})),
    };
    roundtrip(&req);
}

#[test]
fn host_view_response_roundtrip() {
    let resp = HostViewResponse {
        view: BuiltInView::AttentionPattern,
        data: json!({
            "layer": 0,
            "heads": [{"head": 0, "weights": [[0.5, 0.5]]}],
            "seq_len": 1
        }),
    };
    roundtrip(&resp);
}

#[test]
fn view_data_unavailable_error_code() {
    assert_eq!(ErrorCode::ViewDataUnavailable.numeric_code(), -32020);
    assert_eq!(
        ErrorCode::ViewDataUnavailable.severity(),
        Severity::Recoverable
    );
    let json = serde_json::to_string(&ErrorCode::ViewDataUnavailable).unwrap();
    assert_eq!(json, "\"VIEW_DATA_UNAVAILABLE\"");
}

// ===== Event notifications =====

#[test]
fn tick_stopped_event_roundtrip() {
    let evt = TickStoppedEvent {
        position: sample_tick_position(),
        state: Status::Stopped,
    };
    roundtrip(&evt);
}

#[test]
fn tick_heartbeat_event_roundtrip() {
    use rocket_surgeon_protocol::messages::{RankActivity, RankStatus};
    let evt = TickHeartbeatEvent {
        position: sample_tick_position(),
        uptime_seconds: 120.5,
        elapsed_stopped_sec: 5.2,
        per_rank_status: vec![RankStatus {
            rank: 0,
            status: RankActivity::Stopped,
            gpu_memory_used_mb: 8000.0,
            gpu_memory_total_mb: 16384.0,
        }],
    };
    roundtrip(&evt);
}

#[test]
fn probe_fired_event_roundtrip() {
    let evt = ProbeFiredEvent {
        probe_id: "p1".to_owned(),
        point: "llama:0:12:attn.o_proj:output".to_owned(),
        tick_id: 42,
        tensor_summary: Some(sample_tensor_summary()),
        action: ProbeAction::Capture,
        timestamp: "2026-05-14T12:00:00Z".to_owned(),
    };
    roundtrip(&evt);
}

#[test]
fn replay_divergence_event_roundtrip() {
    let evt = ReplayDivergenceEvent {
        tick_id: 50,
        original_tick_id: 42,
        probe_point: "llama:0:12:attn.o_proj:output".to_owned(),
        cosine_similarity: 0.9998,
        max_relative_error: 0.001,
        message: "Minor divergence".to_owned(),
    };
    roundtrip(&evt);
}

#[test]
fn error_event_roundtrip() {
    let evt = ErrorEvent {
        error_code: ErrorCode::GpuOom,
        message: "GPU out of memory".to_owned(),
        details: Some(json!({"device": "cuda:0", "requested_mb": 2048})),
        fatal: true,
    };
    roundtrip(&evt);
}

// ===== Cross-cutting: JSON field name verification =====

#[test]
fn session_state_field_names() {
    let state = sample_session_state();
    let json = serde_json::to_value(&state).unwrap();
    assert!(json.get("session_id").is_some());
    assert!(json.get("model_id").is_some());
    assert!(json.get("available_actions").is_some());
    assert!(json.get("active_probes").is_some());
}

#[test]
fn transport_serde() {
    assert_eq!(
        serde_json::to_string(&Transport::UnixSocket).unwrap(),
        r#""unix_socket""#
    );
}

#[test]
fn wire_format_serde() {
    assert_eq!(
        serde_json::to_string(&WireFormat::Protobuf).unwrap(),
        r#""protobuf""#
    );
}

#[test]
fn execution_mode_serde() {
    assert_eq!(
        serde_json::to_string(&ExecutionMode::Mixed).unwrap(),
        r#""mixed""#
    );
}

#[test]
fn parallelism_serde() {
    assert_eq!(
        serde_json::to_string(&Parallelism::TensorParallel).unwrap(),
        r#""tensor_parallel""#
    );
}

#[test]
fn head_granularity_serde() {
    assert_eq!(
        serde_json::to_string(&HeadGranularity::RequiresUnfused).unwrap(),
        r#""requires_unfused""#
    );
}

// ===== Default values =====

#[test]
fn probe_definition_defaults() {
    let json = json!({
        "id": "p1",
        "point": "llama:*:*:*:output",
        "action": "capture"
    });
    let probe: ProbeDefinition = serde_json::from_value(json).unwrap();
    assert!(probe.enabled);
    assert_eq!(probe.priority, 0);
}

#[test]
fn probe_config_defaults() {
    let json = json!({});
    let config: ProbeConfig = serde_json::from_value(json).unwrap();
    assert!(config.summary);
    assert!(!config.capture_tensor);
}

#[test]
fn intervention_recipe_defaults() {
    let json = json!({
        "id": "int-1",
        "type": "ablate",
        "target": "llama:0:12:mlp:output",
        "params": {}
    });
    let recipe: InterventionRecipe = serde_json::from_value(json).unwrap();
    assert_eq!(recipe.mode, CompositionMode::Additive);
    assert_eq!(recipe.priority, 0);
}

#[test]
fn inspect_request_detail_default() {
    let json = json!({
        "target": "llama:0:12:attn.o_proj:output"
    });
    let req: InspectRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.detail, InspectDetail::Summary);
}

// ===== Edge cases: untagged enum ordering =====

#[test]
fn intervention_params_ablate_from_empty_object() {
    let val: InterventionParams = serde_json::from_value(json!({})).unwrap();
    assert_eq!(val, InterventionParams::Ablate {});
}

#[test]
fn intervention_params_scale_not_confused_with_ablate() {
    let val: InterventionParams = serde_json::from_value(json!({"factor": 2.0})).unwrap();
    assert_eq!(val, InterventionParams::Scale { factor: 2.0 });
}

#[test]
fn intervention_params_clamp_not_confused_with_scale() {
    let val: InterventionParams = serde_json::from_value(json!({"min": -1.0, "max": 1.0})).unwrap();
    assert_eq!(
        val,
        InterventionParams::Clamp {
            min: -1.0,
            max: 1.0,
        }
    );
}

// ===== Edge cases: AddVector untagged discrimination =====

#[test]
fn add_vector_inline_from_array() {
    use rocket_surgeon_protocol::types::AddVector;
    let val: AddVector = serde_json::from_value(json!([1.0, 2.0, 3.0])).unwrap();
    assert_eq!(val, AddVector::Inline(vec![1.0, 2.0, 3.0]));
}

#[test]
fn add_vector_tensor_ref_from_string() {
    use rocket_surgeon_protocol::types::AddVector;
    let val: AddVector = serde_json::from_value(json!("tensor-abc123")).unwrap();
    assert_eq!(val, AddVector::TensorRef("tensor-abc123".to_owned()));
}

// ===== Edge cases: RawMessage negative paths =====

#[test]
fn raw_message_request_into_notification_returns_none() {
    let raw: RawMessage = serde_json::from_value(json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize"
    }))
    .unwrap();
    assert!(raw.into_notification().is_none());
}

#[test]
fn raw_message_notification_into_request_returns_none() {
    let raw: RawMessage = serde_json::from_value(json!({
        "jsonrpc": "2.0",
        "method": "tick.stopped"
    }))
    .unwrap();
    assert!(raw.into_request().is_none());
}

// ===== Schema compliance: deserialize from canonical JSON =====

#[test]
fn initialize_request_from_schema_json() {
    let json = json!({
        "client_name": "rocket-tui",
        "protocol_version": "0.1.0",
        "client_version": "1.0.0",
        "client_capabilities": {"supports_streaming": true}
    });
    let req: InitializeRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.client_name, "rocket-tui");
    assert_eq!(req.protocol_version, "0.1.0");
    assert!(req.client_capabilities.is_some());
}

#[test]
fn step_request_from_schema_json() {
    let json = json!({
        "direction": "forward",
        "count": 5,
        "granularity": "component"
    });
    let req: StepRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.direction, StepDirection::Forward);
    assert_eq!(req.count, 5);
    assert_eq!(req.granularity, Some(TickGranularity::Component));
}

#[test]
fn session_state_from_schema_json() {
    let json = json!({
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "model_id": "abc123",
        "status": "stopped",
        "position": {
            "tick_id": 42,
            "direction": "forward",
            "rank": null,
            "layer": 12,
            "component": "attn.o_proj",
            "event": "output"
        },
        "tick_id": 42,
        "active_probes": ["probe-1"],
        "checkpoints": [],
        "available_actions": ["step", "inspect"]
    });
    let state: SessionState = serde_json::from_value(json).unwrap();
    assert_eq!(state.status, Status::Stopped);
    assert!(state.position.is_some());
    assert_eq!(state.position.unwrap().rank, None);
}

#[test]
fn create_checkpoint_tier_rejects_probe_log() {
    let result: Result<CreateCheckpointTier, _> = serde_json::from_value(json!("probe_log"));
    assert!(result.is_err());
}
