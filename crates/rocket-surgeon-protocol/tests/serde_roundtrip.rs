use rocket_surgeon_protocol::errors::{ErrorCode, ErrorData, Severity};
use rocket_surgeon_protocol::jsonrpc::{
    JSONRPC_VERSION, Notification, RawMessage, Request, RequestId, Response, RpcError,
};
use rocket_surgeon_protocol::messages::{
    AttachRequest, AttachResponse, BranchCompareRequest, BranchCompareResponse, BranchCreatedEvent,
    BranchDropRequest, BranchDropResponse, BranchForkRequest, BranchForkResponse, BranchTier,
    BranchTierChangedEvent, CheckpointRequest, CheckpointResponse, CreateCheckpointTier,
    DetachRequest, DetachResponse, DiscoverMatch, DiscoverRequest, DiscoverResponse, Divergence,
    ErrorEvent, EventType, FocusAnchor, FocusSelector, HostReplayRequest, HostReplayResponse,
    HostViewRequest, HostViewResponse, InitializeRequest, InitializeResponse, InspectDetail,
    InspectRequest, InspectResponse, InterveneRequest, KvCacheEntry, KvEvictedEvent, KvMetric,
    KvOverlay, KvReadRequest, KvReadResponse, KvSlot, KvUpdateEvent, MemoryUsage, ProbeFiredEvent,
    ProbeRequest, ProbeResponse, ReplayDivergenceEvent, ReplayRequest, ReplayResponse,
    ReplayStopAt, StatusRequest, StatusResponse, StepRequest, StepResponse, SubscribeFilter,
    SubscribeRequest, SubscribeResponse, SweepMetric, SweepRequest, SweepResponse, SweepTrial,
    SweepTrialResult, TickHeartbeatEvent, TickStoppedEvent, UnsubscribeRequest,
    UnsubscribeResponse, ViewDefineRequest, ViewDefineResponse, ViewFocusRequest,
    ViewFocusResponse, ViewRequest, ViewResponse,
};
use rocket_surgeon_protocol::types::{
    AblateMode, ActionName, AliasEntry, BuiltInView, Capabilities, CheckpointRef, CheckpointTier,
    ComponentEntry, CompositionMode, DType, EnvelopeMode, ExecutionMode, GranularityScope,
    HeadGranularity, Histogram, InterventionParams, InterventionRecipe, InterventionType,
    Parallelism, Phase, Placement, PlacementType, PositionEnvelope, ProbeAction, ProbeConfig,
    ProbeDefinition, ResponseEnvelope, SessionState, ShardingInfo, Status, StepDirection,
    TensorHandle, TensorStats, TensorSummary, TickClock, TickEvent, TickGranularity, TickLayerInfo,
    TickMapEntry, TickPosition, TopKEntry, Transport, WireFormat, WorldlineSegment, WorldlineState,
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
        phase: Phase::Decode,
        token_position: Some(73),
        clock: None,
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
        worldline: WorldlineState::default(),
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
        id: Some("int-1".to_owned()),
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
    roundtrip(&InterventionParams::Ablate {
        mode: AblateMode::default(),
        reference_run: None,
        reference_tensor_id: None,
    });
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
    roundtrip(&InterventionParams::AttentionMask {
        source_positions: vec![0, 3],
        target_positions: vec![5],
        mask_value: -10000.0,
    });
    roundtrip(&InterventionParams::EmbedSwap {
        position: 5,
        new_token_id: 1234,
    });
    roundtrip(&InterventionParams::EmbedNoise {
        position: 5,
        std: 0.1,
        seed: Some(42),
    });
}

#[test]
fn ablate_mode_serde() {
    assert_eq!(
        serde_json::to_string(&AblateMode::Zero).unwrap(),
        r#""zero""#
    );
    assert_eq!(
        serde_json::to_string(&AblateMode::Mean).unwrap(),
        r#""mean""#
    );
    assert_eq!(
        serde_json::to_string(&AblateMode::Resample).unwrap(),
        r#""resample""#
    );
    assert_eq!(AblateMode::default(), AblateMode::Zero);
}

#[test]
fn ablate_with_mode_mean_roundtrip() {
    let params = InterventionParams::Ablate {
        mode: AblateMode::Mean,
        reference_run: Some("ckpt-baseline".to_owned()),
        reference_tensor_id: None,
    };
    roundtrip(&params);
    let json = serde_json::to_value(&params).unwrap();
    assert_eq!(json["mode"], "mean");
    assert_eq!(json["reference_run"], "ckpt-baseline");
    assert!(json.get("reference_tensor_id").is_none());
}

#[test]
fn ablate_empty_json_defaults_to_zero() {
    let json = serde_json::json!({});
    let params: InterventionParams = serde_json::from_value(json).unwrap();
    match params {
        InterventionParams::Ablate {
            mode,
            reference_run,
            ..
        } => {
            assert_eq!(mode, AblateMode::Zero);
            assert!(reference_run.is_none());
        }
        _ => panic!("expected Ablate variant"),
    }
}

#[test]
fn intervention_type_new_variants_serde() {
    assert_eq!(
        serde_json::to_string(&InterventionType::AttentionMask).unwrap(),
        r#""attention_mask""#
    );
    assert_eq!(
        serde_json::to_string(&InterventionType::EmbedSwap).unwrap(),
        r#""embed_swap""#
    );
    assert_eq!(
        serde_json::to_string(&InterventionType::EmbedNoise).unwrap(),
        r#""embed_noise""#
    );
}

#[test]
fn embed_noise_without_seed_roundtrip() {
    let params = InterventionParams::EmbedNoise {
        position: 3,
        std: 0.05,
        seed: None,
    };
    roundtrip(&params);
    let json = serde_json::to_value(&params).unwrap();
    assert!(json.get("seed").is_none());
}

#[test]
fn attention_mask_roundtrip() {
    let params = InterventionParams::AttentionMask {
        source_positions: vec![0, 3],
        target_positions: vec![5],
        mask_value: -10000.0,
    };
    let json = serde_json::to_value(&params).unwrap();
    assert_eq!(json["source_positions"], serde_json::json!([0, 3]));
    assert_eq!(json["mask_value"], -10000.0);
    roundtrip(&params);
}

#[test]
fn capabilities_includes_new_intervention_types() {
    let caps = Capabilities::phase1_defaults();
    let json = serde_json::to_value(&caps).unwrap();
    let types = json["intervention_types"].as_array().unwrap();
    let type_strs: Vec<&str> = types.iter().map(|v| v.as_str().unwrap()).collect();
    assert!(type_strs.contains(&"attention_mask"));
    assert!(type_strs.contains(&"embed_swap"));
    assert!(type_strs.contains(&"embed_noise"));
}

#[test]
fn capabilities_phase1_defaults_roundtrip() {
    let caps = Capabilities::phase1_defaults();
    roundtrip(&caps);

    let json = serde_json::to_value(&caps).unwrap();
    assert_eq!(json["protocol_version"], "0.3.0");
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
            fired_interventions: vec![],
        }),
    };
    roundtrip(&env);
}

#[test]
fn session_state_roundtrip() {
    roundtrip(&sample_session_state());
}

// ===== EnvelopeMode =====

#[test]
fn envelope_mode_serde() {
    assert_eq!(
        serde_json::to_string(&EnvelopeMode::Full).unwrap(),
        r#""full""#
    );
    assert_eq!(
        serde_json::to_string(&EnvelopeMode::Position).unwrap(),
        r#""position""#
    );
    assert_eq!(
        serde_json::to_string(&EnvelopeMode::None).unwrap(),
        r#""none""#
    );
}

#[test]
fn envelope_mode_default_is_full() {
    assert_eq!(EnvelopeMode::default(), EnvelopeMode::Full);
}

#[test]
fn position_envelope_roundtrip() {
    let env = PositionEnvelope {
        status: Status::Stopped,
        position: Some(sample_tick_position()),
    };
    roundtrip(&env);
}

#[test]
fn step_request_envelope_defaults_to_full() {
    let json = json!({
        "direction": "forward",
        "count": 1
    });
    let req: StepRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.envelope, EnvelopeMode::Full);
}

#[test]
fn step_request_envelope_position() {
    let json = json!({
        "direction": "forward",
        "count": 1,
        "envelope": "position"
    });
    let req: StepRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.envelope, EnvelopeMode::Position);
}

#[test]
fn step_request_run_to_roundtrip() {
    let req = StepRequest {
        direction: StepDirection::Forward,
        count: 0,
        granularity: None,
        envelope: EnvelopeMode::default(),
        run_to: Some("llama:*:12:attn.o_proj:output".to_owned()),
        tokens: None,
    };
    roundtrip(&req);
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["run_to"], "llama:*:12:attn.o_proj:output");
}

#[test]
fn step_request_run_to_completion() {
    let req = StepRequest {
        direction: StepDirection::Forward,
        count: 0,
        granularity: None,
        envelope: EnvelopeMode::default(),
        run_to: Some("completion".to_owned()),
        tokens: None,
    };
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["run_to"], "completion");
}

#[test]
fn step_request_run_to_absent_by_default() {
    let json = json!({
        "direction": "forward",
        "count": 1
    });
    let req: StepRequest = serde_json::from_value(json).unwrap();
    assert!(req.run_to.is_none());
}

#[test]
fn step_request_run_to_omitted_from_json_when_none() {
    let req = StepRequest {
        direction: StepDirection::Forward,
        count: 1,
        granularity: None,
        envelope: EnvelopeMode::default(),
        run_to: None,
        tokens: None,
    };
    let json = serde_json::to_value(&req).unwrap();
    assert!(json.get("run_to").is_none());
}

// ===== TickClock =====

#[test]
fn tick_clock_roundtrip() {
    let clock = TickClock {
        token: 73,
        operator: 42,
        wall_ns: 1_500_000_000,
    };
    roundtrip(&clock);
    let json = serde_json::to_value(clock).unwrap();
    assert_eq!(json["token"], 73);
    assert_eq!(json["operator"], 42);
    assert_eq!(json["wall_ns"], 1_500_000_000u64);
}

#[test]
fn tick_position_with_clock_roundtrip() {
    let pos = TickPosition {
        clock: Some(TickClock {
            token: 73,
            operator: 42,
            wall_ns: 1_500_000_000,
        }),
        ..sample_tick_position()
    };
    roundtrip(&pos);
    let json = serde_json::to_value(&pos).unwrap();
    assert_eq!(json["clock"]["token"], 73);
    assert_eq!(json["clock"]["operator"], 42);
    assert_eq!(json["tick_id"], json["clock"]["operator"]);
}

#[test]
fn tick_position_clock_absent_when_none() {
    let pos = sample_tick_position();
    let json = serde_json::to_value(&pos).unwrap();
    assert!(json.get("clock").is_none());
}

#[test]
fn tick_position_backward_compat_no_clock() {
    let json = json!({
        "tick_id": 42,
        "direction": "forward",
        "rank": null,
        "layer": 12,
        "component": "attn.o_proj",
        "event": "output"
    });
    let pos: TickPosition = serde_json::from_value(json).unwrap();
    assert_eq!(pos.tick_id, 42);
    assert_eq!(pos.clock, None);
}

#[test]
fn tick_position_with_clock_from_json() {
    let json = json!({
        "tick_id": 42,
        "direction": "forward",
        "rank": null,
        "layer": 12,
        "component": "attn.o_proj",
        "event": "output",
        "clock": {
            "token": 73,
            "operator": 42,
            "wall_ns": 1_500_000_000
        }
    });
    let pos: TickPosition = serde_json::from_value(json).unwrap();
    assert!(pos.clock.is_some());
    let clock = pos.clock.unwrap();
    assert_eq!(clock.token, 73);
    assert_eq!(clock.operator, 42);
    assert_eq!(clock.wall_ns, 1_500_000_000);
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
    assert!(json.get("recovery_hint").is_none());
}

#[test]
fn error_data_with_recovery_hint() {
    let mut err = ErrorData::new(ErrorCode::InvalidTarget, "Target not found");
    err.recovery_hint = Some("Did you mean attn.o_proj?".to_owned());
    let json = serde_json::to_value(&err).unwrap();
    assert_eq!(json["recovery_hint"], "Did you mean attn.o_proj?");
    roundtrip(&err);
}

#[test]
fn error_data_recovery_hint_omitted_when_none() {
    let err = ErrorData::new(ErrorCode::InvalidTarget, "Target not found");
    let json = serde_json::to_value(&err).unwrap();
    assert!(json.get("recovery_hint").is_none());
}

#[test]
fn error_data_backward_compat_no_recovery_hint() {
    let json = serde_json::json!({
        "error_code": "INVALID_STATE",
        "numeric_code": -32001,
        "suggestion": "Call initialize first",
        "severity": "recoverable"
    });
    let err: ErrorData = serde_json::from_value(json).unwrap();
    assert!(err.recovery_hint.is_none());
}

#[test]
fn new_error_codes_numeric() {
    assert_eq!(ErrorCode::BranchNotFound.numeric_code(), -32022);
    assert_eq!(ErrorCode::BranchMergeRefused.numeric_code(), -32023);
    assert_eq!(ErrorCode::VramExhausted.numeric_code(), -32024);
    assert_eq!(ErrorCode::CrossRequestKv.numeric_code(), -32025);
    assert_eq!(ErrorCode::KvEvicted.numeric_code(), -32026);
}

#[test]
fn new_error_codes_serde() {
    assert_eq!(
        serde_json::to_string(&ErrorCode::BranchNotFound).unwrap(),
        "\"BRANCH_NOT_FOUND\""
    );
    assert_eq!(
        serde_json::to_string(&ErrorCode::VramExhausted).unwrap(),
        "\"VRAM_EXHAUSTED\""
    );
    assert_eq!(
        serde_json::to_string(&ErrorCode::CrossRequestKv).unwrap(),
        "\"CROSS_REQUEST_KV\""
    );
    assert_eq!(
        serde_json::to_string(&ErrorCode::KvEvicted).unwrap(),
        "\"KV_EVICTED\""
    );
}

#[test]
fn new_error_codes_severity() {
    assert_eq!(ErrorCode::VramExhausted.severity(), Severity::Fatal);
    assert_eq!(ErrorCode::BranchNotFound.severity(), Severity::Recoverable);
    assert_eq!(
        ErrorCode::BranchMergeRefused.severity(),
        Severity::Recoverable
    );
    assert_eq!(ErrorCode::CrossRequestKv.severity(), Severity::Recoverable);
    assert_eq!(ErrorCode::KvEvicted.severity(), Severity::Recoverable);
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
        json!({"client_name": "test", "protocol_version": "0.2.0"}),
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
        "params": {"client_name": "test", "protocol_version": "0.2.0"}
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
        protocol_version: "0.2.0".to_owned(),
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
        component_vocabulary: Vec::new(),
        module_tree: Vec::new(),
        alias_table: Vec::new(),
        tick_map: Vec::new(),
    };
    roundtrip(&resp);
}

#[test]
fn attach_response_with_discovery_fields() {
    let resp = AttachResponse {
        model_id: "m".repeat(64),
        model_family: "llama".to_owned(),
        num_layers: 32,
        num_heads: 32,
        hidden_dim: 4096,
        num_ranks: 1,
        capabilities: Capabilities::phase1_defaults(),
        component_vocabulary: vec![ComponentEntry {
            canonical: "llama:*:0:attn.q:output".to_owned(),
            event: "output".to_owned(),
            tensor_shape: vec![1, 32, 4096],
            category: "attention".to_owned(),
        }],
        module_tree: vec![
            "model".to_owned(),
            "model.layers".to_owned(),
            "model.layers.0".to_owned(),
            "model.layers.0.self_attn".to_owned(),
        ],
        alias_table: vec![AliasEntry {
            canonical: "llama:*:0:attn.q:output".to_owned(),
            aliases: vec!["blocks.0.attn.hook_q".to_owned(), "L0.attn.q".to_owned()],
        }],
        tick_map: vec![TickMapEntry {
            granularity: TickGranularity::Component,
            ticks_per_layer: vec![TickLayerInfo {
                layer: 0,
                components: vec!["attn.q".to_owned(), "attn.k".to_owned()],
                tick_count: 2,
            }],
        }],
    };
    roundtrip(&resp);

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(
        json["component_vocabulary"][0]["canonical"],
        "llama:*:0:attn.q:output"
    );
    assert_eq!(json["alias_table"][0]["aliases"][0], "blocks.0.attn.hook_q");
    assert_eq!(json["tick_map"][0]["granularity"], "component");
}

#[test]
fn attach_response_backward_compat_no_discovery_fields() {
    let json = json!({
        "model_id": "abc",
        "model_family": "llama",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_dim": 4096,
        "num_ranks": 1,
        "capabilities": Capabilities::phase1_defaults()
    });
    let resp: AttachResponse = serde_json::from_value(json).unwrap();
    assert!(resp.component_vocabulary.is_empty());
    assert!(resp.module_tree.is_empty());
    assert!(resp.alias_table.is_empty());
    assert!(resp.tick_map.is_empty());
}

#[test]
fn component_entry_roundtrip() {
    let entry = ComponentEntry {
        canonical: "llama:*:12:mlp:output".to_owned(),
        event: "output".to_owned(),
        tensor_shape: vec![1, 4096],
        category: "mlp".to_owned(),
    };
    roundtrip(&entry);
}

#[test]
fn alias_entry_roundtrip() {
    let entry = AliasEntry {
        canonical: "llama:*:0:attn.q:output".to_owned(),
        aliases: vec!["L0.q".to_owned(), "blocks.0.hook_q".to_owned()],
    };
    roundtrip(&entry);
}

#[test]
fn tick_map_entry_roundtrip() {
    let entry = TickMapEntry {
        granularity: TickGranularity::Component,
        ticks_per_layer: vec![TickLayerInfo {
            layer: 0,
            components: vec!["attn.q".to_owned(), "attn.k".to_owned(), "mlp".to_owned()],
            tick_count: 3,
        }],
    };
    roundtrip(&entry);
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
        envelope: EnvelopeMode::default(),
        run_to: None,
        tokens: None,
    };
    roundtrip(&req);
}

#[test]
fn step_request_with_tokens_roundtrip() {
    let req = StepRequest {
        direction: StepDirection::Forward,
        count: 1,
        granularity: None,
        envelope: EnvelopeMode::default(),
        run_to: None,
        tokens: Some(vec![50256, 464, 3797, 318]),
    };
    let json = serde_json::to_value(&req).unwrap();
    assert!(json.get("tokens").is_some());
    roundtrip(&req);
}

#[test]
fn step_request_without_tokens_omits_field() {
    let req = StepRequest {
        direction: StepDirection::Forward,
        count: 1,
        granularity: None,
        envelope: EnvelopeMode::default(),
        run_to: None,
        tokens: None,
    };
    let json = serde_json::to_value(&req).unwrap();
    assert!(json.get("tokens").is_none());
}

#[test]
fn step_response_roundtrip() {
    let resp = StepResponse {
        ticks_executed: 5,
        stopped_at: sample_tick_position(),
        fired_interventions: vec![],
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
        envelope: EnvelopeMode::default(),
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
            id: Some("int-1".to_owned()),
            intervention_type: InterventionType::Ablate,
            target: "llama:0:12:mlp:output".to_owned(),
            params: InterventionParams::Ablate {
                mode: AblateMode::default(),
                reference_run: None,
                reference_tensor_id: None,
            },
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
        envelope: EnvelopeMode::default(),
        deterministic: None,
        cosine_threshold: None,
        mre_threshold: None,
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
fn replay_request_new_fields_roundtrip() {
    let req = ReplayRequest {
        from_checkpoint: "cp-1".to_owned(),
        interventions: None,
        stop_at: None,
        verify: true,
        envelope: EnvelopeMode::default(),
        deterministic: Some(true),
        cosine_threshold: Some(0.9999),
        mre_threshold: Some(0.0005),
    };
    roundtrip(&req);
}

#[test]
fn replay_request_new_fields_skip_when_none() {
    let req = ReplayRequest {
        from_checkpoint: "cp-1".to_owned(),
        interventions: None,
        stop_at: None,
        verify: true,
        envelope: EnvelopeMode::default(),
        deterministic: None,
        cosine_threshold: None,
        mre_threshold: None,
    };
    let json = serde_json::to_value(&req).unwrap();
    assert!(json.get("deterministic").is_none());
    assert!(json.get("cosine_threshold").is_none());
    assert!(json.get("mre_threshold").is_none());
}

#[test]
fn host_replay_request_roundtrip() {
    let req = HostReplayRequest {
        model_handle: 7,
        checkpoint_id: "cp-42".to_owned(),
        stop_at: Some(ReplayStopAt {
            layer: 12,
            component: "attn.o_proj".to_owned(),
        }),
        interventions: vec![],
        verify: true,
        deterministic: true,
        cosine_threshold: 0.9999,
        mre_threshold: 0.0005,
    };
    roundtrip(&req);
}

#[test]
fn host_replay_response_roundtrip() {
    let resp = HostReplayResponse {
        ticks_replayed: 5,
        stopped_at: sample_tick_position(),
        divergences: vec![],
        verified: true,
    };
    roundtrip(&resp);
}

#[test]
fn worldline_state_default_skips_in_session_state() {
    let state = sample_session_state();
    assert!(state.worldline.is_empty());
    let json = serde_json::to_value(&state).unwrap();
    assert!(
        json.get("worldline").is_none(),
        "empty worldline should be skipped"
    );
}

#[test]
fn worldline_state_non_zero_cursor_with_empty_segments_serializes() {
    // Catches a real wire-format hazard: is_empty() must NOT return true
    // when current_segment is non-default, otherwise skip_serializing_if
    // drops the field and the cursor roundtrips back to 0 on the other side.
    let state = WorldlineState {
        current_segment: 7,
        segments: vec![],
    };
    assert!(
        !state.is_empty(),
        "non-zero current_segment must not be elided"
    );
    roundtrip(&state);
}

#[test]
fn worldline_state_with_segments_roundtrip() {
    let state = WorldlineState {
        current_segment: 1,
        segments: vec![
            WorldlineSegment {
                id: 0,
                parent_segment: None,
                branch_tick: None,
                tick_range: (0, 100),
            },
            WorldlineSegment {
                id: 1,
                parent_segment: Some(0),
                branch_tick: Some(50),
                tick_range: (50, 75),
            },
        ],
    };
    roundtrip(&state);
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
    let req = SubscribeRequest { filter: None };
    roundtrip(&req);
}

#[test]
fn subscribe_filter_events_roundtrip() {
    let req = SubscribeRequest {
        filter: Some(SubscribeFilter {
            events: Some(vec![EventType::TickStopped]),
            layers: None,
            components: None,
        }),
    };
    roundtrip(&req);
    let json = serde_json::to_value(&req).unwrap();
    let filter = &json["filter"];
    assert_eq!(filter["events"][0], "tick.stopped");
    assert!(filter.get("layers").is_none());
    assert!(filter.get("components").is_none());
}

#[test]
fn subscribe_filter_layers_roundtrip() {
    let req = SubscribeRequest {
        filter: Some(SubscribeFilter {
            events: None,
            layers: Some(vec![10, 11, 12]),
            components: None,
        }),
    };
    roundtrip(&req);
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["filter"]["layers"], serde_json::json!([10, 11, 12]));
}

#[test]
fn subscribe_filter_components_roundtrip() {
    let req = SubscribeRequest {
        filter: Some(SubscribeFilter {
            events: None,
            layers: None,
            components: Some(vec!["attn.*".to_owned()]),
        }),
    };
    roundtrip(&req);
}

#[test]
fn subscribe_filter_absent_by_default() {
    let json = serde_json::json!({});
    let req: SubscribeRequest = serde_json::from_value(json).unwrap();
    assert!(req.filter.is_none());
}

#[test]
fn subscribe_filter_omitted_from_json_when_none() {
    let req = SubscribeRequest { filter: None };
    let json = serde_json::to_value(&req).unwrap();
    assert!(json.get("filter").is_none());
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
        envelope: EnvelopeMode::default(),
    };
    roundtrip(&req);
}

#[test]
fn view_request_with_params_roundtrip() {
    let req = ViewRequest {
        view: BuiltInView::AttentionPattern,
        params: Some(json!({"layer": 3, "head": 7})),
        envelope: EnvelopeMode::default(),
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

// ===== KV cache =====

#[test]
fn kv_slot_default_is_both() {
    assert_eq!(KvSlot::default(), KvSlot::Both);
}

#[test]
fn kv_metric_default_is_l2_norm() {
    assert_eq!(KvMetric::default(), KvMetric::L2Norm);
}

#[test]
fn kv_read_request_roundtrip() {
    let req = KvReadRequest {
        layers: Some(vec![0, 1]),
        positions: Some(vec![0, 1, 2]),
        heads: None,
        slot: KvSlot::Both,
        metric: KvMetric::L2Norm,
    };
    roundtrip(&req);
}

#[test]
fn kv_read_request_minimal() {
    let json = serde_json::json!({});
    let req: KvReadRequest = serde_json::from_value(json).unwrap();
    assert!(req.layers.is_none());
    assert_eq!(req.slot, KvSlot::Both);
    assert_eq!(req.metric, KvMetric::L2Norm);
}

#[test]
fn kv_cache_entry_roundtrip() {
    let entry = KvCacheEntry {
        layer: 0,
        position: 5,
        head: 3,
        k_metric: Some(1.23),
        v_metric: Some(4.56),
        overlay: Some(KvOverlay::HeavyHitter),
    };
    roundtrip(&entry);
}

#[test]
fn kv_read_response_roundtrip() {
    let resp = KvReadResponse {
        entries: vec![KvCacheEntry {
            layer: 0,
            position: 0,
            head: 0,
            k_metric: Some(0.5),
            v_metric: None,
            overlay: None,
        }],
    };
    roundtrip(&resp);
}

#[test]
fn kv_overlay_serde() {
    assert_eq!(
        serde_json::to_string(&KvOverlay::Sink).unwrap(),
        r#""sink""#
    );
    assert_eq!(
        serde_json::to_string(&KvOverlay::SharedPrefix).unwrap(),
        r#""shared_prefix""#
    );
}

// ===== Branch =====

#[test]
fn branch_tier_serde() {
    assert_eq!(
        serde_json::to_string(&BranchTier::Live).unwrap(),
        r#""live""#
    );
    assert_eq!(
        serde_json::to_string(&BranchTier::Spilled).unwrap(),
        r#""spilled""#
    );
    assert_eq!(
        serde_json::to_string(&BranchTier::Dropped).unwrap(),
        r#""dropped""#
    );
}

#[test]
fn branch_fork_request_roundtrip() {
    let req = BranchForkRequest {
        from_checkpoint: "ckpt-1".to_owned(),
        name: Some("experiment-a".to_owned()),
    };
    roundtrip(&req);
}

#[test]
fn branch_fork_response_roundtrip() {
    let resp = BranchForkResponse {
        branch_id: "br-001".to_owned(),
        tier: BranchTier::Live,
    };
    roundtrip(&resp);
}

#[test]
fn branch_drop_request_roundtrip() {
    let req = BranchDropRequest {
        branch_id: "br-001".to_owned(),
    };
    roundtrip(&req);
}

#[test]
fn branch_drop_response_roundtrip() {
    let resp = BranchDropResponse {
        branch_id: "br-001".to_owned(),
        freed_mb: Some(128.5),
    };
    roundtrip(&resp);
}

#[test]
fn branch_compare_request_roundtrip() {
    let req = BranchCompareRequest {
        branch_a: "br-001".to_owned(),
        branch_b: "br-002".to_owned(),
    };
    roundtrip(&req);
}

#[test]
fn branch_compare_response_roundtrip() {
    let resp = BranchCompareResponse {
        cosine_similarity: 0.98,
        max_relative_error: 0.02,
        kl_divergence: 0.001,
        per_layer_norm_delta: vec![0.01, 0.02, 0.03],
    };
    roundtrip(&resp);
}

#[test]
fn kv_update_event_roundtrip() {
    let evt = KvUpdateEvent {
        layer: 5,
        new_positions: vec![10, 11],
        total_positions: 100,
    };
    roundtrip(&evt);
}

#[test]
fn kv_evicted_event_roundtrip() {
    let evt = KvEvictedEvent {
        layer: 3,
        evicted_positions: vec![0, 1, 2],
        reason: "cache full".to_owned(),
    };
    roundtrip(&evt);
}

#[test]
fn branch_created_event_roundtrip() {
    let evt = BranchCreatedEvent {
        branch_id: "br-001".to_owned(),
        from_checkpoint: "ckpt-1".to_owned(),
        tier: BranchTier::Live,
    };
    roundtrip(&evt);
}

#[test]
fn branch_tier_changed_event_roundtrip() {
    let evt = BranchTierChangedEvent {
        branch_id: "br-001".to_owned(),
        old_tier: BranchTier::Live,
        new_tier: BranchTier::Dropped,
    };
    roundtrip(&evt);
}

#[test]
fn method_constants_kv_branch() {
    use rocket_surgeon_protocol::messages::method;
    assert_eq!(method::KV_READ, "rocket/kv.read");
    assert_eq!(method::BRANCH_FORK, "rocket/branch.fork");
    assert_eq!(method::BRANCH_DROP, "rocket/branch.drop");
    assert_eq!(method::BRANCH_COMPARE, "rocket/branch.compare");
}

#[test]
fn event_constants_kv_branch() {
    use rocket_surgeon_protocol::messages::event;
    assert_eq!(event::KV_UPDATE, "kv.update");
    assert_eq!(event::KV_EVICTED, "kv.evicted");
    assert_eq!(event::BRANCH_CREATED, "branch.created");
    assert_eq!(event::BRANCH_TIER_CHANGED, "branch.tier_changed");
}

#[test]
fn built_in_view_new_variants_serde() {
    assert_eq!(
        serde_json::to_string(&BuiltInView::TunedLens).unwrap(),
        r#""tuned_lens""#
    );
    assert_eq!(
        serde_json::to_string(&BuiltInView::KvCacheRibbon).unwrap(),
        r#""kv_cache_ribbon""#
    );
    assert_eq!(
        serde_json::to_string(&BuiltInView::WorldlineDag).unwrap(),
        r#""worldline_dag""#
    );
}

// ===== Discover =====

#[test]
fn discover_request_roundtrip() {
    let req = DiscoverRequest {
        pattern: "llama:*:12:*:output".to_owned(),
    };
    roundtrip(&req);
}

#[test]
fn discover_response_with_matches() {
    let resp = DiscoverResponse {
        matches: vec![DiscoverMatch {
            canonical: "attn.o_proj".to_owned(),
            tensor_shape: vec![1, 32, 4096],
            aliases: vec!["o_proj".to_owned()],
        }],
        suggestions: vec![],
    };
    roundtrip(&resp);
    let json = serde_json::to_value(&resp).unwrap();
    assert!(json.get("suggestions").is_none());
}

#[test]
fn discover_response_with_suggestions() {
    let resp = DiscoverResponse {
        matches: vec![],
        suggestions: vec!["attn.o_proj".to_owned()],
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["suggestions"][0], "attn.o_proj");
    roundtrip(&resp);
}

// ===== View Focus =====

#[test]
fn focus_selector_by_position_roundtrip() {
    let sel = FocusSelector::ByPosition { position: 5 };
    roundtrip(&sel);
    let json = serde_json::to_value(&sel).unwrap();
    assert_eq!(json["kind"], "by_position");
    assert_eq!(json["position"], 5);
}

#[test]
fn focus_selector_by_regex_roundtrip() {
    let sel = FocusSelector::ByRegex {
        pattern: "defendant".to_owned(),
    };
    roundtrip(&sel);
}

#[test]
fn focus_selector_by_anchor_roundtrip() {
    let sel = FocusSelector::ByAnchor {
        anchor: FocusAnchor::MaxAttention,
    };
    roundtrip(&sel);
    let json = serde_json::to_value(&sel).unwrap();
    assert_eq!(json["anchor"], "max_attention");
}

#[test]
fn focus_selector_by_range_roundtrip() {
    let sel = FocusSelector::ByRange { start: 0, end: 10 };
    roundtrip(&sel);
}

#[test]
fn focus_anchor_serde() {
    assert_eq!(
        serde_json::to_string(&FocusAnchor::Bos).unwrap(),
        r#""bos""#
    );
    assert_eq!(
        serde_json::to_string(&FocusAnchor::Sink).unwrap(),
        r#""sink""#
    );
    assert_eq!(
        serde_json::to_string(&FocusAnchor::MaxAttention).unwrap(),
        r#""max_attention""#
    );
}

#[test]
fn view_focus_request_roundtrip() {
    let req = ViewFocusRequest {
        selector: FocusSelector::ByPosition { position: 5 },
    };
    roundtrip(&req);
}

#[test]
fn view_focus_response_roundtrip() {
    let resp = ViewFocusResponse {
        position: 5,
        token: serde_json::json!({"id": 1234, "text": "the"}),
        per_layer_summaries: vec![],
    };
    roundtrip(&resp);
}

// ===== Sweep =====

#[test]
fn sweep_request_roundtrip() {
    let req = SweepRequest {
        baseline_checkpoint: "ckpt-clean".to_owned(),
        trials: vec![SweepTrial {
            interventions: vec![InterventionRecipe {
                id: None,
                intervention_type: InterventionType::Scale,
                target: "llama:0:12:attn.o_proj:output".to_owned(),
                params: InterventionParams::Scale { factor: 0.5 },
                condition: None,
                priority: 0,
                mode: CompositionMode::Additive,
            }],
            run_to: Some("completion".to_owned()),
            collect: Some(vec!["llama:*:*:logits:output".to_owned()]),
        }],
        metric: Some(SweepMetric {
            metric_type: "kl_divergence".to_owned(),
            tokens: None,
            position: Some(-1),
        }),
    };
    roundtrip(&req);
}

#[test]
fn sweep_trial_minimal() {
    let trial = SweepTrial {
        interventions: vec![],
        run_to: None,
        collect: None,
    };
    let json = serde_json::to_value(&trial).unwrap();
    assert!(json.get("run_to").is_none());
    assert!(json.get("collect").is_none());
    roundtrip(&trial);
}

#[test]
fn sweep_response_roundtrip() {
    let resp = SweepResponse {
        results: vec![SweepTrialResult {
            trial_index: 0,
            stopped_at: sample_tick_position(),
            collected: vec![],
            metric_value: Some(0.03),
        }],
    };
    roundtrip(&resp);
}

// ===== View Define =====

#[test]
fn view_define_request_roundtrip() {
    let req = ViewDefineRequest {
        name: "my_custom_view".to_owned(),
        spec: serde_json::json!({"type": "composite", "panels": []}),
    };
    roundtrip(&req);
}

#[test]
fn view_define_response_roundtrip() {
    let resp = ViewDefineResponse {
        name: "my_custom_view".to_owned(),
        registered: true,
    };
    roundtrip(&resp);
}

#[test]
fn method_constants_llm_verbs() {
    use rocket_surgeon_protocol::messages::method;
    assert_eq!(method::DISCOVER, "rocket/discover");
    assert_eq!(method::SWEEP, "rocket/sweep");
    assert_eq!(method::VIEW_FOCUS, "rocket/view.focus");
    assert_eq!(method::VIEW_DEFINE, "rocket/view.define");
}

#[test]
fn event_constants_sweep() {
    use rocket_surgeon_protocol::messages::event;
    assert_eq!(event::SPEC_STEP, "spec.step");
    assert_eq!(event::SWEEP_TRIAL_COMPLETE, "sweep.trial_complete");
}

#[test]
fn intervention_recipe_id_optional() {
    let json = serde_json::json!({
        "type": "scale",
        "target": "llama:0:12:attn.o_proj:output",
        "params": {"factor": 0.5}
    });
    let recipe: InterventionRecipe = serde_json::from_value(json).unwrap();
    assert!(recipe.id.is_none());
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
        rank: 3,
    };
    roundtrip(&evt);
}

#[test]
fn probe_fired_event_default_rank_back_compat() {
    let json = r#"{"probe_id":"p1","point":"llama:0:0:attn.o_proj:output","tick_id":1,"tensor_summary":null,"action":"capture","timestamp":"2026-05-14T12:00:00Z"}"#;
    let evt: ProbeFiredEvent =
        serde_json::from_str(json).expect("payload without rank should parse");
    assert_eq!(evt.rank, 0);
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
    assert_eq!(
        val,
        InterventionParams::Ablate {
            mode: AblateMode::default(),
            reference_run: None,
            reference_tensor_id: None,
        }
    );
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
        "protocol_version": "0.2.0",
        "client_version": "1.0.0",
        "client_capabilities": {"supports_streaming": true}
    });
    let req: InitializeRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.client_name, "rocket-tui");
    assert_eq!(req.protocol_version, "0.2.0");
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

// ===== Phase and token_position (protocol 0.2.0) =====

#[test]
fn tick_position_has_phase_field() {
    let pos = sample_tick_position();
    let json = serde_json::to_value(&pos).unwrap();
    assert_eq!(json["phase"]["type"], "decode");
    assert_eq!(json["token_position"], 73);
}

#[test]
fn phase_decode_roundtrip() {
    roundtrip(&Phase::Decode);
    let json = serde_json::to_value(Phase::Decode).unwrap();
    assert_eq!(json["type"], "decode");
}

#[test]
fn phase_prefill_roundtrip() {
    roundtrip(&Phase::Prefill);
    let json = serde_json::to_value(Phase::Prefill).unwrap();
    assert_eq!(json["type"], "prefill");
}

#[test]
fn phase_prefill_chunked_roundtrip() {
    let phase = Phase::PrefillChunked {
        chunk_size: 512,
        chunk_index: 2,
        total_chunks: 4,
    };
    roundtrip(&phase);
    let json = serde_json::to_value(phase).unwrap();
    assert_eq!(json["type"], "prefill_chunked");
    assert_eq!(json["chunk_size"], 512);
    assert_eq!(json["chunk_index"], 2);
    assert_eq!(json["total_chunks"], 4);
}

#[test]
fn tick_position_with_prefill_phase() {
    let pos = TickPosition {
        phase: Phase::Prefill,
        token_position: None,
        ..sample_tick_position()
    };
    roundtrip(&pos);
    let json = serde_json::to_value(&pos).unwrap();
    assert_eq!(json["phase"]["type"], "prefill");
    assert!(json.get("token_position").is_none());
}

#[test]
fn tick_position_forward_compat_no_phase() {
    let json = json!({
        "tick_id": 42,
        "direction": "forward",
        "rank": null,
        "layer": 12,
        "component": "attn.o_proj",
        "event": "output"
    });
    let pos: TickPosition = serde_json::from_value(json).unwrap();
    assert_eq!(pos.phase, Phase::Decode);
    assert_eq!(pos.token_position, None);
}

#[test]
fn tick_position_token_position_absent_when_none() {
    let pos = TickPosition {
        token_position: None,
        ..sample_tick_position()
    };
    let json = serde_json::to_value(&pos).unwrap();
    assert!(json.get("token_position").is_none());
}

#[test]
fn phase_unknown_type_rejected() {
    let json = json!({"type": "speculative_decode"});
    let result: Result<Phase, _> = serde_json::from_value(json);
    assert!(result.is_err());
}

#[test]
fn export_request_roundtrip() {
    use rocket_surgeon_protocol::messages::ExportRequest;
    let req = ExportRequest {
        path: "/tmp/session-abc.tar.gz".into(),
        include_tensors: true,
    };
    roundtrip(&req);
}

#[test]
fn export_request_include_tensors_defaults_true() {
    use rocket_surgeon_protocol::messages::ExportRequest;
    let req: ExportRequest = serde_json::from_value(json!({"path": "/tmp/out.tar.gz"})).unwrap();
    assert!(req.include_tensors);
}

#[test]
fn export_response_roundtrip() {
    use rocket_surgeon_protocol::messages::ExportResponse;
    let resp = ExportResponse {
        path: "/tmp/session-abc.tar.gz".into(),
        size_bytes: 12_345_678,
        artifact_count: 9,
    };
    roundtrip(&resp);
}

#[test]
fn host_export_env_request_roundtrip() {
    use rocket_surgeon_protocol::messages::HostExportEnvRequest;
    roundtrip(&HostExportEnvRequest { model_handle: 42 });
}

#[test]
fn host_export_env_response_roundtrip() {
    use rocket_surgeon_protocol::messages::HostExportEnvResponse;
    let resp = HostExportEnvResponse {
        env: json!({"torch_version": "2.1.0"}),
        model_info: json!({"family": "llama", "layers": 32}),
        prompt: Some(json!({"tokens": [1, 2, 3]})),
    };
    roundtrip(&resp);
}
