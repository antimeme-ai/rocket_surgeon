//! Property-based wire-format conformance for the rocket-surgeon protocol.
//!
//! The hand-written `serde_roundtrip.rs` suite is tier 2-3 (example-based): it
//! pins a handful of concrete values. This suite climbs the oracle hierarchy
//! over the SAME serde surface:
//!
//!   * **Roundtrip (tier 4)** — `from_str(to_string(v)) == v` over *generated*
//!     protocol values. The universal serializer property; generalizes the
//!     example roundtrips into a universally-quantified one.
//!   * **Model-based (tier 6)** — `WorldlineState::is_empty()` must agree with a
//!     simple reference predicate AND with its real wire contract (the
//!     `skip_serializing_if` behaviour inside `SessionState`). This method just
//!     had a wire-format bug; we exercise its boundary exhaustively.
//!   * **Exception-raising (tier 5, 113x mutation-killing odds)** — malformed
//!     JSON, unknown enum variants, and out-of-range numeric fields must FAIL
//!     with an error, never panic and never silently coerce. Almost nobody
//!     writes these; they are the highest-leverage style.
//!   * **Metamorphic (tier 4)** — the parser is idempotent on accepted inputs:
//!     if arbitrary text parses, re-serialising and re-parsing is stable.
//!
//! Generators are measured (see the `generator_distribution_*` tests) so we
//! know we are exercising every variant, not just the trivial ones.
//!
//! Production code is deliberately left untouched: all `Strategy`s are written
//! by hand here rather than `#[derive(Arbitrary)]` on the wire types.

#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;
use std::fmt::Debug;

use proptest::collection::vec;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;
use serde::Serialize;
use serde::de::DeserializeOwned;

use rocket_surgeon_protocol::errors::{ErrorCode, ErrorData, Severity};
use rocket_surgeon_protocol::jsonrpc::{
    Notification, RawMessage, Request, RequestId, Response, RpcError,
};
use rocket_surgeon_protocol::types::{
    AblateMode, ActionName, AddVector, BuiltInView, Capabilities, CheckpointRef, CheckpointTier,
    CompositionMode, DType, EnvelopeMode, ExecutionMode, HeadGranularity, Histogram,
    InterventionParams, InterventionRecipe, InterventionType, Parallelism, Phase, Placement,
    PlacementType, PositionEnvelope, ProbeAction, ProbeConfig, ProbeDefinition, SessionState,
    ShardingInfo, Status, StepDirection, TensorStats, TensorSummary, TickClock, TickEvent,
    TickGranularity, TickPosition, TopKEntry, Transport, WireFormat, WorldlineSegment,
    WorldlineState,
};

// ---------------------------------------------------------------------------
// Oracle helper: serialize -> deserialize must be the identity.
// ---------------------------------------------------------------------------

fn roundtrip<T>(v: &T) -> Result<(), TestCaseError>
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    let json =
        serde_json::to_string(v).map_err(|e| TestCaseError::fail(format!("serialize: {e}")))?;
    let back: T = serde_json::from_str(&json)
        .map_err(|e| TestCaseError::fail(format!("deserialize {json}: {e}")))?;
    prop_assert_eq!(&back, v, "roundtrip mismatch via {}", json);
    Ok(())
}

// ---------------------------------------------------------------------------
// Leaf strategies.
// ---------------------------------------------------------------------------

/// f64 values that survive a JSON round-trip **bit-exactly**.
///
/// IMPORTANT FINDING (see PLATOON-FINDINGS.md): the protocol crate depends on
/// `serde_json` with default features, i.e. WITHOUT `float_roundtrip`.
/// Serialisation (ryu) is shortest-round-trip-correct, but the DEFAULT PARSER
/// is approximate: it recovers a value up to 2 ULPs from the original and is
/// not even idempotent (~10% of finite f64 oscillate between two adjacent
/// representations across repeated round-trips). So a bit-exact identity does
/// NOT hold over arbitrary finite f64. We therefore feed the composite
/// roundtrip properties a domain that IS exactly round-trippable — integers
/// below 2^40 and small power-of-two fractions both have a short exact decimal
/// that the approximate parser still recovers exactly. The imprecise general
/// case is characterised by the `float_wire_*` properties below, which assert
/// the real (tolerant) contract.
fn arb_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        6 => (-(1_i64 << 40)..(1_i64 << 40)).prop_map(|n| n as f64),
        2 => any::<i32>().prop_map(|n| f64::from(n) / 4.0),
        1 => prop::sample::select(vec![
            0.0_f64, -0.0_f64, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.0625,
        ]),
    ]
}

/// The full finite f64 domain, including the extremes that lose up to 1 ULP.
/// Used only by the float characterisation properties, never by the bit-exact
/// composite roundtrips.
fn arb_f64_wide() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(f64::MIN_POSITIVE),
        Just(f64::MAX),
        Just(f64::MIN),
        Just(-f64::MIN_POSITIVE),
        any::<f64>().prop_filter("finite", |x| x.is_finite()),
    ]
}

/// Distance in ULPs between two same-sign finite f64. For IEEE-754, the bit
/// pattern of positive floats is monotonic, so the magnitude of the bit-level
/// difference IS the ULP distance.
fn ulps_between(a: f64, b: f64) -> u64 {
    let ia = i128::from(a.to_bits());
    let ib = i128::from(b.to_bits());
    u64::try_from((ia - ib).unsigned_abs()).unwrap_or(u64::MAX)
}

/// Strings biased toward realistic component paths but including the empty
/// string and arbitrary unicode/control characters to stress JSON escaping.
fn arb_str() -> impl Strategy<Value = String> {
    prop_oneof![
        6 => "[a-zA-Z0-9_.:*\\-]{0,24}",
        2 => Just(String::new()),
        1 => any::<String>(),
    ]
}

fn opt<T: Debug>(s: impl Strategy<Value = T>) -> impl Strategy<Value = Option<T>> {
    prop::option::of(s)
}

// ---------------------------------------------------------------------------
// Enum strategies (Copy enums via select over the full variant set).
// ---------------------------------------------------------------------------

macro_rules! select_strategy {
    ($name:ident -> $ty:ty { $($variant:expr),+ $(,)? }) => {
        fn $name() -> impl Strategy<Value = $ty> {
            prop::sample::select(vec![$($variant),+])
        }
    };
}

select_strategy!(arb_status -> Status {
    Status::Uninitialized, Status::Initialized, Status::Attaching, Status::Stopped,
    Status::Stepping, Status::Inspecting, Status::Modifying, Status::Replaying, Status::Detaching,
});

select_strategy!(arb_action -> ActionName {
    ActionName::Initialize, ActionName::Attach, ActionName::Detach, ActionName::Step,
    ActionName::Inspect, ActionName::Intervene, ActionName::Probe, ActionName::Checkpoint,
    ActionName::Replay, ActionName::Status, ActionName::Subscribe,
});

select_strategy!(arb_step_direction -> StepDirection {
    StepDirection::Forward, StepDirection::Backward,
});

select_strategy!(arb_tick_event -> TickEvent { TickEvent::Input, TickEvent::Output });

select_strategy!(arb_dtype -> DType {
    DType::Float16, DType::Bfloat16, DType::Float32, DType::Float64, DType::Int8,
    DType::Int16, DType::Int32, DType::Int64, DType::Uint8, DType::Bool,
});

select_strategy!(arb_ablate_mode -> AblateMode {
    AblateMode::Zero, AblateMode::Mean, AblateMode::Resample,
});

select_strategy!(arb_composition_mode -> CompositionMode {
    CompositionMode::Additive, CompositionMode::Replace,
});

select_strategy!(arb_checkpoint_tier -> CheckpointTier {
    CheckpointTier::ProbeLog, CheckpointTier::Activation, CheckpointTier::FullSnapshot,
});

select_strategy!(arb_probe_action -> ProbeAction {
    ProbeAction::Capture, ProbeAction::Checkpoint, ProbeAction::Trace,
    ProbeAction::Assert, ProbeAction::Aggregate, ProbeAction::Intervene,
});

select_strategy!(arb_intervention_type -> InterventionType {
    InterventionType::Ablate, InterventionType::Scale, InterventionType::Add,
    InterventionType::Patch, InterventionType::Clamp, InterventionType::RouteOverride,
    InterventionType::AttentionMask, InterventionType::EmbedSwap, InterventionType::EmbedNoise,
});

select_strategy!(arb_tick_granularity -> TickGranularity {
    TickGranularity::Layer, TickGranularity::Component, TickGranularity::Head,
    TickGranularity::RouterPreTopk, TickGranularity::RouterPostTopk,
    TickGranularity::Expert, TickGranularity::MoeLayer,
});

select_strategy!(arb_placement_type -> PlacementType {
    PlacementType::Shard, PlacementType::Replicate, PlacementType::Partial,
});

select_strategy!(arb_execution_mode -> ExecutionMode {
    ExecutionMode::Eager, ExecutionMode::Compiled, ExecutionMode::Mixed,
});

select_strategy!(arb_parallelism -> Parallelism {
    Parallelism::SingleGpu, Parallelism::Ddp, Parallelism::Fsdp,
    Parallelism::TensorParallel, Parallelism::PipelineParallel,
});

select_strategy!(arb_head_granularity -> HeadGranularity {
    HeadGranularity::Native, HeadGranularity::RequiresUnfused, HeadGranularity::Unavailable,
});

select_strategy!(arb_transport -> Transport {
    Transport::Stdio, Transport::UnixSocket, Transport::Tcp, Transport::Websocket,
});

select_strategy!(arb_wire_format -> WireFormat { WireFormat::Json, WireFormat::Protobuf });

select_strategy!(arb_built_in_view -> BuiltInView {
    BuiltInView::ResidualStreamNorm, BuiltInView::AttentionPattern, BuiltInView::HeadOutput,
    BuiltInView::LogitLens, BuiltInView::RoutingDecision, BuiltInView::RoutingEntropy,
    BuiltInView::FeatureAttribution, BuiltInView::SaeActivation, BuiltInView::TunedLens,
    BuiltInView::KvCacheRibbon, BuiltInView::KvCacheDetail, BuiltInView::WorldlineDag,
});

select_strategy!(arb_envelope_mode -> EnvelopeMode {
    EnvelopeMode::Full, EnvelopeMode::Position, EnvelopeMode::None,
});

select_strategy!(arb_severity -> Severity { Severity::Fatal, Severity::Recoverable });

/// The full `ErrorCode` variant set, enumerated exhaustively so the injectivity
/// model-property below is a genuine spec oracle rather than a sample.
const ALL_ERROR_CODES: &[ErrorCode] = &[
    ErrorCode::InvalidState,
    ErrorCode::InvalidTarget,
    ErrorCode::InvalidRecipe,
    ErrorCode::ModelNotAttached,
    ErrorCode::TensorNotFound,
    ErrorCode::CheckpointNotFound,
    ErrorCode::ProbeNotFound,
    ErrorCode::CapabilityNotSupported,
    ErrorCode::SliceOutOfBounds,
    ErrorCode::ResponseTooLarge,
    ErrorCode::HostError,
    ErrorCode::GpuOom,
    ErrorCode::NcclTimeout,
    ErrorCode::ReplayDivergence,
    ErrorCode::UnsupportedModel,
    ErrorCode::CompiledModel,
    ErrorCode::ModelAlreadyAttached,
    ErrorCode::InvalidParams,
    ErrorCode::DuplicateProbeId,
    ErrorCode::InvalidPoint,
    ErrorCode::ViewDataUnavailable,
    ErrorCode::BackendAttachFailed,
    ErrorCode::BranchNotFound,
    ErrorCode::BranchMergeRefused,
    ErrorCode::VramExhausted,
    ErrorCode::CrossRequestKv,
    ErrorCode::KvEvicted,
];

fn arb_error_code() -> impl Strategy<Value = ErrorCode> {
    prop::sample::select(ALL_ERROR_CODES.to_vec())
}

// ---------------------------------------------------------------------------
// Composite strategies.
// ---------------------------------------------------------------------------

fn arb_phase() -> impl Strategy<Value = Phase> {
    prop_oneof![
        Just(Phase::Prefill),
        Just(Phase::Decode),
        (any::<u32>(), any::<u32>(), any::<u32>()).prop_map(
            |(chunk_size, chunk_index, total_chunks)| {
                Phase::PrefillChunked {
                    chunk_size,
                    chunk_index,
                    total_chunks,
                }
            }
        ),
    ]
}

prop_compose! {
    fn arb_tick_clock()(
        token in any::<u64>(),
        operator in any::<u64>(),
        wall_ns in any::<u64>(),
    ) -> TickClock {
        TickClock { token, operator, wall_ns }
    }
}

prop_compose! {
    fn arb_tick_position()(
        tick_id in any::<u64>(),
        direction in arb_step_direction(),
        rank in opt(any::<u32>()),
        layer in any::<u32>(),
        component in arb_str(),
        event in arb_tick_event(),
        replay_of in opt(any::<u64>()),
        phase in arb_phase(),
        token_position in opt(any::<u64>()),
        clock in opt(arb_tick_clock()),
    ) -> TickPosition {
        TickPosition {
            tick_id, direction, rank, layer, component, event,
            replay_of, phase, token_position, clock,
        }
    }
}

prop_compose! {
    fn arb_worldline_segment()(
        id in any::<u32>(),
        parent_segment in opt(any::<u32>()),
        branch_tick in opt(any::<u64>()),
        lo in any::<u64>(),
        hi in any::<u64>(),
    ) -> WorldlineSegment {
        WorldlineSegment { id, parent_segment, branch_tick, tick_range: (lo, hi) }
    }
}

prop_compose! {
    fn arb_worldline_state()(
        current_segment in any::<u32>(),
        segments in vec(arb_worldline_segment(), 0..4),
    ) -> WorldlineState {
        WorldlineState { current_segment, segments }
    }
}

prop_compose! {
    fn arb_checkpoint_ref()(
        checkpoint_id in arb_str(),
        tick_id in any::<u64>(),
        layer_idx in any::<u32>(),
        tier in arb_checkpoint_tier(),
        bookmark in opt(arb_str()),
        created_at in arb_str(),
    ) -> CheckpointRef {
        CheckpointRef { checkpoint_id, tick_id, layer_idx, tier, bookmark, created_at }
    }
}

prop_compose! {
    fn arb_session_state()(
        session_id in arb_str(),
        model_id in opt(arb_str()),
        status in arb_status(),
        position in opt(arb_tick_position()),
        tick_id in opt(any::<u64>()),
        active_probes in vec(arb_str(), 0..3),
        checkpoints in vec(arb_checkpoint_ref(), 0..3),
        available_actions in vec(arb_action(), 0..4),
        worldline in arb_worldline_state(),
    ) -> SessionState {
        SessionState {
            session_id, model_id, status, position, tick_id,
            active_probes, checkpoints, available_actions, worldline,
        }
    }
}

fn arb_add_vector() -> impl Strategy<Value = AddVector> {
    prop_oneof![
        vec(arb_f64(), 0..4).prop_map(AddVector::Inline),
        arb_str().prop_map(AddVector::TensorRef),
    ]
}

fn arb_intervention_params() -> impl Strategy<Value = InterventionParams> {
    prop_oneof![
        arb_f64().prop_map(|factor| InterventionParams::Scale { factor }),
        arb_add_vector().prop_map(|vector| InterventionParams::Add { vector }),
        arb_str().prop_map(|source_tensor_id| InterventionParams::Patch { source_tensor_id }),
        (arb_f64(), arb_f64()).prop_map(|(min, max)| InterventionParams::Clamp { min, max }),
        (any::<u64>(), vec(any::<u64>(), 0..4))
            .prop_map(|(token, experts)| InterventionParams::RouteOverride { token, experts }),
        (vec(any::<u64>(), 0..3), vec(any::<u64>(), 0..3), arb_f64()).prop_map(
            |(source_positions, target_positions, mask_value)| InterventionParams::AttentionMask {
                source_positions,
                target_positions,
                mask_value,
            }
        ),
        (any::<u64>(), any::<u64>()).prop_map(|(position, new_token_id)| {
            InterventionParams::EmbedSwap {
                position,
                new_token_id,
            }
        }),
        (any::<u64>(), arb_f64(), opt(any::<u64>())).prop_map(|(position, std, seed)| {
            InterventionParams::EmbedNoise {
                position,
                std,
                seed,
            }
        }),
        (arb_ablate_mode(), opt(arb_str()), opt(arb_str())).prop_map(
            |(mode, reference_run, reference_tensor_id)| InterventionParams::Ablate {
                mode,
                reference_run,
                reference_tensor_id,
            }
        ),
    ]
}

prop_compose! {
    fn arb_intervention_recipe()(
        id in opt(arb_str()),
        intervention_type in arb_intervention_type(),
        target in arb_str(),
        params in arb_intervention_params(),
        condition in opt(arb_str()),
        priority in any::<i32>(),
        mode in arb_composition_mode(),
    ) -> InterventionRecipe {
        InterventionRecipe { id, intervention_type, target, params, condition, priority, mode }
    }
}

prop_compose! {
    fn arb_probe_config()(
        summary in any::<bool>(),
        capture_tensor in any::<bool>(),
        filter in opt(arb_str()),
        aggregate_fn in opt(arb_str()),
        assertion in opt(arb_str()),
        intervention in opt(arb_intervention_recipe()),
    ) -> ProbeConfig {
        ProbeConfig { summary, capture_tensor, filter, aggregate_fn, assertion, intervention }
    }
}

prop_compose! {
    fn arb_probe_definition()(
        id in arb_str(),
        point in arb_str(),
        action in arb_probe_action(),
        config in opt(arb_probe_config()),
        enabled in any::<bool>(),
        priority in any::<i32>(),
    ) -> ProbeDefinition {
        ProbeDefinition { id, point, action, config, enabled, priority }
    }
}

prop_compose! {
    fn arb_placement()(
        placement_type in arb_placement_type(),
        dim in opt(any::<i32>()),
    ) -> Placement {
        Placement { placement_type, dim }
    }
}

prop_compose! {
    fn arb_sharding_info()(
        mesh in arb_str(),
        placements in vec(arb_placement(), 0..3),
        local_shape in vec(any::<u64>(), 0..4),
        global_shape in vec(any::<u64>(), 0..4),
    ) -> ShardingInfo {
        ShardingInfo { mesh, placements, local_shape, global_shape }
    }
}

prop_compose! {
    fn arb_histogram()(
        bins in any::<u32>(),
        edges in vec(arb_f64(), 0..6),
        counts in vec(any::<u64>(), 0..6),
    ) -> Histogram {
        Histogram { bins, edges, counts }
    }
}

prop_compose! {
    fn arb_tensor_stats()(
        mean in arb_f64(),
        std in arb_f64(),
        min in arb_f64(),
        max in arb_f64(),
        abs_max in arb_f64(),
        sparsity in arb_f64(),
        l2_norm in arb_f64(),
        histogram in arb_histogram(),
    ) -> TensorStats {
        TensorStats { mean, std, min, max, abs_max, sparsity, l2_norm, histogram }
    }
}

prop_compose! {
    fn arb_top_k_entry()(
        index in vec(any::<u64>(), 0..4),
        value in arb_f64(),
    ) -> TopKEntry {
        TopKEntry { index, value }
    }
}

prop_compose! {
    fn arb_tensor_summary()(
        tensor_id in arb_str(),
        shape in vec(any::<u64>(), 0..4),
        dtype in arb_dtype(),
        device in arb_str(),
        sharding in opt(arb_sharding_info()),
        stats in arb_tensor_stats(),
        top_k in vec(arb_top_k_entry(), 0..3),
    ) -> TensorSummary {
        TensorSummary { tensor_id, shape, dtype, device, sharding, stats, top_k }
    }
}

#[allow(clippy::too_many_arguments)]
fn arb_capabilities() -> impl Strategy<Value = Capabilities> {
    (
        (
            arb_str(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            arb_execution_mode(),
            arb_parallelism(),
        ),
        (
            vec(arb_tick_granularity(), 0..3),
            vec(arb_intervention_type(), 0..3),
            vec(arb_built_in_view(), 0..3),
            arb_head_granularity(),
            vec(arb_transport(), 0..3),
            vec(arb_wire_format(), 0..2),
            any::<u64>(),
        ),
        (
            opt(arb_str()),
            opt(arb_str()),
            opt(any::<u32>()),
            opt(any::<u32>()),
            opt(any::<u32>()),
            opt(any::<u32>()),
            opt(any::<u32>()),
            opt(any::<u32>()),
            any::<bool>(),
        ),
    )
        .prop_map(|(a, b, c)| {
            let (
                protocol_version,
                supports_reverse_step,
                supports_checkpointing,
                supports_moe,
                supports_backward,
                supports_sae,
                execution_mode,
                parallelism,
            ) = a;
            let (
                tick_granularities,
                intervention_types,
                built_in_views,
                head_granularity,
                transports,
                wire_formats,
                max_response_bytes,
            ) = b;
            let (
                model_family,
                model_id,
                num_layers,
                num_heads,
                hidden_dim,
                num_ranks,
                num_experts,
                top_k_experts,
                shared_memory_supported,
            ) = c;
            Capabilities {
                protocol_version,
                supports_reverse_step,
                supports_checkpointing,
                supports_moe,
                supports_backward,
                supports_sae,
                execution_mode,
                parallelism,
                tick_granularities,
                intervention_types,
                built_in_views,
                head_granularity,
                transports,
                wire_formats,
                max_response_bytes,
                model_family,
                model_id,
                num_layers,
                num_heads,
                hidden_dim,
                num_ranks,
                num_experts,
                top_k_experts,
                shared_memory_supported,
            }
        })
}

prop_compose! {
    fn arb_error_data()(
        error_code in arb_error_code(),
        numeric_code in opt(any::<i32>()),
        current_state in opt(arb_status()),
        valid_states in opt(vec(arb_status(), 0..3)),
        suggestion in arb_str(),
        severity in arb_severity(),
        recovery_hint in opt(arb_str()),
    ) -> ErrorData {
        ErrorData {
            error_code, numeric_code, current_state, valid_states,
            suggestion, severity, recovery_hint, context: None,
        }
    }
}

fn arb_request_id() -> impl Strategy<Value = RequestId> {
    prop_oneof![
        any::<i64>().prop_map(RequestId::Number),
        arb_str().prop_map(RequestId::String),
    ]
}

// ---------------------------------------------------------------------------
// Roundtrip properties (tier 4). Each generalises the example roundtrips.
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn rt_status(v in arb_status()) { roundtrip(&v)?; }

    #[test]
    fn rt_action(v in arb_action()) { roundtrip(&v)?; }

    #[test]
    fn rt_dtype(v in arb_dtype()) { roundtrip(&v)?; }

    #[test]
    fn rt_phase(v in arb_phase()) { roundtrip(&v)?; }

    #[test]
    fn rt_tick_position(v in arb_tick_position()) { roundtrip(&v)?; }

    #[test]
    fn rt_tick_clock(v in arb_tick_clock()) { roundtrip(&v)?; }

    #[test]
    fn rt_worldline_segment(v in arb_worldline_segment()) { roundtrip(&v)?; }

    #[test]
    fn rt_worldline_state(v in arb_worldline_state()) { roundtrip(&v)?; }

    #[test]
    fn rt_session_state(v in arb_session_state()) { roundtrip(&v)?; }

    #[test]
    fn rt_intervention_params(v in arb_intervention_params()) { roundtrip(&v)?; }

    #[test]
    fn rt_add_vector(v in arb_add_vector()) { roundtrip(&v)?; }

    #[test]
    fn rt_intervention_recipe(v in arb_intervention_recipe()) { roundtrip(&v)?; }

    #[test]
    fn rt_probe_definition(v in arb_probe_definition()) { roundtrip(&v)?; }

    #[test]
    fn rt_tensor_summary(v in arb_tensor_summary()) { roundtrip(&v)?; }

    #[test]
    fn rt_sharding_info(v in arb_sharding_info()) { roundtrip(&v)?; }

    #[test]
    fn rt_checkpoint_ref(v in arb_checkpoint_ref()) { roundtrip(&v)?; }

    #[test]
    fn rt_capabilities(v in arb_capabilities()) { roundtrip(&v)?; }

    #[test]
    fn rt_error_data(v in arb_error_data()) { roundtrip(&v)?; }

    #[test]
    fn rt_request_id(v in arb_request_id()) { roundtrip(&v)?; }

    #[test]
    fn rt_position_envelope(
        status in arb_status(),
        position in opt(arb_tick_position()),
    ) {
        roundtrip(&PositionEnvelope { status, position })?;
    }

    #[test]
    fn rt_envelope_mode(v in arb_envelope_mode()) { roundtrip(&v)?; }
}

// ---------------------------------------------------------------------------
// Float wire-format characterisation (the honest contract for f64 on the wire).
// These REPLACE a false bit-exact claim with the relation that actually holds.
// ---------------------------------------------------------------------------

/// Documented upper bound on the per-round-trip f64 error introduced by
/// `serde_json`'s default approximate parser. Observed max over 200k random
/// finite f64 is 2 ULPs; we assert a safety margin of 4 to stay non-flaky
/// while remaining ~1e-15 relative — three orders tighter than RS's 4-nines
/// fidelity target. If a future `serde_json` bump tightens this to 0, this
/// becomes a (passing) over-approximation; if it regresses past 4, we want to
/// know.
const MAX_WIRE_ULPS: u64 = 4;

proptest! {
    /// The wire format preserves finite f64 to within a small, bounded number
    /// of ULPs, with sign and finiteness intact. This is the real guarantee
    /// (serialisation is exact; the default parser is approximate).
    #[test]
    fn float_wire_within_bounded_ulps(v in arb_f64_wide()) {
        let s = serde_json::to_string(&v).unwrap();
        let back: f64 = serde_json::from_str(&s).unwrap();
        prop_assert!(back.is_finite(), "finite {v} serialised to non-finite via {s}");
        prop_assert_eq!(back.signum(), v.signum(), "sign flipped: {} -> {}", v, back);
        let ulps = ulps_between(v, back);
        prop_assert!(ulps <= MAX_WIRE_ULPS, "{v} -> {back} is {ulps} ULPs (via {s})");
    }

    /// FINDING (PLATOON-FINDINGS): the round-trip is not idempotent and the
    /// error ACCUMULATES — ≈10% of finite f64 walk ~1 ULP per round-trip rather
    /// than settling at a fixed point (observed: 5 ULPs after 8 rounds for
    /// `-2.1445558383837477e241`). The contract that actually holds is that
    /// each individual round-trip adds at most MAX_WIRE_ULPS, so after N rounds
    /// the drift is bounded by N·MAX_WIRE_ULPS. This pins the linear-accumulation
    /// behaviour: a tighter future parser keeps it passing; an unbounded
    /// regression fails it.
    #[test]
    fn float_wire_accumulates_at_most_linearly(v in arb_f64_wide()) {
        let mut cur = v;
        let rounds: u64 = 8;
        for i in 1..=rounds {
            cur = serde_json::from_str(&serde_json::to_string(&cur).unwrap()).unwrap();
            prop_assert!(cur.is_finite());
            let drift = ulps_between(v, cur);
            prop_assert!(
                drift <= i * MAX_WIRE_ULPS,
                "round {i}: {v} -> {cur} drifted {drift} ULPs (> {})", i * MAX_WIRE_ULPS
            );
        }
    }
}

/// HAZARD (recorded in PLATOON-FINDINGS): non-finite f64 silently serialise to
/// JSON `null`, which then fails to deserialise back into the f64 field. A
/// producer that puts NaN/±Inf in a tensor stat emits structurally-valid JSON
/// that the consumer cannot parse — an asymmetric, silent corruption. This test
/// PINS that behaviour so a future fix (e.g. a sentinel encoding) is detected.
#[test]
fn non_finite_floats_serialize_to_null_then_fail_to_parse() {
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let s = serde_json::to_string(&v).expect("serialize non-finite");
        assert_eq!(s, "null", "non-finite {v} did not become null");
        let back: Result<f64, _> = serde_json::from_str(&s);
        assert!(back.is_err(), "null parsed back into f64 for {v}");
    }
}

// ---------------------------------------------------------------------------
// jsonrpc envelope roundtrips — params/id corners matter for the wire.
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn rt_request(
        id in arb_request_id(),
        method in arb_str(),
        has_params in any::<bool>(),
        n in any::<u64>(),
    ) {
        // Request::new normalises null params to None; mirror that so equality holds.
        let params = if has_params { serde_json::json!({ "n": n }) } else { serde_json::Value::Null };
        let req = Request::new(id, method, params);
        roundtrip(&req)?;
    }

    #[test]
    fn rt_notification(method in arb_str(), has_params in any::<bool>(), n in any::<u64>()) {
        let params = if has_params { serde_json::json!({ "n": n }) } else { serde_json::Value::Null };
        roundtrip(&Notification::new(method, params))?;
    }

    #[test]
    fn rt_response_error(id in arb_request_id(), v in arb_error_data()) {
        let resp = Response::error(id, RpcError::from_error_data(v));
        roundtrip(&resp)?;
    }

    #[test]
    fn rt_raw_message(
        jsonrpc in arb_str(),
        id in opt(arb_request_id()),
        method in arb_str(),
    ) {
        let raw = RawMessage { jsonrpc, id, method, params: None };
        roundtrip(&raw)?;
    }
}

// ---------------------------------------------------------------------------
// RawMessage classification metamorphic property: is_notification() <=> no id,
// and into_request()/into_notification() are exactly complementary.
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn raw_message_request_notification_partition(
        jsonrpc in arb_str(),
        id in opt(arb_request_id()),
        method in arb_str(),
    ) {
        let raw = RawMessage { jsonrpc, id: id.clone(), method, params: None };
        let is_notif = raw.is_notification();
        prop_assert_eq!(is_notif, id.is_none());

        // Exactly one of the two conversions succeeds, matching is_notification().
        prop_assert_eq!(raw.clone().into_request().is_some(), !is_notif);
        prop_assert_eq!(raw.into_notification().is_some(), is_notif);
    }
}

// ===========================================================================
// MODEL-BASED PROPERTIES (tier 6): WorldlineState::is_empty()
// ===========================================================================

/// Independent reference model for the contract of `is_empty`.
fn worldline_is_empty_reference(w: &WorldlineState) -> bool {
    w.segments.is_empty() && w.current_segment == 0
}

proptest! {
    /// is_empty() must agree with the structural reference predicate on every
    /// input. A mutation that drops the `current_segment == 0` conjunct (the
    /// exact historical wire-format bug) makes the two disagree.
    #[test]
    fn is_empty_matches_reference(w in arb_worldline_state()) {
        prop_assert_eq!(w.is_empty(), worldline_is_empty_reference(&w));
    }

    /// The REAL contract is behavioural: `is_empty()` is the `skip_serializing_if`
    /// predicate for `SessionState.worldline`. is_empty() == true MUST coincide
    /// with the field being absent from the serialised `SessionState`. This is a
    /// stronger oracle than the structural one — it ties the method to its only
    /// caller's wire effect.
    #[test]
    fn is_empty_iff_field_skipped_in_session_state(state in arb_session_state()) {
        let empty = state.worldline.is_empty();
        let json: serde_json::Value = serde_json::to_value(&state).unwrap();
        let field_present = json.get("worldline").is_some();
        prop_assert_eq!(
            empty,
            !field_present,
            "is_empty()={} but worldline field present={} (json={})",
            empty, field_present, json
        );
    }

    /// Strongest preservation property: embedding ANY worldline in a
    /// SessionState and roundtripping must preserve it byte-for-value. This is
    /// what the historical bug violated — a non-zero `current_segment` with
    /// empty `segments` was silently dropped because `is_empty()` returned true.
    #[test]
    fn session_state_preserves_arbitrary_worldline(
        worldline in arb_worldline_state(),
        session_id in arb_str(),
    ) {
        let state = SessionState {
            session_id,
            model_id: None,
            status: Status::Stopped,
            position: None,
            tick_id: None,
            active_probes: vec![],
            checkpoints: vec![],
            available_actions: vec![],
            worldline: worldline.clone(),
        };
        let json = serde_json::to_string(&state).unwrap();
        let back: SessionState = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(&back.worldline, &worldline, "worldline lost via {}", json);
    }
}

// ===========================================================================
// MODEL-BASED PROPERTIES (tier 6/7): ErrorCode numeric/severity maps
// ===========================================================================

#[test]
fn error_code_numeric_map_is_injective() {
    // Spec oracle over the EXHAUSTIVE variant set: distinct codes -> distinct
    // numbers. A mutation collapsing two arms (e.g. copy-paste -32022 twice)
    // is caught here where per-variant example tests are structurally blind.
    let mut seen: BTreeMap<i32, ErrorCode> = BTreeMap::new();
    for &code in ALL_ERROR_CODES {
        if let Some(prev) = seen.insert(code.numeric_code(), code) {
            panic!(
                "numeric_code collision: {:?} and {:?} both map to {}",
                prev,
                code,
                code.numeric_code()
            );
        }
    }
}

#[test]
fn error_code_serde_names_are_unique_and_roundtrip() {
    let mut seen: BTreeMap<String, ErrorCode> = BTreeMap::new();
    for &code in ALL_ERROR_CODES {
        let name = serde_json::to_string(&code).expect("serialize code");
        // exhaustive roundtrip
        let back: ErrorCode = serde_json::from_str(&name).expect("deserialize code");
        assert_eq!(back, code, "code roundtrip failed for {name}");
        if let Some(prev) = seen.insert(name.clone(), code) {
            panic!("serde-name collision: {prev:?} and {code:?} share {name}");
        }
    }
}

#[test]
fn fatal_codes_are_exactly_the_documented_set() {
    // Spec oracle: severity() must classify exactly HostError, GpuOom,
    // NcclTimeout, VramExhausted as Fatal and everything else as Recoverable.
    let fatal: Vec<ErrorCode> = ALL_ERROR_CODES
        .iter()
        .copied()
        .filter(|c| c.severity() == Severity::Fatal)
        .collect();
    assert_eq!(
        fatal,
        vec![
            ErrorCode::HostError,
            ErrorCode::GpuOom,
            ErrorCode::NcclTimeout,
            ErrorCode::VramExhausted,
        ]
    );
}

proptest! {
    /// Metamorphic/model: constructing an ErrorData from a code (with any
    /// suggestion text) and lifting it into an RpcError must carry the code's
    /// canonical numeric value and severity — no field drift.
    #[test]
    fn error_data_new_is_consistent_with_code(code in arb_error_code(), msg in arb_str()) {
        let data = ErrorData::new(code, msg.clone());
        prop_assert_eq!(data.numeric_code, Some(code.numeric_code()));
        prop_assert_eq!(data.severity, code.severity());
        prop_assert_eq!(&data.suggestion, &msg);

        let rpc = RpcError::from_error_data(data);
        prop_assert_eq!(rpc.code, code.numeric_code());
        prop_assert_eq!(&rpc.message, &msg);
    }
}

// ===========================================================================
// EXCEPTION-RAISING PROPERTIES (tier 5): invalid input -> error, not panic /
// not silent coercion. 113x mutation-killing odds; almost nobody writes these.
// ===========================================================================

/// Valid `snake_case` names for Status — anything else must be rejected.
const VALID_STATUS_NAMES: &[&str] = &[
    "uninitialized",
    "initialized",
    "attaching",
    "stopped",
    "stepping",
    "inspecting",
    "modifying",
    "replaying",
    "detaching",
];

const VALID_DTYPE_NAMES: &[&str] = &[
    "float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool",
];

proptest! {
    /// An unknown enum variant string must produce a deserialization error,
    /// never silently map to a default or panic.
    #[test]
    fn unknown_status_variant_rejected(s in "[a-z_]{0,16}") {
        prop_assume!(!VALID_STATUS_NAMES.contains(&s.as_str()));
        let json = format!("\"{s}\"");
        let parsed: Result<Status, _> = serde_json::from_str(&json);
        prop_assert!(parsed.is_err(), "accepted bogus Status {:?}", s);
    }

    #[test]
    fn unknown_dtype_variant_rejected(s in "[a-z0-9_]{0,16}") {
        prop_assume!(!VALID_DTYPE_NAMES.contains(&s.as_str()));
        let json = format!("\"{s}\"");
        let parsed: Result<DType, _> = serde_json::from_str(&json);
        prop_assert!(parsed.is_err(), "accepted bogus DType {:?}", s);
    }

    /// u64 fields must reject negative inputs rather than wrapping/coercing.
    #[test]
    fn negative_tick_id_rejected(neg in i64::MIN..0_i64) {
        let json = serde_json::json!({
            "tick_id": neg,
            "direction": "forward",
            "layer": 0,
            "component": "x",
            "event": "input",
        });
        let parsed: Result<TickPosition, _> = serde_json::from_value(json);
        prop_assert!(parsed.is_err(), "accepted negative tick_id {}", neg);
    }

    /// u32 fields must reject values beyond u32::MAX rather than truncating.
    #[test]
    fn out_of_range_layer_rejected(big in (u64::from(u32::MAX) + 1)..=u64::MAX) {
        let json = serde_json::json!({
            "tick_id": 0,
            "direction": "forward",
            "layer": big,
            "component": "x",
            "event": "input",
        });
        let parsed: Result<TickPosition, _> = serde_json::from_value(json);
        prop_assert!(parsed.is_err(), "accepted out-of-range layer {}", big);
    }

    /// A missing required field must error (here: TickPosition without `event`).
    #[test]
    fn missing_required_field_rejected(tick_id in any::<u64>()) {
        let json = serde_json::json!({
            "tick_id": tick_id,
            "direction": "forward",
            "layer": 0,
            "component": "x",
            // no "event"
        });
        let parsed: Result<TickPosition, _> = serde_json::from_value(json);
        prop_assert!(parsed.is_err(), "accepted TickPosition missing required event");
    }

    /// Fuzz: arbitrary text must never panic the parser. If it parses, the
    /// parser is idempotent (metamorphic): re-serialise + re-parse is stable.
    #[test]
    fn arbitrary_text_never_panics_and_is_idempotent(s in ".{0,80}") {
        if let Ok(state) = serde_json::from_str::<SessionState>(&s) {
            let reser = serde_json::to_string(&state).expect("re-serialize accepted value");
            let again: SessionState =
                serde_json::from_str(&reser).expect("re-parse own output");
            prop_assert_eq!(state, again);
        }
        // Reaching here without panicking is the implicit-oracle half.
    }
}

// ===========================================================================
// DOCUMENTED HAZARD: untagged InterventionParams silently coerces a malformed
// payload into a no-op Ablate. Recorded in PLATOON-FINDINGS. The first test
// PINS the current (surprising) behaviour so a future fix is detected; the
// #[ignore]d test states the DESIRED behaviour (reject), currently failing.
// ===========================================================================

#[test]
fn untagged_params_typo_coerces_to_ablate_current_behaviour() {
    // A client that means Scale but mistypes the value type gets NO error.
    let malformed = serde_json::json!({ "factor": "zero-point-five" });
    let parsed: InterventionParams =
        serde_json::from_value(malformed).expect("untagged fallthrough accepts it");
    assert!(
        matches!(parsed, InterventionParams::Ablate { .. }),
        "documents that a typo'd Scale silently becomes a no-op Ablate: {parsed:?}"
    );
}

#[test]
fn untagged_params_incomplete_clamp_coerces_to_ablate_current_behaviour() {
    // Clamp missing its `max` should be an error; instead it falls through.
    let malformed = serde_json::json!({ "min": 1.0 });
    let parsed: InterventionParams =
        serde_json::from_value(malformed).expect("untagged fallthrough accepts it");
    assert!(
        matches!(parsed, InterventionParams::Ablate { .. }),
        "documents that an incomplete Clamp silently becomes Ablate: {parsed:?}"
    );
}

#[test]
#[ignore = "DESIRED behaviour: malformed intervention params should error. \
            Currently fails because InterventionParams is #[serde(untagged)] \
            with an all-optional Ablate fallback. See PLATOON-FINDINGS.md."]
fn untagged_params_typo_should_be_rejected() {
    let malformed = serde_json::json!({ "factor": "zero-point-five" });
    let parsed: Result<InterventionParams, _> = serde_json::from_value(malformed);
    assert!(
        parsed.is_err(),
        "a mistyped Scale should be rejected, not coerced to Ablate"
    );
}

// ===========================================================================
// GENERATOR DISTRIBUTION EVIDENCE ("measure what you generate"). proptest has
// no built-in classify/collect, so we sample the strategy directly and assert
// every variant is well-represented. Run with --nocapture to see the tally.
// ===========================================================================

fn intervention_variant(p: &InterventionParams) -> &'static str {
    match p {
        InterventionParams::Scale { .. } => "Scale",
        InterventionParams::Add { .. } => "Add",
        InterventionParams::Patch { .. } => "Patch",
        InterventionParams::Clamp { .. } => "Clamp",
        InterventionParams::RouteOverride { .. } => "RouteOverride",
        InterventionParams::AttentionMask { .. } => "AttentionMask",
        InterventionParams::EmbedSwap { .. } => "EmbedSwap",
        InterventionParams::EmbedNoise { .. } => "EmbedNoise",
        InterventionParams::Ablate { .. } => "Ablate",
    }
}

#[test]
fn generator_distribution_intervention_params() {
    const N: usize = 5000;
    let mut runner = TestRunner::deterministic();
    let strat = arb_intervention_params();
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for _ in 0..N {
        let value = strat.new_tree(&mut runner).unwrap().current();
        *counts.entry(intervention_variant(&value)).or_default() += 1;
    }
    eprintln!("InterventionParams distribution over {N} samples: {counts:?}");

    // 9 variants, ~556 expected each. Require each to appear in non-trivial
    // numbers so no variant is starved (the "94% trivial" failure mode).
    assert_eq!(counts.len(), 9, "missing variants: {counts:?}");
    for (name, &c) in &counts {
        assert!(
            c >= N / 20,
            "variant {name} under-represented: {c}/{N} (<5%)"
        );
    }
}

#[test]
fn generator_distribution_tick_position_optionals() {
    // Classify TickPosition by which optional/skipped fields are populated and
    // by Phase variant, to prove we exercise the skip_serializing_if corners
    // rather than always producing the same dense shape.
    const N: usize = 5000;
    let mut runner = TestRunner::deterministic();
    let strat = arb_tick_position();
    let mut has_clock = 0usize;
    let mut has_replay = 0usize;
    let mut has_token = 0usize;
    let mut phase_chunked = 0usize;
    for _ in 0..N {
        let p = strat.new_tree(&mut runner).unwrap().current();
        has_clock += usize::from(p.clock.is_some());
        has_replay += usize::from(p.replay_of.is_some());
        has_token += usize::from(p.token_position.is_some());
        phase_chunked += usize::from(matches!(p.phase, Phase::PrefillChunked { .. }));
    }
    eprintln!(
        "TickPosition over {N}: clock={has_clock} replay_of={has_replay} \
         token_position={has_token} phase_chunked={phase_chunked}"
    );
    // Each corner should be hit by a meaningful fraction (Some ~50%, chunked ~1/3).
    for (label, c) in [
        ("clock", has_clock),
        ("replay_of", has_replay),
        ("token_position", has_token),
        ("phase_chunked", phase_chunked),
    ] {
        assert!(c >= N / 10, "{label} corner under-exercised: {c}/{N}");
        assert!(c <= N - N / 10, "{label} corner over-saturated: {c}/{N}");
    }
}
