//! Stateful model-based & exception-raising property tests for the daemon
//! session FSM (B002 / callsign DELTA).
//!
//! ## What this tests
//!
//! [`crate::session::Session`] is a `Status` state machine
//! (`Uninitialized → Initialized → Stopped`) plus a registry of checkpoints,
//! a tick cursor, and a worldline segment tree. The existing suite pins this
//! with ~60 example-based tests (oracle tier 2-3). This module climbs to:
//!
//! * **Model-based stateful testing** (tier 6). We generate random sequences
//!   of FSM actions — *including illegal ones* — drive them against the real
//!   `Session` and an independently-written abstract [`Model`] in lockstep, and
//!   after every action assert (a) the real Ok/Err outcome matches the model's
//!   prediction (down to the [`ErrorCode`]), and (b) a battery of invariants
//!   holds: `session_id` stability, `available_actions` ↔ `status` agreement,
//!   `model_id` presence ↔ `Stopped`, checkpoint-list projection equality, and
//!   full worldline-tree structural equality. This is the Hughes (2016)
//!   "abstract model in parallel" pattern.
//! * **Exception-raising properties** (tier 5, the 113×-effective category):
//!   universally-quantified assertions that invalid inputs in each state raise
//!   the *right* error — never a panic, never a silent no-op.
//! * **Generator-distribution measurement**: a sampling test that classifies
//!   the generated corpus and asserts it is not dominated by trivial inputs.
//!
//! ## Modeling choices
//!
//! * The state guard for `rocket/checkpoint` lives in
//!   `dispatch::handle_checkpoint` (it calls `require_stopped` before invoking
//!   the `Session::checkpoint_*` methods, which have no internal precondition).
//!   The harness reproduces that composition: it calls `require_stopped` first,
//!   mirroring the real dispatch contract, so we test the *system* behavior.
//! * `advance_worldline_segment` is an unguarded mutator the daemon main loop
//!   calls during stepping/replay. We exercise it directly; the model mirrors
//!   its (corrected) logic — see `PLATOON-FINDINGS.md` for the self-parented
//!   root defect this caught.
//! * Replay is excluded from the stateful command set (its worker-fallback
//!   tick arithmetic makes a parallel model fragile); replay's FSM edges and
//!   not-found behavior are covered by the focused exception properties below.

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

use rocket_surgeon_protocol::errors::ErrorCode;
use rocket_surgeon_protocol::messages::{
    AttachRequest, InitializeRequest, ReplayRequest, StepRequest,
};
use rocket_surgeon_protocol::types::{
    ActionName, EnvelopeMode, Phase, Status, StepDirection, TickEvent, TickPosition,
    WorldlineSegment, WorldlineState,
};

use crate::session::Session;

const PROTOCOL_VERSION: &str = "0.3.0";
const SUPPORTED_FAMILIES: &[&str] = &["llama", "mixtral", "gpt-neox", "gpt2"];

// ── request constructors ────────────────────────────────────────────────────

fn init_req(version: &str) -> InitializeRequest {
    InitializeRequest {
        client_name: "proptest-client".to_owned(),
        protocol_version: version.to_owned(),
        client_version: None,
        client_capabilities: None,
    }
}

fn attach_req(family: &str, compiled: bool) -> AttachRequest {
    AttachRequest {
        model_path: "/models/test".to_owned(),
        model_family: family.to_owned(),
        device: "cuda:0".to_owned(),
        dtype: None,
        num_ranks: 1,
        config: compiled.then(|| serde_json::json!({ "execution_mode": "compiled" })),
    }
}

fn step_req() -> StepRequest {
    StepRequest {
        direction: StepDirection::Forward,
        count: 1,
        granularity: None,
        envelope: EnvelopeMode::None,
        run_to: None,
        tokens: None,
    }
}

fn tick_pos(tick_id: u64, layer: u32) -> TickPosition {
    TickPosition {
        tick_id,
        direction: StepDirection::Forward,
        rank: Some(0),
        layer,
        component: "attn.q_proj".to_owned(),
        event: TickEvent::Output,
        replay_of: None,
        phase: Phase::default(),
        token_position: None,
        clock: None,
    }
}

fn replay_req(from: &str) -> ReplayRequest {
    ReplayRequest {
        from_checkpoint: from.to_owned(),
        interventions: None,
        stop_at: None,
        verify: true,
        envelope: EnvelopeMode::None,
        deterministic: None,
        cosine_threshold: None,
        mre_threshold: None,
    }
}

// ── action language ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Family {
    Supported(usize),
    Unsupported,
    Compiled,
}

#[derive(Debug, Clone)]
enum Action {
    Initialize {
        good_version: bool,
    },
    Attach {
        family: Family,
    },
    Detach,
    Status,
    Step {
        tick_id: u64,
        layer: u32,
    },
    CheckpointCreate,
    CheckpointDelete {
        use_existing: bool,
        pick: usize,
        rand_id: String,
    },
    CheckpointRestore {
        use_existing: bool,
        pick: usize,
        rand_id: String,
    },
    AdvanceWorldline {
        branch_tick: u64,
    },
}

fn family_strategy() -> impl Strategy<Value = Family> {
    prop_oneof![
        6 => (0usize..SUPPORTED_FAMILIES.len()).prop_map(Family::Supported),
        1 => Just(Family::Unsupported),
        1 => Just(Family::Compiled),
    ]
}

fn action_strategy() -> impl Strategy<Value = Action> {
    let ckpt_ref = (any::<bool>(), 0usize..8, "[a-z]{1,6}");
    prop_oneof![
        2 => (0u8..10).prop_map(|n| Action::Initialize { good_version: n != 0 }),
        2 => family_strategy().prop_map(|family| Action::Attach { family }),
        1 => Just(Action::Detach),
        1 => Just(Action::Status),
        4 => (0u64..100, 0u32..7).prop_map(|(tick_id, layer)| Action::Step { tick_id, layer }),
        3 => Just(Action::CheckpointCreate),
        2 => ckpt_ref.clone().prop_map(|(use_existing, pick, rand_id)| {
            Action::CheckpointDelete { use_existing, pick, rand_id }
        }),
        2 => ckpt_ref.prop_map(|(use_existing, pick, rand_id)| {
            Action::CheckpointRestore { use_existing, pick, rand_id }
        }),
        3 => (0u64..50).prop_map(|branch_tick| Action::AdvanceWorldline { branch_tick }),
    ]
}

fn sequence_strategy() -> impl Strategy<Value = Vec<Action>> {
    prop::collection::vec(action_strategy(), 1..40)
}

// ── abstract model ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Checkpoint {
    id: String,
    tick_id: u64,
    layer: u32,
}

/// Independently-written reference model of the session FSM. It encodes the
/// *intended* behavior; mismatches against the real `Session` are bugs in one
/// or the other (and a deliberate corrected-vs-buggy mismatch is how we caught
/// the self-parented-root defect).
#[derive(Debug, Clone)]
struct Model {
    status: Status,
    session_id: Option<String>,
    model_attached: bool,
    tick_id: Option<u64>,
    layer: Option<u32>,
    checkpoints: Vec<Checkpoint>,
    worldline: WorldlineState,
}

impl Model {
    fn new() -> Self {
        Self {
            status: Status::Uninitialized,
            session_id: None,
            model_attached: false,
            tick_id: None,
            layer: None,
            checkpoints: Vec::new(),
            worldline: WorldlineState::default(),
        }
    }

    /// Corrected mirror of `Session::advance_worldline_segment`: the *root*
    /// segment (created when the tree is empty) has no parent and no branch
    /// point; every later segment branches from the current cursor.
    fn advance_worldline(&mut self, branch_tick: u64) {
        let is_first = self.worldline.segments.is_empty();
        let cur = self.worldline.current_segment as usize;
        if cur < self.worldline.segments.len() {
            self.worldline.segments[cur].tick_range.1 = branch_tick;
        }
        let new_id = self.worldline.segments.len() as u32;
        self.worldline.segments.push(WorldlineSegment {
            id: new_id,
            parent_segment: (!is_first).then_some(self.worldline.current_segment),
            branch_tick: (!is_first).then_some(branch_tick),
            tick_range: (branch_tick, 0),
        });
        self.worldline.current_segment = new_id;
    }

    fn step_worldline(&mut self, tick_id: u64) {
        let cur = self.worldline.current_segment as usize;
        if cur < self.worldline.segments.len() {
            self.worldline.segments[cur].tick_range.1 = tick_id;
        }
    }
}

type Outcome = Result<(), ErrorCode>;

fn code_of<T>(r: Result<T, crate::session::SessionError>) -> Outcome {
    r.map(|_| ()).map_err(|e| e.error_data().error_code)
}

/// Exact mirror of `Session::update_available_actions`.
fn expected_actions(status: Status) -> Vec<ActionName> {
    match status {
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
        _ => vec![],
    }
}

/// Outcome statistics, accumulated across a corpus for distribution analysis.
#[derive(Debug, Default, Clone)]
struct Stats {
    commands: usize,
    rejections: usize,
    reached_stopped: bool,
    checkpoints_created: usize,
    worldline_advances: usize,
    restores_ok: usize,
    deletes_ok: usize,
    attaches_ok: usize,
    detaches_ok: usize,
}

/// Drive one action sequence against the real `Session` and the `Model` in
/// lockstep. Returns accumulated stats on success, or a human-readable
/// invariant/oracle-violation message on failure (so both the proptest
/// property and the distribution sampler can share this engine).
#[allow(clippy::too_many_lines)]
fn run_sequence(seq: &[Action]) -> Result<Stats, String> {
    let mut session = Session::new();
    let mut model = Model::new();
    let mut stats = Stats::default();
    let mut ckpt_counter: u64 = 0;
    let mut first_session_id: Option<String> = None;

    for (i, action) in seq.iter().enumerate() {
        // (1) Predict expected outcome from the *pre-action* model.
        // (2) Apply to the real session.
        // (3) Compare, then mutate the model on success.
        let expected: Outcome;
        let actual: Outcome;

        match action {
            Action::Initialize { good_version } => {
                let version = if *good_version {
                    PROTOCOL_VERSION
                } else {
                    "0.0.0-nope"
                };
                expected = if model.status != Status::Uninitialized {
                    Err(ErrorCode::InvalidState)
                } else if !good_version {
                    Err(ErrorCode::InvalidParams)
                } else {
                    Ok(())
                };
                actual = code_of(session.initialize(&init_req(version)));
                if actual.is_ok() {
                    model.status = Status::Initialized;
                    model.session_id = Some(session.state().session_id.clone());
                }
            }
            Action::Attach { family } => {
                let (fam_str, compiled) = match family {
                    Family::Supported(idx) => (SUPPORTED_FAMILIES[*idx], false),
                    Family::Unsupported => ("totally-made-up-arch", false),
                    Family::Compiled => (SUPPORTED_FAMILIES[0], true),
                };
                expected = if model.status == Status::Stopped || model.model_attached {
                    Err(ErrorCode::ModelAlreadyAttached)
                } else if model.status != Status::Initialized {
                    Err(ErrorCode::InvalidState)
                } else if compiled {
                    Err(ErrorCode::CompiledModel)
                } else if !SUPPORTED_FAMILIES.contains(&fam_str) {
                    Err(ErrorCode::UnsupportedModel)
                } else {
                    Ok(())
                };
                actual = code_of(session.attach(&attach_req(fam_str, compiled), 7, 3, 256));
                if actual.is_ok() {
                    model.status = Status::Stopped;
                    model.model_attached = true;
                    model.tick_id = None;
                    model.layer = None;
                    stats.attaches_ok += 1;
                }
            }
            Action::Detach => {
                expected = if model.status == Status::Initialized || !model.model_attached {
                    Err(ErrorCode::ModelNotAttached)
                } else {
                    Ok(())
                };
                actual = code_of(session.detach());
                if actual.is_ok() {
                    model.status = Status::Initialized;
                    model.model_attached = false;
                    model.tick_id = None;
                    model.layer = None;
                    model.checkpoints.clear();
                    stats.detaches_ok += 1;
                }
            }
            Action::Status => {
                expected = if model.status == Status::Uninitialized {
                    Err(ErrorCode::InvalidState)
                } else {
                    Ok(())
                };
                actual = code_of(session.status());
            }
            Action::Step { tick_id, layer } => {
                expected = if model.status == Status::Stopped {
                    Ok(())
                } else {
                    Err(ErrorCode::ModelNotAttached)
                };
                actual = session
                    .step(&step_req(), &tick_pos(*tick_id, *layer), false, vec![])
                    .map(|_| ())
                    .map_err(|e| e.error_data().error_code);
                if actual.is_ok() {
                    model.tick_id = Some(*tick_id);
                    model.layer = Some(*layer);
                    model.step_worldline(*tick_id);
                }
            }
            Action::CheckpointCreate => {
                let id = format!("cp-{ckpt_counter}");
                ckpt_counter += 1;
                expected = if model.status == Status::Stopped {
                    Ok(())
                } else {
                    Err(ErrorCode::ModelNotAttached)
                };
                actual = match session.require_stopped("rocket/checkpoint") {
                    Ok(()) => {
                        session.checkpoint_create_with_id(None, Some(id.clone()));
                        Ok(())
                    }
                    Err(e) => Err(e.error_data().error_code),
                };
                if actual.is_ok() {
                    model.checkpoints.push(Checkpoint {
                        id,
                        tick_id: model.tick_id.unwrap_or(0),
                        layer: model.layer.unwrap_or(0),
                    });
                    stats.checkpoints_created += 1;
                }
            }
            Action::CheckpointDelete {
                use_existing,
                pick,
                rand_id,
            } => {
                let id = pick_id(&model, *use_existing, *pick, rand_id);
                let found = model.checkpoints.iter().any(|c| c.id == id);
                expected = if model.status != Status::Stopped {
                    Err(ErrorCode::ModelNotAttached)
                } else if found {
                    Ok(())
                } else {
                    Err(ErrorCode::CheckpointNotFound)
                };
                actual = match session.require_stopped("rocket/checkpoint") {
                    Ok(()) => code_of(session.checkpoint_delete(&id)),
                    Err(e) => Err(e.error_data().error_code),
                };
                if actual.is_ok() {
                    model.checkpoints.retain(|c| c.id != id);
                    stats.deletes_ok += 1;
                }
            }
            Action::CheckpointRestore {
                use_existing,
                pick,
                rand_id,
            } => {
                let id = pick_id(&model, *use_existing, *pick, rand_id);
                let found = model.checkpoints.iter().find(|c| c.id == id).cloned();
                expected = if model.status != Status::Stopped {
                    Err(ErrorCode::ModelNotAttached)
                } else if found.is_some() {
                    Ok(())
                } else {
                    Err(ErrorCode::CheckpointNotFound)
                };
                actual = match session.require_stopped("rocket/checkpoint") {
                    Ok(()) => code_of(session.checkpoint_restore(&id)),
                    Err(e) => Err(e.error_data().error_code),
                };
                if actual.is_ok() {
                    let cp = found.expect("Ok implies the checkpoint exists");
                    model.tick_id = Some(cp.tick_id);
                    model.layer = Some(cp.layer);
                    stats.restores_ok += 1;
                }
            }
            Action::AdvanceWorldline { branch_tick } => {
                // Unguarded mutator (mirrors the daemon main loop). Always
                // applied to both sides; outcome is always Ok.
                expected = Ok(());
                session.advance_worldline_segment(*branch_tick);
                actual = Ok(());
                model.advance_worldline(*branch_tick);
                stats.worldline_advances += 1;
            }
        }

        stats.commands += 1;
        if actual.is_err() {
            stats.rejections += 1;
        }
        if model.status == Status::Stopped {
            stats.reached_stopped = true;
        }

        // ── oracle: outcome must match the model's prediction exactly ──
        if expected != actual {
            return Err(format!(
                "step {i} {action:?}: outcome mismatch — model predicted {expected:?}, \
                 session returned {actual:?} (status {:?})",
                model.status
            ));
        }

        // ── invariants on the post-action session vs model ──
        if let Err(msg) = check_invariants(&session, &model, &mut first_session_id) {
            return Err(format!("step {i} {action:?}: {msg}"));
        }
    }

    Ok(stats)
}

fn pick_id(model: &Model, use_existing: bool, pick: usize, rand_id: &str) -> String {
    if use_existing && !model.checkpoints.is_empty() {
        model.checkpoints[pick % model.checkpoints.len()].id.clone()
    } else {
        // Prefix guarantees disjointness from the "cp-N" minted ids.
        format!("missing-{rand_id}")
    }
}

#[allow(clippy::too_many_lines)]
fn check_invariants(
    session: &Session,
    model: &Model,
    first_session_id: &mut Option<String>,
) -> Result<(), String> {
    let st = session.state();

    // status agreement + only the three synchronous states are ever reached.
    if st.status != model.status {
        return Err(format!(
            "status diverged: session {:?} vs model {:?}",
            st.status, model.status
        ));
    }
    if !matches!(
        st.status,
        Status::Uninitialized | Status::Initialized | Status::Stopped
    ) {
        return Err(format!(
            "session entered an intermediate state {:?} — the synchronous FSM \
             must only ever be Uninitialized/Initialized/Stopped",
            st.status
        ));
    }

    // available_actions is a pure function of status.
    let want = expected_actions(st.status);
    if st.available_actions != want {
        return Err(format!(
            "available_actions {:?} != expected {:?} for status {:?}",
            st.available_actions, want, st.status
        ));
    }

    // model_id present iff Stopped.
    if st.model_id.is_some() != (st.status == Status::Stopped) {
        return Err(format!(
            "model_id presence ({}) disagrees with Stopped ({:?})",
            st.model_id.is_some(),
            st.status
        ));
    }

    // session_id: empty iff Uninitialized, and stable once minted.
    if st.session_id.is_empty() != (st.status == Status::Uninitialized) {
        return Err(format!(
            "session_id emptiness ({}) disagrees with Uninitialized ({:?})",
            st.session_id.is_empty(),
            st.status
        ));
    }
    if !st.session_id.is_empty() {
        match first_session_id {
            Some(prev) if prev != &st.session_id => {
                return Err(format!(
                    "session_id changed from {prev} to {} — must be stable for the \
                     connection lifetime across attach/detach",
                    st.session_id
                ));
            }
            None => *first_session_id = Some(st.session_id.clone()),
            _ => {}
        }
    }

    // checkpoint registry projects to the model's (id, tick, layer) list.
    if st.checkpoints.len() != model.checkpoints.len() {
        return Err(format!(
            "checkpoint count {} != model {}",
            st.checkpoints.len(),
            model.checkpoints.len()
        ));
    }
    for (real, m) in st.checkpoints.iter().zip(&model.checkpoints) {
        let real_proj = (real.checkpoint_id.as_str(), real.tick_id, real.layer_idx);
        let model_proj = (m.id.as_str(), m.tick_id, m.layer);
        if real_proj != model_proj {
            return Err(format!(
                "checkpoint projection mismatch: session {real_proj:?} vs model {model_proj:?}"
            ));
        }
    }
    if st.status != Status::Stopped && !st.checkpoints.is_empty() {
        return Err("checkpoints must be empty when not Stopped".to_owned());
    }

    // worldline: full structural equality against the model...
    if session.worldline() != &model.worldline {
        return Err(format!(
            "worldline diverged:\n  session {:?}\n  model   {:?}",
            session.worldline(),
            model.worldline
        ));
    }
    // ...and an independent structural invariant on the tree shape.
    check_worldline_shape(session.worldline())?;

    Ok(())
}

/// A well-formed worldline is an in-order forest where ids are assigned
/// densely in push order, the first segment is a parentless root, and every
/// later segment branches from a strictly-earlier segment.
fn check_worldline_shape(w: &WorldlineState) -> Result<(), String> {
    for (i, seg) in w.segments.iter().enumerate() {
        if seg.id != i as u32 {
            return Err(format!("segment {i} has non-dense id {}", seg.id));
        }
        if i == 0 {
            if seg.parent_segment.is_some() {
                return Err(format!(
                    "root segment must have parent_segment=None, got {:?} \
                     (self-parented root is structurally invalid)",
                    seg.parent_segment
                ));
            }
        } else {
            match seg.parent_segment {
                Some(p) if p < seg.id => {}
                other => {
                    return Err(format!(
                        "segment {} has invalid parent {:?} (must be Some(p) with p < id)",
                        seg.id, other
                    ));
                }
            }
        }
    }
    if !w.segments.is_empty() && (w.current_segment as usize) >= w.segments.len() {
        return Err(format!(
            "current_segment {} out of range (len {})",
            w.current_segment,
            w.segments.len()
        ));
    }
    Ok(())
}

// ── the stateful property ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(384))]

    /// Random legal *and illegal* action sequences keep the real session in
    /// lockstep with the abstract model: every outcome and every invariant.
    #[test]
    fn fsm_matches_model_under_random_action_sequences(seq in sequence_strategy()) {
        if let Err(msg) = run_sequence(&seq) {
            prop_assert!(false, "{}", msg);
        }
    }
}

// ── generator-distribution measurement ──────────────────────────────────────

/// "If 94% of your generated inputs are trivial, you're not testing."
/// Sample the sequence generator and assert the corpus is rich: it reaches
/// `Stopped`, exercises a healthy band of rejected (illegal) actions, and
/// actually creates checkpoints, advances worldlines, and restores.
#[test]
fn generator_distribution_is_non_trivial() {
    let mut runner = TestRunner::deterministic();
    let strat = sequence_strategy();

    let n = 500usize;
    let mut total_commands = 0usize;
    let mut total_rejections = 0usize;
    let mut reached_stopped = 0usize;
    let mut any_checkpoints = 0usize;
    let mut any_advances = 0usize;
    let mut any_restores = 0usize;
    let mut any_attach = 0usize;
    let mut any_detach = 0usize;

    for _ in 0..n {
        let tree = strat
            .new_tree(&mut runner)
            .expect("strategy produces a value");
        let seq = tree.current();
        let stats = run_sequence(&seq).expect("sampled sequence stays in lockstep");
        total_commands += stats.commands;
        total_rejections += stats.rejections;
        reached_stopped += usize::from(stats.reached_stopped);
        any_checkpoints += usize::from(stats.checkpoints_created > 0);
        any_advances += usize::from(stats.worldline_advances > 0);
        any_restores += usize::from(stats.restores_ok > 0);
        any_attach += usize::from(stats.attaches_ok > 0);
        any_detach += usize::from(stats.detaches_ok > 0);
    }

    let rej_frac = total_rejections as f64 / total_commands as f64;
    let stopped_frac = reached_stopped as f64 / n as f64;

    eprintln!(
        "FSM generator distribution over {n} sequences ({total_commands} commands):\n  \
         reached Stopped: {:.1}%\n  \
         rejected (illegal) commands: {:.1}%\n  \
         sequences w/ checkpoint create: {:.1}%\n  \
         sequences w/ worldline advance: {:.1}%\n  \
         sequences w/ successful restore: {:.1}%\n  \
         sequences w/ successful attach: {:.1}%   detach: {:.1}%",
        stopped_frac * 100.0,
        rej_frac * 100.0,
        any_checkpoints as f64 / n as f64 * 100.0,
        any_advances as f64 / n as f64 * 100.0,
        any_restores as f64 / n as f64 * 100.0,
        any_attach as f64 / n as f64 * 100.0,
        any_detach as f64 / n as f64 * 100.0,
    );

    // The corpus must exercise BOTH regions: enough sequences must reach the
    // post-attach `Stopped` region (so Stopped-only verbs are tested) *and*
    // enough must stay pre-attach (so the illegal-transition paths are tested).
    // Thresholds sit ~25% below the observed distribution — tight enough to
    // catch a degenerate generator, loose enough not to be flaky.
    let frac = |k: usize| k as f64 / n as f64;
    assert!(
        (0.35..0.95).contains(&stopped_frac),
        "reached-Stopped fraction {:.1}% outside 35–95% — generator no longer covers \
         both the post-attach and pre-attach regions",
        stopped_frac * 100.0
    );
    assert!(
        (0.10..0.85).contains(&rej_frac),
        "rejected-command fraction {:.1}% is outside the healthy 10–85% band",
        rej_frac * 100.0
    );
    assert!(
        frac(any_checkpoints) > 0.20,
        "too few sequences create checkpoints"
    );
    assert!(
        frac(any_advances) > 0.40,
        "too few sequences advance the worldline"
    );
    assert!(
        frac(any_restores) > 0.02,
        "too few sequences perform a successful restore"
    );
    assert!(frac(any_attach) > 0.35, "too few sequences attach a model");
    assert!(
        frac(any_detach) > 0.05,
        "too few sequences perform a successful detach"
    );
}

// ── focused exception-raising properties (the 113×-effective category) ───────

fn initialized_session() -> Session {
    let mut s = Session::new();
    s.initialize(&init_req(PROTOCOL_VERSION)).unwrap();
    s
}

fn stopped_session() -> Session {
    let mut s = initialized_session();
    s.attach(&attach_req("llama", false), 7, 3, 256).unwrap();
    s
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Any protocol version that is not the exact supported string is rejected
    /// with `INVALID_PARAMS` — never accepted, never a panic.
    #[test]
    fn initialize_wrong_version_is_invalid_params(version in ".{0,12}") {
        prop_assume!(version != PROTOCOL_VERSION);
        let mut s = Session::new();
        let err = s.initialize(&init_req(&version)).unwrap_err();
        prop_assert_eq!(err.error_data().error_code, ErrorCode::InvalidParams);
        // and the session stays put.
        prop_assert_eq!(s.state().status, Status::Uninitialized);
    }

    /// `initialize` is rejected with `INVALID_STATE` from any non-fresh state,
    /// regardless of the (otherwise-valid) version supplied.
    #[test]
    fn initialize_when_already_initialized_is_invalid_state(attach_first in any::<bool>()) {
        let mut s = if attach_first { stopped_session() } else { initialized_session() };
        let err = s.initialize(&init_req(PROTOCOL_VERSION)).unwrap_err();
        prop_assert_eq!(err.error_data().error_code, ErrorCode::InvalidState);
    }

    /// Any family not in the supported set is rejected with `UNSUPPORTED_MODEL`
    /// from `Initialized` — never silently accepted.
    #[test]
    fn attach_unsupported_family_is_unsupported_model(family in "[a-z][a-z0-9_-]{0,15}") {
        prop_assume!(!SUPPORTED_FAMILIES.contains(&family.as_str()));
        let mut s = initialized_session();
        let err = s.attach(&attach_req(&family, false), 7, 3, 256).unwrap_err();
        prop_assert_eq!(err.error_data().error_code, ErrorCode::UnsupportedModel);
        prop_assert_eq!(s.state().status, Status::Initialized);
    }

    /// `attach` from `Uninitialized` is an `INVALID_STATE` transition for any
    /// family (the state check precedes the family check).
    #[test]
    fn attach_from_uninitialized_is_invalid_state(family in "[a-z]{1,8}") {
        let mut s = Session::new();
        let err = s.attach(&attach_req(&family, false), 7, 3, 256).unwrap_err();
        prop_assert_eq!(err.error_data().error_code, ErrorCode::InvalidState);
    }

    /// On a stopped session whose registry is empty, restoring/deleting/
    /// replaying *any* checkpoint id reports `CHECKPOINT_NOT_FOUND` — not a
    /// panic, not a silent success.
    #[test]
    fn unknown_checkpoint_id_is_not_found(id in ".{0,16}") {
        let mut s = stopped_session();
        prop_assert_eq!(
            s.checkpoint_restore(&id).unwrap_err().error_data().error_code,
            ErrorCode::CheckpointNotFound
        );
        prop_assert_eq!(
            s.checkpoint_delete(&id).unwrap_err().error_data().error_code,
            ErrorCode::CheckpointNotFound
        );
        prop_assert_eq!(
            s.replay(&replay_req(&id), None).unwrap_err().error_data().error_code,
            ErrorCode::CheckpointNotFound
        );
    }

    /// `rocket/discover` patterns that are not exactly five non-empty
    /// colon-separated segments are rejected with `INVALID_PARAMS` and never
    /// panic. We bias the generator toward wrong arities and empty segments.
    #[test]
    fn malformed_discover_pattern_is_invalid_params(pattern in "[a-z*]{0,6}(:[a-z*]{0,6}){0,6}") {
        let segs: Vec<&str> = pattern.split(':').collect();
        let well_formed = segs.len() == 5 && segs.iter().all(|s| !s.is_empty());
        prop_assume!(!well_formed);
        let s = stopped_session();
        let err = s.discover(&pattern).unwrap_err();
        prop_assert_eq!(err.error_data().error_code, ErrorCode::InvalidParams);
    }

    /// Stopped-only verbs are rejected with `MODEL_NOT_ATTACHED` from every
    /// pre-attach state, for arbitrary payloads.
    #[test]
    fn stopped_verbs_rejected_before_attach(
        initialize_first in any::<bool>(),
        tick_id in any::<u64>(),
        layer in 0u32..64,
    ) {
        let mut s = if initialize_first { initialized_session() } else { Session::new() };
        prop_assert_eq!(
            s.step(&step_req(), &tick_pos(tick_id, layer), false, vec![])
                .unwrap_err().error_data().error_code,
            ErrorCode::ModelNotAttached
        );
        prop_assert_eq!(
            s.discover("llama:*:*:*:output").unwrap_err().error_data().error_code,
            ErrorCode::ModelNotAttached
        );
        prop_assert_eq!(
            s.replay(&replay_req("any"), None).unwrap_err().error_data().error_code,
            ErrorCode::ModelNotAttached
        );
        // checkpoint guard lives in dispatch::handle_checkpoint via require_stopped.
        prop_assert_eq!(
            s.require_stopped("rocket/checkpoint").unwrap_err().error_data().error_code,
            ErrorCode::ModelNotAttached
        );
    }

    /// `detach` from any state without an attached model is `MODEL_NOT_ATTACHED`.
    #[test]
    fn detach_without_model_is_not_attached(initialize_first in any::<bool>()) {
        let mut s = if initialize_first { initialized_session() } else { Session::new() };
        prop_assert_eq!(
            s.detach().unwrap_err().error_data().error_code,
            ErrorCode::ModelNotAttached
        );
    }
}
