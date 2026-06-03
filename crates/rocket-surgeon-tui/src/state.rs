pub mod cache;
pub mod reducer;

use rocket_surgeon_protocol::types::{Capabilities, InterventionRecipe, Status, TickPosition};

use crate::input::mode::Mode;
use crate::state::cache::TensorCache;

/// Default capacity for the TUI's in-memory tensor cache. Sized to comfortably
/// hold all components on a single layer for ~8 layers' worth of probe firings
/// at typical configurations — `TensorDetail` reads from it for the cursor's
/// focus.
pub const DEFAULT_TENSOR_CACHE_CAPACITY: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ViewId(pub u32);

#[derive(Debug)]
pub struct UiState {
    pub session: SessionSnapshot,
    pub cursor: CursorState,
    pub mode: Mode,
    pub views: Vec<ViewSlot>,
    /// Shown in the status bar; incremented once daemon requests are wired
    /// (BEAD-0015 slice 2). Always `0` in slice 1 — no producer yet.
    pub pending_requests: u32,
    pub status_line: String,
    pub command_buffer: String,
    /// LRU-bounded cache of captured tensor summaries, keyed by
    /// `(tick_id, probe_point)`. Populated by the inspect response path
    /// (separate slice); read by `TensorDetail`.
    pub tensor_cache: TensorCache,
}

#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    pub status: Status,
    /// Daemon-populated; written once the daemon link is wired (slice 2).
    #[allow(dead_code)]
    pub position: Option<TickPosition>,
    pub capabilities: Option<Capabilities>,
    /// Daemon-populated; written once the interventions flow is wired (slice 2).
    #[allow(dead_code)]
    pub active_interventions: Vec<InterventionRecipe>,
    /// Daemon-populated; set from the `connected` handshake (slice 2).
    #[allow(dead_code)]
    pub protocol_version: String,
}

#[derive(Debug, Clone)]
pub struct CursorState {
    pub layer: u32,
    pub component: String,
    pub token_position: u64,
    /// In-flight scaffolding: read once multi-view focus routing lands
    /// (BEAD-0015 slice 5).
    #[allow(dead_code)]
    pub focused_view: ViewId,
}

#[derive(Debug, Clone)]
pub struct ViewSlot {
    pub id: ViewId,
    pub kind: ViewKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewKind {
    LayerStack,
    // TensorDetail is implemented (slice 5b) but not in the default layout
    // — selectable once panel-swap commands land. The variant is constructed
    // in tests; the bin target still flags it dead until then.
    #[allow(dead_code)]
    TensorDetail,
    // In-flight scaffolding: view kinds for the planned panel set, built out
    // per-panel in BEAD-0015 slice 5. `StatusBar` and `CommandLine` are
    // backed by components and used in the default layout; the rest are not
    // yet constructed.
    #[allow(dead_code)]
    ProbeWatch,
    #[allow(dead_code)]
    Timeline,
    #[allow(dead_code)]
    KvCache,
    #[allow(dead_code)]
    Worldline,
    CommandLine,
    StatusBar,
}

pub fn initial_ui_state() -> UiState {
    UiState {
        session: SessionSnapshot {
            status: Status::Uninitialized,
            position: None,
            capabilities: None,
            active_interventions: Vec::new(),
            protocol_version: String::new(),
        },
        cursor: CursorState {
            layer: 0,
            component: String::new(),
            token_position: 0,
            focused_view: ViewId(0),
        },
        mode: Mode::default(),
        views: Vec::new(),
        pending_requests: 0,
        status_line: String::new(),
        command_buffer: String::new(),
        tensor_cache: TensorCache::new(DEFAULT_TENSOR_CACHE_CAPACITY),
    }
}
