pub mod cache;
pub mod reducer;

use rocket_surgeon_protocol::types::{Capabilities, InterventionRecipe, Status, TickPosition};

use crate::input::mode::Mode;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ViewId(pub u32);

#[derive(Debug, Clone)]
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
    // In-flight scaffolding: view kinds for the planned panel set, built out
    // per-panel in BEAD-0015 slice 5. The compositor already dispatches on
    // `CommandLine`; the rest are not yet constructed.
    #[allow(dead_code)]
    TensorDetail,
    #[allow(dead_code)]
    ProbeWatch,
    #[allow(dead_code)]
    Timeline,
    #[allow(dead_code)]
    KvCache,
    #[allow(dead_code)]
    Worldline,
    #[allow(dead_code)]
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
    }
}
