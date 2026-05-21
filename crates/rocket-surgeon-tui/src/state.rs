pub mod cache;
pub mod diff;
pub mod reducer;

use std::collections::HashSet;

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
    pub pending_requests: u32,
    pub status_line: String,
    pub command_buffer: String,
    pub dirty: HashSet<ViewId>,
}

#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    pub status: Status,
    pub position: Option<TickPosition>,
    pub capabilities: Option<Capabilities>,
    // In-flight scaffolding: populated/read once the interventions panel is
    // wired up; the diff engine already compares this field in its tests.
    #[allow(dead_code)]
    pub active_interventions: Vec<InterventionRecipe>,
    pub protocol_version: String,
}

#[derive(Debug, Clone)]
pub struct CursorState {
    pub layer: u32,
    pub component: String,
    pub token_position: u64,
    // In-flight scaffolding: read once multi-view focus routing lands; the
    // diff engine already compares this field in its tests.
    #[allow(dead_code)]
    pub focused_view: ViewId,
}

#[derive(Debug, Clone)]
pub struct ViewSlot {
    pub id: ViewId,
    pub kind: ViewKind,
    pub data_deps: Vec<DataDep>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewKind {
    LayerStack,
    // In-flight scaffolding: these view kinds are defined for the planned
    // panel set; the compositor already dispatches on `CommandLine`. They are
    // not yet constructed by the bin, so `dead_code` is a false positive.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataDep {
    SessionStatus,
    CursorPosition,
    // In-flight scaffolding: data dependencies for the tensor and
    // interventions panels; the diff engine already references them in tests.
    #[allow(dead_code)]
    TensorAt {
        layer: u32,
        component: String,
    },
    #[allow(dead_code)]
    Interventions,
    Mode,
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
        dirty: HashSet::new(),
    }
}
