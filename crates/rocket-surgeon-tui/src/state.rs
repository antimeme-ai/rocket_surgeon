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
    pub active_interventions: Vec<InterventionRecipe>,
    pub protocol_version: String,
}

#[derive(Debug, Clone)]
pub struct CursorState {
    pub layer: u32,
    pub component: String,
    pub token_position: u64,
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
    TensorDetail,
    ProbeWatch,
    Timeline,
    KvCache,
    Worldline,
    CommandLine,
    StatusBar,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataDep {
    SessionStatus,
    CursorPosition,
    TensorAt { layer: u32, component: String },
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
