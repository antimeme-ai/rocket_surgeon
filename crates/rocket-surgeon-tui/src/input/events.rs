#[derive(Debug, Clone, PartialEq)]
pub enum InputEvent {
    Navigation(NavigationEvent),
    Command(CommandEvent),
    Mode(ModeEvent),
    Quit,
    Resize { width: u16, height: u16 },
}

// Some navigation intents are reserved for the daemon-connected view layer.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum NavigationEvent {
    Up,
    Down,
    Left,
    Right,
    PageUp,
    PageDown,
    Home,
    End,
    ZoomIn,
    ZoomOut,
    JumpTo(JumpTarget),
    ContinuousAdjust { axis: Axis, value: f32 },
}

// Command cancellation is reserved for the command executor slice.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CommandEvent {
    Char(char),
    Execute,
    Cancel,
    Backspace,
    TabComplete,
    HistoryPrev,
    HistoryNext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeEvent {
    EnterCommand,
    EnterInspect,
    EnterIntervene,
    ExitToNormal,
}

// Continuous adjustment axes are reserved for richer tensor/detail views.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Axis {
    Layer,
    TokenPosition,
    Head,
    Custom(String),
}

// Jump targets are reserved for command-mode navigation commands.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum JumpTarget {
    Layer(u32),
    Token(u64),
    Component(String),
}
