#[derive(Debug, Clone, PartialEq)]
pub enum InputEvent {
    Navigation(NavigationEvent),
    Command(CommandEvent),
    Mode(ModeEvent),
    Quit,
    Resize { width: u16, height: u16 },
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
pub enum Axis {
    Layer,
    TokenPosition,
    Head,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum JumpTarget {
    Layer(u32),
    Token(u64),
    Component(String),
}
