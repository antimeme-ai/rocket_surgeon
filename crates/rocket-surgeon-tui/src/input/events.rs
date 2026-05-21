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
    // In-flight scaffolding: absolute jumps and continuous (mouse/analog)
    // adjustment are reduced and unit-tested, but the terminal decoder does
    // not emit them yet. `dead_code` is a false positive here.
    #[allow(dead_code)]
    JumpTo(JumpTarget),
    #[allow(dead_code)]
    ContinuousAdjust {
        axis: Axis,
        value: f32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandEvent {
    Char(char),
    Execute,
    // In-flight scaffolding: explicit command cancel is handled by the
    // reducer but not yet emitted by the terminal decoder.
    #[allow(dead_code)]
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

// In-flight scaffolding: `Axis` and `JumpTarget` describe the payloads of the
// not-yet-emitted `NavigationEvent::ContinuousAdjust` / `JumpTo` variants (see
// the note above). They are intentional API, so `dead_code` is a false
// positive against the bin-only build.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Axis {
    Layer,
    TokenPosition,
    Head,
    Custom(String),
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JumpTarget {
    Layer(u32),
    Token(u64),
    Component(String),
}
