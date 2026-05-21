use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};

use super::events::{CommandEvent, InputEvent, ModeEvent, NavigationEvent};
use super::mode::Mode;

pub fn decode(event: &Event, mode: Mode) -> Option<InputEvent> {
    match event {
        Event::Key(key) => decode_key(*key, mode),
        Event::Resize(w, h) => Some(InputEvent::Resize {
            width: *w,
            height: *h,
        }),
        _ => None,
    }
}

fn decode_key(key: KeyEvent, mode: Mode) -> Option<InputEvent> {
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return decode_ctrl(key.code, mode);
    }

    match mode {
        Mode::Normal => decode_normal(key.code),
        Mode::Command => decode_command(key.code),
        Mode::Inspect => decode_inspect(key.code),
        Mode::Intervene => decode_intervene(key.code),
    }
}

fn decode_ctrl(code: KeyCode, _mode: Mode) -> Option<InputEvent> {
    match code {
        KeyCode::Char('c' | 'q') => Some(InputEvent::Quit),
        _ => None,
    }
}

fn decode_normal(code: KeyCode) -> Option<InputEvent> {
    match code {
        // Navigation
        KeyCode::Up | KeyCode::Char('k') => Some(InputEvent::Navigation(NavigationEvent::Up)),
        KeyCode::Down | KeyCode::Char('j') => Some(InputEvent::Navigation(NavigationEvent::Down)),
        KeyCode::Left | KeyCode::Char('h') => Some(InputEvent::Navigation(NavigationEvent::Left)),
        KeyCode::Right | KeyCode::Char('l') => Some(InputEvent::Navigation(NavigationEvent::Right)),
        KeyCode::PageUp => Some(InputEvent::Navigation(NavigationEvent::PageUp)),
        KeyCode::PageDown => Some(InputEvent::Navigation(NavigationEvent::PageDown)),
        KeyCode::Home => Some(InputEvent::Navigation(NavigationEvent::Home)),
        KeyCode::End => Some(InputEvent::Navigation(NavigationEvent::End)),
        KeyCode::Char('+' | '=') => Some(InputEvent::Navigation(NavigationEvent::ZoomIn)),
        KeyCode::Char('-') => Some(InputEvent::Navigation(NavigationEvent::ZoomOut)),

        // Mode transitions
        KeyCode::Char(':') => Some(InputEvent::Mode(ModeEvent::EnterCommand)),
        KeyCode::Char('i') => Some(InputEvent::Mode(ModeEvent::EnterInspect)),
        KeyCode::Char('I') => Some(InputEvent::Mode(ModeEvent::EnterIntervene)),

        KeyCode::Char('q') => Some(InputEvent::Quit),
        _ => None,
    }
}

fn decode_command(code: KeyCode) -> Option<InputEvent> {
    match code {
        KeyCode::Enter => Some(InputEvent::Command(CommandEvent::Execute)),
        KeyCode::Esc => Some(InputEvent::Mode(ModeEvent::ExitToNormal)),
        KeyCode::Backspace => Some(InputEvent::Command(CommandEvent::Backspace)),
        KeyCode::Tab => Some(InputEvent::Command(CommandEvent::TabComplete)),
        KeyCode::Up => Some(InputEvent::Command(CommandEvent::HistoryPrev)),
        KeyCode::Down => Some(InputEvent::Command(CommandEvent::HistoryNext)),
        KeyCode::Char(c) => Some(InputEvent::Command(CommandEvent::Char(c))),
        _ => None,
    }
}

fn decode_inspect(code: KeyCode) -> Option<InputEvent> {
    match code {
        KeyCode::Esc => Some(InputEvent::Mode(ModeEvent::ExitToNormal)),
        // Navigation still works in inspect mode
        KeyCode::Up | KeyCode::Char('k') => Some(InputEvent::Navigation(NavigationEvent::Up)),
        KeyCode::Down | KeyCode::Char('j') => Some(InputEvent::Navigation(NavigationEvent::Down)),
        KeyCode::Left | KeyCode::Char('h') => Some(InputEvent::Navigation(NavigationEvent::Left)),
        KeyCode::Right | KeyCode::Char('l') => Some(InputEvent::Navigation(NavigationEvent::Right)),
        KeyCode::Char('+' | '=') => Some(InputEvent::Navigation(NavigationEvent::ZoomIn)),
        KeyCode::Char('-') => Some(InputEvent::Navigation(NavigationEvent::ZoomOut)),
        _ => None,
    }
}

fn decode_intervene(code: KeyCode) -> Option<InputEvent> {
    match code {
        KeyCode::Esc => Some(InputEvent::Mode(ModeEvent::ExitToNormal)),
        KeyCode::Up | KeyCode::Char('k') => Some(InputEvent::Navigation(NavigationEvent::Up)),
        KeyCode::Down | KeyCode::Char('j') => Some(InputEvent::Navigation(NavigationEvent::Down)),
        KeyCode::Left | KeyCode::Char('h') => Some(InputEvent::Navigation(NavigationEvent::Left)),
        KeyCode::Right | KeyCode::Char('l') => Some(InputEvent::Navigation(NavigationEvent::Right)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEventKind, KeyEventState};

    fn key(code: KeyCode) -> Event {
        Event::Key(KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
    }

    fn ctrl(code: KeyCode) -> Event {
        Event::Key(KeyEvent {
            code,
            modifiers: KeyModifiers::CONTROL,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
    }

    // Normal mode navigation
    #[test]
    fn normal_vim_keys() {
        assert_eq!(
            decode(&key(KeyCode::Char('j')), Mode::Normal),
            Some(InputEvent::Navigation(NavigationEvent::Down))
        );
        assert_eq!(
            decode(&key(KeyCode::Char('k')), Mode::Normal),
            Some(InputEvent::Navigation(NavigationEvent::Up))
        );
        assert_eq!(
            decode(&key(KeyCode::Char('h')), Mode::Normal),
            Some(InputEvent::Navigation(NavigationEvent::Left))
        );
        assert_eq!(
            decode(&key(KeyCode::Char('l')), Mode::Normal),
            Some(InputEvent::Navigation(NavigationEvent::Right))
        );
    }

    #[test]
    fn normal_arrow_keys() {
        assert_eq!(
            decode(&key(KeyCode::Up), Mode::Normal),
            Some(InputEvent::Navigation(NavigationEvent::Up))
        );
        assert_eq!(
            decode(&key(KeyCode::Down), Mode::Normal),
            Some(InputEvent::Navigation(NavigationEvent::Down))
        );
    }

    #[test]
    fn normal_zoom() {
        assert_eq!(
            decode(&key(KeyCode::Char('+')), Mode::Normal),
            Some(InputEvent::Navigation(NavigationEvent::ZoomIn))
        );
        assert_eq!(
            decode(&key(KeyCode::Char('-')), Mode::Normal),
            Some(InputEvent::Navigation(NavigationEvent::ZoomOut))
        );
    }

    // Mode transitions
    #[test]
    fn colon_enters_command() {
        assert_eq!(
            decode(&key(KeyCode::Char(':')), Mode::Normal),
            Some(InputEvent::Mode(ModeEvent::EnterCommand))
        );
    }

    #[test]
    fn i_enters_inspect() {
        assert_eq!(
            decode(&key(KeyCode::Char('i')), Mode::Normal),
            Some(InputEvent::Mode(ModeEvent::EnterInspect))
        );
    }

    #[test]
    fn shift_i_enters_intervene() {
        assert_eq!(
            decode(&key(KeyCode::Char('I')), Mode::Normal),
            Some(InputEvent::Mode(ModeEvent::EnterIntervene))
        );
    }

    #[test]
    fn esc_exits_command() {
        assert_eq!(
            decode(&key(KeyCode::Esc), Mode::Command),
            Some(InputEvent::Mode(ModeEvent::ExitToNormal))
        );
    }

    #[test]
    fn esc_exits_inspect() {
        assert_eq!(
            decode(&key(KeyCode::Esc), Mode::Inspect),
            Some(InputEvent::Mode(ModeEvent::ExitToNormal))
        );
    }

    // Command mode
    #[test]
    fn command_chars() {
        assert_eq!(
            decode(&key(KeyCode::Char('a')), Mode::Command),
            Some(InputEvent::Command(CommandEvent::Char('a')))
        );
    }

    #[test]
    fn command_enter_executes() {
        assert_eq!(
            decode(&key(KeyCode::Enter), Mode::Command),
            Some(InputEvent::Command(CommandEvent::Execute))
        );
    }

    #[test]
    fn command_history() {
        assert_eq!(
            decode(&key(KeyCode::Up), Mode::Command),
            Some(InputEvent::Command(CommandEvent::HistoryPrev))
        );
        assert_eq!(
            decode(&key(KeyCode::Down), Mode::Command),
            Some(InputEvent::Command(CommandEvent::HistoryNext))
        );
    }

    // Ctrl combos
    #[test]
    fn ctrl_c_quits() {
        assert_eq!(
            decode(&ctrl(KeyCode::Char('c')), Mode::Normal),
            Some(InputEvent::Quit)
        );
    }

    #[test]
    fn ctrl_c_quits_from_any_mode() {
        assert_eq!(
            decode(&ctrl(KeyCode::Char('c')), Mode::Command),
            Some(InputEvent::Quit)
        );
        assert_eq!(
            decode(&ctrl(KeyCode::Char('c')), Mode::Inspect),
            Some(InputEvent::Quit)
        );
    }

    // Inspect mode retains navigation
    #[test]
    fn inspect_navigation() {
        assert_eq!(
            decode(&key(KeyCode::Char('j')), Mode::Inspect),
            Some(InputEvent::Navigation(NavigationEvent::Down))
        );
    }

    // Resize
    #[test]
    fn resize_event() {
        assert_eq!(
            decode(&Event::Resize(120, 40), Mode::Normal),
            Some(InputEvent::Resize {
                width: 120,
                height: 40
            })
        );
    }
}
