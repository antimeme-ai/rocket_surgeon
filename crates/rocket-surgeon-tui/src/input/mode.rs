#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Command,
    Inspect,
    Intervene,
}

impl Mode {
    pub fn transition(self, target: Mode) -> Option<Mode> {
        match (self, target) {
            (from, to) if from == to => None,
            // From Normal, can enter any mode
            (Mode::Normal, _) => Some(target),
            // From any mode, can return to Normal
            (_, Mode::Normal) => Some(Mode::Normal),
            // Direct transitions between non-Normal modes not allowed —
            // must go through Normal first
            _ => None,
        }
    }
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_to_command() {
        assert_eq!(Mode::Normal.transition(Mode::Command), Some(Mode::Command));
    }

    #[test]
    fn normal_to_inspect() {
        assert_eq!(Mode::Normal.transition(Mode::Inspect), Some(Mode::Inspect));
    }

    #[test]
    fn normal_to_intervene() {
        assert_eq!(
            Mode::Normal.transition(Mode::Intervene),
            Some(Mode::Intervene)
        );
    }

    #[test]
    fn command_to_normal() {
        assert_eq!(Mode::Command.transition(Mode::Normal), Some(Mode::Normal));
    }

    #[test]
    fn inspect_to_normal() {
        assert_eq!(Mode::Inspect.transition(Mode::Normal), Some(Mode::Normal));
    }

    #[test]
    fn same_mode_is_no_op() {
        assert_eq!(Mode::Normal.transition(Mode::Normal), None);
        assert_eq!(Mode::Command.transition(Mode::Command), None);
    }

    #[test]
    fn command_to_inspect_rejected() {
        assert_eq!(Mode::Command.transition(Mode::Inspect), None);
    }

    #[test]
    fn inspect_to_intervene_rejected() {
        assert_eq!(Mode::Inspect.transition(Mode::Intervene), None);
    }

    #[test]
    fn default_is_normal() {
        assert_eq!(Mode::default(), Mode::Normal);
    }
}
