#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Normal,
    Command,
    Inspect,
    Intervene,
}

impl Mode {
    pub fn transition(self, target: Self) -> Option<Self> {
        match (self, target) {
            (from, to) if from == to => None,
            // From Normal, can enter any mode
            (Self::Normal, _) => Some(target),
            // From any mode, can return to Normal
            (_, Self::Normal) => Some(Self::Normal),
            // Direct transitions between non-Normal modes not allowed —
            // must go through Normal first
            _ => None,
        }
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
