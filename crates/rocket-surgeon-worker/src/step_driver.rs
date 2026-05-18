use rocket_surgeon_protocol::types::TickGranularity;

pub struct StepPlan {
    pub ticks_to_drain: u32,
    pub granularity: TickGranularity,
}

pub fn plan_step(count: u32, granularity: Option<TickGranularity>) -> StepPlan {
    StepPlan {
        ticks_to_drain: count,
        granularity: granularity.unwrap_or(TickGranularity::Component),
    }
}

pub fn is_layer_boundary(current_layer: Option<u32>, new_layer: u32) -> bool {
    match current_layer {
        None => false,
        Some(prev) => new_layer != prev,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_step_defaults_to_component() {
        let plan = plan_step(3, None);
        assert_eq!(plan.ticks_to_drain, 3);
        assert_eq!(plan.granularity, TickGranularity::Component);
    }

    #[test]
    fn plan_step_respects_explicit_granularity() {
        let plan = plan_step(1, Some(TickGranularity::Layer));
        assert_eq!(plan.granularity, TickGranularity::Layer);
    }

    #[test]
    fn is_layer_boundary_detects_change() {
        assert!(is_layer_boundary(Some(0), 1));
        assert!(!is_layer_boundary(Some(0), 0));
        assert!(!is_layer_boundary(None, 0));
    }
}
