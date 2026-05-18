#![allow(dead_code)]

use crate::adapter::{ComponentMap, MappedComponent};
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

pub fn lookup_component<'a>(
    map: &'a ComponentMap,
    module_path: &str,
    call_index: u32,
) -> Option<&'a MappedComponent> {
    map.components
        .iter()
        .find(|c| c.module_path == module_path && c.call_index == call_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::ModuleMapping;

    fn sample_map() -> ComponentMap {
        ComponentMap {
            components: vec![
                MappedComponent {
                    module_path: "model.layers.0.input_layernorm".into(),
                    canonical: "ln1".into(),
                    layer_index: Some(0),
                    call_index: 0,
                    mapping: ModuleMapping::Direct {
                        canonical: "ln1".into(),
                    },
                    probe_point: "model:0:0:ln1:0:fwd".into(),
                },
                MappedComponent {
                    module_path: "model.layers.0.self_attn.q_proj".into(),
                    canonical: "q_proj".into(),
                    layer_index: Some(0),
                    call_index: 0,
                    mapping: ModuleMapping::Direct {
                        canonical: "q_proj".into(),
                    },
                    probe_point: "model:0:0:q_proj:0:fwd".into(),
                },
                MappedComponent {
                    module_path: "model.layers.1.input_layernorm".into(),
                    canonical: "ln1".into(),
                    layer_index: Some(1),
                    call_index: 0,
                    mapping: ModuleMapping::Direct {
                        canonical: "ln1".into(),
                    },
                    probe_point: "model:0:1:ln1:0:fwd".into(),
                },
            ],
            model_family: "llama".into(),
            vocabulary: vec!["ln1".into(), "q_proj".into()],
        }
    }

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

    #[test]
    fn lookup_component_finds_by_path_and_index() {
        let map = sample_map();
        let comp = lookup_component(&map, "model.layers.0.self_attn.q_proj", 0);
        assert!(comp.is_some());
        assert_eq!(comp.unwrap().canonical, "q_proj");
    }

    #[test]
    fn lookup_component_returns_none_for_unknown() {
        let map = sample_map();
        let comp = lookup_component(&map, "nonexistent.path", 0);
        assert!(comp.is_none());
    }
}
