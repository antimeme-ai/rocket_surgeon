#![allow(dead_code)]

use rocket_surgeon_probes::grammar::ProbePoint;

use crate::adapter::ComponentMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMode {
    None,
    Summary,
    Full,
}

pub fn capture_mode(active_probes_count: usize) -> CaptureMode {
    if active_probes_count == 0 {
        CaptureMode::None
    } else {
        CaptureMode::Summary
    }
}

pub fn should_capture(
    map: &ComponentMap,
    module_path: &str,
    call_index: u32,
    active_probes: &[(rocket_surgeon_protocol::types::ProbeDefinition, ProbePoint)],
) -> bool {
    let component = map
        .components
        .iter()
        .find(|c| c.module_path == module_path && c.call_index == call_index);

    let Some(component) = component else {
        return false;
    };

    let Ok(target) = ProbePoint::parse(&component.probe_point) else {
        return false;
    };

    active_probes.iter().any(|(_, pp)| pp.matches(&target))
}

pub fn probe_matches_target(probe_point: &str, target: &str) -> bool {
    let Ok(point) = ProbePoint::parse(probe_point) else {
        return false;
    };
    let Ok(tgt) = ProbePoint::parse(target) else {
        return false;
    };
    point.matches(&tgt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{ComponentMap, MappedComponent, ModuleMapping};
    use rocket_surgeon_protocol::types::{ProbeAction, ProbeDefinition};

    fn sample_component_map() -> ComponentMap {
        ComponentMap {
            components: vec![
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
                    module_path: "model.layers.0.self_attn.k_proj".into(),
                    canonical: "k_proj".into(),
                    layer_index: Some(0),
                    call_index: 0,
                    mapping: ModuleMapping::Direct {
                        canonical: "k_proj".into(),
                    },
                    probe_point: "model:0:0:k_proj:0:fwd".into(),
                },
            ],
            model_family: "llama".into(),
            vocabulary: vec!["q_proj".into(), "k_proj".into()],
        }
    }

    fn make_active(point: &str) -> (ProbeDefinition, ProbePoint) {
        let def = ProbeDefinition {
            id: "test".to_owned(),
            point: point.to_owned(),
            action: ProbeAction::Capture,
            config: None,
            enabled: true,
            priority: 0,
        };
        let pp = ProbePoint::parse(point).unwrap();
        (def, pp)
    }

    #[test]
    fn probe_matches_exact_path() {
        let map = sample_component_map();
        let active = vec![make_active("model:0:0:q_proj:0:fwd")];
        assert!(should_capture(
            &map,
            "model.layers.0.self_attn.q_proj",
            0,
            &active
        ));
    }

    #[test]
    fn probe_does_not_match_different_component() {
        let map = sample_component_map();
        let active = vec![make_active("model:0:0:q_proj:0:fwd")];
        assert!(!should_capture(
            &map,
            "model.layers.0.self_attn.k_proj",
            0,
            &active
        ));
    }

    #[test]
    fn wildcard_layer_matches_any_layer() {
        let map = sample_component_map();
        let active = vec![make_active("model:0:*:q_proj:0:fwd")];
        assert!(should_capture(
            &map,
            "model.layers.0.self_attn.q_proj",
            0,
            &active
        ));
    }

    #[test]
    fn wildcard_component_matches_all() {
        let map = sample_component_map();
        let active = vec![make_active("model:0:0:*:0:fwd")];
        assert!(should_capture(
            &map,
            "model.layers.0.self_attn.q_proj",
            0,
            &active
        ));
        assert!(should_capture(
            &map,
            "model.layers.0.self_attn.k_proj",
            0,
            &active
        ));
    }

    #[test]
    fn empty_active_probes_matches_nothing() {
        let map = sample_component_map();
        let active: Vec<(ProbeDefinition, ProbePoint)> = vec![];
        assert!(!should_capture(
            &map,
            "model.layers.0.self_attn.q_proj",
            0,
            &active
        ));
    }

    #[test]
    fn capture_policy_none_for_inactive() {
        assert_eq!(capture_mode(0), CaptureMode::None);
    }

    #[test]
    fn capture_policy_summary_is_default() {
        assert_eq!(capture_mode(1), CaptureMode::Summary);
    }

    #[test]
    fn grammar_based_matching_works() {
        assert!(probe_matches_target(
            "model:0:0:q_proj:0:fwd",
            "*:*:*:*:*:*"
        ));
        assert!(!probe_matches_target(
            "model:0:0:q_proj:0:fwd",
            "model:0:0:k_proj:0:fwd"
        ));
    }
}
