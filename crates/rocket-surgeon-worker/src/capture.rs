#![allow(dead_code)]

use crate::adapter::ComponentMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMode {
    None,
    Summary,
    Full,
}

pub fn capture_mode(active_probes: &[String]) -> CaptureMode {
    if active_probes.is_empty() {
        CaptureMode::None
    } else {
        CaptureMode::Summary
    }
}

pub fn should_capture(
    map: &ComponentMap,
    module_path: &str,
    call_index: u32,
    active_probes: &[String],
) -> bool {
    let component = map
        .components
        .iter()
        .find(|c| c.module_path == module_path && c.call_index == call_index);

    let Some(component) = component else {
        return false;
    };

    for probe_pattern in active_probes {
        if probe_matches(&component.probe_point, probe_pattern) {
            return true;
        }
    }
    false
}

pub fn probe_matches_target(probe_point: &str, target: &str) -> bool {
    probe_matches(probe_point, target)
}

fn probe_matches(probe_point: &str, pattern: &str) -> bool {
    let point_parts: Vec<&str> = probe_point.split(':').collect();
    let pattern_parts: Vec<&str> = pattern.split(':').collect();

    if point_parts.len() != pattern_parts.len() {
        return false;
    }

    for (pp, pat) in point_parts.iter().zip(pattern_parts.iter()) {
        if *pat == "*" {
            continue;
        }
        if pp != pat {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{ComponentMap, MappedComponent, ModuleMapping};

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

    #[test]
    fn probe_matches_exact_path() {
        let map = sample_component_map();
        let active = vec!["model:0:0:q_proj:0:fwd".to_owned()];
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
        let active = vec!["model:0:0:q_proj:0:fwd".to_owned()];
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
        let active = vec!["model:0:*:q_proj:0:fwd".to_owned()];
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
        let active = vec!["model:0:0:*:0:fwd".to_owned()];
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
        let active: Vec<String> = vec![];
        assert!(!should_capture(
            &map,
            "model.layers.0.self_attn.q_proj",
            0,
            &active
        ));
    }

    #[test]
    fn capture_policy_none_for_inactive() {
        assert_eq!(capture_mode(&[]), CaptureMode::None);
    }

    #[test]
    fn capture_policy_summary_is_default() {
        assert_eq!(
            capture_mode(&["model:0:0:q_proj:0:fwd".to_owned()]),
            CaptureMode::Summary
        );
    }
}
