use std::collections::HashMap;

use rocket_surgeon_protocol::types::ProbeDefinition;

use crate::grammar::{ParseError, ProbePoint};

#[derive(Debug, Clone, thiserror::Error)]
pub enum RegistryError {
    #[error("probe not found: {id}")]
    NotFound { id: String },
    #[error("duplicate probe id: {id}")]
    DuplicateId { id: String },
    #[error("invalid probe point: {0}")]
    InvalidPoint(#[from] ParseError),
}

#[derive(Debug)]
struct StoredProbe {
    definition: ProbeDefinition,
    parsed_point: ProbePoint,
    seq: u64,
}

#[derive(Debug, Default)]
pub struct ProbeRegistry {
    probes: HashMap<String, StoredProbe>,
    next_seq: u64,
}

impl ProbeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn define(&mut self, probe: ProbeDefinition) -> Result<String, RegistryError> {
        if self.probes.contains_key(&probe.id) {
            return Err(RegistryError::DuplicateId { id: probe.id });
        }
        let parsed_point = ProbePoint::parse(&probe.point)?;
        let seq = self.next_seq;
        self.next_seq += 1;
        let id = probe.id.clone();
        self.probes.insert(
            id.clone(),
            StoredProbe {
                definition: probe,
                parsed_point,
                seq,
            },
        );
        Ok(id)
    }

    pub fn get(&self, id: &str) -> Option<&ProbeDefinition> {
        self.probes.get(id).map(|s| &s.definition)
    }

    pub fn list(&self) -> Vec<ProbeDefinition> {
        let mut probes: Vec<_> = self.probes.values().collect();
        probes.sort_by_key(|s| (s.definition.priority, s.seq));
        probes.into_iter().map(|s| s.definition.clone()).collect()
    }

    pub fn enable(&mut self, id: &str) -> Result<ProbeDefinition, RegistryError> {
        let stored = self
            .probes
            .get_mut(id)
            .ok_or_else(|| RegistryError::NotFound { id: id.to_owned() })?;
        stored.definition.enabled = true;
        Ok(stored.definition.clone())
    }

    pub fn disable(&mut self, id: &str) -> Result<ProbeDefinition, RegistryError> {
        let stored = self
            .probes
            .get_mut(id)
            .ok_or_else(|| RegistryError::NotFound { id: id.to_owned() })?;
        stored.definition.enabled = false;
        Ok(stored.definition.clone())
    }

    pub fn remove(&mut self, id: &str) -> Result<ProbeDefinition, RegistryError> {
        self.probes
            .remove(id)
            .map(|s| s.definition)
            .ok_or_else(|| RegistryError::NotFound { id: id.to_owned() })
    }

    pub fn matching(&self, target: &ProbePoint) -> Vec<ProbeDefinition> {
        let mut matched: Vec<_> = self
            .probes
            .values()
            .filter(|s| s.parsed_point.matches(target))
            .collect();
        matched.sort_by_key(|s| (s.definition.priority, s.seq));
        matched.into_iter().map(|s| s.definition.clone()).collect()
    }

    pub fn matching_enabled(&self, target: &ProbePoint) -> Vec<ProbeDefinition> {
        let mut matched: Vec<_> = self
            .probes
            .values()
            .filter(|s| s.definition.enabled && s.parsed_point.matches(target))
            .collect();
        matched.sort_by_key(|s| (s.definition.priority, s.seq));
        matched.into_iter().map(|s| s.definition.clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.probes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.probes.is_empty()
    }

    pub fn active_probe_ids(&self) -> Vec<String> {
        self.probes
            .values()
            .filter(|s| s.definition.enabled)
            .map(|s| s.definition.id.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rocket_surgeon_protocol::types::ProbeAction;

    use super::*;

    fn capture_probe(id: &str, point: &str) -> ProbeDefinition {
        ProbeDefinition {
            id: id.to_owned(),
            point: point.to_owned(),
            action: ProbeAction::Capture,
            config: None,
            enabled: true,
            priority: 0,
        }
    }

    #[test]
    fn define_and_get() {
        let mut reg = ProbeRegistry::new();
        let id = reg
            .define(capture_probe("p1", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();
        assert_eq!(id, "p1");
        assert!(reg.get("p1").is_some());
        assert_eq!(
            reg.get("p1").unwrap().point,
            "llama:0:12:attn.o_proj:0:output"
        );
    }

    #[test]
    fn define_duplicate_rejected() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();
        let err = reg
            .define(capture_probe("p1", "llama:0:12:mlp:0:output"))
            .unwrap_err();
        assert!(matches!(err, RegistryError::DuplicateId { .. }));
    }

    #[test]
    fn define_invalid_point_rejected() {
        let mut reg = ProbeRegistry::new();
        let err = reg.define(capture_probe("p1", "bad")).unwrap_err();
        assert!(matches!(err, RegistryError::InvalidPoint(_)));
    }

    #[test]
    fn list_returns_all_sorted() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();
        reg.define(capture_probe("p2", "llama:0:12:mlp:0:output"))
            .unwrap();
        let list = reg.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, "p1");
        assert_eq!(list[1].id, "p2");
    }

    #[test]
    fn enable_and_disable() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();

        let probe = reg.disable("p1").unwrap();
        assert!(!probe.enabled);

        let probe = reg.enable("p1").unwrap();
        assert!(probe.enabled);
    }

    #[test]
    fn enable_already_enabled_is_idempotent() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();
        let probe = reg.enable("p1").unwrap();
        assert!(probe.enabled);
    }

    #[test]
    fn disable_already_disabled_is_idempotent() {
        let mut reg = ProbeRegistry::new();
        let mut p = capture_probe("p1", "llama:0:12:attn.o_proj:0:output");
        p.enabled = false;
        reg.define(p).unwrap();
        let probe = reg.disable("p1").unwrap();
        assert!(!probe.enabled);
    }

    #[test]
    fn enable_nonexistent_returns_not_found() {
        let mut reg = ProbeRegistry::new();
        let err = reg.enable("ghost").unwrap_err();
        assert!(matches!(err, RegistryError::NotFound { .. }));
    }

    #[test]
    fn disable_nonexistent_returns_not_found() {
        let mut reg = ProbeRegistry::new();
        let err = reg.disable("ghost").unwrap_err();
        assert!(matches!(err, RegistryError::NotFound { .. }));
    }

    #[test]
    fn remove_probe() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();
        let removed = reg.remove("p1").unwrap();
        assert_eq!(removed.id, "p1");
        assert!(reg.get("p1").is_none());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn remove_nonexistent_returns_not_found() {
        let mut reg = ProbeRegistry::new();
        let err = reg.remove("ghost").unwrap_err();
        assert!(matches!(err, RegistryError::NotFound { .. }));
    }

    #[test]
    fn matching_wildcard_finds_all() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();
        reg.define(capture_probe("p2", "llama:0:13:mlp:0:input"))
            .unwrap();

        let wildcard = ProbePoint::parse("*:*:*:*:*:*").unwrap();
        let matched = reg.matching(&wildcard);
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn matching_specific_filters_correctly() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();
        reg.define(capture_probe("p2", "llama:0:13:mlp:0:input"))
            .unwrap();

        let target = ProbePoint::parse("llama:0:12:attn.o_proj:0:output").unwrap();
        let matched = reg.matching(&target);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].id, "p1");
    }

    #[test]
    fn matching_partial_wildcard_probe() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "llama:*:*:mlp:*:output"))
            .unwrap();
        reg.define(capture_probe("p2", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();

        let target = ProbePoint::parse("llama:0:5:mlp:0:output").unwrap();
        let matched = reg.matching(&target);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].id, "p1");
    }

    #[test]
    fn matching_returns_empty_when_no_match() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();

        let target = ProbePoint::parse("mixtral:0:0:mlp:0:input").unwrap();
        let matched = reg.matching(&target);
        assert!(matched.is_empty());
    }

    #[test]
    fn matching_wildcard_probe_point() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "*:*:*:*:*:*")).unwrap();

        let target = ProbePoint::parse("llama:0:12:attn.o_proj:0:output").unwrap();
        let matched = reg.matching(&target);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].id, "p1");
    }

    #[test]
    fn matching_enabled_excludes_disabled() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "*:*:*:*:*:*")).unwrap();
        reg.define(capture_probe("p2", "*:*:*:*:*:*")).unwrap();
        reg.disable("p2").unwrap();

        let wildcard = ProbePoint::parse("*:*:*:*:*:*").unwrap();
        let matched = reg.matching_enabled(&wildcard);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].id, "p1");
    }

    #[test]
    fn matching_sorted_by_priority() {
        let mut reg = ProbeRegistry::new();
        let mut p1 = capture_probe("p1", "*:*:*:*:*:*");
        p1.priority = 10;
        let mut p2 = capture_probe("p2", "*:*:*:*:*:*");
        p2.priority = 0;
        let mut p3 = capture_probe("p3", "*:*:*:*:*:*");
        p3.priority = 5;

        reg.define(p1).unwrap();
        reg.define(p2).unwrap();
        reg.define(p3).unwrap();

        let wildcard = ProbePoint::parse("*:*:*:*:*:*").unwrap();
        let matched = reg.matching(&wildcard);
        assert_eq!(matched.len(), 3);
        assert_eq!(matched[0].id, "p2");
        assert_eq!(matched[1].id, "p3");
        assert_eq!(matched[2].id, "p1");
    }

    #[test]
    fn matching_enabled_sorted_by_priority() {
        let mut reg = ProbeRegistry::new();
        let mut p1 = capture_probe("p1", "*:*:*:*:*:*");
        p1.priority = 10;
        let mut p2 = capture_probe("p2", "*:*:*:*:*:*");
        p2.priority = 0;

        reg.define(p1).unwrap();
        reg.define(p2).unwrap();

        let wildcard = ProbePoint::parse("*:*:*:*:*:*").unwrap();
        let matched = reg.matching_enabled(&wildcard);
        assert_eq!(matched[0].id, "p2");
        assert_eq!(matched[1].id, "p1");
    }

    #[test]
    fn priority_ties_broken_by_insertion_order() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("first", "*:*:*:*:*:*")).unwrap();
        reg.define(capture_probe("second", "*:*:*:*:*:*")).unwrap();
        reg.define(capture_probe("third", "*:*:*:*:*:*")).unwrap();

        let wildcard = ProbePoint::parse("*:*:*:*:*:*").unwrap();
        let matched = reg.matching(&wildcard);
        assert_eq!(matched[0].id, "first");
        assert_eq!(matched[1].id, "second");
        assert_eq!(matched[2].id, "third");
    }

    #[test]
    fn active_probe_ids() {
        let mut reg = ProbeRegistry::new();
        reg.define(capture_probe("p1", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();
        reg.define(capture_probe("p2", "llama:0:13:mlp:0:input"))
            .unwrap();
        reg.disable("p2").unwrap();

        let active = reg.active_probe_ids();
        assert_eq!(active.len(), 1);
        assert!(active.contains(&"p1".to_owned()));
    }

    #[test]
    fn is_empty_and_len() {
        let mut reg = ProbeRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);

        reg.define(capture_probe("p1", "llama:0:12:attn.o_proj:0:output"))
            .unwrap();
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);
    }
}
