use std::collections::HashMap;

pub struct InternTable {
    names: HashMap<String, u64>,
    next_iid: u64,
}

impl Default for InternTable {
    fn default() -> Self {
        Self::new()
    }
}

impl InternTable {
    pub fn new() -> Self {
        Self {
            names: HashMap::new(),
            next_iid: 1,
        }
    }

    pub fn intern(&mut self, name: &str) -> u64 {
        if let Some(&iid) = self.names.get(name) {
            return iid;
        }
        let iid = self.next_iid;
        self.next_iid += 1;
        self.names.insert(name.to_owned(), iid);
        iid
    }

    pub fn get(&self, name: &str) -> Option<u64> {
        self.names.get(name).copied()
    }

    pub fn entries(&self) -> impl Iterator<Item = (u64, &str)> {
        self.names.iter().map(|(name, &iid)| (iid, name.as_str()))
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_returns_stable_iid() {
        let mut table = InternTable::new();
        let iid1 = table.intern("L0::attn::q_proj");
        let iid2 = table.intern("L0::attn::q_proj");
        assert_eq!(iid1, iid2);
    }

    #[test]
    fn intern_assigns_unique_iids() {
        let mut table = InternTable::new();
        let iid1 = table.intern("L0::attn::q_proj");
        let iid2 = table.intern("L0::attn::k_proj");
        assert_ne!(iid1, iid2);
    }

    #[test]
    fn iids_start_at_one() {
        let mut table = InternTable::new();
        let iid = table.intern("first");
        assert_eq!(iid, 1);
    }

    #[test]
    fn get_returns_none_for_unknown() {
        let table = InternTable::new();
        assert_eq!(table.get("unknown"), None);
    }

    #[test]
    fn get_returns_iid_for_known() {
        let mut table = InternTable::new();
        let iid = table.intern("known");
        assert_eq!(table.get("known"), Some(iid));
    }

    #[test]
    fn entries_returns_all_pairs() {
        let mut table = InternTable::new();
        table.intern("alpha");
        table.intern("beta");
        table.intern("gamma");
        let mut entries: Vec<(u64, String)> = table
            .entries()
            .map(|(iid, name)| (iid, name.to_owned()))
            .collect();
        entries.sort_by_key(|(iid, _)| *iid);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].1, "alpha");
        assert_eq!(entries[1].1, "beta");
        assert_eq!(entries[2].1, "gamma");
    }

    #[test]
    fn len_tracks_unique_entries() {
        let mut table = InternTable::new();
        assert_eq!(table.len(), 0);
        table.intern("a");
        table.intern("b");
        table.intern("a");
        assert_eq!(table.len(), 2);
    }
}
