use std::collections::VecDeque;
use std::time::Instant;

const DEFAULT_MAX_ENTRIES: usize = 10_000;

#[derive(Debug)]
#[allow(dead_code)]
pub struct TraceEntry {
    timestamp_us: u64,
    direction: Direction,
    raw: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Inbound,
    Outbound,
}

#[derive(Debug)]
pub struct TraceLog {
    entries: VecDeque<TraceEntry>,
    max_entries: usize,
    total_recorded: u64,
    epoch: Instant,
}

impl TraceLog {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries: DEFAULT_MAX_ENTRIES,
            total_recorded: 0,
            epoch: Instant::now(),
        }
    }

    pub fn record(&mut self, direction: Direction, raw: &str) {
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(TraceEntry {
            timestamp_us: self.epoch.elapsed().as_micros() as u64,
            direction,
            raw: raw.to_owned(),
        });
        self.total_recorded += 1;
    }

    pub fn len(&self) -> u64 {
        self.total_recorded
    }

    #[allow(dead_code)]
    pub fn buffered(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[allow(dead_code)]
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for entry in &self.entries {
            let dir = match entry.direction {
                Direction::Inbound => "in",
                Direction::Outbound => "out",
            };
            let line = serde_json::json!({
                "t": entry.timestamp_us,
                "d": dir,
                "msg": serde_json::from_str::<serde_json::Value>(&entry.raw)
                    .unwrap_or_else(|_| serde_json::Value::String(entry.raw.clone())),
            });
            out.push_str(&line.to_string());
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_log() {
        let log = TraceLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert_eq!(log.to_jsonl(), "");
    }

    #[test]
    fn trace_log_records_messages() {
        let mut log = TraceLog::new();
        log.record(
            Direction::Inbound,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
        );
        log.record(
            Direction::Outbound,
            r#"{"jsonrpc":"2.0","id":1,"result":{}}"#,
        );
        assert_eq!(log.len(), 2);
        assert_eq!(log.buffered(), 2);
    }

    #[test]
    fn trace_log_is_jsonl_format() {
        let mut log = TraceLog::new();
        log.record(
            Direction::Inbound,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
        );
        log.record(
            Direction::Outbound,
            r#"{"jsonrpc":"2.0","id":1,"result":{}}"#,
        );

        let jsonl = log.to_jsonl();
        let lines: Vec<&str> = jsonl.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["d"], "in");
        assert!(first["t"].is_u64());
        assert!(first["msg"].is_object());

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["d"], "out");
    }

    #[test]
    fn trace_log_preserves_order() {
        let mut log = TraceLog::new();
        for i in 0..5 {
            log.record(Direction::Inbound, &format!(r#"{{"seq":{i}}}"#));
        }
        let jsonl = log.to_jsonl();
        let lines: Vec<&str> = jsonl.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 5);
        for (i, line) in lines.iter().enumerate() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["msg"]["seq"], i);
        }
    }

    #[test]
    fn trace_log_handles_invalid_json() {
        let mut log = TraceLog::new();
        log.record(Direction::Inbound, "not json at all");
        let jsonl = log.to_jsonl();
        let entry: serde_json::Value = serde_json::from_str(jsonl.trim_end()).unwrap();
        assert_eq!(entry["msg"], "not json at all");
    }

    #[test]
    fn timestamps_are_monotonic() {
        let mut log = TraceLog::new();
        log.record(Direction::Inbound, r#"{"a":1}"#);
        log.record(Direction::Outbound, r#"{"b":2}"#);
        let jsonl = log.to_jsonl();
        let lines: Vec<&str> = jsonl.trim_end().split('\n').collect();
        let t0: u64 = serde_json::from_str::<serde_json::Value>(lines[0]).unwrap()["t"]
            .as_u64()
            .unwrap();
        let t1: u64 = serde_json::from_str::<serde_json::Value>(lines[1]).unwrap()["t"]
            .as_u64()
            .unwrap();
        assert!(t1 >= t0);
    }

    #[test]
    fn ring_buffer_evicts_oldest() {
        let mut log = TraceLog::new();
        log.max_entries = 3;
        for i in 0..5 {
            log.record(Direction::Inbound, &format!(r#"{{"seq":{i}}}"#));
        }
        assert_eq!(log.len(), 5);
        assert_eq!(log.buffered(), 3);
        let jsonl = log.to_jsonl();
        let lines: Vec<&str> = jsonl.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 3);
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["msg"]["seq"], 2);
    }
}
