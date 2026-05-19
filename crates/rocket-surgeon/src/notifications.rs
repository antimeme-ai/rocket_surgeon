use std::io::Write;

use rocket_surgeon_protocol::jsonrpc::Notification;
use rocket_surgeon_transport::framing::{FramingError, write_message};

pub fn send_notification(
    writer: &mut impl Write,
    seq: &mut u64,
    method: &str,
    mut params: serde_json::Value,
) -> Result<(), FramingError> {
    if let Some(obj) = params.as_object_mut() {
        obj.insert("seq".to_owned(), (*seq).into());
    }
    let notification = Notification::new(method, params);
    let json = serde_json::to_string(&notification).expect("serialize notification");
    write_message(writer, &json)?;
    *seq += 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocket_surgeon_transport::framing::read_message;
    use std::io::Cursor;

    #[test]
    fn send_notification_injects_seq_and_increments() {
        let mut buf = Vec::new();
        let mut seq = 1u64;
        let params = serde_json::json!({"position": {"tick_id": 42}});
        send_notification(&mut buf, &mut seq, "tick.stopped", params).unwrap();

        assert_eq!(seq, 2);
        let msg = read_message(&mut Cursor::new(buf)).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&msg).unwrap();
        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["method"], "tick.stopped");
        assert!(parsed.get("id").is_none());
        assert_eq!(parsed["params"]["seq"], 1);
        assert_eq!(parsed["params"]["position"]["tick_id"], 42);
    }

    #[test]
    fn send_notification_seq_starts_at_zero() {
        let mut buf = Vec::new();
        let mut seq = 0u64;
        send_notification(
            &mut buf,
            &mut seq,
            "tick.heartbeat",
            serde_json::json!({"uptime_seconds": 1.0}),
        )
        .unwrap();

        assert_eq!(seq, 1);
        let msg = read_message(&mut Cursor::new(buf)).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&msg).unwrap();
        assert_eq!(parsed["params"]["seq"], 0);
    }

    #[test]
    fn send_notification_multiple_increments_seq() {
        let mut buf = Vec::new();
        let mut seq = 0u64;

        send_notification(&mut buf, &mut seq, "tick.stopped", serde_json::json!({})).unwrap();
        send_notification(&mut buf, &mut seq, "probe.fired", serde_json::json!({})).unwrap();
        send_notification(&mut buf, &mut seq, "tick.heartbeat", serde_json::json!({})).unwrap();

        assert_eq!(seq, 3);

        let mut cursor = Cursor::new(buf);
        let m1: serde_json::Value =
            serde_json::from_str(&read_message(&mut cursor).unwrap()).unwrap();
        let m2: serde_json::Value =
            serde_json::from_str(&read_message(&mut cursor).unwrap()).unwrap();
        let m3: serde_json::Value =
            serde_json::from_str(&read_message(&mut cursor).unwrap()).unwrap();

        assert_eq!(m1["params"]["seq"], 0);
        assert_eq!(m2["params"]["seq"], 1);
        assert_eq!(m3["params"]["seq"], 2);
    }
}
