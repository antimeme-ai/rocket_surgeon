use std::io::Write;

use rocket_surgeon_protocol::jsonrpc::Notification;
use rocket_surgeon_protocol::messages::{EventType, SubscribeFilter, event};
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

/// Send a notification only if it passes the subscriber's [`SubscribeFilter`].
///
/// This is the event fan-out filter for `subscribe-filter.feature`. A `None`
/// filter (no subscription filter negotiated) delivers everything, exactly
/// like [`send_notification`]. When a filter is present, an event that fails
/// [`event_matches_filter`] is silently dropped — it is neither written to
/// the wire nor does it consume a `seq` number, so the subscriber's view of
/// the event stream stays gap-free for the events it did ask for.
pub fn send_notification_filtered(
    writer: &mut impl Write,
    seq: &mut u64,
    method: &str,
    params: serde_json::Value,
    filter: Option<&SubscribeFilter>,
) -> Result<(), FramingError> {
    if !event_matches_filter(filter, method, &params) {
        return Ok(());
    }
    send_notification(writer, seq, method, params)
}

/// Map a notification method string to its [`EventType`].
fn method_to_event_type(method: &str) -> Option<EventType> {
    match method {
        event::TICK_STOPPED => Some(EventType::TickStopped),
        event::TICK_HEARTBEAT => Some(EventType::TickHeartbeat),
        event::PROBE_FIRED => Some(EventType::ProbeFired),
        event::REPLAY_DIVERGENCE => Some(EventType::ReplayDivergence),
        event::ERROR => Some(EventType::Error),
        _ => None,
    }
}

/// Extract the `(layer, component)` an event pertains to, if it has one.
///
/// `tick.stopped` carries a `TickPosition` with explicit `layer`/`component`
/// fields. `probe.fired` carries a `point` string of the canonical form
/// `model:rank:layer:component:call_index:event`, from which the layer and
/// component segment are pulled. Events with no spatial coordinate (e.g.
/// `tick.heartbeat`) return `(None, None)`.
fn event_coordinates(method: &str, params: &serde_json::Value) -> (Option<u32>, Option<String>) {
    match method {
        event::TICK_STOPPED => {
            let pos = &params["position"];
            let layer = pos["layer"].as_u64().and_then(|n| u32::try_from(n).ok());
            let component = pos["component"].as_str().map(str::to_owned);
            (layer, component)
        }
        event::PROBE_FIRED => {
            let Some(point) = params["point"].as_str() else {
                return (None, None);
            };
            // model:rank:layer:component:call_index:event
            let mut parts = point.split(':');
            let _model = parts.next();
            let _rank = parts.next();
            let layer = parts.next().and_then(|s| s.parse::<u32>().ok());
            let component = parts.next().map(str::to_owned);
            (layer, component)
        }
        _ => (None, None),
    }
}

/// Glob match supporting a single trailing `*` wildcard, e.g. `attn.*`.
///
/// A bare `*` matches anything; a pattern with no `*` is an exact match.
/// This deliberately mirrors the simple prefix-glob shape used by probe
/// component patterns rather than full shell globbing.
fn component_glob_matches(pattern: &str, component: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        component.starts_with(prefix)
    } else {
        pattern == component
    }
}

/// Decide whether an event should be delivered to a subscriber.
///
/// Per `subscribe-filter.feature` the three filter dimensions are `ANDed`: an
/// event must satisfy every dimension the subscriber specified. A dimension
/// left unset (`None`) is unconstrained. A `None` filter delivers everything.
///
/// - `events`: the event's [`EventType`] must be in the allow-list.
/// - `layers`: an event that carries a layer coordinate must have that layer
///   in the allow-list; an event with no layer coordinate (e.g.
///   `tick.heartbeat`) is unaffected by this dimension.
/// - `components`: an event that carries a component must match at least one
///   glob pattern; an event with no component is unaffected.
pub fn event_matches_filter(
    filter: Option<&SubscribeFilter>,
    method: &str,
    params: &serde_json::Value,
) -> bool {
    let Some(filter) = filter else {
        return true;
    };

    if let Some(allowed) = &filter.events {
        match method_to_event_type(method) {
            Some(et) if allowed.contains(&et) => {}
            // Unknown event types and disallowed ones are both filtered out
            // when an explicit event allow-list is in force.
            _ => return false,
        }
    }

    let (layer, component) = event_coordinates(method, params);

    if let Some(allowed_layers) = &filter.layers {
        if let Some(layer) = layer {
            if !allowed_layers.contains(&layer) {
                return false;
            }
        }
    }

    if let Some(patterns) = &filter.components {
        if let Some(component) = &component {
            if !patterns
                .iter()
                .any(|p| component_glob_matches(p, component))
            {
                return false;
            }
        }
    }

    true
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

    // --- SubscribeFilter tests (TCK subscribe-filter.feature) ---

    fn tick_stopped_params(layer: u32, component: &str) -> serde_json::Value {
        serde_json::json!({
            "position": { "layer": layer, "component": component },
            "state": "stopped",
        })
    }

    fn probe_fired_params(point: &str) -> serde_json::Value {
        serde_json::json!({ "point": point, "probe_id": "p1", "tick_id": 1 })
    }

    #[test]
    fn no_filter_delivers_everything() {
        assert!(event_matches_filter(
            None,
            event::PROBE_FIRED,
            &probe_fired_params("llama:0:5:mlp:0:output"),
        ));
        assert!(event_matches_filter(
            None,
            event::TICK_HEARTBEAT,
            &serde_json::json!({}),
        ));
    }

    #[test]
    fn filter_by_event_type_admits_listed_and_drops_others() {
        // Scenario: Filter by event type.
        let filter = SubscribeFilter {
            events: Some(vec![EventType::TickStopped]),
            layers: None,
            components: None,
        };
        assert!(event_matches_filter(
            Some(&filter),
            event::TICK_STOPPED,
            &tick_stopped_params(3, "attn.q_proj"),
        ));
        assert!(!event_matches_filter(
            Some(&filter),
            event::PROBE_FIRED,
            &probe_fired_params("llama:0:3:mlp:0:output"),
        ));
    }

    #[test]
    fn filter_by_layer_range_admits_in_range_drops_out_of_range() {
        // Scenario: Filter by layer range.
        let filter = SubscribeFilter {
            events: None,
            layers: Some(vec![10, 11, 12]),
            components: None,
        };
        // probe.fired on layer 11 -> delivered.
        assert!(event_matches_filter(
            Some(&filter),
            event::PROBE_FIRED,
            &probe_fired_params("llama:0:11:mlp:0:output"),
        ));
        // probe.fired on layer 4 -> dropped.
        assert!(!event_matches_filter(
            Some(&filter),
            event::PROBE_FIRED,
            &probe_fired_params("llama:0:4:mlp:0:output"),
        ));
        // tick.stopped also honours the layer filter.
        assert!(event_matches_filter(
            Some(&filter),
            event::TICK_STOPPED,
            &tick_stopped_params(12, "mlp"),
        ));
        assert!(!event_matches_filter(
            Some(&filter),
            event::TICK_STOPPED,
            &tick_stopped_params(31, "mlp"),
        ));
    }

    #[test]
    fn layer_filter_does_not_drop_events_without_a_layer() {
        // tick.heartbeat has no layer coordinate; a layer filter must not
        // silently swallow it.
        let filter = SubscribeFilter {
            events: None,
            layers: Some(vec![10]),
            components: None,
        };
        assert!(event_matches_filter(
            Some(&filter),
            event::TICK_HEARTBEAT,
            &serde_json::json!({ "uptime_seconds": 1.0 }),
        ));
    }

    #[test]
    fn filter_by_component_pattern_uses_trailing_glob() {
        // Scenario: Filter by component pattern.
        let filter = SubscribeFilter {
            events: None,
            layers: None,
            components: Some(vec!["attn.*".to_owned()]),
        };
        assert!(event_matches_filter(
            Some(&filter),
            event::TICK_STOPPED,
            &tick_stopped_params(0, "attn.q_proj"),
        ));
        assert!(!event_matches_filter(
            Some(&filter),
            event::TICK_STOPPED,
            &tick_stopped_params(0, "mlp.gate_proj"),
        ));
        // probe.fired component also honours the glob.
        assert!(event_matches_filter(
            Some(&filter),
            event::PROBE_FIRED,
            &probe_fired_params("llama:0:5:attn.o_proj:0:output"),
        ));
    }

    #[test]
    fn filter_dimensions_are_anded() {
        // events AND layers AND components must all be satisfied.
        let filter = SubscribeFilter {
            events: Some(vec![EventType::TickStopped]),
            layers: Some(vec![10]),
            components: Some(vec!["attn.*".to_owned()]),
        };
        // Right event, right layer, right component -> delivered.
        assert!(event_matches_filter(
            Some(&filter),
            event::TICK_STOPPED,
            &tick_stopped_params(10, "attn.q_proj"),
        ));
        // Right event + component but wrong layer -> dropped.
        assert!(!event_matches_filter(
            Some(&filter),
            event::TICK_STOPPED,
            &tick_stopped_params(11, "attn.q_proj"),
        ));
        // Right event + layer but wrong component -> dropped.
        assert!(!event_matches_filter(
            Some(&filter),
            event::TICK_STOPPED,
            &tick_stopped_params(10, "mlp.gate"),
        ));
    }

    #[test]
    fn bare_wildcard_component_matches_anything() {
        assert!(component_glob_matches("*", "anything.at.all"));
        assert!(component_glob_matches("attn.*", "attn.q_proj"));
        assert!(component_glob_matches("attn.q_proj", "attn.q_proj"));
        assert!(!component_glob_matches("attn.q_proj", "attn.k_proj"));
    }

    #[test]
    fn send_notification_filtered_drops_non_matching_event_without_consuming_seq() {
        let filter = SubscribeFilter {
            events: Some(vec![EventType::TickStopped]),
            layers: None,
            components: None,
        };
        let mut buf = Vec::new();
        let mut seq = 0u64;

        // probe.fired is filtered out: nothing written, seq untouched.
        send_notification_filtered(
            &mut buf,
            &mut seq,
            event::PROBE_FIRED,
            probe_fired_params("llama:0:1:mlp:0:output"),
            Some(&filter),
        )
        .unwrap();
        assert!(buf.is_empty());
        assert_eq!(seq, 0);

        // tick.stopped passes: written, seq advances.
        send_notification_filtered(
            &mut buf,
            &mut seq,
            event::TICK_STOPPED,
            tick_stopped_params(0, "embed"),
            Some(&filter),
        )
        .unwrap();
        assert!(!buf.is_empty());
        assert_eq!(seq, 1);
    }

    #[test]
    fn send_notification_filtered_with_none_filter_delivers() {
        let mut buf = Vec::new();
        let mut seq = 0u64;
        send_notification_filtered(
            &mut buf,
            &mut seq,
            event::PROBE_FIRED,
            probe_fired_params("llama:0:1:mlp:0:output"),
            None,
        )
        .unwrap();
        assert!(!buf.is_empty());
        assert_eq!(seq, 1);
    }
}
