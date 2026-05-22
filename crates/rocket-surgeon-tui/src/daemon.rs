//! The daemon link: the application's bidirectional channel to the daemon.
//!
//! Connects to the rs-daemon over its Unix socket via [`client::Connection`],
//! performs the JSON-RPC handshake, and subscribes to the event stream. It
//! then services both directions: daemon notifications become [`Action`]s on
//! the loop's channel — the terminal's co-equal source (BEAD-0015 slice 2) —
//! and application [`Effect`]s become `rocket/*` requests (slice 4).
//!
//! Reconnection is a later refinement: on any disconnect the task emits
//! [`DaemonEvent::Disconnected`] and ends.

use std::sync::Arc;

use rocket_surgeon_protocol::jsonrpc::{Notification, Response};
use rocket_surgeon_protocol::types::TickPosition;
use tokio::net::UnixStream;
use tokio::sync::{broadcast, mpsc};

use crate::action::{Action, DaemonEvent, Effect};
use crate::client::connection::{ClientError, Connection};

/// Protocol version this client speaks.
const PROTOCOL_VERSION: &str = "0.3.0";

/// Spawn the daemon-link task and return the sender for app→daemon effects.
///
/// The task owns the connection for its lifetime: it emits [`DaemonEvent`]s
/// into `tx`, turns [`Effect`]s received on the returned channel into
/// `rocket/*` requests, and on any disconnect emits `Disconnected` exactly
/// once and ends.
pub fn spawn(socket: String, tx: mpsc::Sender<Action>) -> mpsc::Sender<Effect> {
    let (effect_tx, effect_rx) = mpsc::channel(64);
    tokio::spawn(async move {
        if let Err(e) = run(&socket, &tx, effect_rx).await {
            tracing::warn!(error = %e, "daemon link ended");
        }
        let _ = tx.send(Action::Daemon(DaemonEvent::Disconnected)).await;
    });
    effect_tx
}

/// Connect to the daemon socket, then drive the handshake + event loop.
async fn run(
    socket: &str,
    tx: &mpsc::Sender<Action>,
    effect_rx: mpsc::Receiver<Effect>,
) -> Result<(), ClientError> {
    let stream = UnixStream::connect(socket).await?;
    let (read, write) = stream.into_split();
    let (notif_tx, _) = broadcast::channel(256);
    let conn = Connection::spawn(read, write, notif_tx);
    let notifications = conn.subscribe();
    drive(conn, notifications, tx, effect_rx).await
}

/// Handshake, subscribe, then service both directions of the link: daemon
/// notifications become [`Action`]s, application [`Effect`]s become `rocket/*`
/// requests.
///
/// Split from [`run`] so the transport can be a `duplex` pipe under test.
async fn drive(
    conn: Arc<Connection>,
    mut notifications: broadcast::Receiver<Notification>,
    tx: &mpsc::Sender<Action>,
    mut effect_rx: mpsc::Receiver<Effect>,
) -> Result<(), ClientError> {
    // Handshake: initialize, then announce the link is up.
    let resp = checked(
        conn.request(
            "initialize",
            serde_json::json!({
                "client_name": "rocket-surgeon-tui",
                "protocol_version": PROTOCOL_VERSION,
            }),
        )
        .await?,
    )?;
    // The initialize result is the daemon's `ResponseEnvelope<InitializeResponse>`:
    // `{ state, data: { capabilities: { protocol_version } } }`.
    let protocol_version = resp
        .result
        .as_ref()
        .and_then(|r| r["data"]["capabilities"]["protocol_version"].as_str())
        .unwrap_or(PROTOCOL_VERSION)
        .to_owned();
    send(
        tx,
        Action::Daemon(DaemonEvent::Connected { protocol_version }),
    )
    .await?;

    // Subscribe to the unfiltered event stream.
    checked(
        conn.request("rocket/subscribe", serde_json::json!({}))
            .await?,
    )?;

    // `conn` is held for the task's lifetime — it is the request handle for
    // effect dispatch. Slice 2 dropped it here so a notification-stream close
    // would surface `Disconnected`; that close can no longer fire while `conn`
    // lives, so a dead link is now detected on the request path instead (see
    // `dispatch_effect`). Silent idle-link death is the reconnection slice.
    loop {
        tokio::select! {
            notification = notifications.recv() => match notification {
                Ok(notification) => {
                    if let Some(action) = map_notification(&notification) {
                        send(tx, action).await?;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    tracing::warn!(skipped, "daemon notification stream lagged");
                }
            },
            effect = effect_rx.recv() => match effect {
                Some(effect) => dispatch_effect(&conn, &effect).await?,
                // The application dropped the effect sender — it is shutting
                // down; end the link cleanly.
                None => break,
            },
        }
    }
    Ok(())
}

/// Turn an [`Effect`] into a `rocket/*` request on the link.
///
/// A transport failure (`Closed` / `Cancelled`) is fatal — it ends the task so
/// `Disconnected` fires. An RPC-level rejection (e.g. a wrong session state)
/// is logged and survived: it is the daemon's verdict on one request, not a
/// dead link — so, unlike the handshake, the response is not run through
/// [`checked`].
async fn dispatch_effect(conn: &Connection, effect: &Effect) -> Result<(), ClientError> {
    let (method, params) = match effect {
        Effect::RequestStep { count } => (
            "rocket/step",
            serde_json::json!({ "direction": "forward", "count": count }),
        ),
    };
    match conn.request(method, params).await {
        Ok(response) => {
            if let Some(error) = response.error {
                tracing::warn!(
                    method,
                    code = error.code,
                    message = %error.message,
                    "daemon rejected request"
                );
            }
            Ok(())
        }
        Err(e @ (ClientError::Closed | ClientError::Cancelled)) => Err(e),
        Err(e) => {
            tracing::warn!(method, error = %e, "request failed");
            Ok(())
        }
    }
}

/// Treat a JSON-RPC-level error in a response as a link error.
fn checked(resp: Response) -> Result<Response, ClientError> {
    if let Some(err) = resp.error {
        return Err(ClientError::Rpc {
            code: err.code,
            message: err.message,
        });
    }
    Ok(resp)
}

/// Send an action to the loop, treating a closed channel as a link error.
async fn send(tx: &mpsc::Sender<Action>, action: Action) -> Result<(), ClientError> {
    tx.send(action).await.map_err(|_| ClientError::Closed)
}

/// Map a daemon notification to an [`Action`], or `None` when the TUI does not
/// (yet) act on it.
fn map_notification(notification: &Notification) -> Option<Action> {
    match notification.method.as_str() {
        "tick.stopped" => {
            let params = notification.params.as_ref()?;
            let position: TickPosition =
                serde_json::from_value(params.get("position")?.clone()).ok()?;
            Some(Action::Daemon(DaemonEvent::TickStopped(position)))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::connection::read_content_length_message;
    use rocket_surgeon_protocol::jsonrpc::Request;
    use tokio::io::{AsyncWriteExt, BufReader, duplex, split};

    fn tick_stopped_notification() -> Notification {
        Notification::new(
            "tick.stopped",
            serde_json::json!({
                "position": {
                    "tick_id": 7,
                    "direction": "forward",
                    "rank": 0,
                    "layer": 2,
                    "component": "attn.o_proj",
                    "event": "output",
                    "replay_of": null,
                    "phase": {"type": "decode"},
                    "token_position": null,
                    "clock": null
                }
            }),
        )
    }

    #[test]
    fn maps_tick_stopped_to_action() {
        match map_notification(&tick_stopped_notification()) {
            Some(Action::Daemon(DaemonEvent::TickStopped(pos))) => {
                assert_eq!(pos.tick_id, 7);
                assert_eq!(pos.layer, 2);
                assert_eq!(pos.component, "attn.o_proj");
            }
            other => panic!("expected TickStopped, got {other:?}"),
        }
    }

    #[test]
    fn unknown_notification_maps_to_none() {
        let notification = Notification::new("tick.heartbeat", serde_json::json!({}));
        assert!(map_notification(&notification).is_none());
    }

    #[test]
    fn tick_stopped_without_position_maps_to_none() {
        let notification = Notification::new("tick.stopped", serde_json::json!({}));
        assert!(map_notification(&notification).is_none());
    }

    fn frame(body: &str) -> Vec<u8> {
        format!("Content-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
    }

    /// `drive` against a scripted server: the handshake emits `Connected` with
    /// the daemon's reported protocol version, and `drive` returns once the
    /// effect channel closes — the application-shutdown path.
    #[tokio::test]
    async fn drive_handshake_emits_connected_then_exits() {
        let (client, server) = duplex(8192);
        let (client_read, client_write) = split(client);
        let (notif_tx, _) = broadcast::channel(64);
        let conn = Connection::spawn(client_read, client_write, notif_tx);
        let notifications = conn.subscribe();
        let (action_tx, mut action_rx) = mpsc::channel(16);
        let (effect_tx, effect_rx) = mpsc::channel::<Effect>(16);
        // No effects queued: once the handshake completes, the closed effect
        // channel ends `drive`.
        drop(effect_tx);

        let server = tokio::spawn(async move {
            let (server_read, mut server_write) = split(server);
            let mut reader = BufReader::new(server_read);
            // initialize -> envelope-wrapped capabilities
            let raw = read_content_length_message(&mut reader).await.unwrap();
            let req: Request = serde_json::from_str(&raw).unwrap();
            let resp = Response::success(
                req.id,
                serde_json::json!({
                    "state": {},
                    "data": {"capabilities": {"protocol_version": "9.9.9"}}
                }),
            );
            let body = serde_json::to_string(&resp).unwrap();
            server_write.write_all(&frame(&body)).await.unwrap();
            // rocket/subscribe -> ok
            let raw = read_content_length_message(&mut reader).await.unwrap();
            let req: Request = serde_json::from_str(&raw).unwrap();
            let body =
                serde_json::to_string(&Response::success(req.id, serde_json::json!({}))).unwrap();
            server_write.write_all(&frame(&body)).await.unwrap();
        });

        drive(conn, notifications, &action_tx, effect_rx)
            .await
            .unwrap();
        server.await.unwrap();

        match action_rx.recv().await {
            Some(Action::Daemon(DaemonEvent::Connected { protocol_version })) => {
                assert_eq!(protocol_version, "9.9.9", "read from the response envelope");
            }
            other => panic!("expected Connected, got {other:?}"),
        }
    }

    /// An `Effect` on the channel becomes a `rocket/step` request on the wire,
    /// carrying the direction and count.
    #[tokio::test]
    async fn drive_dispatches_step_effect_as_request() {
        let (client, server) = duplex(8192);
        let (client_read, client_write) = split(client);
        let (notif_tx, _) = broadcast::channel(64);
        let conn = Connection::spawn(client_read, client_write, notif_tx);
        let notifications = conn.subscribe();
        let (action_tx, _action_rx) = mpsc::channel(16);
        let (effect_tx, effect_rx) = mpsc::channel::<Effect>(16);

        // Queue one step, then close the channel so `drive` exits after it.
        effect_tx
            .send(Effect::RequestStep { count: 3 })
            .await
            .unwrap();
        drop(effect_tx);

        let server = tokio::spawn(async move {
            let (server_read, mut server_write) = split(server);
            let mut reader = BufReader::new(server_read);
            // initialize, then subscribe — the handshake.
            for _ in 0..2 {
                let raw = read_content_length_message(&mut reader).await.unwrap();
                let req: Request = serde_json::from_str(&raw).unwrap();
                let body = serde_json::to_string(&Response::success(req.id, serde_json::json!({})))
                    .unwrap();
                server_write.write_all(&frame(&body)).await.unwrap();
            }
            // the dispatched effect.
            let raw = read_content_length_message(&mut reader).await.unwrap();
            let req: Request = serde_json::from_str(&raw).unwrap();
            let body =
                serde_json::to_string(&Response::success(req.id.clone(), serde_json::json!({})))
                    .unwrap();
            server_write.write_all(&frame(&body)).await.unwrap();
            req
        });

        drive(conn, notifications, &action_tx, effect_rx)
            .await
            .unwrap();
        let step = server.await.unwrap();

        assert_eq!(step.method, "rocket/step");
        let params = step.params.expect("step request carries params");
        assert_eq!(params["direction"], "forward");
        assert_eq!(params["count"], 3);
        // The wire params must satisfy the daemon's `StepRequest` contract.
        let parsed: rocket_surgeon_protocol::messages::StepRequest =
            serde_json::from_value(params).expect("params deserialize as StepRequest");
        assert_eq!(parsed.count, 3);
        assert_eq!(
            parsed.direction,
            rocket_surgeon_protocol::types::StepDirection::Forward
        );
    }
}
