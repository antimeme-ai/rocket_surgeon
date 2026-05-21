//! The daemon link: a second event source for the application loop.
//!
//! Connects to the rs-daemon over its Unix socket via [`client::Connection`],
//! performs the JSON-RPC handshake, subscribes to the event stream, and maps
//! daemon notifications into [`Action`]s on the loop's channel — the
//! terminal's co-equal source (BEAD-0015 slice 2).
//!
//! Reconnection is a later refinement: on any disconnect the task emits
//! [`DaemonEvent::Disconnected`] and ends.

use std::sync::Arc;

use rocket_surgeon_protocol::jsonrpc::{Notification, Response};
use rocket_surgeon_protocol::types::TickPosition;
use tokio::net::UnixStream;
use tokio::sync::{broadcast, mpsc};

use crate::action::{Action, DaemonEvent};
use crate::client::connection::{ClientError, Connection};

/// Protocol version this client speaks.
const PROTOCOL_VERSION: &str = "0.3.0";

/// Spawn the daemon-link task. It owns the connection for its lifetime and
/// emits [`DaemonEvent`]s into `tx`; on any disconnect it emits
/// `Disconnected` exactly once and ends.
pub fn spawn(socket: String, tx: mpsc::Sender<Action>) {
    tokio::spawn(async move {
        if let Err(e) = run(&socket, &tx).await {
            tracing::warn!(error = %e, "daemon link ended");
        }
        let _ = tx.send(Action::Daemon(DaemonEvent::Disconnected)).await;
    });
}

/// Connect to the daemon socket, then drive the handshake + notification loop.
async fn run(socket: &str, tx: &mpsc::Sender<Action>) -> Result<(), ClientError> {
    let stream = UnixStream::connect(socket).await?;
    let (read, write) = stream.into_split();
    let (notif_tx, _) = broadcast::channel(256);
    let conn = Connection::spawn(read, write, notif_tx);
    let notifications = conn.subscribe();
    drive(conn, notifications, tx).await
}

/// Handshake, subscribe, and forward notifications until the link closes.
///
/// Split from [`run`] so the transport can be a `duplex` pipe under test.
async fn drive(
    conn: Arc<Connection>,
    mut notifications: broadcast::Receiver<Notification>,
    tx: &mpsc::Sender<Action>,
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

    // Drop the connection handle before the loop: `Connection` holds a
    // notification-channel sender, so while `conn` is alive `recv()` would
    // park forever after a disconnect and `Disconnected` would never fire.
    // Once dropped, the read task's sender is the only one — it closes the
    // channel when the transport dies.
    drop(conn);

    // Forward notifications as actions until the connection closes.
    loop {
        match notifications.recv().await {
            Ok(notification) => {
                if let Some(action) = map_notification(&notification) {
                    send(tx, action).await?;
                }
            }
            Err(broadcast::error::RecvError::Closed) => break,
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                tracing::warn!(skipped, "daemon notification stream lagged");
            }
        }
    }
    Ok(())
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
    /// the daemon's reported protocol version (read from the response
    /// envelope), and `drive` returns when the transport closes.
    #[tokio::test]
    async fn drive_handshake_emits_connected_then_exits_on_close() {
        let (client, server) = duplex(8192);
        let (client_read, client_write) = split(client);
        let (notif_tx, _) = broadcast::channel(64);
        let conn = Connection::spawn(client_read, client_write, notif_tx);
        let notifications = conn.subscribe();
        let (action_tx, mut action_rx) = mpsc::channel(16);

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
            // drop the server: the transport closes, `drive` should return.
        });

        drive(conn, notifications, &action_tx).await.unwrap();
        server.await.unwrap();

        match action_rx.recv().await {
            Some(Action::Daemon(DaemonEvent::Connected { protocol_version })) => {
                assert_eq!(protocol_version, "9.9.9", "read from the response envelope");
            }
            other => panic!("expected Connected, got {other:?}"),
        }
    }
}
