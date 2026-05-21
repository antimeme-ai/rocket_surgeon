use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rocket_surgeon_protocol::jsonrpc::{Notification, RawMessage, Request, RequestId, Response};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{broadcast, mpsc, oneshot};

const MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
const MAX_HEADER_COUNT: usize = 16;
const MAX_HEADER_LINE_LEN: usize = 1024;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("transport closed")]
    Closed,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("framing: missing Content-Length")]
    MissingContentLength,
    #[error("framing: invalid Content-Length")]
    InvalidContentLength,
    #[error("response cancelled")]
    Cancelled,
    #[error("rpc error: code={code} message={message}")]
    Rpc { code: i32, message: String },
    #[error("message too large (max {MAX_MESSAGE_SIZE} bytes)")]
    MessageTooLarge,
    #[error("too many headers (max {MAX_HEADER_COUNT})")]
    TooManyHeaders,
    #[error("header line too long (max {MAX_HEADER_LINE_LEN} bytes)")]
    HeaderLineTooLong,
}

type PendingMap = Arc<Mutex<HashMap<RequestId, oneshot::Sender<Result<Response, ClientError>>>>>;

fn lock_pending(
    pending: &PendingMap,
) -> std::sync::MutexGuard<'_, HashMap<RequestId, oneshot::Sender<Result<Response, ClientError>>>> {
    pending
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub struct Connection {
    outgoing_tx: mpsc::Sender<OutgoingMessage>,
    notification_tx: broadcast::Sender<Notification>,
    next_id: AtomicU64,
    pending: PendingMap,
}

enum OutgoingMessage {
    Raw(Vec<u8>),
}

impl Connection {
    pub fn spawn<R, W>(
        reader: R,
        writer: W,
        notification_tx: broadcast::Sender<Notification>,
    ) -> Arc<Self>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let (outgoing_tx, outgoing_rx) = mpsc::channel(64);
        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));

        let conn = Arc::new(Self {
            outgoing_tx,
            notification_tx: notification_tx.clone(),
            next_id: AtomicU64::new(1),
            pending: Arc::clone(&pending),
        });

        tokio::spawn(write_loop(outgoing_rx, writer));
        tokio::spawn(read_loop(reader, notification_tx, Arc::clone(&pending)));

        conn
    }

    pub async fn request(
        &self,
        method: impl Into<String>,
        params: serde_json::Value,
    ) -> Result<Response, ClientError> {
        let raw = self.next_id.fetch_add(1, Ordering::Relaxed);
        let id = RequestId::Number((raw % (i64::MAX as u64)) as i64);
        let req = Request::new(id.clone(), method, params);

        let body = serde_json::to_string(&req)?;
        let frame = format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes();

        let (tx, rx) = oneshot::channel();
        lock_pending(&self.pending).insert(id, tx);

        self.outgoing_tx
            .send(OutgoingMessage::Raw(frame))
            .await
            .map_err(|_| ClientError::Closed)?;

        rx.await.map_err(|_| ClientError::Cancelled)?
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Notification> {
        self.notification_tx.subscribe()
    }
}

pub type ConnectFn = Box<
    dyn Fn(
            broadcast::Sender<Notification>,
        ) -> Pin<Box<dyn Future<Output = Result<Arc<Connection>, ClientError>> + Send>>
        + Send
        + Sync,
>;

/// Retry-wrapping client. Retained, fully tested, for the reconnection slice;
/// `daemon.rs` currently drives a bare [`Connection`].
#[allow(dead_code)]
pub struct ReconnectingClient {
    conn: tokio::sync::RwLock<Arc<Connection>>,
    connect: ConnectFn,
    notification_tx: broadcast::Sender<Notification>,
    max_retries: u32,
    base_delay: Duration,
}

#[allow(dead_code)]
impl ReconnectingClient {
    pub fn new(
        conn: Arc<Connection>,
        connect: ConnectFn,
        notification_tx: broadcast::Sender<Notification>,
    ) -> Self {
        Self {
            conn: tokio::sync::RwLock::new(conn),
            connect,
            notification_tx,
            max_retries: 5,
            base_delay: Duration::from_millis(100),
        }
    }

    pub async fn request(
        &self,
        method: impl Into<String>,
        params: serde_json::Value,
    ) -> Result<Response, ClientError> {
        let method = method.into();
        let conn = self.conn.read().await.clone();
        match conn.request(&method, params.clone()).await {
            Ok(resp) => Ok(resp),
            Err(ClientError::Closed | ClientError::Cancelled) => {
                let new_conn = self.reconnect().await?;
                new_conn.request(&method, params).await
            }
            Err(e) => Err(e),
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Notification> {
        self.notification_tx.subscribe()
    }

    pub async fn connection(&self) -> Arc<Connection> {
        self.conn.read().await.clone()
    }

    async fn reconnect(&self) -> Result<Arc<Connection>, ClientError> {
        for attempt in 0..self.max_retries {
            let delay = self.base_delay * 2u32.saturating_pow(attempt);
            tokio::time::sleep(delay).await;

            match (self.connect)(self.notification_tx.clone()).await {
                Ok(new_conn) => {
                    *self.conn.write().await = new_conn.clone();
                    return Ok(new_conn);
                }
                Err(_) if attempt + 1 < self.max_retries => {}
                Err(e) => return Err(e),
            }
        }
        Err(ClientError::Closed)
    }
}

pub async fn read_content_length_message<R: AsyncBufReadExt + Unpin>(
    reader: &mut R,
) -> Result<String, ClientError> {
    let mut content_length: Option<usize> = None;
    let mut line = String::new();
    let mut header_count = 0usize;

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            return Err(ClientError::Closed);
        }

        if line.len() > MAX_HEADER_LINE_LEN {
            return Err(ClientError::HeaderLineTooLong);
        }

        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }

        header_count += 1;
        if header_count > MAX_HEADER_COUNT {
            return Err(ClientError::TooManyHeaders);
        }

        if let Some((key, value)) = trimmed.split_once(':') {
            if key.eq_ignore_ascii_case("content-length") {
                content_length = Some(
                    value
                        .trim()
                        .parse()
                        .map_err(|_| ClientError::InvalidContentLength)?,
                );
            }
        }
    }

    let len = content_length.ok_or(ClientError::MissingContentLength)?;
    if len > MAX_MESSAGE_SIZE {
        return Err(ClientError::MessageTooLarge);
    }
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).await?;
    String::from_utf8(body)
        .map_err(|e| ClientError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
}

async fn write_loop<W: AsyncWrite + Unpin>(
    mut outgoing_rx: mpsc::Receiver<OutgoingMessage>,
    mut writer: W,
) {
    while let Some(OutgoingMessage::Raw(frame)) = outgoing_rx.recv().await {
        if writer.write_all(&frame).await.is_err() {
            return;
        }
        if writer.flush().await.is_err() {
            return;
        }
    }
}

async fn read_loop<R: AsyncRead + Unpin + Send>(
    reader: R,
    notification_tx: broadcast::Sender<Notification>,
    pending: PendingMap,
) {
    let mut reader = BufReader::new(reader);

    loop {
        let Ok(msg) = read_content_length_message(&mut reader).await else {
            let drained: Vec<_> = {
                let mut map = lock_pending(&pending);
                map.drain().collect()
            };
            for (_, tx) in drained {
                let _ = tx.send(Err(ClientError::Closed));
            }
            return;
        };

        if let Ok(resp) = serde_json::from_str::<Response>(&msg) {
            let pending_tx = lock_pending(&pending).remove(&resp.id);
            if let Some(tx) = pending_tx {
                let _ = tx.send(Ok(resp));
            }
            continue;
        }

        if let Ok(raw) = serde_json::from_str::<RawMessage>(&msg) {
            if let Some(notif) = raw.into_notification() {
                let _ = notification_tx.send(notif);
                continue;
            }
        }

        tracing::warn!(msg_len = msg.len(), "dropping unparseable message");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    fn frame_message(body: &str) -> Vec<u8> {
        format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
    }

    #[tokio::test]
    async fn request_response_roundtrip() {
        let (client_stream, mut server_stream) = duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_stream);
        let (notification_tx, _) = broadcast::channel(256);

        let conn = Connection::spawn(client_read, client_write, notification_tx);

        let server_handle = tokio::spawn(async move {
            let mut reader = BufReader::new(&mut server_stream);
            let msg = read_content_length_message(&mut reader).await.unwrap();
            let req: Request = serde_json::from_str(&msg).unwrap();

            let resp = Response::success(req.id, serde_json::json!({"protocol_version": "0.3.0"}));
            let body = serde_json::to_string(&resp).unwrap();
            let frame = frame_message(&body);

            server_stream.write_all(&frame).await.unwrap();
            server_stream.flush().await.unwrap();
        });

        let resp = conn
            .request(
                "initialize",
                serde_json::json!({"client_name": "test", "protocol_version": "0.3.0"}),
            )
            .await
            .unwrap();

        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert_eq!(result["protocol_version"], "0.3.0");

        server_handle.await.unwrap();
    }

    #[tokio::test]
    async fn notification_forwarded_to_subscriber() {
        let (client_stream, mut server_stream) = duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_stream);
        let (notification_tx, _) = broadcast::channel(256);

        let _conn = Connection::spawn(client_read, client_write, notification_tx.clone());
        let mut sub = notification_tx.subscribe();

        let notif = Notification::new(
            "tick.stopped",
            serde_json::json!({"position": {"tick_id": 42}}),
        );
        let body = serde_json::to_string(&notif).unwrap();
        let frame = frame_message(&body);

        server_stream.write_all(&frame).await.unwrap();
        server_stream.flush().await.unwrap();

        let received = tokio::time::timeout(std::time::Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(received.method, "tick.stopped");
    }

    #[tokio::test]
    async fn reconnects_after_disconnect() {
        use std::sync::atomic::AtomicUsize;

        let (notification_tx, _) = broadcast::channel(256);
        let attempt = Arc::new(AtomicUsize::new(0));
        let (server_tx, mut server_rx) = mpsc::channel::<tokio::io::DuplexStream>(4);

        let attempt_clone = Arc::clone(&attempt);
        let connect: ConnectFn = Box::new(move |ntx| {
            let server_tx = server_tx.clone();
            let attempt = Arc::clone(&attempt_clone);
            Box::pin(async move {
                attempt.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let (client_stream, server_stream) = tokio::io::duplex(4096);
                server_tx
                    .send(server_stream)
                    .await
                    .map_err(|_| ClientError::Closed)?;
                let (r, w) = tokio::io::split(client_stream);
                Ok(Connection::spawn(r, w, ntx))
            })
        });

        // Initial connection
        let (client_stream, first_server) = tokio::io::duplex(4096);
        let (r, w) = tokio::io::split(client_stream);
        let initial_conn = Connection::spawn(r, w, notification_tx.clone());
        let client = Arc::new(ReconnectingClient::new(
            initial_conn,
            connect,
            notification_tx,
        ));

        // Drop first server to simulate disconnect
        drop(first_server);

        // Spawn a task to handle the reconnected server
        let server_handle = tokio::spawn(async move {
            let mut server_stream = server_rx.recv().await.unwrap();
            let mut reader = BufReader::new(&mut server_stream);
            let msg = read_content_length_message(&mut reader).await.unwrap();
            let req: Request = serde_json::from_str(&msg).unwrap();

            let resp = Response::success(req.id, serde_json::json!({"ok": true}));
            let body = serde_json::to_string(&resp).unwrap();
            let frame = frame_message(&body);

            server_stream.write_all(&frame).await.unwrap();
            server_stream.flush().await.unwrap();
        });

        let resp = client
            .request("rocket/status", serde_json::json!({}))
            .await
            .unwrap();

        assert!(resp.result.is_some());
        assert_eq!(resp.result.unwrap()["ok"], true);
        assert!(attempt.load(std::sync::atomic::Ordering::Relaxed) >= 1);

        server_handle.await.unwrap();
    }

    #[tokio::test]
    async fn closed_transport_returns_error() {
        let (client_stream, server_stream) = duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_stream);
        let (notification_tx, _) = broadcast::channel(256);

        let conn = Connection::spawn(client_read, client_write, notification_tx);

        drop(server_stream);

        let result = conn.request("initialize", serde_json::json!({})).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn rejects_oversized_content_length() {
        let msg = b"Content-Length: 999999999\r\n\r\n";
        let mut reader = tokio::io::BufReader::new(&msg[..]);
        let result = read_content_length_message(&mut reader).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ClientError::MessageTooLarge => {}
            other => panic!("expected MessageTooLarge, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_too_many_headers() {
        use std::fmt::Write;

        let mut msg = String::new();
        for i in 0..20 {
            writeln!(msg, "X-Header-{i}: value\r").unwrap();
        }
        msg.push_str("\r\n");
        let mut reader = tokio::io::BufReader::new(msg.as_bytes());
        let result = read_content_length_message(&mut reader).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ClientError::TooManyHeaders => {}
            other => panic!("expected TooManyHeaders, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_oversized_header_line() {
        let long_value = "x".repeat(2048);
        let msg = format!("Content-Length: {long_value}\r\n\r\n");
        let mut reader = tokio::io::BufReader::new(msg.as_bytes());
        let result = read_content_length_message(&mut reader).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ClientError::HeaderLineTooLong => {}
            other => panic!("expected HeaderLineTooLong, got {other:?}"),
        }
    }
}
