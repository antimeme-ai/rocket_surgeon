use std::collections::HashSet;
use std::sync::Arc;

use rocket_surgeon_protocol::jsonrpc::Notification;
use rocket_surgeon_protocol::messages::{
    method, EventType, SubscribeFilter, SubscribeRequest,
};
use tokio::sync::broadcast;

use super::connection::{ClientError, Connection};

pub struct SubscriptionManager {
    conn: Arc<Connection>,
    active_events: HashSet<EventType>,
    active_layers: Option<Vec<u32>>,
    active_components: Option<Vec<String>>,
}

impl SubscriptionManager {
    pub fn new(conn: Arc<Connection>) -> Self {
        Self {
            conn,
            active_events: HashSet::new(),
            active_layers: None,
            active_components: None,
        }
    }

    pub fn receiver(&self) -> broadcast::Receiver<Notification> {
        self.conn.subscribe()
    }

    pub async fn update_filter(
        &mut self,
        events: HashSet<EventType>,
        layers: Option<Vec<u32>>,
        components: Option<Vec<String>>,
    ) -> Result<(), ClientError> {
        if events == self.active_events
            && layers == self.active_layers
            && components == self.active_components
        {
            return Ok(());
        }

        let filter = if events.is_empty() && layers.is_none() && components.is_none() {
            None
        } else {
            let event_list = if events.is_empty() {
                None
            } else {
                let mut sorted: Vec<EventType> = events.iter().copied().collect();
                sorted.sort_by_key(|e| format!("{e:?}"));
                Some(sorted)
            };
            Some(SubscribeFilter {
                events: event_list,
                layers: layers.clone(),
                components: components.clone(),
            })
        };

        let req = SubscribeRequest { filter };
        let params = serde_json::to_value(&req).map_err(ClientError::Json)?;
        let resp = self.conn.request(method::SUBSCRIBE, params).await?;

        if let Some(err) = resp.error {
            return Err(ClientError::Rpc {
                code: err.code,
                message: err.message,
            });
        }

        self.active_events = events;
        self.active_layers = layers;
        self.active_components = components;
        Ok(())
    }

    pub async fn unsubscribe(&mut self) -> Result<(), ClientError> {
        let params = serde_json::to_value(&serde_json::json!({})).unwrap();
        let resp = self.conn.request(method::UNSUBSCRIBE, params).await?;

        if let Some(err) = resp.error {
            return Err(ClientError::Rpc {
                code: err.code,
                message: err.message,
            });
        }

        self.active_events.clear();
        self.active_layers = None;
        self.active_components = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocket_surgeon_protocol::jsonrpc::Response;
    use rocket_surgeon_protocol::messages::{EventType, SubscribeResponse};
    use rocket_surgeon_protocol::types::Status;
    use tokio::io::{duplex, AsyncWriteExt, BufReader};

    use super::super::connection::read_content_length_message;

    fn frame_message(body: &str) -> Vec<u8> {
        format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
    }

    async fn respond_to_subscribe(server: &mut (impl tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin)) {
        let mut reader = BufReader::new(&mut *server);
        let msg = read_content_length_message(&mut reader).await.unwrap();
        let req: rocket_surgeon_protocol::jsonrpc::Request =
            serde_json::from_str(&msg).unwrap();

        let sub_resp = SubscribeResponse {
            available_events: vec![EventType::TickStopped, EventType::ProbeFired],
            status: Status::Stopped,
        };
        let resp = Response::success(req.id, serde_json::to_value(&sub_resp).unwrap());
        let body = serde_json::to_string(&resp).unwrap();
        let frame = frame_message(&body);
        server.write_all(&frame).await.unwrap();
        server.flush().await.unwrap();
    }

    #[tokio::test]
    async fn subscribe_sends_filter_to_server() {
        let (client_stream, mut server_stream) = duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_stream);
        let conn = Connection::spawn(client_read, client_write);
        let mut mgr = SubscriptionManager::new(conn);

        let server_handle = tokio::spawn(async move {
            respond_to_subscribe(&mut server_stream).await;
            server_stream
        });

        let mut events = HashSet::new();
        events.insert(EventType::TickStopped);
        mgr.update_filter(events, None, None).await.unwrap();

        assert_eq!(mgr.active_events.len(), 1);
        assert!(mgr.active_events.contains(&EventType::TickStopped));

        server_handle.await.unwrap();
    }

    #[tokio::test]
    async fn no_op_when_filter_unchanged() {
        let (client_stream, mut server_stream) = duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_stream);
        let conn = Connection::spawn(client_read, client_write);
        let mut mgr = SubscriptionManager::new(conn);

        let server_handle = tokio::spawn(async move {
            respond_to_subscribe(&mut server_stream).await;
            server_stream
        });

        let mut events = HashSet::new();
        events.insert(EventType::TickStopped);
        mgr.update_filter(events.clone(), None, None).await.unwrap();

        let _server_stream = server_handle.await.unwrap();

        // Second call with same filter should not send any request
        // If it did, it would hang since server isn't reading
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            mgr.update_filter(events, None, None),
        )
        .await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }

    #[tokio::test]
    async fn unsubscribe_clears_state() {
        let (client_stream, mut server_stream) = duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_stream);
        let conn = Connection::spawn(client_read, client_write);
        let mut mgr = SubscriptionManager::new(conn);

        let server_handle = tokio::spawn(async move {
            // First: handle subscribe
            respond_to_subscribe(&mut server_stream).await;
            // Second: handle unsubscribe
            let mut reader = BufReader::new(&mut server_stream);
            let msg = read_content_length_message(&mut reader).await.unwrap();
            let req: rocket_surgeon_protocol::jsonrpc::Request =
                serde_json::from_str(&msg).unwrap();

            let unsub_resp = rocket_surgeon_protocol::messages::UnsubscribeResponse {
                status: Status::Stopped,
            };
            let resp = Response::success(req.id, serde_json::to_value(&unsub_resp).unwrap());
            let body = serde_json::to_string(&resp).unwrap();
            let frame = frame_message(&body);
            server_stream.write_all(&frame).await.unwrap();
            server_stream.flush().await.unwrap();
            server_stream
        });

        let mut events = HashSet::new();
        events.insert(EventType::TickStopped);
        mgr.update_filter(events, None, None).await.unwrap();
        assert_eq!(mgr.active_events.len(), 1);

        mgr.unsubscribe().await.unwrap();
        assert!(mgr.active_events.is_empty());
        assert!(mgr.active_layers.is_none());
        assert!(mgr.active_components.is_none());

        server_handle.await.unwrap();
    }
}
