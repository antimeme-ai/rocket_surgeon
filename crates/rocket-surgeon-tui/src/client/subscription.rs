use std::collections::HashSet;

use rocket_surgeon_protocol::messages::{EventType, SubscribeFilter, SubscribeRequest, method};

use super::connection::{ClientError, ReconnectingClient};

pub struct SubscriptionState {
    pub active_events: HashSet<EventType>,
    pub active_layers: Option<Vec<u32>>,
    pub active_components: Option<Vec<String>>,
}

pub fn initial_subscription_state() -> SubscriptionState {
    SubscriptionState {
        active_events: HashSet::new(),
        active_layers: None,
        active_components: None,
    }
}

pub async fn update_filter(
    state: &mut SubscriptionState,
    client: &ReconnectingClient,
    events: HashSet<EventType>,
    layers: Option<Vec<u32>>,
    components: Option<Vec<String>>,
) -> Result<(), ClientError> {
    if events == state.active_events
        && layers == state.active_layers
        && components == state.active_components
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
            sorted.sort();
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
    let resp = client.request(method::SUBSCRIBE, params).await?;

    if let Some(err) = resp.error {
        return Err(ClientError::Rpc {
            code: err.code,
            message: err.message,
        });
    }

    state.active_events = events;
    state.active_layers = layers;
    state.active_components = components;
    Ok(())
}

pub async fn unsubscribe(
    state: &mut SubscriptionState,
    client: &ReconnectingClient,
) -> Result<(), ClientError> {
    let params = serde_json::to_value(serde_json::json!({})).unwrap();
    let resp = client.request(method::UNSUBSCRIBE, params).await?;

    if let Some(err) = resp.error {
        return Err(ClientError::Rpc {
            code: err.code,
            message: err.message,
        });
    }

    state.active_events.clear();
    state.active_layers = None;
    state.active_components = None;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocket_surgeon_protocol::jsonrpc::Response;
    use rocket_surgeon_protocol::messages::{EventType, SubscribeResponse};
    use rocket_surgeon_protocol::types::Status;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use tokio::io::{AsyncWriteExt, BufReader, duplex};
    use tokio::sync::broadcast;

    use super::super::connection::{ConnectFn, Connection, read_content_length_message};

    fn frame_message(body: &str) -> Vec<u8> {
        format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
    }

    fn dummy_connect() -> ConnectFn {
        Box::new(|_ntx| {
            Box::pin(async { Err(ClientError::Closed) })
                as Pin<Box<dyn Future<Output = Result<Arc<Connection>, ClientError>> + Send>>
        })
    }

    async fn respond_to_subscribe(
        server: &mut (impl tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin),
    ) {
        let mut reader = BufReader::new(&mut *server);
        let msg = read_content_length_message(&mut reader).await.unwrap();
        let req: rocket_surgeon_protocol::jsonrpc::Request = serde_json::from_str(&msg).unwrap();

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

    fn make_client(
        notification_tx: broadcast::Sender<rocket_surgeon_protocol::jsonrpc::Notification>,
        client_read: impl tokio::io::AsyncRead + Unpin + Send + 'static,
        client_write: impl tokio::io::AsyncWrite + Unpin + Send + 'static,
    ) -> ReconnectingClient {
        let conn = Connection::spawn(client_read, client_write, notification_tx.clone());
        ReconnectingClient::new(conn, dummy_connect(), notification_tx)
    }

    #[tokio::test]
    async fn subscribe_sends_filter_to_server() {
        let (client_stream, mut server_stream) = duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_stream);
        let (notification_tx, _) = broadcast::channel(256);
        let client = make_client(notification_tx, client_read, client_write);
        let mut sub_state = initial_subscription_state();

        let server_handle = tokio::spawn(async move {
            respond_to_subscribe(&mut server_stream).await;
            server_stream
        });

        let mut events = HashSet::new();
        events.insert(EventType::TickStopped);
        update_filter(&mut sub_state, &client, events, None, None)
            .await
            .unwrap();

        assert_eq!(sub_state.active_events.len(), 1);
        assert!(sub_state.active_events.contains(&EventType::TickStopped));

        server_handle.await.unwrap();
    }

    #[tokio::test]
    async fn no_op_when_filter_unchanged() {
        let (client_stream, mut server_stream) = duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_stream);
        let (notification_tx, _) = broadcast::channel(256);
        let client = make_client(notification_tx, client_read, client_write);
        let mut sub_state = initial_subscription_state();

        let server_handle = tokio::spawn(async move {
            respond_to_subscribe(&mut server_stream).await;
            server_stream
        });

        let mut events = HashSet::new();
        events.insert(EventType::TickStopped);
        update_filter(&mut sub_state, &client, events.clone(), None, None)
            .await
            .unwrap();

        let _server_stream = server_handle.await.unwrap();

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            update_filter(&mut sub_state, &client, events, None, None),
        )
        .await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }

    #[tokio::test]
    async fn unsubscribe_clears_state() {
        let (client_stream, mut server_stream) = duplex(4096);
        let (client_read, client_write) = tokio::io::split(client_stream);
        let (notification_tx, _) = broadcast::channel(256);
        let client = make_client(notification_tx, client_read, client_write);
        let mut sub_state = initial_subscription_state();

        let server_handle = tokio::spawn(async move {
            respond_to_subscribe(&mut server_stream).await;
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
        update_filter(&mut sub_state, &client, events, None, None)
            .await
            .unwrap();
        assert_eq!(sub_state.active_events.len(), 1);

        unsubscribe(&mut sub_state, &client).await.unwrap();
        assert!(sub_state.active_events.is_empty());
        assert!(sub_state.active_layers.is_none());
        assert!(sub_state.active_components.is_none());

        server_handle.await.unwrap();
    }
}
