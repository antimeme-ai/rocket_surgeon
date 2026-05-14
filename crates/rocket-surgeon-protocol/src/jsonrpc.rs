use serde::{Deserialize, Serialize};

use crate::errors::ErrorData;

pub const JSONRPC_VERSION: &str = "2.0";

// ---------------------------------------------------------------------------
// Request ID
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    Number(i64),
    String(String),
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub jsonrpc: String,
    pub id: RequestId,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl Request {
    #[must_use]
    pub fn new(id: RequestId, method: impl Into<String>, params: serde_json::Value) -> Self {
        let params = if params.is_null() { None } else { Some(params) };
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id,
            method: method.into(),
            params,
        }
    }
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Response {
    pub jsonrpc: String,
    pub id: RequestId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl Response {
    #[must_use]
    pub fn success(id: RequestId, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id,
            result: Some(result),
            error: None,
        }
    }

    #[must_use]
    pub fn error(id: RequestId, error: RpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

// ---------------------------------------------------------------------------
// Notification (no id, no response expected)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Notification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl Notification {
    #[must_use]
    pub fn new(method: impl Into<String>, params: serde_json::Value) -> Self {
        let params = if params.is_null() { None } else { Some(params) };
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            method: method.into(),
            params,
        }
    }
}

// ---------------------------------------------------------------------------
// Error object
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<ErrorData>,
}

impl RpcError {
    #[must_use]
    pub fn from_error_data(error_data: ErrorData) -> Self {
        Self {
            code: error_data
                .numeric_code
                .unwrap_or_else(|| error_data.error_code.numeric_code()),
            message: error_data.suggestion.clone(),
            data: Some(error_data),
        }
    }
}

pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const INTERNAL_ERROR: i32 = -32603;

// ---------------------------------------------------------------------------
// Incoming message (request or notification — determined by presence of id)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawMessage {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<RequestId>,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl RawMessage {
    #[must_use]
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }

    #[must_use]
    pub fn into_request(self) -> Option<Request> {
        self.id.map(|id| Request {
            jsonrpc: self.jsonrpc,
            id,
            method: self.method,
            params: self.params,
        })
    }

    #[must_use]
    pub fn into_notification(self) -> Option<Notification> {
        if self.id.is_some() {
            return None;
        }
        Some(Notification {
            jsonrpc: self.jsonrpc,
            method: self.method,
            params: self.params,
        })
    }
}
