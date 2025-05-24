// src/common.rs - Unified core infrastructure for cancellable operations

use serde_json::Value;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::utils::{ProxyError, Logger};

/// Unified wrapper for cancellable HTTP requests
pub struct CancellableRequest {
    client: reqwest::Client,
    token: CancellationToken,
    request_id: String,
    logger: Logger,
    timeout_seconds: u64,
}

impl CancellableRequest {
    pub fn new(client: reqwest::Client, token: CancellationToken, logger: Logger, timeout_seconds: u64) -> Self {
        let request_id = format!("req_{}", chrono::Utc::now().timestamp_millis());
        Self { client, token, request_id, logger, timeout_seconds }
    }

    /// Make a cancellable HTTP request that can be aborted mid-flight
    pub async fn make_request(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<Value>,
    ) -> Result<reqwest::Response, ProxyError> {
        let mut request_builder = self.client.request(method, url);

        if let Some(body) = body {
            request_builder = request_builder
                .header("Content-Type", "application/json")
                .json(&body);
        }

        request_builder = request_builder.timeout(Duration::from_secs(self.timeout_seconds));

        self.logger.log(&format!("🌐 [{}] Starting request to LM Studio: {}", self.request_id, url));

        let request_future = request_builder.send();

        // Use tokio::select to race between the request and cancellation
        tokio::select! {
            result = request_future => {
                match result {
                    Ok(response) => {
                        self.logger.log(&format!("✅ [{}] Request completed successfully", self.request_id));
                        Ok(response)
                    },
                    Err(err) => {
                        self.logger.log(&format!("❌ [{}] Request failed: {}", self.request_id, err));
                        Err(ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", err)))
                    }
                }
            }
            
            // Request was cancelled
            _ = self.token.cancelled() => {
                self.logger.log(&format!("🚫 [{}] HTTP request to LM Studio was cancelled by client disconnection", self.request_id));
                Err(ProxyError::request_cancelled())
            }
        }
    }
}

/// Generic helper for handling cancellable JSON responses
pub async fn handle_json_response(response: reqwest::Response, cancellation_token: CancellationToken) -> Result<Value, ProxyError> {
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let response_future = response.json::<Value>();

    tokio::select! {
        result = response_future => {
            result.map_err(|e| ProxyError::internal_server_error(&format!("Failed to parse LM Studio response: {}", e)))
        }
        _ = cancellation_token.cancelled() => {
            Err(ProxyError::request_cancelled())
        }
    }
}

/// Generic helper for handling cancellable text responses
pub async fn handle_text_response(response: reqwest::Response, cancellation_token: CancellationToken) -> Result<String, ProxyError> {
    if cancellation_token.is_cancelled() {
        return Err(ProxyError::request_cancelled());
    }

    let response_future = response.text();

    tokio::select! {
        result = response_future => {
            result.map_err(|e| ProxyError::internal_server_error(&format!("Failed to read LM Studio response: {}", e)))
        }
        _ = cancellation_token.cancelled() => {
            Err(ProxyError::request_cancelled())
        }
    }
}
