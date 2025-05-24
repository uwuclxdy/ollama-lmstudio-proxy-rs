// src/common.rs - Unified core infrastructure for cancellable operations

use serde_json::Value;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::utils::ProxyError;

/// Unified wrapper for cancellable HTTP requests
pub struct CancellableRequest {
    client: reqwest::Client,
    token: CancellationToken,
    request_id: String,
}

impl CancellableRequest {
    pub fn new(client: reqwest::Client, token: CancellationToken) -> Self {
        let request_id = format!("req_{}", chrono::Utc::now().timestamp_millis());
        Self { client, token, request_id }
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

        // Add a timeout to prevent hanging requests
        request_builder = request_builder.timeout(Duration::from_secs(300)); // 5 minutes max

        log::info!("🌐 [{}] Starting request to LM Studio: {}", self.request_id, url);

        let request_future = request_builder.send();

        // Use tokio::select to race between the request and cancellation
        tokio::select! {
            // Request completes normally
            result = request_future => {
                match result {
                    Ok(response) => {
                        log::info!("✅ [{}] Request completed successfully", self.request_id);
                        Ok(response)
                    },
                    Err(err) => {
                        log::warn!("❌ [{}] Request failed: {}", self.request_id, err);
                        Err(ProxyError::internal_server_error(&format!("Failed to reach LM Studio: {}", err)))
                    }
                }
            }
            // Request was cancelled - this is the key part!
            _ = self.token.cancelled() => {
                log::warn!("🚫 [{}] HTTP request to LM Studio was cancelled by client disconnection", self.request_id);
                // Return immediately - the HTTP request will be dropped and cancelled
                Err(ProxyError::request_cancelled())
            }
        }
    }
}

/// Generic helper for handling cancellable JSON responses
pub async fn handle_cancellable_json_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken,
) -> Result<Value, ProxyError> {
    // Check if cancelled before processing response
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
pub async fn handle_cancellable_text_response(
    response: reqwest::Response,
    cancellation_token: CancellationToken,
) -> Result<String, ProxyError> {
    // Check if cancelled before processing response
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
